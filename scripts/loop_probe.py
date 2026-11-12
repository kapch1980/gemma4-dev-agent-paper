"""Does a frequency/presence penalty stop run3's runaway loops? Test it on the Mac (MLX), free.

Takes the run3 Kaggle traces (Exp 29, results/eval10-run3), finds the first agent step in each task that ran
to the ~4,096 thinking budget (a loop, Exp 30), rebuilds the exact prompt before that step, and lets the run3
adapter write that step again with and without a penalty. Same temperature 0.2 / top_p 0.95 as the submission.

The penalties copy vLLM's OpenAI-style ones (they only count tokens generated in this step, not the prompt):
    logit[t] -= frequency_penalty * count[t] + presence_penalty * (count[t] > 0)
We checked (Exp 32) that the grader's harness (adk_submission 0.2.13 + ADK LiteLlm) passes both through to vLLM.

A generation is stopped early and marked "loop" once it is clearly repeating (one line >= 8 times in the last
4,000 characters, or the last 3,000 characters compress below 15%). Otherwise it runs until the turn ends or
--max-tokens (about the Kaggle thinking budget plus a tool call).

Usage:
  .venv-mlx/bin/python scripts/loop_probe.py --adapter adapters-peft/run3-full --out results/loop-probe-run3.jsonl
"""
import argparse
import collections
import copy
import glob
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft import convert  # noqa: E402
from train_lora_mlx import ROOT, TARGETS, to_template_messages  # noqa: E402
from verbosity_probe import peft_to_mlx  # noqa: E402


def is_loop(text):
    tail = text[-4000:]
    lines = [l.strip() for l in tail.splitlines() if len(l.strip()) >= 4]
    if lines and collections.Counter(lines).most_common(1)[0][1] >= 8:
        return True
    t3 = text[-3000:]
    return len(t3) >= 3000 and len(zlib.compress(t3.encode())) / len(t3.encode()) < 0.15


def make_penalty(prompt_len, freq, pres):
    def proc(tokens, logits):
        gen = np.array(tokens[prompt_len:]).astype(np.int64)
        if gen.size == 0:
            return logits
        import mlx.core as mx
        counts = np.bincount(gen, minlength=logits.shape[-1])[: logits.shape[-1]].astype(np.float32)
        pen = freq * counts + pres * (counts > 0)
        return logits - mx.array(pen)[None, :].astype(logits.dtype)
    return proc


def loop_prompts(traces_dir, min_tokens):
    out = []
    for f in sorted(glob.glob(str(Path(traces_dir) / 'trace_*.json'))):
        tid = Path(f).stem.removeprefix('trace_')
        t = json.load(open(f))
        steps = [s for s in t['steps'] if s.get('source') == 'agent' and s.get('metrics')]
        big = [i for i, s in enumerate(steps) if s['metrics']['completion_tokens'] >= min_tokens]
        if not big:
            continue
        msgs, _ = convert(t, tid, require_final=False)
        msgs = to_template_messages(msgs)
        a_idx = [i for i, m in enumerate(msgs) if m['role'] == 'assistant']
        i = big[0]
        out.append((tid, i + 1, msgs[:a_idx[i]], steps[i]['metrics']['completion_tokens']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(Path('~/models/gemma-4-31b-mlx-4bit').expanduser()))
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--traces', default=str(ROOT / 'results/eval10-run3/results/traces'))
    ap.add_argument('--min-tokens', type=int, default=3000, help='a Kaggle step this long counts as a loop')
    ap.add_argument('--settings', default='none:0:0,freq0.3:0.3:0', help='name:frequency:presence,...')
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--max-tokens', type=int, default=4300)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import mlx.core as mx
    _init = mx.distributed.init
    mx.distributed.init = lambda strict=False, backend='any': _init(strict=False, backend='ring')
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers

    cfg = json.loads((Path(args.adapter) / 'adapter_config.json').read_text())
    rank, alpha = cfg['r'], cfg['lora_alpha']
    model, tok = load(args.model)
    hf_tok = tok._tokenizer if hasattr(tok, '_tokenizer') else tok
    model.freeze()
    linear_to_lora_layers(model, len(model.layers), {'rank': rank, 'scale': alpha / rank, 'dropout': 0.0, 'keys': TARGETS})
    w = peft_to_mlx(args.adapter)
    model.load_weights(list(w.items()), strict=False)
    loras = [m for _, m in model.named_modules() if hasattr(m, 'lora_a')]
    assert len(loras) * 2 == len(w), (len(loras), len(w))
    print(f'adapter: {len(loras)} LoRA layers, r {rank}, alpha {alpha} (full strength)', flush=True)

    tools = json.load(open(ROOT / 'data/sft/tools.json'))
    kw = dict(tools=tools, enable_thinking=True, tokenize=True, return_dict=False)
    items = [(tid, turn, hf_tok.apply_chat_template(m, add_generation_prompt=True, **kw), kt)
             for tid, turn, m, kt in loop_prompts(args.traces, args.min_tokens)]
    print(f'{len(items)} loop steps:', [(tid, turn, len(p), kt) for tid, turn, p, kt in items], flush=True)

    settings = [(n, float(f), float(p)) for n, f, p in (s.split(':') for s in args.settings.split(','))]
    sampler = make_sampler(temp=0.2, top_p=0.95)
    out = Path(args.out)
    done = {(r['id'], r['setting'], r['seed']) for r in map(json.loads, open(out))} if out.exists() else set()
    with open(out, 'a') as fh:
        for tid, turn, p, kaggle_tokens in items:
            todo = [(seed, st) for seed in map(int, args.seeds.split(',')) for st in settings
                    if (tid, st[0], seed) not in done]
            if not todo:
                continue
            # Read the long prompt once (all but its last token) and reuse a copy of that cache for every run.
            t0 = time.time()
            base_cache = make_prompt_cache(model)
            for i in range(0, len(p) - 1, 2048):
                model(mx.array(p[i:min(i + 2048, len(p) - 1)])[None], cache=base_cache)
                mx.eval([c.state for c in base_cache])
            prefill_s = round(time.time() - t0, 1)
            for seed, (name, freq, pres) in todo:
                mx.random.seed(seed * 1000 + turn)
                cache = copy.deepcopy(base_cache)
                # Only the last prompt token goes through generate, so every token the processor sees after the
                # first one was generated in this step: the same thing vLLM's penalties count.
                procs = [make_penalty(1, freq, pres)] if (freq or pres) else None
                t0, text, n_tok, reason, looped = time.time(), '', 0, None, False
                for r in stream_generate(model, tok, p[-1:], max_tokens=args.max_tokens, sampler=sampler,
                                         logits_processors=procs, prompt_cache=cache):
                    text += r.text
                    n_tok, reason = r.generation_tokens, r.finish_reason
                    if n_tok % 100 == 0 and n_tok >= 600 and is_loop(text):
                        looped, reason = True, 'loop'
                        break
                del cache
                rec = {'id': tid, 'turn': turn, 'setting': name, 'frequency_penalty': freq,
                       'presence_penalty': pres, 'seed': seed, 'prompt_tokens': len(p),
                       'kaggle_tokens': kaggle_tokens, 'gen_tokens': n_tok, 'finish': reason,
                       'looped': looped or (n_tok >= args.max_tokens - 5 and is_loop(text)),
                       'hit_cap': n_tok >= args.max_tokens - 5, 'seconds': round(time.time() - t0, 1),
                       'prefill_seconds': prefill_s, 'text': text}
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
                print(f"seed {seed} {tid:14s} turn {turn:2d} {name:8s}: {n_tok:5d} tokens, {reason}"
                      f"{' LOOP' if rec['looped'] else ''}, {rec['seconds']}s", flush=True)
            del base_cache
            mx.clear_cache()

if __name__ == '__main__':
    main()
