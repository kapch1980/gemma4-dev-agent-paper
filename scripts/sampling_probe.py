"""Does Gemma still write well-formed tool calls at temperature 1.0 + top_k 64 (v6a) vs 0.2 (b024)? MLX, base model.

Rebuilds real prompts from Kaggle base-model traces (agent steps 1, 4 and 8 of each task), lets the base model
write that step again under each sampling setting, and checks the raw output's tool call:
  ok        <|tool_call>call:NAME{...}<tool_call|> with a known tool and its required arguments
  malformed a call that is cut, has an unknown name, or lacks a required argument (the Exp 32 failure kinds)
  no_call   the step ends with text only
  cap       hit --max-tokens

Usage: .venv-mlx/bin/python scripts/sampling_probe.py --out results/sampling-probe-v6a.jsonl
"""
import argparse
import collections
import copy
import glob
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft import convert  # noqa: E402
from train_lora_mlx import ROOT, to_template_messages  # noqa: E402

REQUIRED = {'run_command': {'command'}, 'read_file': {'filepath'}, 'edit_file': {'filepath', 'old_string', 'new_string'},
            'write_file': {'filepath', 'content'}, 'get_status': set(), 'submit_patch': set()}
Q = '<|"|>'


def parse_args(s):
    """Parse Gemma 4 call arguments 'k:v,k:<|"|>str<|"|>,...' into a dict of keys; None if malformed."""
    keys, i = set(), 0
    while i < len(s):
        m = re.match(r'\s*([A-Za-z_]\w*)\s*:', s[i:])
        if not m:
            return None
        keys.add(m.group(1))
        i += m.end()
        if s.startswith(Q, i):
            j = s.find(Q, i + len(Q))
            if j < 0:
                return None
            i = j + len(Q)
        else:
            depth = 0
            while i < len(s) and not (depth == 0 and s[i] == ','):
                depth += s[i] in '[{'
                depth -= s[i] in ']}'
                i += 1
        if i < len(s):
            if s[i] != ',':
                return None
            i += 1
    return keys


def classify(text, hit_cap):
    m = re.search(r'<\|tool_call>call:(\w+)\{(.*)\}<tool_call\|>', text, re.S)
    if not m:
        if '<|tool_call>' in text or 'call:' in text.split('<channel|>')[-1]:
            return 'cap' if hit_cap else 'malformed', None
        return ('cap' if hit_cap else 'no_call'), None
    name, args = m.group(1), m.group(2)
    keys = parse_args(args)
    if name not in REQUIRED or keys is None or not REQUIRED[name] <= keys:
        return 'malformed', name
    return 'ok', name


def prompts(dirs, steps):
    out = []
    for d in dirs:
        for f in sorted(glob.glob(str(ROOT / d / 'results/traces/trace_*.json'))):
            tid = Path(f).stem.removeprefix('trace_')
            msgs, _ = convert(json.load(open(f)), tid, require_final=False)
            msgs = to_template_messages(msgs)
            a_idx = [i for i, m in enumerate(msgs) if m['role'] == 'assistant']
            for s in steps:
                if s <= len(a_idx):
                    out.append((f'{d.split("/")[-1]}:{tid}', s, msgs[:a_idx[s - 1]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(Path('~/models/gemma-4-31b-mlx-4bit').expanduser()))
    ap.add_argument('--traces', default='results/eval10-base-short')
    ap.add_argument('--steps', default='1,4,8')
    ap.add_argument('--limit', type=int, default=12, help='max prompts')
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--max-tokens', type=int, default=3000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import mlx.core as mx
    _init = mx.distributed.init
    mx.distributed.init = lambda strict=False, backend='any': _init(strict=False, backend='ring')
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(args.model)
    hf_tok = tok._tokenizer if hasattr(tok, '_tokenizer') else tok
    tools = json.load(open(ROOT / 'data/sft/tools.json'))
    kw = dict(tools=tools, enable_thinking=True, tokenize=True, return_dict=False)
    items = prompts(args.traces.split(','), [int(s) for s in args.steps.split(',')])
    # spread across tasks: round-robin by step so short and long prompts both appear
    items = sorted(items, key=lambda x: (x[1], x[0]))[:: max(1, len(items) // args.limit)][: args.limit]
    items = [(k, s, hf_tok.apply_chat_template(m, add_generation_prompt=True, **kw)) for k, s, m in items]
    print(f'{len(items)} prompts:', [(k, s, len(p)) for k, s, p in items], flush=True)

    settings = {'b024_t0.2': dict(temp=0.2, top_p=0.95), 'v6a_t1.0_k64': dict(temp=1.0, top_p=0.95, top_k=64)}
    out = Path(args.out)
    done = {(r['id'], r['step'], r['setting'], r['seed']) for r in map(json.loads, open(out))} if out.exists() else set()
    with open(out, 'a') as fh:
        for key, step, p in items:
            todo = [(sd, st) for sd in map(int, args.seeds.split(',')) for st in settings if (key, step, st, sd) not in done]
            if not todo:
                continue
            t0 = time.time()
            base = make_prompt_cache(model)
            for i in range(0, len(p) - 1, 2048):
                model(mx.array(p[i:min(i + 2048, len(p) - 1)])[None], cache=base)
                mx.eval([c.state for c in base])
            pre = round(time.time() - t0, 1)
            for seed, st in todo:
                mx.random.seed(seed * 1000 + step)
                cache = copy.deepcopy(base)
                t0, text, n = time.time(), '', 0
                for r in stream_generate(model, tok, p[-1:], max_tokens=args.max_tokens,
                                         sampler=make_sampler(**settings[st]), prompt_cache=cache):
                    text += r.text
                    n = r.generation_tokens
                    if '<tool_call|>' in text:
                        break
                del cache
                verdict, name = classify(text, n >= args.max_tokens - 2)
                rec = {'id': key, 'step': step, 'setting': st, 'seed': seed, 'prompt_tokens': len(p), 'gen_tokens': n,
                       'verdict': verdict, 'tool': name, 'seconds': round(time.time() - t0, 1), 'prefill_s': pre,
                       'tail': text[-600:]}
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
                print(f'{key:32s} step {step:2d} {st:13s} seed {seed}: {verdict:9s} {name or "":12s} {n:5d} tok {rec["seconds"]}s', flush=True)
            del base
            mx.clear_cache()
    rows = [json.loads(l) for l in open(out)]
    for st in settings:
        c = collections.Counter(r['verdict'] for r in rows if r['setting'] == st)
        print(st, dict(c))


if __name__ == '__main__':
    main()
