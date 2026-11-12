"""Does a LoRA make Gemma write more per agent step? Measure it on the Mac (MLX), no vLLM needed.

Rebuilds real agent prompts from base-model traces (same 10 tasks as the Kaggle 10-task checks), cuts each
trace just before an assistant turn, and lets the model write that turn at several adapter strengths
(0 = base model). Counts the generated tokens (thinking + answer + tool call) until the turn ends.
Same sampling as the submission: temperature 0.2, top_p 0.95. Same seed for every strength.

Usage:
  .venv-mlx/bin/python scripts/verbosity_probe.py --adapter <peft dir> --out results/verbosity-run3.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft import convert  # noqa: E402
from train_lora_mlx import ROOT, TARGETS, to_template_messages  # noqa: E402

EVAL10 = ['fastapi_14349', 'fastapi_14360', 'rich_3006', 'rich_3043', 'rich_3067', 'rich_3718',
          'fastapi_13713', 'fastapi_14583', 'rich_4070', 'rich_3180']


def peft_to_mlx(path):
    """PEFT A [r, in], B [out, r]  ->  MLX lora_a [in, r], lora_b [r, out] (inverse of mlx_to_peft.py)."""
    import mlx.core as mx
    out = {}
    for k, v in mx.load(str(Path(path) / 'adapter_model.safetensors')).items():
        # base_model.model.model.language_model.layers.N.mlp.down_proj.lora_A.weight
        assert k.startswith('base_model.model.model.language_model.layers.') and k.endswith('.weight'), k
        mod, kind = k.removesuffix('.weight').rsplit('.', 1)
        name = 'language_model.model.' + mod.removeprefix('base_model.model.model.language_model.')
        out[f"{name}.{ {'lora_A': 'lora_a', 'lora_B': 'lora_b'}[kind]}"] = v.T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(Path('~/models/gemma-4-31b-mlx-4bit').expanduser()))
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--traces', default=str(ROOT / 'results/eval30-base-c5/results/traces'))
    ap.add_argument('--strengths', default='0,0.5,1')
    ap.add_argument('--turns', default='1,4,8', help='assistant turn numbers (1 = first) to regenerate')
    ap.add_argument('--max-tokens', type=int, default=3000)
    ap.add_argument('--max-prompt', type=int, default=14000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import mlx.core as mx
    _init = mx.distributed.init
    mx.distributed.init = lambda strict=False, backend='any': _init(strict=False, backend='ring')
    from mlx_lm import load, stream_generate
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
    # Check the weights really landed (load_weights with strict=False would silently skip bad names).
    k0 = next(iter(w))
    got = dict(__import__('mlx.utils', fromlist=['tree_flatten']).tree_flatten(model.parameters()))[k0]
    assert mx.allclose(got.astype(mx.float32), w[k0].astype(mx.float32)).item(), 'adapter weights not loaded'
    print(f'adapter: {len(loras)} LoRA layers, r {rank}, alpha {alpha}', flush=True)

    tools = json.load(open(ROOT / 'data/sft/tools.json'))
    kw = dict(tools=tools, enable_thinking=True, tokenize=True, return_dict=False)
    prompts = []
    for tid in EVAL10:
        msgs, _ = convert(json.load(open(Path(args.traces) / f'trace_{tid}.json')), tid, require_final=False)
        msgs = to_template_messages(msgs)
        a_idx = [i for i, m in enumerate(msgs) if m['role'] == 'assistant']
        for t in map(int, args.turns.split(',')):
            if t > len(a_idx):
                continue
            p = hf_tok.apply_chat_template(msgs[:a_idx[t - 1]], add_generation_prompt=True, **kw)
            if len(p) <= args.max_prompt:
                prompts.append((tid, t, p))
    print(f'{len(prompts)} prompts, lengths {min(len(p) for *_, p in prompts)}-{max(len(p) for *_, p in prompts)} tokens', flush=True)

    sampler = make_sampler(temp=0.2, top_p=0.95)
    done = set()
    out = Path(args.out)
    if out.exists():
        done = {(r['id'], r['turn'], r['strength']) for r in map(json.loads, open(out))}
    with open(out, 'a') as fh:
        for n, (tid, t, p) in enumerate(prompts):
            for f in map(float, args.strengths.split(',')):
                if (tid, t, f) in done:
                    continue
                for m in loras:
                    m.scale = alpha / rank * f
                mx.random.seed(1000 + n)
                t0, text, n_tok, reason = time.time(), [], 0, None
                for r in stream_generate(model, tok, p, max_tokens=args.max_tokens, sampler=sampler):
                    text.append(r.text)
                    n_tok = r.generation_tokens
                    reason = r.finish_reason
                    tps, ptps = r.generation_tps, r.prompt_tps
                rec = {'id': tid, 'turn': t, 'strength': f, 'prompt_tokens': len(p), 'gen_tokens': n_tok,
                       'finish': reason, 'seconds': round(time.time() - t0, 1), 'gen_tps': round(tps, 1),
                       'prompt_tps': round(ptps, 1), 'text': ''.join(text)}
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
                print(f"{tid:14s} turn {t} x{f:g}: {n_tok:5d} tokens ({reason}), prompt {len(p)}, "
                      f"{rec['seconds']}s, {rec['gen_tps']} tok/s", flush=True)


if __name__ == '__main__':
    main()
