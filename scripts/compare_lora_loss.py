"""Sanity check a trained LoRA before submitting: loss with and without it on the same windows.

Training windows should get a lower loss with the adapter (it learned them). Windows from runs
it never saw (unsolved tasks, still Gemma's own outputs) should not get much worse; if they do,
the adapter has damaged the model's general behaviour and should not be submitted.

Usage:
  .venv-mlx/bin/python scripts/compare_lora_loss.py --model ~/models/gemma-4-31b-mlx-4bit --adapter adapters-mlx/run1
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft import convert  # noqa: E402
from train_lora_mlx import ROOT, batches, encode, masked_loss  # noqa: E402

# (task id, window index): the iter-23 loss spike (fastapi_9555 #1) plus two short end-of-run windows.
TRAIN_WINDOWS = [('fastapi_9555', 1), ('rich_2943', 1), ('rich_3521', 1)]
HELD_OUT_LEN = 8192


def held_out_examples(n):
    """First window of n unsolved runs, converted like the training data."""
    ids = [l.strip() for l in open(ROOT / 'results/unsolved_27.txt') if l.strip()]
    traces = {p.stem.removeprefix('trace_'): p for p in ROOT.glob('results/collect-*/results/traces/trace_*.json')}
    out = []
    for tid in ids:
        if tid not in traces:
            continue
        trace = json.load(open(traces[tid]))
        # Unsolved runs often stop on a tool result; cut back to the last assistant step.
        while trace['steps'] and trace['steps'][-1].get('source') != 'agent':
            trace['steps'].pop()
        try:
            msgs, _ = convert(trace, tid)
        except AssertionError:
            continue
        out.append({'id': tid, 'messages': msgs})
        if len(out) == n:
            break
    return out


def losses(model, windows):
    import mlx.core as mx
    res = []
    for name, (toks, mask) in windows:
        t = time.time()
        x, pos, y = next(batches([(toks, mask)], loop=False))
        loss, _ = masked_loss(model, x, pos, y)
        mx.eval(loss)
        res.append(loss.item())
        print(f'  {name:22s} len {len(toks):6d} loss {res[-1]:.4f}  ({time.time() - t:.0f}s)', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--held-out', type=int, default=2)
    args = ap.parse_args()

    import mlx.core as mx
    _init = mx.distributed.init
    mx.distributed.init = lambda strict=False, backend='any': _init(strict=False, backend='ring')
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    cfg = json.loads((Path(args.adapter) / 'adapter_config.json').read_text())
    tools = json.load(open(ROOT / 'data/sft/tools.json'))
    model, tok = load(args.model)
    tok = tok._tokenizer if hasattr(tok, '_tokenizer') else tok

    train = {json.loads(l)['id']: json.loads(l) for l in open(ROOT / 'data/sft/train_24.jsonl')}
    train_w = [(f'train {tid}#{i}', encode(tok, train[tid], tools, cfg['max_len'])[0][i]) for tid, i in TRAIN_WINDOWS]
    held_w = [(f'held-out {ex["id"]}', encode(tok, ex, tools, HELD_OUT_LEN)[0][0]) for ex in held_out_examples(args.held_out)]
    windows = train_w + held_w

    model.eval()
    print('base model:', flush=True)
    base = losses(model, windows)
    linear_to_lora_layers(model, len(model.layers), cfg['lora_parameters'])
    model.load_weights(str(Path(args.adapter) / 'adapters.safetensors'), strict=False)
    model.eval()
    print('with LoRA:', flush=True)
    lora = losses(model, windows)

    n = len(train_w)
    mean = lambda v: sum(v) / len(v)
    tb, tl, hb, hl = mean(base[:n]), mean(lora[:n]), mean(base[n:]), mean(lora[n:])
    print(f'\ntrain windows    base {tb:.4f}  lora {tl:.4f}')
    print(f'held-out windows base {hb:.4f}  lora {hl:.4f}  ({(hl / hb - 1) * 100:+.1f}%)')
    ok = all(math.isfinite(v) for v in lora) and tl < tb and hl <= hb * 1.15
    print('VERDICT:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
