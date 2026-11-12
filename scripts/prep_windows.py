"""Tokenize the training windows and the loss-gate windows once, locally, for GPU training elsewhere.

Uses exactly the windowing and masks of train_lora_mlx.py (same tokenizer and chat template), so a
cloud run trains on the same tokens as the MLX run. The gate windows are the ones compare_lora_loss.py
uses, plus more held-out runs, since loss checks are cheap on a big GPU.

Usage:
  .venv-mlx/bin/python scripts/prep_windows.py --tokenizer ~/models/gemma-4-31B-it-qat-unquantized
Writes data/sft/windows_<max_len>.json: {"train": [[tokens, mask], ...], "gate": [{"name", "kind", "tokens", "mask"}, ...]}
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_lora_loss import HELD_OUT_LEN, TRAIN_WINDOWS, held_out_examples  # noqa: E402
from train_lora_mlx import ROOT, encode, load_examples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--data', default=str(ROOT / 'data/sft/train_24.jsonl'))
    ap.add_argument('--max-len', type=int, default=16384)
    ap.add_argument('--held-out', type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(Path(args.tokenizer).expanduser()))
    tools = json.load(open(ROOT / 'data/sft/tools.json'))

    train = load_examples(args.data, tok, tools, args.max_len)
    by_id = {json.loads(l)['id']: json.loads(l) for l in open(args.data)}
    gate = []
    for tid, i in TRAIN_WINDOWS:
        toks, mask = encode(tok, by_id[tid], tools, args.max_len)[0][i]
        gate.append({'name': f'{tid}#{i}', 'kind': 'train', 'tokens': toks, 'mask': mask})
    for ex in held_out_examples(args.held_out):
        toks, mask = encode(tok, ex, tools, HELD_OUT_LEN)[0][0]
        gate.append({'name': ex['id'], 'kind': 'held-out', 'tokens': toks, 'mask': mask})

    out = ROOT / f'data/sft/windows_{args.max_len}.json'
    out.write_text(json.dumps({'max_len': args.max_len, 'train': train, 'gate': gate}))
    print(f'{len(train)} train windows, {sum(len(t) for t, _ in train)} tokens, '
          f'{sum(sum(m) for _, m in train)} trained; {len(gate)} gate windows -> {out} '
          f'({out.stat().st_size / 2 ** 20:.1f} MiB)')


if __name__ == '__main__':
    main()
