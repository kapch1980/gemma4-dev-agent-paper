"""Train a LoRA on Gemma's own successful agent runs with MLX (Apple Silicon).

Loss is computed on every assistant turn (reasoning + tool call), not on the system
prompt, the task, or tool outputs. mlx-lm's built-in mask only covers the last turn,
so this script builds per-token masks itself and plugs them into mlx-lm's trainer.

Usage:
  # 1. tokenize and report lengths/masks only (no weights needed)
  .venv-mlx/bin/python scripts/train_lora_mlx.py --check-data --model ~/models/gemma-4-31B-it-qat-unquantized
  # 2. train
  .venv-mlx/bin/python scripts/train_lora_mlx.py --model ~/models/gemma-4-31b-mlx-4bit --out adapters-mlx/run1
"""
import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
           'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']


def to_template_messages(messages):
    """Trace reasoning is logged as the assistant message; Gemma's template renders it as a thought.

    The last assistant message is the visible final answer. Consecutive assistant messages
    without tool calls are one model turn (thought, then answer), so they are merged.
    """
    out = []
    last = len(messages) - 1
    for i, m in enumerate(messages):
        if m['role'] == 'assistant':
            if i == last:
                m = {**m}
            else:
                m = {**m, 'reasoning_content': m['content'], 'content': ''}
            for tc in m.get('tool_calls', []):
                # The template wants arguments as a mapping, not a JSON string.
                tc['function']['arguments'] = json.loads(tc['function']['arguments'])
            prev = out[-1] if out else None
            if prev and prev['role'] == 'assistant' and not prev.get('tool_calls'):
                prev['reasoning_content'] = '\n'.join(filter(None, [prev.get('reasoning_content'), m.get('reasoning_content')]))
                prev['content'] = m.get('content', '')
                if m.get('tool_calls'):
                    prev['tool_calls'] = m['tool_calls']
                continue
        out.append(m)
    return out


def encode(tokenizer, example, tools, max_len):
    """Return a list of (tokens, mask) windows; mask[i] = 1 if token i is produced by the assistant.

    Runs longer than max_len are split at assistant-turn boundaries. Every window starts with the
    system prompt and the task, then continues from a later turn, much like the harness's own
    context compaction drops older turns. Each assistant turn is trained exactly once.
    """
    msgs = to_template_messages(json.loads(json.dumps(example['messages'])))
    kw = dict(tools=tools, enable_thinking=True, tokenize=True, return_dict=False)
    full = tokenizer.apply_chat_template(msgs, **kw)
    bounds = []  # bounds[i] = token index where message i ends
    for i in range(len(msgs)):
        prefix = tokenizer.apply_chat_template(msgs[:i + 1], **kw)
        assert full[:len(prefix)] == prefix, f"{example['id']}: template is not prefix-stable at message {i}"
        bounds.append(len(prefix))
    mask = [0] * len(full)
    for i, m in enumerate(msgs):
        if m['role'] == 'assistant':
            start = len(tokenizer.apply_chat_template(msgs[:i], add_generation_prompt=True, **kw))
            for j in range(start, bounds[i]):
                mask[j] = 1
    header = bounds[1]  # system + task
    # '<|turn>model\n': later windows start mid-way through the model's turn, so reopen it.
    turn_open = tokenizer.apply_chat_template(msgs[:2], add_generation_prompt=True, **kw)[header:]
    windows, a = [], 2
    while a < len(msgs):
        b = a
        while b + 1 < len(msgs) and header + len(turn_open) + bounds[b + 1] - bounds[a - 1] <= max_len:
            b += 1
        # end each window after an assistant turn's tool result, so the next window starts with a turn
        while b + 1 < len(msgs) and msgs[b + 1]['role'] != 'assistant' and b > a:
            b -= 1
        seg = slice(bounds[a - 1], bounds[b])
        lead = full[:header] if a == 2 else full[:header] + turn_open
        toks = (lead + full[seg])[:max_len]
        msk = ([0] * len(lead) + mask[seg])[:max_len]
        windows.append((toks, msk))
        a = b + 1
    return windows, len(full), sum(mask)


def load_examples(path, tokenizer, tools, max_len):
    data = []
    for line in open(path):
        ex = json.loads(line)
        windows, n, n_train = encode(tokenizer, ex, tools, max_len)
        kept = sum(sum(m) for _, m in windows)
        print(f"{ex['id']:16s} tokens={n:6d} assistant={n_train:5d} windows={len(windows)} "
              f"lengths={[len(t) for t, _ in windows]}" + (f"  LOST {n_train - kept} assistant tokens" if kept < n_train else ''))
        data.extend(windows)
    return data


def batches(data, loop=True, seed=0):
    """Yield (inputs, positions, targets): only positions whose next token is the assistant's are scored."""
    import mlx.core as mx
    rng = random.Random(seed)
    while True:
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in order:
            toks, mask = data[i]
            pos = [j for j in range(len(toks) - 1) if mask[j + 1]]
            yield mx.array(toks[:-1])[None], mx.array(pos), mx.array([toks[j + 1] for j in pos])
        if not loop:
            return


LOSS_CHUNK = 1024


def masked_loss(model, x, pos, y):
    """Cross-entropy on assistant tokens only, without materialising the full logits.

    Gemma's vocabulary is 262k, so logits for a 16k window would need ~17 GB. Instead take the
    hidden states at the scored positions and project them in chunks; each chunk is checkpointed
    so its logits are recomputed in the backward pass rather than kept in memory.
    """
    import mlx.core as mx
    import mlx.nn as nn
    lm = model.language_model
    h = lm.model(x)[0][pos]

    def chunk_ce(yc):
        # Targets are closed over, not passed in: mx.checkpoint differentiates every argument.
        def f(hc):
            logits = lm.model.embed_tokens.as_linear(hc).astype(mx.float32)
            if lm.final_logit_softcapping is not None:
                logits = mx.tanh(logits / lm.final_logit_softcapping) * lm.final_logit_softcapping
            return nn.losses.cross_entropy(logits, yc).sum()
        return mx.checkpoint(f)

    n = y.shape[0]
    total = sum(chunk_ce(y[s:s + LOSS_CHUNK])(h[s:s + LOSS_CHUNK]) for s in range(0, n, LOSS_CHUNK))
    return total / n, mx.array(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', default=str(ROOT / 'data/sft/train_24.jsonl'))
    ap.add_argument('--tools', default=str(ROOT / 'data/sft/tools.json'))
    ap.add_argument('--out', default=str(ROOT / 'adapters-mlx/run1'))
    ap.add_argument('--max-len', type=int, default=16384)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=float, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=float, default=2)
    ap.add_argument('--limit', type=int, default=0, help='use only the first N examples (speed/memory test)')
    ap.add_argument('--check-data', action='store_true')
    args = ap.parse_args()

    tools = json.load(open(args.tools))
    if args.check_data:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        data = load_examples(args.data, tok, tools, args.max_len)
        total = sum(len(t) for t, _ in data)
        print(f'{len(data)} windows, {total} tokens, {sum(sum(m) for _, m in data)} trained tokens')
        return

    import mlx.core as mx
    import mlx.optimizers as optim

    # Single machine: skip MPI (Anaconda ships MPICH, which aborts MLX's MPI backend).
    _init = mx.distributed.init
    mx.distributed.init = lambda strict=False, backend='any': _init(strict=False, backend='ring')
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten
    import mlx.nn as nn
    import time

    model, tok = load(args.model)
    data = load_examples(args.data, tok._tokenizer if hasattr(tok, '_tokenizer') else tok, tools, args.max_len)
    if args.limit:
        data = data[:args.limit]
    model.freeze()
    lora_cfg = {'rank': args.rank, 'scale': args.alpha / args.rank, 'dropout': 0.0, 'keys': TARGETS}
    linear_to_lora_layers(model, len(model.layers), lora_cfg)
    grad_checkpoint(model.layers[0])  # recompute each layer in the backward pass instead of storing it
    n_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f'trainable parameters: {n_params / 1e6:.1f}M, windows: {len(data)}', flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    iters = max(1, math.ceil(args.epochs * len(data)))
    (out / 'adapter_config.json').write_text(json.dumps({
        'fine_tune_type': 'lora', 'num_layers': len(model.layers), 'lora_parameters': lora_cfg,
        'model': args.model, 'data': args.data, 'max_len': args.max_len, 'lr': args.lr, 'iters': iters,
    }, indent=2))
    opt = optim.AdamW(learning_rate=optim.cosine_decay(args.lr, iters, end=args.lr * 0.1), weight_decay=0.0)
    loss_and_grad = nn.value_and_grad(model, masked_loss)
    model.train()

    def save(name):
        mx.save_safetensors(str(out / name), dict(tree_flatten(model.trainable_parameters())))

    start = time.time()
    for it, batch in zip(range(1, iters + 1), batches(data)):
        t = time.time()
        (loss, ntok), grads = loss_and_grad(model, *batch)
        opt.update(model, grads)
        mx.eval(loss, model.trainable_parameters(), opt.state)
        L = batch[0].shape[1]
        print(f'iter {it}/{iters} loss {loss.item():.4f} len {L} trained {ntok.item()} '
              f'{L / (time.time() - t):.0f} tok/s peak {mx.get_peak_memory() / 1e9:.1f} GB '
              f'elapsed {(time.time() - start) / 3600:.2f} h', flush=True)
        if it % len(data) == 0 or it == iters:
            save('adapters.safetensors')
    print('done', flush=True)


if __name__ == '__main__':
    main()
