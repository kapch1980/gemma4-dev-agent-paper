"""Train a LoRA with PyTorch + PEFT on a CUDA GPU (the same recipe as train_lora_mlx.py), then gate it.

Input is data/sft/windows_<len>.json from prep_windows.py: the exact token windows and assistant
masks of the MLX run, in the same shuffled order. Output is a PEFT adapter in the layout vLLM loads
(same keys and config as scripts/mlx_to_peft.py), saved at a few checkpoints, plus a loss gate:
base vs LoRA loss on training windows and on windows from runs it never saw, at several strengths.

Usage (normally via scripts/train_lora_modal.py):
  python scripts/train_lora_torch.py --model /models/gemma --windows windows_16384.json --out out/run2
On 4x L4 (Kaggle, 24 GB each) the bf16 model doesn't fit with 16k-token activations, so --quant4 loads
the frozen base in 4-bit NF4 (bitsandbytes) spread over all GPUs; the LoRA itself still trains in bf16.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
LOSS_CHUNK = 1024
SERVED_BASE = 'google/gemma-4-31b-it-qat-w4a16-ct'
ATTN_CHUNK = 1024  # query rows per attention slice (chunked_sdpa)
MLP_CHUNK = 2048  # tokens per MLP slice (chunk_mlps)


def register_chunked_sdpa():
    """Register attn_implementation='chunked_sdpa': SDPA computed in slices of ATTN_CHUNK query rows.

    Gemma 4's global layers use head_dim 512, which the flash kernels don't support, and every layer gets an
    explicit mask, so SDPA falls back to the math kernel and builds the full (heads x L x L) score table:
    ~14 GB for one layer at 16k tokens, more than a 22 GB L4 has left. Slicing the queries keeps one slice's
    table in memory; with gradients on, each slice is checkpointed so backward also recomputes one slice at a
    time. The result is the same attention, just computed in pieces.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def chunked(module, query, key, value, attention_mask, **kw):
        q_len = query.shape[2]
        causal_none = attention_mask is None and key.shape[2] == q_len and getattr(module, 'is_causal', True)
        if q_len <= ATTN_CHUNK or (attention_mask is None and not causal_none):
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

        def piece(q, k, v, m):
            return sdpa_attention_forward(module, q, k, v, m, **kw)[0]

        outs = []
        for s in range(0, q_len, ATTN_CHUNK):
            e = min(s + ATTN_CHUNK, q_len)
            q, k, v = query[:, :, s:e], key, value
            if causal_none:
                # Plain causal layer (transformers passes no mask): rows s..e-1 see keys 0..row, so keys
                # past e are never used; build the boolean mask (True = attend) for this slice.
                k, v = key[:, :, :e], value[:, :, :e]
                rows = torch.arange(s, e, device=query.device)[:, None]
                m = (torch.arange(e, device=query.device)[None, :] <= rows)[None, None]
            else:
                m = attention_mask[..., s:e, :] if attention_mask.shape[-2] == q_len else attention_mask
            if torch.is_grad_enabled():
                outs.append(checkpoint(piece, q, k, v, m, use_reentrant=False))
            else:
                outs.append(piece(q, k, v, m))
        return torch.cat(outs, dim=1), None  # sdpa output is (batch, q_len, heads, head_dim)

    AttentionInterface.register('chunked_sdpa', chunked)
    # Without this, transformers builds no mask for an unknown name and sliding layers lose their window.
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
    AttentionMaskInterface.register('chunked_sdpa', ALL_MASK_ATTENTION_FUNCTIONS['sdpa'])


def chunk_mlps(model):
    """Run every text MLP in slices of MLP_CHUNK tokens, each checkpointed.

    The MLP treats each token independently, so slicing gives exactly the same result. Without it a 16k
    window makes several (tokens x 21504) tensors per layer during backward, including a float32 copy of
    the down_proj input for the LoRA (1.31 GiB each), which overflowed a 22 GB L4 (training v2).
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    n = 0
    for mod in model.modules():
        if type(mod).__name__ != 'Gemma4TextMLP':
            continue
        orig = mod.forward

        def fwd(x, orig=orig):
            if x.dim() != 3 or x.shape[1] <= MLP_CHUNK:
                return orig(x)
            parts = []
            for s in range(0, x.shape[1], MLP_CHUNK):
                xs = x[:, s:s + MLP_CHUNK]
                parts.append(checkpoint(orig, xs, use_reentrant=False) if torch.is_grad_enabled() else orig(xs))
            return torch.cat(parts, dim=1)

        mod.forward = fwd
        n += 1
    return n


def gpu_report():
    import torch
    return ' '.join(f'{torch.cuda.memory_allocated(d) / 2**30:.1f}/{torch.cuda.max_memory_allocated(d) / 2**30:.1f}'
                    for d in range(torch.cuda.device_count()))


def masked_loss(model, toks, mask, device):
    """Mean cross-entropy on assistant tokens only, without materialising logits for every position.

    The vocabulary is 262k, so full logits for a 16k window would take ~17 GB. Take the hidden states
    at the scored positions and project them in checkpointed chunks instead.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    base = model.get_base_model()
    # With the model split over several GPUs, inputs go to the embedding's GPU and the loss is
    # computed wherever the last layer put the hidden states.
    x = torch.tensor(toks[:-1], device=base.get_input_embeddings().weight.device)[None]
    h = base.model.language_model(input_ids=x, use_cache=False).last_hidden_state[0]
    pos = torch.tensor([j for j in range(len(toks) - 1) if mask[j + 1]], dtype=torch.long, device=h.device)
    y = torch.tensor(toks, device=h.device)[1:][pos]
    h = h[pos]
    cap = base.config.get_text_config().final_logit_softcapping

    def chunk_ce(hc, yc):
        logits = base.lm_head(hc).float()
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return F.cross_entropy(logits, yc.to(logits.device), reduction='sum')

    total = sum(checkpoint(chunk_ce, h[s:s + LOSS_CHUNK], y[s:s + LOSS_CHUNK], use_reentrant=False)
                for s in range(0, len(pos), LOSS_CHUNK))
    return total / len(pos), len(pos)


def set_strength(model, factor):
    """Scale every LoRA update by factor (0 = base model, 1 = as trained); same as serving alpha*factor."""
    for m in model.modules():
        if hasattr(m, 'lora_alpha') and isinstance(getattr(m, 'scaling', None), dict):
            for k in m.scaling:
                m.scaling[k] = m.lora_alpha[k] / m.r[k] * factor


def save_adapter(model, out, rank, alpha):
    """Write adapter_model.safetensors (bf16) + adapter_config.json exactly like mlx_to_peft.py."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    modules = sorted({k.split('.lora_')[0].rsplit('.', 1)[1] for k in sd})
    (out / 'adapter_config.json').write_text(json.dumps({
        'base_model_name_or_path': SERVED_BASE, 'bias': 'none', 'fan_in_fan_out': False,
        'inference_mode': True, 'init_lora_weights': True, 'layers_to_transform': None,
        'lora_alpha': int(alpha) if float(alpha).is_integer() else alpha,
        'lora_dropout': 0.0, 'modules_to_save': None, 'peft_type': 'LORA', 'r': rank,
        'target_modules': modules, 'task_type': 'CAUSAL_LM', 'use_dora': False, 'use_rslora': False,
    }, indent=2))
    return len(sd)


def gate(model, windows, device, strengths=(0.0, 0.5, 1.0)):
    """Loss per gate window at each strength; PASS if train loss drops and held-out rises <= 15%."""
    import torch
    model.eval()
    res = {}
    with torch.no_grad():
        for f in strengths:
            set_strength(model, f)
            res[f] = [masked_loss(model, w['tokens'], w['mask'], device)[0].item() for w in windows]
    set_strength(model, 1.0)
    model.train()
    mean = lambda v: sum(v) / len(v)
    kinds = [w['kind'] for w in windows]
    pick = lambda v, k: [x for x, kk in zip(v, kinds) if kk == k]
    lines, verdict = [], {}
    for i, w in enumerate(windows):
        lines.append(f"  {w['kind']:8s} {w['name']:16s} len {len(w['tokens']):6d} " +
                     ' '.join(f'x{f:g} {res[f][i]:.4f}' for f in strengths))
    tb, hb = mean(pick(res[0.0], 'train')), mean(pick(res[0.0], 'held-out'))
    for f in strengths[1:]:
        tl, hl = mean(pick(res[f], 'train')), mean(pick(res[f], 'held-out'))
        ok = all(math.isfinite(v) for v in res[f]) and tl < tb and hl <= hb * 1.15
        verdict[f] = ok
        lines.append(f'  strength x{f:g}: train {tb:.4f} -> {tl:.4f}, held-out {hb:.4f} -> {hl:.4f} '
                     f'({(hl / hb - 1) * 100:+.1f}%)  {"PASS" if ok else "FAIL"}')
    return '\n'.join(lines), verdict


def peak_gb():
    import torch
    if not torch.cuda.is_available():
        return 0
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9


def train(model_dir, windows_path, out, rank=16, alpha=32, lr=5e-5, epochs=1.0, saves=2, limit=0, log=print,
          quant4=False, gpu_gib=20):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma4ForConditionalGeneration

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(out)
    data = json.load(open(windows_path))
    train_w, gate_w = data['train'], data['gate']
    if limit:
        train_w = sorted(train_w, key=lambda w: -len(w[0]))[:limit]  # longest first: worst-case memory
    t0 = time.time()
    if quant4:
        register_chunked_sdpa()
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        max_memory = {i: f'{gpu_gib}GiB' for i in range(torch.cuda.device_count())}
        model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='auto',
                                                               max_memory=max_memory, quantization_config=bnb,
                                                               attn_implementation='chunked_sdpa')
    else:
        model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=device,
                                                               attn_implementation='sdpa')
    log(f'loaded model in {time.time() - t0:.0f}s')
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()  # frozen embeddings would otherwise cut the checkpointed graph
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias='none',
                     target_modules=r'.*language_model.*\.(' + '|'.join(TARGETS) + ')')
    model = get_peft_model(model, cfg)
    log(f'MLPs sliced: {chunk_mlps(model)}')
    if device == 'cuda':
        dm = getattr(model.get_base_model(), 'hf_device_map', {}) or {}
        per = {}
        for k, v in dm.items():
            per[v] = per.get(v, 0) + (1 if '.layers.' in k else 0)
        log(f'device map: decoder layers per device {per}; GiB allocated/peak per GPU after load: {gpu_report()}')
    params = [p for p in model.parameters() if p.requires_grad]
    log(f'trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f}M (dtype {params[0].dtype}), '
        f'windows: {len(train_w)}, device {device}')

    iters = max(1, math.ceil(epochs * len(train_w)))
    save_at = sorted({math.ceil(iters * (k + 1) / saves) for k in range(saves)})
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    # Cosine decay from lr to lr/10 over the run, as in the MLX run.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, iters) / iters)))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'train_config.json').write_text(json.dumps({'rank': rank, 'alpha': alpha, 'lr': lr, 'epochs': epochs,
                                                       'iters': iters, 'windows': len(train_w), 'save_at': save_at}))

    def order():
        rng = random.Random(0)  # same shuffle as train_lora_mlx.batches
        while True:
            idx = list(range(len(train_w)))
            rng.shuffle(idx)
            yield from idx

    model.train()
    if device == 'cuda':
        # Memory pre-check: forward + backward on the longest window, no optimizer step, gradients dropped.
        # A window that doesn't fit fails here in minutes instead of partway through the run.
        toks, mask = max(train_w, key=lambda w: len(w[0]))
        t = time.time()
        try:
            loss, n = masked_loss(model, toks, mask, device)
            loss.backward()
        except torch.OutOfMemoryError:
            log(f'pre-check OUT OF MEMORY on len {len(toks)}; GiB allocated/peak per GPU: {gpu_report()}')
            raise
        opt.zero_grad(set_to_none=True)
        per_gpu = [torch.cuda.max_memory_allocated(d) / 2**30 for d in range(torch.cuda.device_count())]
        log(f'pre-check longest window len {len(toks)}: {time.time() - t:.0f}s, peak per GPU (GiB) '
            + ' '.join(f'{g:.1f}' for g in per_gpu))
    start = time.time()
    for it, i in zip(range(1, iters + 1), order()):
        t = time.time()
        if device == 'cuda':
            for d in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(d)
        toks, mask = train_w[i]
        loss, n = masked_loss(model, toks, mask, device)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        peak = peak_gb()
        log(f'iter {it}/{iters} loss {loss.item():.4f} len {len(toks)} trained {n} '
            f'{len(toks) / (time.time() - t):.0f} tok/s peak {peak:.1f} GB elapsed {(time.time() - start) / 60:.1f} min')
        if it in save_at:
            for f, name in ((1.0, 'full'), (0.5, 'half')):
                n_t = save_adapter(model, out / f'iter{it}-{name}' / 'main_lora', rank, alpha * f)
            report, verdict = gate(model, gate_w, device)
            (out / f'iter{it}-gate.txt').write_text(report + '\n')
            log(f'saved iter {it} ({n_t} tensors); gate:\n{report}')
    log('done')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--windows', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=float, default=32)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--epochs', type=float, default=1)
    ap.add_argument('--saves', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--quant4', action='store_true', help='4-bit base split over all GPUs (Kaggle 4x L4)')
    ap.add_argument('--gpu-gib', type=int, default=20)
    a = ap.parse_args()
    train(a.model, a.windows, a.out, a.rank, a.alpha, a.lr, a.epochs, a.saves, a.limit,
          log=lambda s: print(s, flush=True), quant4=a.quant4, gpu_gib=a.gpu_gib)


if __name__ == '__main__':
    main()
