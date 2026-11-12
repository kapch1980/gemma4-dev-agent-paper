"""Convert an MLX LoRA adapter (scripts/train_lora_mlx.py) to a PEFT adapter vLLM can load.

Usage: .venv-mlx/bin/python scripts/mlx_to_peft.py adapters-mlx/run1 submission_adapters/main_lora

MLX computes  y = W x + scale * (x @ lora_a) @ lora_b   with lora_a [in, r], lora_b [r, out].
PEFT computes y = W x + (alpha / r) * B (A x)           with A [r, in],     B [out, r].
So A = lora_a.T, B = lora_b.T and alpha = scale * r. Weights are saved in bfloat16.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx


def main(src, dst):
    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / 'adapter_config.json').read_text())['lora_parameters']
    rank, scale = cfg['rank'], cfg['scale']
    weights = mx.load(str(src / 'adapters.safetensors'))
    out, modules = {}, set()
    for k, v in weights.items():
        # language_model.model.layers.N.self_attn.q_proj.lora_a
        assert k.startswith('language_model.model.layers.'), k
        path, kind = k.rsplit('.', 1)
        base = 'base_model.model.model.language_model.' + path.removeprefix('language_model.model.')
        name = {'lora_a': 'lora_A', 'lora_b': 'lora_B'}[kind]
        out[f'{base}.{name}.weight'] = v.T.astype(mx.bfloat16)
        modules.add(path.rsplit('.', 1)[1])
    dst.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst / 'adapter_model.safetensors'), out, metadata={'format': 'pt'})
    (dst / 'adapter_config.json').write_text(json.dumps({
        'base_model_name_or_path': 'google/gemma-4-31b-it-qat-w4a16-ct', 'bias': 'none', 'fan_in_fan_out': False,
        'inference_mode': True, 'init_lora_weights': True, 'layers_to_transform': None, 'lora_alpha': int(scale * rank) if float(scale * rank).is_integer() else scale * rank,
        'lora_dropout': 0.0, 'modules_to_save': None, 'peft_type': 'LORA', 'r': rank,
        'target_modules': sorted(modules), 'task_type': 'CAUSAL_LM', 'use_dora': False, 'use_rslora': False,
    }, indent=2))
    size = (dst / 'adapter_model.safetensors').stat().st_size / 2 ** 20
    print(f'{len(out)} tensors, modules {sorted(modules)}, r={rank}, alpha={scale * rank}, {size:.0f} MiB -> {dst}')


if __name__ == '__main__':
    main(*sys.argv[1:])
