"""Export a nanoGPT checkpoint to a HF folder that the BabyLM eval pipeline can load.

uv run python export_hf.py out/ckpt.pt hf_models/babylm_bidir [tokenizers/babylm10m]
"""

import shutil
import sys

import torch

from hf.configuration_nanogpt import NanoGPTConfig
from hf.modeling_nanogpt import NanoGPTForCausalLM

ckpt_path, out_dir = sys.argv[1], sys.argv[2]
tokenizer_dir = sys.argv[3] if len(sys.argv) > 3 else 'tokenizers/babylm10m'

ckpt = torch.load(ckpt_path, map_location='cpu')
args = {k: v for k, v in ckpt['model_args'].items() if k != 'dropout'}
state_dict = {
    k.removeprefix('_orig_mod.'): v
    for k, v in ckpt['model'].items()
    if not k.endswith('lm_head.weight')  # tied to wte
}

config = NanoGPTConfig(**args)
config.auto_map = {
    'AutoConfig': 'configuration_nanogpt.NanoGPTConfig',
    'AutoModel': 'modeling_nanogpt.NanoGPTModel',
    'AutoModelForCausalLM': 'modeling_nanogpt.NanoGPTForCausalLM',
}
model = NanoGPTForCausalLM(config)
model.load_state_dict(state_dict)
model.save_pretrained(out_dir)

for f in ('hf/configuration_nanogpt.py', 'hf/modeling_nanogpt.py'):
    shutil.copy(f, out_dir)
for f in ('tokenizer.json', 'tokenizer_config.json'):
    shutil.copy(f'{tokenizer_dir}/{f}', out_dir)
print(f'saved {args} to {out_dir}')
