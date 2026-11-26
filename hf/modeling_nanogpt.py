"""HF wrapper around nanoGPT checkpoints, for the BabyLM eval pipeline.

Parameter names match model.py's state dict, so checkpoints load directly.
Both causal and bidirectional models are evaluated with the BabyLM `causal`
backend: logits[:, t] is always the prediction for token t + 1 given tokens
0..t. For the bidirectional model this needs one forward pass per prefix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

from .configuration_nanogpt import NanoGPTConfig


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head

    def forward(self, x, mask):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, -1).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        return x + self.mlp(self.ln_2(x))


class NanoGPTModel(PreTrainedModel):
    """Returns hidden states from a single pass (used for finetuning)."""

    config_class = NanoGPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.post_init()

    def _init_weights(self, module):
        pass  # weights always come from a checkpoint

    def hidden(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # positions start at 0 on the first real token (handles left padding)
        pos = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        mask = attention_mask.bool()[:, None, :].expand(B, T, T)  # key padding
        if self.config.is_causal:
            mask = mask.tril()
        # always attend to self, so all-padding rows don't produce NaNs
        mask = mask | torch.eye(T, dtype=torch.bool, device=input_ids.device)

        x = self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x, mask)
        return self.transformer.ln_f(x)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return BaseModelOutput(last_hidden_state=self.hidden(input_ids, attention_mask))


class NanoGPTForCausalLM(NanoGPTModel):
    """Returns logits where logits[:, t] predicts token t + 1 from tokens 0..t."""

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.config.is_causal:
            h = self.hidden(input_ids, attention_mask)
        else:
            # bidirectional: encode each prefix separately, keep its last position
            T = input_ids.size(1)
            h = torch.stack(
                [
                    self.hidden(input_ids[:, : t + 1], attention_mask[:, : t + 1])[
                        :, -1
                    ]
                    for t in range(T)
                ],
                dim=1,
            )
        logits = F.linear(h, self.transformer.wte.weight)  # tied lm_head
        return CausalLMOutput(logits=logits)
