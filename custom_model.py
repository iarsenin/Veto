"""
HCRG (Head-Level Context-Dependent Repression Gates) architecture.

Based on nanoGPT (https://github.com/karpathy/nanoGPT) with the HCRG modification
from "Learning to Veto: Mitigating Hallucinations via Head-Level Repression Gates
in Language Models."

Core change: each transformer Block learns a context-dependent scalar gate per
attention head, conditioned on the pre-attention residual state.  Gates are
computed as:

    g = sigmoid(W_gate @ x_current + b_gate)     # shape (B, T, n_head)

and applied to head outputs before they are merged by the output projection:

    y_gated[:,h,:] = g[:,h] * head_output[:,h,:]

Bias is initialised to +5 so that sigmoid(+5) ≈ 0.993, keeping essentially
all head magnitude at step-0 while leaving gradients open to learn repression.

IsoFLOPs note: gate adds (n_embd * n_head + n_head) params per layer.
For the two experimental grids this is < 0.1 % of total parameters,
well inside the 1 % margin, so no dimension reduction is required.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


_GATE_BIAS_INIT = 5.0  # sigmoid(5) ≈ 0.993 — gates start nearly fully open


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch does not support bias=False natively)."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class HCRGCausalSelfAttention(nn.Module):
    """
    Causal self-attention with HCRG gating support.

    When a `gates` tensor of shape (B, T, n_head) is supplied, each head
    output is scaled by the corresponding gate value before the outputs are
    concatenated and passed through the output projection.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x, gates=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hs)

        # ----- HCRG: apply per-head repression gates -----
        if gates is not None:
            # gates: (B, T, n_head) -> (B, n_head, T, 1) for broadcast over head_size
            gates_4d = gates.permute(0, 2, 1).unsqueeze(-1)
            y = y * gates_4d
        # --------------------------------------------------

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class HCRGBlock(nn.Module):
    """
    Transformer block with Head-Level Context-Dependent Repression Gates.

    gate_proj maps the pre-attention residual state to one scalar per head:
        g = sigmoid(gate_proj(x))    # (B, T, n_head)

    Conditioning on x (the raw residual, before LayerNorm) ensures the gate
    sees a representation that already aggregates prior-context information
    from earlier layers, while remaining strictly causal and KV-cache
    compatible (only the current token's vector is consumed at decode time).
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = HCRGCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        # One gate scalar per head; bias init to _GATE_BIAS_INIT preserves
        # near-unity head amplitude at initialisation.
        self.gate_proj = nn.Linear(config.n_embd, config.n_head, bias=True)

    def forward(self, x):
        # Compute repression gates from pre-attention residual state
        gates = torch.sigmoid(self.gate_proj(x))   # (B, T, n_head)
        x = x + self.attn(self.ln_1(x), gates=gates)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    """HCRG-augmented GPT.  Drop-in replacement for the baseline GPT class."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([HCRGBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
        gate_params = sum(
            p.numel() for n, p in self.named_parameters() if 'gate_proj' in n
        )
        baseline_params = self.get_num_params() - gate_params
        overhead_pct = 100.0 * gate_params / baseline_params
        print(f"HCRG gate parameters: {gate_params:,} ({overhead_pct:.3f}% overhead)")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # After generic init, override gate_proj biases to the negative constant
        # so that sigmoid output ≈ 1 at initialisation (gates fully open).
        if isinstance(module, nn.Linear) and module.bias is not None:
            if module.weight.shape[0] <= 64:
                # Heuristic: gate_proj has small output dim (n_head).
                # More precisely, we tag these in __init__ after apply().
                pass  # handled below via named_parameters walk

    def _init_gate_biases(self):
        """Set gate_proj biases to _GATE_BIAS_INIT after the generic weight init."""
        for name, module in self.named_modules():
            if name.endswith('gate_proj'):
                torch.nn.init.constant_(module.bias, _GATE_BIAS_INIT)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Sequence length {t} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ── post-construction gate bias fix ──────────────────────────────────────────
_original_gpt_init = GPT.__init__


def _patched_gpt_init(self, config):
    _original_gpt_init(self, config)
    self._init_gate_biases()


GPT.__init__ = _patched_gpt_init
