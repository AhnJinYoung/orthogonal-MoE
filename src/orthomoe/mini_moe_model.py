"""A tiny MoE language model for smoke tests and low-cost pretraining experiments.

This is not intended to match Gemma/Qwen. It gives you a controlled model where
all aggregation variants are native and easy to debug before patching a 26B/35B
checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregations import MoEAggregator, build_aggregator
from .losses import output_orthogonality_loss


@dataclass
class MiniMoEConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_layers: int = 8
    num_heads: int = 8
    num_experts: int = 8
    top_k: int = 2
    max_position_embeddings: int = 2048
    dropout: float = 0.0
    aggregation: Dict[str, Any] | str = None
    aux_ortho_coef: float = 0.0

    def __post_init__(self):
        if self.aggregation is None:
            self.aggregation = {"name": "standard"}


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    def __init__(self, cfg: MiniMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(cfg.hidden_size, cfg.intermediate_size) for _ in range(cfg.num_experts)])
        self.aggregator: MoEAggregator = build_aggregator(cfg.aggregation)
        self.last_stats = None
        self.aux_loss = None

    def set_aggregator(self, aggregation: Dict[str, Any] | str | MoEAggregator) -> None:
        self.aggregator = aggregation if isinstance(aggregation, MoEAggregator) else build_aggregator(aggregation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        flat = x.reshape(-1, dim)
        logits = self.router(flat)
        probs = F.softmax(logits.float(), dim=-1)
        gates, indices = torch.topk(probs, self.cfg.top_k, dim=-1)
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        gates = gates.to(dtype=flat.dtype)

        tokens, top_k = indices.shape
        expert_outputs = flat.new_zeros(tokens, top_k, dim)
        with torch.no_grad():
            mask = F.one_hot(indices, num_classes=self.cfg.num_experts).permute(2, 1, 0)
            hits = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).flatten()
        for expert_idx in hits.tolist():
            top_pos, token_idx = torch.where(mask[expert_idx])
            expert_outputs[token_idx, top_pos] = self.experts[expert_idx](flat[token_idx])

        y, stats = self.aggregator(expert_outputs, gates, indices)
        self.last_stats = {k: v.detach() for k, v in stats.items()}
        if self.cfg.aux_ortho_coef > 0:
            self.aux_loss = self.cfg.aux_ortho_coef * output_orthogonality_loss(expert_outputs, gates)
        else:
            self.aux_loss = None
        return y.reshape(batch, seq, dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: MiniMoEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.moe = SparseMoE(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = residual + self.dropout(h)
        residual = x
        h = self.moe(self.ln2(x))
        return residual + self.dropout(h)


class MiniMoELM(nn.Module):
    def __init__(self, cfg: MiniMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_positions = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def set_aggregator(self, aggregation: Dict[str, Any] | str | MoEAggregator) -> None:
        for layer in self.layers:
            layer.moe.set_aggregator(aggregation)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        batch, seq = input_ids.shape
        if seq > self.cfg.max_position_embeddings:
            raise ValueError(f"Sequence length {seq} exceeds max_position_embeddings={self.cfg.max_position_embeddings}")
        pos = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, seq)
        x = self.embed_tokens(input_ids) + self.embed_positions(pos)
        causal_mask = torch.triu(torch.ones(seq, seq, device=input_ids.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask)
        x = self.final_ln(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            aux_losses = [layer.moe.aux_loss for layer in self.layers if layer.moe.aux_loss is not None]
            if aux_losses:
                loss = loss + torch.stack([x.float() for x in aux_losses]).mean()
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, out_dir: str) -> None:
        import json
        from pathlib import Path

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "mini_moe_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        torch.save(self.state_dict(), out / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, path: str):
        import json
        from pathlib import Path

        p = Path(path)
        with open(p / "mini_moe_config.json", "r", encoding="utf-8") as f:
            cfg = MiniMoEConfig(**json.load(f))
        model = cls(cfg)
        model.load_state_dict(torch.load(p / "pytorch_model.bin", map_location="cpu"))
        return model
