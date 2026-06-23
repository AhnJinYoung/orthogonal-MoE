"""Token-wise MoE aggregation variants.

The code assumes expert_outputs has shape [tokens, top_k, hidden_dim] and contains
RAW expert outputs. Gates has shape [tokens, top_k] and is usually normalized by
the router. The returned tensor has shape [tokens, hidden_dim].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
Stats = Dict[str, Tensor]


@dataclass
class AggregatorConfig:
    name: str = "standard"
    lam: float = 1.0
    eps: float = 1e-6
    positive_only: bool = True
    stop_anchor: bool = True
    preserve_norm: bool = False
    max_rescale: float = 2.0
    novelty_alpha: float = 1.0
    whitening_shrinkage: float = 1e-3
    other_scale: float = 1.0


class MoEAggregator(nn.Module):
    """Base interface for all aggregators."""

    variant_name: str = "base"

    def forward(
        self,
        expert_outputs: Tensor,
        gates: Tensor,
        selected_experts: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Stats]:
        raise NotImplementedError

    def _base_stats(self, expert_outputs: Tensor, gates: Tensor, aggregated: Tensor) -> Stats:
        with torch.no_grad():
            stats: Stats = {
                "moe_output_norm": aggregated.detach().float().norm(dim=-1).mean(),
                "gate_entropy": _entropy(gates.detach().float()).mean(),
                "gate_top1_mass": gates.detach().float()[:, 0].mean() if gates.ndim == 2 else torch.tensor(0.0),
            }
            if expert_outputs.shape[1] > 1:
                a = expert_outputs[:, :1, :].detach().float()
                b = expert_outputs[:, 1:, :].detach().float()
                cos = F.cosine_similarity(a, b, dim=-1)
                novelty = torch.sqrt(torch.clamp(1.0 - cos.square(), min=0.0))
                stats.update(
                    {
                        "cos_top1_mean": cos.mean(),
                        "cos_top1_abs_mean": cos.abs().mean(),
                        "cos_top1_pos_mean": cos.clamp_min(0.0).mean(),
                        "novelty_top1_mean": novelty.mean(),
                    }
                )
            else:
                z = aggregated.detach().float().new_tensor(0.0)
                stats.update(
                    {
                        "cos_top1_mean": z,
                        "cos_top1_abs_mean": z,
                        "cos_top1_pos_mean": z,
                        "novelty_top1_mean": z,
                    }
                )
            return stats


class StandardAggregation(MoEAggregator):
    variant_name = "standard"

    def forward(self, expert_outputs: Tensor, gates: Tensor, selected_experts: Optional[Tensor] = None):
        gates = gates.to(dtype=expert_outputs.dtype)
        y = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)
        return y, self._base_stats(expert_outputs, gates, y)


class Top1OrthogonalAggregation(MoEAggregator):
    """Project non-top experts away from the top-1 expert direction.

    If positive_only=True, only remove the positively aligned component. That
    keeps useful negative/corrective components intact.
    """

    variant_name = "top1_orthogonal"

    def __init__(
        self,
        lam: float = 1.0,
        eps: float = 1e-6,
        positive_only: bool = True,
        stop_anchor: bool = True,
        preserve_norm: bool = False,
        max_rescale: float = 2.0,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.positive_only = positive_only
        self.stop_anchor = stop_anchor
        self.preserve_norm = preserve_norm
        self.max_rescale = max_rescale

    def forward(self, expert_outputs: Tensor, gates: Tensor, selected_experts: Optional[Tensor] = None):
        gates = gates.to(dtype=expert_outputs.dtype)
        if expert_outputs.shape[1] <= 1 or self.lam == 0.0:
            y = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)
            return y, self._base_stats(expert_outputs, gates, y)

        a = expert_outputs[:, :1, :]
        b = expert_outputs[:, 1:, :]
        anchor = a.detach() if self.stop_anchor else a
        denom = anchor.square().sum(dim=-1, keepdim=True).clamp_min(self.eps)
        coeff = (b * anchor).sum(dim=-1, keepdim=True) / denom
        if self.positive_only:
            coeff = coeff.clamp_min(0.0)
        b_new = b - self.lam * coeff * anchor

        if self.preserve_norm:
            old_norm = b.norm(dim=-1, keepdim=True)
            new_norm = b_new.norm(dim=-1, keepdim=True)
            scale = (old_norm / (new_norm + self.eps)).clamp(max=self.max_rescale)
            b_new = b_new * scale

        y = gates[:, :1, None] * a + torch.sum(gates[:, 1:, None] * b_new, dim=1, keepdim=True)
        y = y.squeeze(1)
        stats = self._base_stats(expert_outputs, gates, y)
        with torch.no_grad():
            old_other = b.detach().float().norm(dim=-1)
            new_other = b_new.detach().float().norm(dim=-1)
            stats["projection_norm_ratio"] = (new_other / (old_other + self.eps)).mean()
        return y, stats


class WeightedSumTop1ProjectionAggregation(MoEAggregator):
    """Project the weighted sum of all non-top experts once.

    This is algebraically equivalent to projecting each non-top expert for the
    full signed linear projection, but cheaper. It is not equivalent to
    positive-only projection.
    """

    variant_name = "weighted_sum_top1_projection"

    def __init__(self, lam: float = 1.0, eps: float = 1e-6, stop_anchor: bool = True) -> None:
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.stop_anchor = stop_anchor

    def forward(self, expert_outputs: Tensor, gates: Tensor, selected_experts: Optional[Tensor] = None):
        gates = gates.to(dtype=expert_outputs.dtype)
        if expert_outputs.shape[1] <= 1 or self.lam == 0.0:
            y = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)
            return y, self._base_stats(expert_outputs, gates, y)
        a = expert_outputs[:, :1, :]
        other_sum = torch.sum(gates[:, 1:, None] * expert_outputs[:, 1:, :], dim=1, keepdim=True)
        anchor = a.detach() if self.stop_anchor else a
        denom = anchor.square().sum(dim=-1, keepdim=True).clamp_min(self.eps)
        coeff = (other_sum * anchor).sum(dim=-1, keepdim=True) / denom
        other_new = other_sum - self.lam * coeff * anchor
        y = (gates[:, :1, None] * a + other_new).squeeze(1)
        stats = self._base_stats(expert_outputs, gates, y)
        return y, stats


class GramSchmidtAggregation(MoEAggregator):
    """Sequentially orthogonalize selected experts in router order."""

    variant_name = "gram_schmidt"

    def __init__(
        self,
        lam: float = 1.0,
        eps: float = 1e-6,
        positive_only: bool = False,
        preserve_norm: bool = False,
        max_rescale: float = 2.0,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.positive_only = positive_only
        self.preserve_norm = preserve_norm
        self.max_rescale = max_rescale

    def forward(self, expert_outputs: Tensor, gates: Tensor, selected_experts: Optional[Tensor] = None):
        gates = gates.to(dtype=expert_outputs.dtype)
        k = expert_outputs.shape[1]
        if k <= 1 or self.lam == 0.0:
            y = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)
            return y, self._base_stats(expert_outputs, gates, y)

        qs = []
        for i in range(k):
            q = expert_outputs[:, i : i + 1, :]
            old_norm = q.norm(dim=-1, keepdim=True)
            for prev in qs:
                denom = prev.square().sum(dim=-1, keepdim=True).clamp_min(self.eps)
                coeff = (q * prev).sum(dim=-1, keepdim=True) / denom
                if self.positive_only:
                    coeff = coeff.clamp_min(0.0)
                q = q - self.lam * coeff * prev
            if self.preserve_norm:
                new_norm = q.norm(dim=-1, keepdim=True)
                scale = (old_norm / (new_norm + self.eps)).clamp(max=self.max_rescale)
                q = q * scale
            qs.append(q)
        q_stack = torch.cat(qs, dim=1)
        y = torch.sum(q_stack * gates.unsqueeze(-1), dim=1)
        stats = self._base_stats(expert_outputs, gates, y)
        with torch.no_grad():
            stats["projection_norm_ratio"] = (
                q_stack.detach().float().norm(dim=-1) / (expert_outputs.detach().float().norm(dim=-1) + self.eps)
            ).mean()
        return y, stats


class WhiteningAggregation(MoEAggregator):
    """Symmetric per-token whitening of selected expert outputs.

    For each token, form C = E E^T and use C^{-1/2} E. Because top_k is small,
    torch.linalg.eigh over [tokens, top_k, top_k] is usually acceptable.
    """

    variant_name = "whitening"

    def __init__(
        self,
        lam: float = 1.0,
        eps: float = 1e-6,
        whitening_shrinkage: float = 1e-3,
        preserve_norm: bool = True,
        max_rescale: float = 2.0,
    ) -> None:
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.whitening_shrinkage = whitening_shrinkage
        self.preserve_norm = preserve_norm
        self.max_rescale = max_rescale

    def forward(self, expert_outputs: Tensor, gates: Tensor, selected_experts: Optional[Tensor] = None):
        gates = gates.to(dtype=expert_outputs.dtype)
        k = expert_outputs.shape[1]
        if k <= 1 or self.lam == 0.0:
            y = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)
            return y, self._base_stats(expert_outputs, gates, y)

        x = expert_outputs
        x_float = x.float()
        gram = torch.matmul(x_float, x_float.transpose(-1, -2))
        diag_mean = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(self.eps)
        eye = torch.eye(k, device=x.device, dtype=torch.float32).unsqueeze(0)
        gram = gram + eye * (self.whitening_shrinkage * diag_mean.unsqueeze(-1) + self.eps)
        evals, evecs = torch.linalg.eigh(gram)
        inv_sqrt = evecs @ torch.diag_embed(torch.rsqrt(evals.clamp_min(self.eps))) @ evecs.transpose(-1, -2)
        white = torch.matmul(inv_sqrt, x_float).to(dtype=x.dtype)
        x_new = (1.0 - self.lam) * x + self.lam * white

        if self.preserve_norm:
            old_norm = x.norm(dim=-1, keepdim=True)
            new_norm = x_new.norm(dim=-1, keepdim=True)
            scale = (old_norm / (new_norm + self.eps)).clamp(max=self.max_rescale)
            x_new = x_new * scale

        y = torch.sum(x_new * gates.unsqueeze(-1), dim=1)
        stats = self._base_stats(expert_outputs, gates, y)
        with torch.no_grad():
            stats["projection_norm_ratio"] = (
                x_new.detach().float().norm(dim=-1) / (expert_outputs.detach().float().norm(dim=-1) + self.eps)
            ).mean()
        return y, stats


class NoveltyGatedAggregation(MoEAggregator):
    """Keep expert outputs unchanged but reweight gates by novelty to top-1."""

    variant_name = "novelty_gated"

    def __init__(
        self,
        alpha: float = 1.0,
        eps: float = 1e-6,
        positive_only: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.positive_only = positive_only

    def forward(self, expert_outputs: Tensor, gates: Tensor, selected_experts: Optional[Tensor] = None):
        gates = gates.to(dtype=expert_outputs.dtype)
        k = expert_outputs.shape[1]
        if k <= 1 or self.alpha == 0.0:
            y = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)
            return y, self._base_stats(expert_outputs, gates, y)

        a = expert_outputs[:, :1, :].detach().float()
        b = expert_outputs[:, 1:, :].detach().float()
        cos = F.cosine_similarity(a, b, dim=-1)
        if self.positive_only:
            novelty_other = torch.sqrt(torch.clamp(1.0 - cos.clamp_min(0.0).square(), min=0.0))
        else:
            novelty_other = torch.sqrt(torch.clamp(1.0 - cos.square(), min=0.0))
        novelty = torch.cat([torch.ones_like(gates[:, :1].float()), novelty_other], dim=1)
        new_gates = gates.float() * (novelty + self.eps).pow(self.alpha)
        new_gates = new_gates / new_gates.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        new_gates = new_gates.to(dtype=expert_outputs.dtype)
        y = torch.sum(expert_outputs * new_gates.unsqueeze(-1), dim=1)
        stats = self._base_stats(expert_outputs, new_gates, y)
        with torch.no_grad():
            stats["gate_delta_abs_mean"] = (new_gates.detach().float() - gates.detach().float()).abs().mean()
        return y, stats


class NormMatchedShrinkageAggregation(MoEAggregator):
    """Control: shrink non-top expert outputs without changing direction."""

    variant_name = "norm_matched_shrinkage"

    def __init__(self, other_scale: float = 0.75) -> None:
        super().__init__()
        self.other_scale = other_scale

    def forward(self, expert_outputs: Tensor, gates: Tensor, selected_experts: Optional[Tensor] = None):
        gates = gates.to(dtype=expert_outputs.dtype)
        if expert_outputs.shape[1] <= 1:
            y = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)
            return y, self._base_stats(expert_outputs, gates, y)
        x = expert_outputs.clone()
        x[:, 1:, :] = x[:, 1:, :] * self.other_scale
        y = torch.sum(x * gates.unsqueeze(-1), dim=1)
        return y, self._base_stats(expert_outputs, gates, y)


class RandomAnchorProjectionAggregation(MoEAggregator):
    """Control: project non-anchor experts away from a random selected expert."""

    variant_name = "random_anchor_projection"

    def __init__(self, lam: float = 1.0, eps: float = 1e-6, positive_only: bool = True) -> None:
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.positive_only = positive_only

    def forward(self, expert_outputs: Tensor, gates: Tensor, selected_experts: Optional[Tensor] = None):
        gates = gates.to(dtype=expert_outputs.dtype)
        tokens, k, _ = expert_outputs.shape
        if k <= 1 or self.lam == 0.0:
            y = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)
            return y, self._base_stats(expert_outputs, gates, y)
        anchor_idx = torch.randint(low=0, high=k, size=(tokens,), device=expert_outputs.device)
        anchor = expert_outputs[torch.arange(tokens, device=expert_outputs.device), anchor_idx].unsqueeze(1).detach()
        denom = anchor.square().sum(dim=-1, keepdim=True).clamp_min(self.eps)
        coeff = (expert_outputs * anchor).sum(dim=-1, keepdim=True) / denom
        if self.positive_only:
            coeff = coeff.clamp_min(0.0)
        x_new = expert_outputs - self.lam * coeff * anchor
        x_new[torch.arange(tokens, device=expert_outputs.device), anchor_idx] = expert_outputs[
            torch.arange(tokens, device=expert_outputs.device), anchor_idx
        ]
        y = torch.sum(x_new * gates.unsqueeze(-1), dim=1)
        return y, self._base_stats(expert_outputs, gates, y)


def _entropy(p: Tensor, eps: float = 1e-9) -> Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def build_aggregator(config: Mapping[str, Any] | AggregatorConfig | str) -> MoEAggregator:
    if isinstance(config, str):
        cfg = AggregatorConfig(name=config)
    elif isinstance(config, AggregatorConfig):
        cfg = config
    else:
        data = dict(config)
        if "lambda" in data and "lam" not in data:
            data["lam"] = data.pop("lambda")
        cfg = AggregatorConfig(**{k: v for k, v in data.items() if k in AggregatorConfig.__dataclass_fields__})

    name = cfg.name.lower()
    if name in {"standard", "baseline", "default"}:
        return StandardAggregation()
    if name in {"top1_orthogonal", "top1_ortho", "positive_top1_ortho"}:
        return Top1OrthogonalAggregation(
            lam=cfg.lam,
            eps=cfg.eps,
            positive_only=cfg.positive_only,
            stop_anchor=cfg.stop_anchor,
            preserve_norm=cfg.preserve_norm,
            max_rescale=cfg.max_rescale,
        )
    if name in {"weighted_sum_top1_projection", "aggregate_projection", "sum_projection"}:
        return WeightedSumTop1ProjectionAggregation(lam=cfg.lam, eps=cfg.eps, stop_anchor=cfg.stop_anchor)
    if name in {"gram_schmidt", "gs"}:
        return GramSchmidtAggregation(
            lam=cfg.lam,
            eps=cfg.eps,
            positive_only=cfg.positive_only,
            preserve_norm=cfg.preserve_norm,
            max_rescale=cfg.max_rescale,
        )
    if name in {"whitening", "symmetric_whitening"}:
        return WhiteningAggregation(
            lam=cfg.lam,
            eps=cfg.eps,
            whitening_shrinkage=cfg.whitening_shrinkage,
            preserve_norm=cfg.preserve_norm,
            max_rescale=cfg.max_rescale,
        )
    if name in {"novelty_gated", "novelty_gate"}:
        return NoveltyGatedAggregation(alpha=cfg.novelty_alpha, eps=cfg.eps, positive_only=cfg.positive_only)
    if name in {"norm_matched_shrinkage", "shrinkage_control", "shrink"}:
        return NormMatchedShrinkageAggregation(other_scale=cfg.other_scale)
    if name in {"random_anchor_projection", "random_anchor"}:
        return RandomAnchorProjectionAggregation(lam=cfg.lam, eps=cfg.eps, positive_only=cfg.positive_only)
    raise ValueError(f"Unknown aggregation variant: {cfg.name}")
