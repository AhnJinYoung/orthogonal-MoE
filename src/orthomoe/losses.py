"""Auxiliary losses for MoE specialization experiments."""
from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def output_orthogonality_loss(
    expert_outputs: Tensor,
    gates: Tensor | None = None,
    positive_only: bool = False,
    squared: bool = True,
    detach_gates: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Penalize selected expert-output cosine similarity.

    Args:
        expert_outputs: [tokens, top_k, hidden_dim], raw expert outputs.
        gates: Optional [tokens, top_k] gate weights. If provided, pair losses are
            weighted by gate_i * gate_j.
        positive_only: penalize only positive cosine overlap.
        squared: use cos^2. If False, use abs(cos) or relu(cos).
        detach_gates: do not backprop through router weights when weighting.
    """
    if expert_outputs.shape[1] <= 1:
        return expert_outputs.new_zeros(())

    x = F.normalize(expert_outputs.float(), dim=-1, eps=eps)
    sim = torch.matmul(x, x.transpose(-1, -2))
    k = sim.shape[-1]
    mask = torch.triu(torch.ones(k, k, device=sim.device, dtype=torch.bool), diagonal=1)
    vals = sim[:, mask]
    if positive_only:
        vals = vals.clamp_min(0.0)
    else:
        vals = vals.abs()
    if squared:
        vals = vals.square()

    if gates is not None:
        g = gates.float().detach() if detach_gates else gates.float()
        pair_weights = torch.matmul(g.unsqueeze(-1), g.unsqueeze(-2))[:, mask]
        return (vals * pair_weights).sum() / pair_weights.sum().clamp_min(eps)
    return vals.mean()


def aggregate_aux_losses(model: torch.nn.Module) -> Tensor:
    """Sum differentiable orthogonality losses stored by patched expert modules."""
    losses = []
    for module in model.modules():
        loss = getattr(module, "_orthomoe_aux_loss", None)
        if loss is not None:
            losses.append(loss)
    if not losses:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return torch.zeros((), device=device)
    return torch.stack([x.float() for x in losses]).mean()


def reset_aux_losses(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "_orthomoe_aux_loss"):
            module._orthomoe_aux_loss = None


def router_z_loss(router_logits: Tensor, coef: float = 1e-4) -> Tensor:
    """Switch-style router z-loss, useful for stabilizing router logits."""
    log_z = torch.logsumexp(router_logits.float(), dim=-1)
    return coef * log_z.square().mean()
