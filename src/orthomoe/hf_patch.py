"""Monkey-patching utilities for Hugging Face MoE models.

The safest path for Gemma4/Qwen3/Qwen3.5 MoE checkpoints is to patch the tensor
expert collection modules. These modules expose:
    gate_up_proj: [num_experts, 2 * intermediate_dim, hidden_dim]
    down_proj:    [num_experts, hidden_dim, intermediate_dim]
    act_fn
    forward(hidden_states, top_k_index, top_k_weights)

Patching the expert collection preserves the surrounding router and decoder layer
implementation while replacing only the aggregation of the selected expert outputs.
"""
from __future__ import annotations

import re
import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Pattern, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregations import MoEAggregator, build_aggregator
from .losses import output_orthogonality_loss


@dataclass
class PatchReport:
    patched_tensor_experts: List[str]
    patched_modulelist_blocks: List[str]

    @property
    def total(self) -> int:
        return len(self.patched_tensor_experts) + len(self.patched_modulelist_blocks)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "patched_tensor_experts": self.patched_tensor_experts,
            "patched_modulelist_blocks": self.patched_modulelist_blocks,
            "total": self.total,
        }


def patch_model(
    model: nn.Module,
    aggregation: Mapping[str, Any] | str | MoEAggregator,
    *,
    module_name_regex: str | None = None,
    collect_stats: bool = True,
    aux_ortho_coef: float = 0.0,
    aux_positive_only: bool = False,
    patch_modulelist_blocks: bool = True,
    modulelist_return_router_logits: bool | None = None,
) -> PatchReport:
    """Patch MoE aggregation inside a loaded Hugging Face model.

    Args:
        model: Loaded HF model.
        aggregation: aggregation config dict/name or aggregator module.
        module_name_regex: optional regex to patch a subset of module names.
        collect_stats: store scalar stats from the last forward pass.
        aux_ortho_coef: if >0, store a differentiable selected-output
            orthogonality loss in each patched expert module. The training loop
            can add aggregate_aux_losses(model) * coef to LM loss.
        patch_modulelist_blocks: fallback for Mixtral-like SparseMoeBlock modules
            with gate + ModuleList experts.
        modulelist_return_router_logits: set True for blocks whose original
            forward returns (hidden_states, router_logits). None tries to infer
            from class name and is conservative.
    """
    aggregator = aggregation if isinstance(aggregation, MoEAggregator) else build_aggregator(aggregation)
    pattern = re.compile(module_name_regex) if module_name_regex else None
    patched_tensor: List[str] = []
    patched_blocks: List[str] = []

    for name, module in model.named_modules():
        if pattern is not None and not pattern.search(name):
            continue
        if _is_tensor_experts_module(module):
            patch_tensor_experts_module(
                module,
                aggregator,
                collect_stats=collect_stats,
                aux_ortho_coef=aux_ortho_coef,
                aux_positive_only=aux_positive_only,
            )
            patched_tensor.append(name)

    if not patched_tensor and patch_modulelist_blocks:
        for name, module in model.named_modules():
            if pattern is not None and not pattern.search(name):
                continue
            if _is_modulelist_sparse_moe_block(module):
                patch_modulelist_sparse_block(
                    module,
                    aggregator,
                    collect_stats=collect_stats,
                    aux_ortho_coef=aux_ortho_coef,
                    aux_positive_only=aux_positive_only,
                    return_router_logits=modulelist_return_router_logits,
                )
                patched_blocks.append(name)

    return PatchReport(patched_tensor, patched_blocks)


def set_aggregator(model: nn.Module, aggregation: Mapping[str, Any] | str | MoEAggregator) -> int:
    """Update all patched modules to use a new aggregator."""
    aggregator = aggregation if isinstance(aggregation, MoEAggregator) else build_aggregator(aggregation)
    count = 0
    for module in model.modules():
        if getattr(module, "_orthomoe_patched", False):
            module._orthomoe_aggregator = aggregator
            count += 1
    if count == 0:
        raise RuntimeError("No patched MoE modules found. Call patch_model(...) first.")
    return count


def collect_moe_stats(model: nn.Module, prefix: str = "") -> Dict[str, float]:
    """Collect mean scalar stats from the most recent forward pass."""
    buckets: Dict[str, List[float]] = {}
    for module in model.modules():
        stats = getattr(module, "_orthomoe_last_stats", None)
        if not stats:
            continue
        for key, value in stats.items():
            if value is None:
                continue
            try:
                scalar = float(value.detach().float().mean().item())
            except Exception:
                continue
            buckets.setdefault(prefix + key, []).append(scalar)
    return {key: sum(vals) / max(len(vals), 1) for key, vals in buckets.items()}


def clear_moe_stats(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "_orthomoe_last_stats"):
            module._orthomoe_last_stats = None


def unpatch_model(model: nn.Module) -> int:
    """Restore original forward methods where possible."""
    count = 0
    for module in model.modules():
        original = getattr(module, "_orthomoe_original_forward", None)
        if original is not None:
            module.forward = original
            for attr in [
                "_orthomoe_original_forward",
                "_orthomoe_aggregator",
                "_orthomoe_patched",
                "_orthomoe_last_stats",
                "_orthomoe_aux_loss",
                "_orthomoe_collect_stats",
            ]:
                if hasattr(module, attr):
                    delattr(module, attr)
            count += 1
    return count


def _is_tensor_experts_module(module: nn.Module) -> bool:
    return (
        hasattr(module, "gate_up_proj")
        and hasattr(module, "down_proj")
        and hasattr(module, "act_fn")
        and hasattr(module, "num_experts")
        and isinstance(getattr(module, "gate_up_proj"), torch.nn.Parameter)
        and isinstance(getattr(module, "down_proj"), torch.nn.Parameter)
    )


def _is_modulelist_sparse_moe_block(module: nn.Module) -> bool:
    return hasattr(module, "gate") and hasattr(module, "experts") and isinstance(getattr(module, "experts"), nn.ModuleList)


def patch_tensor_experts_module(
    module: nn.Module,
    aggregator: MoEAggregator,
    *,
    collect_stats: bool = True,
    aux_ortho_coef: float = 0.0,
    aux_positive_only: bool = False,
) -> None:
    if getattr(module, "_orthomoe_patched", False):
        module._orthomoe_aggregator = aggregator
        module._orthomoe_collect_stats = collect_stats
        module._orthomoe_aux_ortho_coef = aux_ortho_coef
        module._orthomoe_aux_positive_only = aux_positive_only
        return
    module._orthomoe_original_forward = module.forward
    module._orthomoe_aggregator = aggregator
    module._orthomoe_collect_stats = collect_stats
    module._orthomoe_aux_ortho_coef = aux_ortho_coef
    module._orthomoe_aux_positive_only = aux_positive_only
    module._orthomoe_last_stats = None
    module._orthomoe_aux_loss = None
    module._orthomoe_patched = True
    module.forward = types.MethodType(_tensor_experts_forward, module)


def _tensor_experts_forward(self: nn.Module, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
    """Replacement for Gemma4/Qwen tensor-expert forward."""
    tokens, hidden_dim = hidden_states.shape
    top_k = top_k_index.shape[1]
    expert_outputs = hidden_states.new_zeros(tokens, top_k, hidden_dim)

    # Create [num_experts, top_k, tokens] mask, then process only hit experts.
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=int(self.num_experts)).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).flatten()

    for expert_idx in expert_hit.tolist():
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden = self.act_fn(gate) * up
        current_hidden = F.linear(current_hidden, self.down_proj[expert_idx])
        expert_outputs[token_idx, top_k_pos] = current_hidden.to(expert_outputs.dtype)

    y, stats = self._orthomoe_aggregator(expert_outputs, top_k_weights, top_k_index)
    if getattr(self, "_orthomoe_collect_stats", True):
        self._orthomoe_last_stats = {key: value.detach() for key, value in stats.items()}
    else:
        self._orthomoe_last_stats = None

    coef = float(getattr(self, "_orthomoe_aux_ortho_coef", 0.0) or 0.0)
    if coef > 0.0:
        self._orthomoe_aux_loss = output_orthogonality_loss(
            expert_outputs,
            top_k_weights,
            positive_only=bool(getattr(self, "_orthomoe_aux_positive_only", False)),
        ) * coef
    else:
        self._orthomoe_aux_loss = None
    return y.to(hidden_states.dtype)


def patch_modulelist_sparse_block(
    module: nn.Module,
    aggregator: MoEAggregator,
    *,
    collect_stats: bool = True,
    aux_ortho_coef: float = 0.0,
    aux_positive_only: bool = False,
    return_router_logits: bool | None = None,
) -> None:
    if getattr(module, "_orthomoe_patched", False):
        module._orthomoe_aggregator = aggregator
        return
    module._orthomoe_original_forward = module.forward
    module._orthomoe_aggregator = aggregator
    module._orthomoe_collect_stats = collect_stats
    module._orthomoe_aux_ortho_coef = aux_ortho_coef
    module._orthomoe_aux_positive_only = aux_positive_only
    module._orthomoe_return_router_logits = return_router_logits
    module._orthomoe_last_stats = None
    module._orthomoe_aux_loss = None
    module._orthomoe_patched = True
    module.forward = types.MethodType(_modulelist_sparse_block_forward, module)


def _modulelist_sparse_block_forward(self: nn.Module, hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
    """Generic replacement for Mixtral-like blocks with gate + ModuleList experts."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_dim)

    gate_out = self.gate(flat)
    if isinstance(gate_out, tuple):
        if len(gate_out) >= 3:
            router_logits, routing_weights, selected_experts = gate_out[:3]
        else:
            router_logits = gate_out[0]
            routing_weights, selected_experts = _topk_from_logits(self, router_logits)
    else:
        router_logits = gate_out
        routing_weights, selected_experts = _topk_from_logits(self, router_logits)

    tokens, top_k = selected_experts.shape
    expert_outputs = flat.new_zeros(tokens, top_k, hidden_dim)
    with torch.no_grad():
        expert_mask = F.one_hot(selected_experts, num_classes=len(self.experts)).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).flatten()

    for expert_idx in expert_hit.tolist():
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        current = self.experts[expert_idx](flat[token_idx])
        expert_outputs[token_idx, top_k_pos] = current.to(expert_outputs.dtype)

    y, stats = self._orthomoe_aggregator(expert_outputs, routing_weights, selected_experts)
    if getattr(self, "_orthomoe_collect_stats", True):
        self._orthomoe_last_stats = {key: value.detach() for key, value in stats.items()}

    coef = float(getattr(self, "_orthomoe_aux_ortho_coef", 0.0) or 0.0)
    if coef > 0.0:
        self._orthomoe_aux_loss = output_orthogonality_loss(
            expert_outputs,
            routing_weights,
            positive_only=bool(getattr(self, "_orthomoe_aux_positive_only", False)),
        ) * coef
    else:
        self._orthomoe_aux_loss = None

    y = y.reshape(batch_size, sequence_length, hidden_dim).to(hidden_states.dtype)
    ret = getattr(self, "_orthomoe_return_router_logits", None)
    if ret is None:
        class_name = self.__class__.__name__.lower()
        ret = "mixtral" in class_name or "dbrx" in class_name
    if ret:
        return y, router_logits
    return y


def _topk_from_logits(module: nn.Module, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    top_k = int(
        getattr(module, "top_k", None)
        or getattr(module, "num_experts_per_tok", None)
        or getattr(getattr(module, "config", object()), "num_experts_per_tok", None)
        or getattr(getattr(module, "config", object()), "top_k", None)
        or 2
    )
    routing_weights = F.softmax(router_logits.float(), dim=-1)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    routing_weights = routing_weights.to(router_logits.dtype)
    return routing_weights, selected_experts
