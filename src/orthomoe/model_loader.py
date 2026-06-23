from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .hf_patch import patch_model
from .resources import build_load_memory_kwargs, log_memory
from .utils import dtype_from_string


def load_tokenizer(model_cfg: Dict[str, Any]):
    model_id = model_cfg["model_id"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _quantization_config(model_cfg: Dict[str, Any]):
    q = model_cfg.get("quantization")
    if not q:
        return None
    from transformers import BitsAndBytesConfig

    qtype = str(q.get("type", "")).lower()
    if qtype in {"4bit", "nf4"}:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(q.get("double_quant", True)),
            bnb_4bit_compute_dtype=dtype_from_string(q.get("compute_dtype", "bfloat16")),
        )
    if qtype in {"8bit", "int8"}:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization type: {qtype}")


def load_hf_causal_lm(
    model_cfg: Dict[str, Any],
    *,
    for_training: bool = False,
    res_cfg: Dict[str, Any] | None = None,
):
    """Load a HF causal LM. Falls back to AutoModelForMultimodalLM if needed.

    Args:
        model_cfg: the ``model`` block of the config.
        for_training: skip ``device_map`` sharding when preparing for training.
        res_cfg: optional ``resources`` block. When it specifies a memory
            budget, a ``max_memory`` map plus disk ``offload_folder`` is passed
            to ``from_pretrained`` so loading cannot blow past the pod's RAM
            limit (the common exit-code-137 trigger).
    """
    res_cfg = dict(res_cfg or {})
    model_id = model_cfg["model_id"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    dtype = dtype_from_string(model_cfg.get("dtype", "bfloat16"))
    attn_impl = model_cfg.get("attn_implementation", None)
    quant_config = _quantization_config(model_cfg)

    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": bool(model_cfg.get("low_cpu_mem_usage", True)),
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    if not for_training:
        kwargs["device_map"] = model_cfg.get("device_map", "auto")
        # Bound host RAM during sharded loading and offload overflow to disk.
        kwargs.update(build_load_memory_kwargs(model_cfg, res_cfg))

    init = str(model_cfg.get("init", "pretrained")).lower()
    if init in {"scratch", "scratch_from_config", "from_config"}:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
        if dtype not in {None, "auto"}:
            model = model.to(dtype=dtype)
        return model

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as causal_error:
        if not bool(model_cfg.get("allow_multimodal_fallback", True)):
            raise
        try:
            from transformers import AutoModelForMultimodalLM
        except Exception as import_error:
            raise RuntimeError(
                "AutoModelForCausalLM failed and AutoModelForMultimodalLM is not available. "
                "Install a recent transformers build for Gemma4/Qwen3.5 multimodal checkpoints."
            ) from causal_error
        try:
            return AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
        except Exception as multi_error:
            raise RuntimeError(
                "Failed to load with both AutoModelForCausalLM and AutoModelForMultimodalLM. "
                "For text-only experiments, prefer the base/causal-LM checkpoint when available."
            ) from multi_error


def load_model_and_tokenizer(cfg: Dict[str, Any], *, for_training: bool = False):
    res_cfg = cfg.get("resources", {}) or {}
    tokenizer = load_tokenizer(cfg["model"])
    log_memory("before-load")
    model = load_hf_causal_lm(cfg["model"], for_training=for_training, res_cfg=res_cfg)
    log_memory("after-load")

    patch_cfg = cfg.get("patch", {})
    if patch_cfg.get("enabled", True):
        report = patch_model(
            model,
            cfg.get("aggregation", {"name": "standard"}),
            module_name_regex=patch_cfg.get("module_name_regex"),
            collect_stats=bool(patch_cfg.get("collect_stats", True)),
            aux_ortho_coef=float(patch_cfg.get("aux_ortho_coef", 0.0) or 0.0),
            aux_positive_only=bool(patch_cfg.get("aux_positive_only", False)),
            patch_modulelist_blocks=bool(patch_cfg.get("patch_modulelist_blocks", True)),
            modulelist_return_router_logits=patch_cfg.get("modulelist_return_router_logits", None),
        )
        if report.total == 0:
            raise RuntimeError(
                "No MoE expert modules were patched. Check that the model is an MoE checkpoint "
                "and inspect module names with scripts/inspect_model.py."
            )
        model._orthomoe_patch_report = report.as_dict()
    return model, tokenizer


def prepare_for_training(model, train_cfg: Dict[str, Any]):
    """Enable gradient checkpointing, LoRA, and trainable parameter filtering."""
    if train_cfg.get("gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    if train_cfg.get("use_lora", False):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if train_cfg.get("prepare_kbit", False):
            model = prepare_model_for_kbit_training(model)
        lora_cfg = train_cfg.get("lora", {})
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias=lora_cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        model = get_peft_model(model, peft_config)

    trainable_regex = train_cfg.get("trainable_regex")
    if trainable_regex:
        import re

        pat = re.compile(trainable_regex)
        for name, param in model.named_parameters():
            param.requires_grad = bool(pat.search(name))

    return model
