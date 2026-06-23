from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import torch
from tqdm.auto import tqdm

from .data import build_lm_dataloader, move_batch_to_device
from .hf_patch import clear_moe_stats, collect_moe_stats, set_aggregator
from .model_loader import load_model_and_tokenizer
from .utils import Timer, append_jsonl, cuda_memory, load_yaml, save_json, set_seed


def get_input_device(model) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_loss(outputs):
    if hasattr(outputs, "loss"):
        return outputs.loss
    if isinstance(outputs, dict) and "loss" in outputs:
        return outputs["loss"]
    raise RuntimeError("Model output does not contain loss. Make sure labels are provided.")


def default_variants() -> list[dict[str, Any]]:
    return [
        {"name": "standard", "aggregation": {"name": "standard"}},
        {
            "name": "top1_pos_lam025",
            "aggregation": {"name": "top1_orthogonal", "lambda": 0.25, "positive_only": True, "stop_anchor": True},
        },
        {
            "name": "top1_pos_lam050",
            "aggregation": {"name": "top1_orthogonal", "lambda": 0.50, "positive_only": True, "stop_anchor": True},
        },
        {
            "name": "top1_signed_lam050",
            "aggregation": {"name": "top1_orthogonal", "lambda": 0.50, "positive_only": False, "stop_anchor": True},
        },
        {"name": "novelty_gated", "aggregation": {"name": "novelty_gated", "novelty_alpha": 1.0}},
        {"name": "gram_schmidt", "aggregation": {"name": "gram_schmidt", "lambda": 0.50}},
        {"name": "whitening", "aggregation": {"name": "whitening", "lambda": 0.25, "preserve_norm": True}},
    ]


@torch.no_grad()
def evaluate_variant(model, dataloader, variant: dict[str, Any], *, max_batches: int | None = None) -> dict[str, Any]:
    set_aggregator(model, variant["aggregation"])
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    device = get_input_device(model)
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    stats_accum: dict[str, list[float]] = {}

    with Timer() as timer:
        for step, batch in enumerate(tqdm(dataloader, desc=f"eval:{variant['name']}", leave=False)):
            if max_batches is not None and step >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            clear_moe_stats(model)
            outputs = model(**batch)
            loss = extract_loss(outputs)
            labels = batch.get("labels", batch["input_ids"])
            tokens = int((labels[:, 1:] != -100).sum().item())
            total_loss += float(loss.detach().float().item()) * max(tokens, 1)
            total_tokens += max(tokens, 1)
            total_batches += 1
            stats = collect_moe_stats(model)
            for key, value in stats.items():
                stats_accum.setdefault(key, []).append(value)

    mean_loss = total_loss / max(total_tokens, 1)
    row = {
        "variant": variant["name"],
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "tokens": total_tokens,
        "batches": total_batches,
        "seconds": timer.elapsed,
        "tokens_per_second": total_tokens / max(timer.elapsed, 1e-9),
    }
    row.update(cuda_memory())
    for key, vals in stats_accum.items():
        row[f"stat_{key}"] = sum(vals) / max(len(vals), 1)
    return row


def run_benchmark(config_path: str, output: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 42)))
    output_dir = Path(output or cfg.get("project", {}).get("output_dir", "outputs/orthomoe"))
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "benchmark.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    model, tokenizer = load_model_and_tokenizer(cfg, for_training=False)
    patch_report = getattr(model, "_orthomoe_patch_report", {})
    save_json({"config": cfg, "patch_report": patch_report}, output_dir / "run_config.json")

    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    dataloader = build_lm_dataloader(
        tokenizer,
        data_cfg,
        split=data_cfg.get("eval_split", "validation"),
        batch_size=int(eval_cfg.get("batch_size", 1)),
        block_size=int(data_cfg.get("block_size", 1024)),
        max_samples=data_cfg.get("max_eval_samples"),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 2)),
    )

    variants = cfg.get("variants") or default_variants()
    rows = []
    for variant in variants:
        row = evaluate_variant(model, dataloader, variant, max_batches=eval_cfg.get("max_batches"))
        rows.append(row)
        append_jsonl(row, jsonl_path)
        print(row)

    csv_path = output_dir / "benchmark.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved benchmark: {jsonl_path}")
    print(f"Saved CSV: {csv_path}")
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None, help="Output directory. Defaults to project.output_dir")
    args = parser.parse_args()
    run_benchmark(args.config, args.output)


if __name__ == "__main__":
    main()
