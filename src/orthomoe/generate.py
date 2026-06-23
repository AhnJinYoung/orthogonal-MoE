from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from .hf_patch import set_aggregator
from .model_loader import load_model_and_tokenizer
from .utils import append_jsonl, load_yaml, set_seed


def input_device(model):
    for p in model.parameters():
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def generate_for_variant(model, tokenizer, variant: dict[str, Any], prompts: list[str], gen_cfg: dict[str, Any]):
    set_aggregator(model, variant["aggregation"])
    model.eval()
    device = input_device(model)
    rows = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 128)),
            do_sample=bool(gen_cfg.get("do_sample", False)),
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.95)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        rows.append({"variant": variant["name"], "prompt": prompt, "text": text})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    model, tokenizer = load_model_and_tokenizer(cfg, for_training=False)
    prompts = cfg.get("generation", {}).get(
        "prompts",
        [
            "Explain mixture-of-experts routing in one paragraph.",
            "Write a small PyTorch function for cosine similarity.",
        ],
    )
    variants = cfg.get("generation", {}).get("variants") or cfg.get("variants", [])[:3]
    out_path = Path(args.output)
    if out_path.exists():
        out_path.unlink()
    for variant in variants:
        for row in generate_for_variant(model, tokenizer, variant, prompts, cfg.get("generation", {})):
            append_jsonl(row, out_path)
            print(f"[{row['variant']}] {row['text'][:160].replace(chr(10), ' ')}...")
    print(f"Saved generations to {out_path}")


if __name__ == "__main__":
    main()
