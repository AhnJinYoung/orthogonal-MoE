#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

from transformers import AutoConfig

from orthomoe.model_loader import load_hf_causal_lm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--load", action="store_true", help="Actually load weights and print module names. Without this, prints config only.")
    args = parser.parse_args()
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    print(cfg)
    if not args.load:
        return
    model = load_hf_causal_lm({"model_id": args.model_id, "dtype": "bfloat16", "device_map": "auto"}, for_training=False)
    counts = Counter(type(m).__name__ for m in model.modules())
    print("\nTop module classes:")
    for name, count in counts.most_common(40):
        print(f"{count:5d} {name}")
    print("\nLikely tensor expert modules:")
    for name, m in model.named_modules():
        if all(hasattr(m, attr) for attr in ["gate_up_proj", "down_proj", "act_fn", "num_experts"]):
            print(name, type(m).__name__)
    print("\nLikely sparse blocks:")
    for name, m in model.named_modules():
        if hasattr(m, "gate") and hasattr(m, "experts"):
            print(name, type(m).__name__)


if __name__ == "__main__":
    main()
