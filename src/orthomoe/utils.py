from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
import yaml


def load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(data: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def append_jsonl(row: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dtype_from_string(name: str | None):
    if name is None:
        return None
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "full"}:
        return torch.float32
    if name in {"auto"}:
        return "auto"
    raise ValueError(f"Unknown dtype: {name}")


def cuda_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"cuda_allocated_gb": 0.0, "cuda_reserved_gb": 0.0, "cuda_peak_allocated_gb": 0.0}
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "cuda_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def flatten_dict(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten_dict(v, prefix=key + "."))
        else:
            out[key] = v
    return out
