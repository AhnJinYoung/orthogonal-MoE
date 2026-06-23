from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


def load_text_dataset(data_cfg: Dict[str, Any], split: str):
    """Load either a Hugging Face dataset or local text/json/jsonl files."""
    from datasets import load_dataset

    if data_cfg.get("dataset_name"):
        name = data_cfg["dataset_name"]
        subset = data_cfg.get("dataset_config_name")
        streaming = bool(data_cfg.get("streaming", False))
        return load_dataset(name, subset, split=split, streaming=streaming)

    path = data_cfg.get("data_files")
    if not path:
        raise ValueError("Set data.dataset_name or data.data_files in the config.")
    extension = str(path).split(".")[-1]
    if extension == "txt":
        return load_dataset("text", data_files={split: path}, split=split)
    if extension in {"json", "jsonl"}:
        return load_dataset("json", data_files={split: path}, split=split)
    raise ValueError(f"Unsupported local data extension: {extension}")


def build_lm_dataloader(
    tokenizer,
    data_cfg: Dict[str, Any],
    *,
    split: str,
    batch_size: int,
    block_size: int,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    num_workers: int = 2,
):
    """Tokenize text and return a standard causal-LM dataloader."""
    ds = load_text_dataset(data_cfg, split)
    text_column = data_cfg.get("text_column", "text")

    if max_samples is not None and not data_cfg.get("streaming", False):
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))

    def tokenize(batch):
        texts = batch[text_column]
        texts = [x if isinstance(x, str) else "" for x in texts]
        return tokenizer(texts, add_special_tokens=False)

    remove_columns = None if data_cfg.get("streaming", False) else list(ds.column_names)
    tokenized = ds.map(tokenize, batched=True, remove_columns=remove_columns)

    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
            if tokenizer.eos_token_id is not None:
                concatenated.append(tokenizer.eos_token_id)
        total_length = (len(concatenated) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
        attention_mask = [[1] * block_size for _ in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": [x.copy() for x in input_ids]}

    lm_ds = tokenized.map(group_texts, batched=True)
    if max_samples is not None and data_cfg.get("streaming", False):
        lm_ds = lm_ds.take(max_samples)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(
        lm_ds,
        batch_size=batch_size,
        shuffle=shuffle and not data_cfg.get("streaming", False),
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device | str):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
