from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_scheduler

from .data import build_lm_dataloader
from .hf_patch import collect_moe_stats
from .losses import aggregate_aux_losses, reset_aux_losses
from .mini_moe_model import MiniMoEConfig, MiniMoELM
from .model_loader import load_model_and_tokenizer, load_tokenizer, prepare_for_training
from .resources import MemoryGuard, apply_resource_limits, log_memory
from .utils import append_jsonl, cuda_memory, load_yaml, save_json, set_seed


def extract_loss(outputs):
    if hasattr(outputs, "loss"):
        return outputs.loss
    if isinstance(outputs, dict) and "loss" in outputs:
        return outputs["loss"]
    raise RuntimeError("Model output does not contain loss")


def build_optimizer(model, train_cfg: Dict[str, Any]):
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(train_cfg.get("learning_rate", 2e-5))
    wd = float(train_cfg.get("weight_decay", 0.1))
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if opt_name in {"paged_adamw_8bit", "adamw_8bit"}:
        import bitsandbytes as bnb

        cls = bnb.optim.PagedAdamW8bit if opt_name == "paged_adamw_8bit" else bnb.optim.AdamW8bit
        return cls(params, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95), eps=1e-8)


def save_model(model, tokenizer, output_dir: Path, accelerator: Accelerator, name: str):
    save_dir = output_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(save_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
    else:
        accelerator.save(unwrapped.state_dict(), save_dir / "pytorch_model.bin")
    if accelerator.is_main_process and tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)


def build_model_for_train(cfg: Dict[str, Any]):
    train_cfg = cfg.get("training", {})
    mode = str(train_cfg.get("mode", "continue_pretrain")).lower()
    if mode == "mini":
        tokenizer = load_tokenizer(cfg["model"])
        vocab_size = int(train_cfg.get("mini", {}).get("vocab_size", len(tokenizer)))
        mini_cfg = MiniMoEConfig(vocab_size=vocab_size, **{k: v for k, v in train_cfg.get("mini", {}).items() if k != "vocab_size"})
        mini_cfg.aggregation = cfg.get("aggregation", {"name": "standard"})
        mini_cfg.aux_ortho_coef = float(cfg.get("patch", {}).get("aux_ortho_coef", 0.0) or 0.0)
        return MiniMoELM(mini_cfg), tokenizer

    model, tokenizer = load_model_and_tokenizer(cfg, for_training=True)
    model = prepare_for_training(model, train_cfg)
    return model, tokenizer


def train(config_path: str, output: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    apply_resource_limits(cfg)
    memory_guard = MemoryGuard.from_config(cfg)
    train_cfg = cfg.get("training", {})
    set_seed(int(cfg.get("seed", 42)))
    output_dir = Path(output or train_cfg.get("output_dir") or cfg.get("project", {}).get("output_dir", "outputs/orthomoe_train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output_dir / "train_config.json")

    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(train_cfg.get("mixed_precision", "bf16")),
    )

    model, tokenizer = build_model_for_train(cfg)
    dataloader = build_lm_dataloader(
        tokenizer,
        cfg.get("data", {}),
        split=cfg.get("data", {}).get("train_split", "train"),
        batch_size=int(train_cfg.get("micro_batch_size", 1)),
        block_size=int(cfg.get("data", {}).get("block_size", 1024)),
        max_samples=cfg.get("data", {}).get("max_train_samples"),
        shuffle=True,
        num_workers=int(cfg.get("data", {}).get("num_workers", 2)),
    )

    optimizer = build_optimizer(model, train_cfg)
    max_steps = int(train_cfg.get("max_steps", 1000))
    lr_scheduler = get_scheduler(
        name=str(train_cfg.get("lr_scheduler", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(train_cfg.get("warmup_steps", max(1, max_steps // 20))),
        num_training_steps=max_steps,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    model.train()
    rows = []
    jsonl_path = output_dir / "train_log.jsonl"
    if accelerator.is_main_process and jsonl_path.exists():
        jsonl_path.unlink()

    global_step = 0
    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process, desc="train")
    while global_step < max_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                reset_aux_losses(model)
                outputs = model(**batch)
                lm_loss = extract_loss(outputs)
                aux_loss = aggregate_aux_losses(model)
                loss = lm_loss + aux_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    max_norm = train_cfg.get("max_grad_norm")
                    if max_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), float(max_norm))
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress.update(1)
                    memory_guard.check(f"train:step{global_step}")

                    if accelerator.is_main_process and global_step % int(train_cfg.get("log_every", 10)) == 0:
                        stats = collect_moe_stats(model)
                        row = {
                            "step": global_step,
                            "loss": float(loss.detach().float().item()),
                            "lm_loss": float(lm_loss.detach().float().item()),
                            "aux_loss": float(aux_loss.detach().float().item()),
                            "lr": float(lr_scheduler.get_last_lr()[0]),
                            "ppl_est": math.exp(min(float(lm_loss.detach().float().item()), 20.0)),
                        }
                        row.update(stats)
                        row.update(cuda_memory())
                        rows.append(row)
                        append_jsonl(row, jsonl_path)
                        progress.set_postfix(loss=row["loss"], lr=row["lr"])

                    if global_step % int(train_cfg.get("save_every", 500)) == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            save_model(model, tokenizer, output_dir, accelerator, f"checkpoint-{global_step}")

                    if global_step >= max_steps:
                        break
        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_model(model, tokenizer, output_dir, accelerator, "final")
        if rows:
            pd.DataFrame(rows).to_csv(output_dir / "train_log.csv", index=False)
        print(f"Saved training run to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    train(args.config, args.output)


if __name__ == "__main__":
    main()
