"""Memory-safe streaming metrics over output logits.

The benchmark already reports loss/perplexity. This module adds the rest of a
detailed language-model evaluation plus measures of how *expressive* the output
distribution is, so we can see how the orthogonal-aggregation variants change
the model's predictions:

* ``perplexity`` / ``bits_per_token``  -- standard LM quality.
* ``token_accuracy`` / ``top5_accuracy`` -- argmax next-token correctness.
* ``pred_entropy`` -- mean entropy of the softmax over the vocabulary. Higher
  entropy => the model spreads probability over more tokens (more hedging /
  more expressive support); lower => sharper, more confident predictions.
* ``effective_classes`` = exp(entropy) -- number of tokens the distribution
  effectively competes over.
* ``logit_margin`` -- top1 minus top2 logit; a direct expressiveness/confidence
  signal that does not require a full softmax.
* ``top1_prob`` -- mean probability mass on the argmax token.

Everything is accumulated in chunks over the flattened token dimension so we
never hold a full ``[tokens, vocab]`` softmax in memory at once -- important for
26B/35B checkpoints with ~256k vocab on RAM/VRAM-constrained pods.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


class LogitMetricAccumulator:
    """Accumulate detailed next-token metrics across batches.

    Args:
        chunk_size: number of tokens processed per softmax chunk.
        sample_cap: max per-token values retained (per field) for distribution
            plots. Kept on CPU as float16 to bound memory.
        sample_stride: keep one in every ``sample_stride`` valid tokens.
    """

    def __init__(self, *, chunk_size: int = 2048, sample_cap: int = 50000, sample_stride: int = 1) -> None:
        self.chunk_size = int(chunk_size)
        self.sample_cap = int(sample_cap)
        self.sample_stride = max(1, int(sample_stride))

        self.total_tokens = 0
        self.sum_nll = 0.0
        self.sum_entropy = 0.0
        self.sum_margin = 0.0
        self.sum_top1_prob = 0.0
        self.correct_top1 = 0
        self.correct_top5 = 0

        self._samples: Dict[str, list] = {"nll": [], "entropy": [], "margin": [], "top1_prob": []}
        self._seen_valid = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Consume one batch of logits/labels (causal shift handled here)."""
        if logits.dim() == 3:
            logits = logits[:, :-1, :]
            labels = labels[:, 1:]
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)

        valid = labels != -100
        if valid.any():
            logits = logits[valid]
            labels = labels[valid]
        else:
            return

        n = labels.shape[0]
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            self._update_chunk(logits[start:end].float(), labels[start:end])

    def _update_chunk(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        # top-2 logits give both the margin and the top-1 accuracy cheaply.
        top2_val, top2_idx = logits.topk(2, dim=-1)
        margin = top2_val[:, 0] - top2_val[:, 1]
        pred1 = top2_idx[:, 0]
        top1_prob = probs.gather(-1, pred1.unsqueeze(-1)).squeeze(-1)

        top5_idx = logits.topk(5, dim=-1).indices
        correct5 = (top5_idx == targets.unsqueeze(-1)).any(dim=-1)

        m = targets.shape[0]
        self.total_tokens += m
        self.sum_nll += float(nll.sum().item())
        self.sum_entropy += float(entropy.sum().item())
        self.sum_margin += float(margin.sum().item())
        self.sum_top1_prob += float(top1_prob.sum().item())
        self.correct_top1 += int((pred1 == targets).sum().item())
        self.correct_top5 += int(correct5.sum().item())

        self._stash_samples(nll=nll, entropy=entropy, margin=margin, top1_prob=top1_prob)

    def _stash_samples(self, **fields: torch.Tensor) -> None:
        if sum(len(v) for v in self._samples.values()) >= self.sample_cap * len(self._samples):
            return
        m = next(iter(fields.values())).shape[0]
        # Honour the stride against the global token counter for an even sample.
        idx = torch.arange(m)
        keep = ((self._seen_valid + idx) % self.sample_stride == 0)
        self._seen_valid += m
        if not keep.any():
            return
        budget = self.sample_cap - len(self._samples["nll"])
        if budget <= 0:
            return
        sel = keep.nonzero(as_tuple=False).flatten()[:budget].to(next(iter(fields.values())).device)
        for name, tensor in fields.items():
            self._samples[name].append(tensor[sel].detach().to("cpu", torch.float16))

    def result(self) -> Dict[str, float]:
        """Aggregate scalar metrics. Safe to call with zero tokens."""
        n = max(self.total_tokens, 1)
        mean_nll = self.sum_nll / n
        mean_entropy = self.sum_entropy / n
        return {
            "eval_tokens": self.total_tokens,
            "loss_logit": mean_nll,
            "perplexity": math.exp(min(mean_nll, 20.0)),
            "bits_per_token": mean_nll / LN2,
            "token_accuracy": self.correct_top1 / n,
            "top5_accuracy": self.correct_top5 / n,
            "pred_entropy": mean_entropy,
            "pred_entropy_bits": mean_entropy / LN2,
            "effective_classes": math.exp(min(mean_entropy, 20.0)),
            "logit_margin": self.sum_margin / n,
            "top1_prob": self.sum_top1_prob / n,
        }

    def samples(self) -> Dict[str, np.ndarray]:
        """Concatenated per-token samples for distribution plots."""
        out: Dict[str, np.ndarray] = {}
        for name, chunks in self._samples.items():
            if chunks:
                out[name] = torch.cat(chunks).float().numpy()
            else:
                out[name] = np.empty(0, dtype=np.float32)
        return out
