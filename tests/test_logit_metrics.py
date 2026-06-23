import math

import torch

from orthomoe.logit_metrics import LogitMetricAccumulator


def test_logit_metrics_basic_invariants():
    torch.manual_seed(0)
    acc = LogitMetricAccumulator(chunk_size=7, sample_cap=100, sample_stride=2)
    B, S, V = 2, 8, 50
    for _ in range(3):
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))
        labels[0, 0] = -100  # dropped by the causal shift anyway
        acc.update(logits, labels)

    r = acc.result()
    # 3 batches * B * (S-1) shifted positions, all valid.
    assert r["eval_tokens"] == 3 * B * (S - 1)
    assert 0.0 <= r["token_accuracy"] <= 1.0
    assert r["top5_accuracy"] >= r["token_accuracy"]
    assert math.isclose(r["bits_per_token"], r["loss_logit"] / math.log(2), rel_tol=1e-6)
    assert r["perplexity"] > 0.0
    assert r["effective_classes"] >= 1.0

    samples = acc.samples()
    for field in ("nll", "entropy", "margin", "top1_prob"):
        assert field in samples
        assert samples[field].ndim == 1


def test_logit_metrics_ignore_index_only():
    acc = LogitMetricAccumulator()
    logits = torch.randn(1, 4, 10)
    labels = torch.full((1, 4), -100)
    acc.update(logits, labels)
    assert acc.total_tokens == 0
    # result() must be safe even with zero tokens.
    assert acc.result()["eval_tokens"] == 0


def test_perfect_predictions_high_accuracy():
    V = 12
    targets = torch.tensor([[3, 5, 7, 1]])
    logits = torch.full((1, 4, V), -10.0)
    # Make position t strongly predict targets[t+1] (next-token convention).
    for t in range(3):
        logits[0, t, targets[0, t + 1]] = 10.0
    acc = LogitMetricAccumulator()
    acc.update(logits, targets)
    r = acc.result()
    assert r["token_accuracy"] == 1.0
    assert r["top1_prob"] > 0.99
