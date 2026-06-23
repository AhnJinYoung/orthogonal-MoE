import torch

from orthomoe.aggregations import build_aggregator


def test_all_aggregators_shape():
    torch.manual_seed(0)
    x = torch.randn(16, 4, 32)
    gates = torch.softmax(torch.randn(16, 4), dim=-1)
    variants = [
        {"name": "standard"},
        {"name": "top1_orthogonal", "lambda": 0.5},
        {"name": "weighted_sum_top1_projection", "lambda": 0.5},
        {"name": "gram_schmidt", "lambda": 0.5},
        {"name": "whitening", "lambda": 0.25},
        {"name": "novelty_gated"},
        {"name": "norm_matched_shrinkage"},
        {"name": "random_anchor_projection"},
    ]
    for cfg in variants:
        agg = build_aggregator(cfg)
        y, stats = agg(x, gates)
        assert y.shape == (16, 32)
        assert torch.isfinite(y).all()
        assert "gate_entropy" in stats
