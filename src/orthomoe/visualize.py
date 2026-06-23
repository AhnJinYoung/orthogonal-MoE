from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless: no display on pods
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import read_jsonl


def load_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame(read_jsonl(path))


def bar_plot(df: pd.DataFrame, y: str, out_path: Path, title: str, *, ascending: bool = True) -> None:
    if y not in df.columns or df[y].isna().all():
        return
    plot_df = df.dropna(subset=[y]).sort_values(y, ascending=ascending)
    fig = plt.figure(figsize=(max(8, len(plot_df) * 1.2), 5))
    bars = plt.bar(plot_df["variant"], plot_df[y])
    # Mark the standard baseline so deltas are easy to read at a glance.
    for bar, name in zip(bars, plot_df["variant"]):
        if str(name) == "standard":
            bar.set_color("#444444")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def scatter_plot(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    if x not in df.columns or y not in df.columns:
        return
    sub = df.dropna(subset=[x, y])
    if sub.empty:
        return
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(sub[x], sub[y])
    for _, row in sub.iterrows():
        plt.annotate(str(row["variant"]), (row[x], row[y]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def delta_vs_standard_plot(df: pd.DataFrame, columns: list[str], out_path: Path) -> None:
    """Grouped bars of each variant's metric relative to the standard baseline.

    Shows *how much* the output distribution changes per variant, normalised so
    different-scale metrics are comparable on one axis.
    """
    cols = [c for c in columns if c in df.columns and not df[c].isna().all()]
    if not cols or "standard" not in set(df["variant"]):
        return
    base = df[df["variant"] == "standard"].iloc[0]
    others = df[df["variant"] != "standard"]
    if others.empty:
        return
    variants = list(others["variant"])
    x = np.arange(len(variants))
    width = 0.8 / max(len(cols), 1)
    fig = plt.figure(figsize=(max(9, len(variants) * 1.4), 5))
    for i, col in enumerate(cols):
        denom = abs(base[col]) if abs(base[col]) > 1e-9 else 1.0
        rel = [(row[col] - base[col]) / denom for _, row in others.iterrows()]
        plt.bar(x + i * width, rel, width, label=col)
    plt.axhline(0.0, color="#888888", linewidth=0.8)
    plt.xticks(x + width * (len(cols) - 1) / 2, variants, rotation=35, ha="right")
    plt.ylabel("relative change vs standard")
    plt.title("Output-distribution change per variant (vs standard baseline)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _load_logit_samples(results: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load per-variant per-token sample arrays dumped by the benchmark."""
    logit_dir = results.parent / "logit_stats"
    samples: Dict[str, Dict[str, np.ndarray]] = {}
    if not logit_dir.is_dir():
        return samples
    for npz in sorted(logit_dir.glob("*.npz")):
        with np.load(npz) as data:
            samples[npz.stem] = {k: data[k] for k in data.files}
    return samples


def distribution_overlay(samples: Dict[str, Dict[str, np.ndarray]], field: str, out_path: Path, title: str, xlabel: str) -> None:
    """Overlay per-variant histograms of a per-token quantity (e.g. entropy).

    This is the core "how expressive did the model become" view: it shows the
    full shape of the predictive distribution per variant, not just the mean.
    """
    present = {name: s[field] for name, s in samples.items() if field in s and s[field].size > 0}
    if not present:
        return
    all_vals = np.concatenate(list(present.values()))
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(all_vals.min()), float(all_vals.max()) + 1e-6
    bins = np.linspace(lo, hi, 60)
    fig = plt.figure(figsize=(8, 5))
    for name, vals in present.items():
        plt.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.6, label=name)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summary_heatmap(df: pd.DataFrame, columns: list[str], out_path: Path) -> None:
    """Z-scored heatmap of all variants x metrics for a one-glance comparison."""
    cols = [c for c in columns if c in df.columns and not df[c].isna().all()]
    if not cols or df.empty:
        return
    mat = df[cols].to_numpy(dtype=float)
    mean = np.nanmean(mat, axis=0, keepdims=True)
    std = np.nanstd(mat, axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    z = (mat - mean) / std
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.1), max(4, len(df) * 0.5)))
    im = ax.imshow(z, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(list(df["variant"]), fontsize=8)
    fig.colorbar(im, ax=ax, label="z-score across variants")
    ax.set_title("Per-variant metric heatmap (z-scored per column)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def visualize(results: str | Path, outdir: str | Path) -> Path:
    results = Path(results)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_results(results)
    df.to_csv(outdir / "results_table.csv", index=False)

    # Quality / efficiency.
    bar_plot(df, "perplexity", outdir / "perplexity_by_variant.png", "Perplexity by aggregation variant")
    bar_plot(df, "bits_per_token", outdir / "bits_per_token_by_variant.png", "Bits/token by aggregation variant")
    bar_plot(df, "loss", outdir / "loss_by_variant.png", "Validation loss by aggregation variant")
    bar_plot(df, "tokens_per_second", outdir / "throughput_by_variant.png", "Throughput by aggregation variant", ascending=False)
    bar_plot(df, "cuda_peak_allocated_gb", outdir / "peak_memory_by_variant.png", "Peak CUDA memory")

    # Prediction quality.
    bar_plot(df, "token_accuracy", outdir / "token_accuracy_by_variant.png", "Next-token top-1 accuracy", ascending=False)
    bar_plot(df, "top5_accuracy", outdir / "top5_accuracy_by_variant.png", "Next-token top-5 accuracy", ascending=False)

    # Output-logit expressiveness.
    bar_plot(df, "pred_entropy_bits", outdir / "pred_entropy_by_variant.png", "Predictive entropy (bits) by variant")
    bar_plot(df, "effective_classes", outdir / "effective_classes_by_variant.png", "Effective #classes (exp entropy) by variant")
    bar_plot(df, "logit_margin", outdir / "logit_margin_by_variant.png", "Top1-Top2 logit margin by variant")
    bar_plot(df, "top1_prob", outdir / "top1_prob_by_variant.png", "Top-1 softmax confidence by variant")

    # MoE geometry.
    bar_plot(df, "stat_cos_top1_abs_mean", outdir / "cos_top1_abs_by_variant.png", "Mean |cos(top1, other)|")
    bar_plot(df, "stat_novelty_top1_mean", outdir / "novelty_by_variant.png", "Mean selected-expert novelty", ascending=False)

    # Relationships.
    scatter_plot(df, "stat_cos_top1_abs_mean", "perplexity", outdir / "ppl_vs_cos_top1_abs.png", "Perplexity vs selected-expert redundancy")
    scatter_plot(df, "tokens_per_second", "perplexity", outdir / "ppl_vs_throughput.png", "Perplexity vs throughput")
    scatter_plot(df, "pred_entropy_bits", "perplexity", outdir / "ppl_vs_entropy.png", "Perplexity vs predictive entropy")
    scatter_plot(df, "logit_margin", "token_accuracy", outdir / "accuracy_vs_margin.png", "Accuracy vs logit margin")

    # Cross-metric summaries.
    expressiveness_cols = ["perplexity", "bits_per_token", "token_accuracy", "pred_entropy_bits", "logit_margin", "top1_prob"]
    delta_vs_standard_plot(df, expressiveness_cols, outdir / "expressiveness_delta_vs_standard.png")
    summary_heatmap(
        df,
        expressiveness_cols + ["stat_cos_top1_abs_mean", "stat_novelty_top1_mean", "tokens_per_second"],
        outdir / "metric_heatmap.png",
    )

    # Per-token distribution overlays (the "how the logits change" view).
    samples = _load_logit_samples(results)
    if samples:
        distribution_overlay(samples, "entropy", outdir / "dist_entropy.png", "Per-token predictive entropy distribution", "entropy (nats)")
        distribution_overlay(samples, "margin", outdir / "dist_margin.png", "Per-token top1-top2 logit margin distribution", "logit margin")
        distribution_overlay(samples, "top1_prob", outdir / "dist_top1_prob.png", "Per-token top-1 probability distribution", "top-1 probability")
        distribution_overlay(samples, "nll", outdir / "dist_nll.png", "Per-token negative log-likelihood distribution", "NLL (nats)")
    return outdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="benchmark.jsonl or benchmark.csv")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    out = visualize(args.results, args.outdir)
    print(f"Saved figures to {out}")


if __name__ == "__main__":
    main()
