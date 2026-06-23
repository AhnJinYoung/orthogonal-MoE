from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import read_jsonl


def load_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame(read_jsonl(path))


def bar_plot(df: pd.DataFrame, y: str, out_path: Path, title: str) -> None:
    if y not in df.columns:
        return
    plot_df = df.sort_values(y, ascending=True)
    fig = plt.figure(figsize=(max(8, len(plot_df) * 1.2), 5))
    plt.bar(plot_df["variant"], plot_df[y])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def scatter_plot(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    if x not in df.columns or y not in df.columns:
        return
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(df[x], df[y])
    for _, row in df.iterrows():
        plt.annotate(str(row["variant"]), (row[x], row[y]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def visualize(results: str | Path, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_results(results)
    df.to_csv(outdir / "results_table.csv", index=False)

    bar_plot(df, "perplexity", outdir / "perplexity_by_variant.png", "Perplexity by aggregation variant")
    bar_plot(df, "loss", outdir / "loss_by_variant.png", "Validation loss by aggregation variant")
    bar_plot(df, "tokens_per_second", outdir / "throughput_by_variant.png", "Throughput by aggregation variant")
    bar_plot(df, "stat_cos_top1_abs_mean", outdir / "cos_top1_abs_by_variant.png", "Mean |cos(top1, other)|")
    bar_plot(df, "stat_novelty_top1_mean", outdir / "novelty_by_variant.png", "Mean selected-expert novelty")
    bar_plot(df, "cuda_peak_allocated_gb", outdir / "peak_memory_by_variant.png", "Peak CUDA memory")
    scatter_plot(
        df,
        "stat_cos_top1_abs_mean",
        "perplexity",
        outdir / "ppl_vs_cos_top1_abs.png",
        "Perplexity vs selected-expert redundancy",
    )
    scatter_plot(
        df,
        "tokens_per_second",
        "perplexity",
        outdir / "ppl_vs_throughput.png",
        "Perplexity vs throughput",
    )
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
