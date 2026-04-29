"""Generate the scaling-study figure for the ICML/NExT paper."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = ROOT / "smacv2_poc" / "results_scaling" / "logs"
OUT = Path(__file__).resolve().parent / "scaling_performance.png"


def load_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(LOG_DIR.glob("*.csv")):
        match = re.match(r"(\dv\d)_balanced_cond(\d)_seed(\d+)\.csv", path.name)
        if not match:
            continue

        scale, cond_raw, seed_raw = match.groups()
        cond = int(cond_raw)
        seed = int(seed_raw)

        # This run was interrupted around 1.13M steps; seed 3 is the completed
        # replacement run for 9v9 Condition 1.
        if scale == "9v9" and cond == 1 and seed == 2:
            continue

        with path.open(newline="") as handle:
            data = list(csv.DictReader(handle))
        if not data or int(data[-1]["step"]) < 1_900_000:
            continue

        win_rates = [float(row["win_rate"]) for row in data]
        rows.append(
            {
                "scale": scale,
                "cond": cond,
                "seed": seed,
                "last10": mean(win_rates[-10:]),
                "auc": mean(win_rates),
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> dict[tuple[str, int], dict[str, tuple[float, float]]]:
    summary: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}
    for scale in ["3v3", "6v6", "9v9"]:
        for cond in [1, 3]:
            values = [row for row in rows if row["scale"] == scale and row["cond"] == cond]
            summary[(scale, cond)] = {}
            for metric in ["last10", "auc"]:
                metric_values = [float(row[metric]) for row in values]
                summary[(scale, cond)][metric] = (
                    mean(metric_values),
                    stdev(metric_values) if len(metric_values) > 1 else 0.0,
                )
    return summary


def main() -> None:
    rows = load_rows()
    summary = summarize(rows)

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        },
    )

    scales = ["3v3", "6v6", "9v9"]
    x = list(range(len(scales)))
    colors = {1: "#2F5FA8", 3: "#D9822B"}
    labels = {1: "Cond. 1: Full attention", 3: "Cond. 3: Role+MLP"}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    metrics = [("last10", "Late training win rate"), ("auc", "Mean win rate over training")]

    for ax, (metric, title) in zip(axes, metrics):
        for cond in [1, 3]:
            y = [summary[(scale, cond)][metric][0] for scale in scales]
            err = [summary[(scale, cond)][metric][1] for scale in scales]
            ax.errorbar(
                x,
                y,
                yerr=err,
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                capsize=3,
                color=colors[cond],
                label=labels[cond],
            )
            for xi, yi in zip(x, y):
                ax.text(xi, yi + 0.035, f"{yi:.2f}", ha="center", va="bottom", fontsize=7, color=colors[cond])

        ax.set_title(title)
        ax.set_xticks(x, scales)
        ax.set_ylim(0, 0.9)
        ax.set_xlabel("Team scale")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Win rate")
    axes[1].legend(loc="upper right", frameon=False)
    fig.text(
        0.01,
        0.01,
        "Mean +/- std over three completed seeds. 9v9 Cond. 1 excludes interrupted seed 2 and uses seed 3.",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")


if __name__ == "__main__":
    main()
