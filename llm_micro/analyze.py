"""Analyze dispatch results for the LLM routing-legibility probe.

The redesigned probe separates:
- surface legibility: are role/type semantics visible in the dispatch names?
- latent competence: does the planner still pick the correct specialist when
  aliases hide the canonical role names?
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from llm_micro.scenarios import EXPLICIT_LABELS, IMPLICIT_LABELS

ROLES = EXPLICIT_LABELS


def decoded_dispatch(record: dict) -> str | None:
    dispatched = record.get("dispatched")
    if not dispatched:
        return None
    return record.get("surface_to_role", {}).get(dispatched)


def build_matrix(records: list[dict], condition: str, mode: str):
    if mode not in {"surface", "latent"}:
        raise ValueError(f"Unknown matrix mode: {mode}")

    cols = ROLES if mode == "latent" or condition == "explicit" else IMPLICIT_LABELS
    mat = np.zeros((3, 3))
    row_totals = np.zeros(3)

    for record in records:
        if record["condition"] != condition:
            continue
        label = record.get("dispatched") if mode == "surface" else decoded_dispatch(record)
        if label is None or label not in cols:
            continue
        i = ROLES.index(record["context_role"])
        j = cols.index(label)
        mat[i, j] += 1
        row_totals[i] += 1

    row_totals[row_totals == 0] = 1
    return mat / row_totals[:, None], cols


def delta_max_per_context(records: list[dict], condition: str, mode: str = "latent"):
    cols = ROLES if mode == "latent" or condition == "explicit" else IMPLICIT_LABELS
    out = {}
    for ctx in ROLES:
        dists = {}
        for formation in ["Clustered", "Spread"]:
            counts = np.zeros(3)
            total = 0
            for record in records:
                if record["condition"] != condition or record["context_role"] != ctx:
                    continue
                if record["formation"] != formation:
                    continue
                label = record.get("dispatched") if mode == "surface" else decoded_dispatch(record)
                if label is None or label not in cols:
                    continue
                counts[cols.index(label)] += 1
                total += 1
            dists[formation] = counts / max(total, 1)
        out[ctx] = float(np.max(np.abs(dists["Clustered"] - dists["Spread"])))
    return out


def accuracy(records: list[dict], condition: str, mode: str) -> float:
    subset = [r for r in records if r["condition"] == condition]
    if not subset:
        return 0.0
    if mode == "surface":
        hits = sum(r.get("dispatched") == r.get("expected_dispatch") for r in subset)
    elif mode == "latent":
        hits = sum(decoded_dispatch(r) == r["context_role"] for r in subset)
    else:
        raise ValueError(f"Unknown accuracy mode: {mode}")
    return hits / len(subset)


def plot_routing(mat, cols, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=cols,
        yticklabels=ROLES,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "dispatch probability"},
    )
    ax.set_xlabel("Dispatched")
    ax.set_ylabel("Scenario family")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def plot_accuracy(surface_explicit, surface_implicit, latent_explicit, latent_implicit, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Explicit", "Implicit"]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, [surface_explicit, surface_implicit], w, label="surface-name accuracy")
    ax.bar(x + w / 2, [latent_explicit, latent_implicit], w, label="latent-role accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Planner accuracy: visible labels vs anonymized aliases")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def plot_delta_max(deltas_explicit, deltas_implicit, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(ROLES))
    w = 0.35
    ax.bar(x - w / 2, [deltas_explicit[r] for r in ROLES], w, label="explicit (latent role)")
    ax.bar(x + w / 2, [deltas_implicit[r] for r in ROLES], w, label="implicit (latent role)")
    ax.set_xticks(x)
    ax.set_xticklabels(ROLES)
    ax.set_ylabel("Delta_max")
    ax.set_title("Dispatch shift across formations")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def matrix_std(mat):
    return float(np.std(mat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="llm_micro/results/dispatches.json")
    ap.add_argument("--outdir", type=str, default="llm_micro/results")
    args = ap.parse_args()

    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_total = len(records)
    n_parsed = sum(1 for r in records if r.get("dispatched") is not None)
    print(f"Parsed {n_parsed}/{n_total} dispatches")

    mat_exp_surface, cols_exp_surface = build_matrix(records, "explicit", "surface")
    mat_imp_surface, cols_imp_surface = build_matrix(records, "implicit", "surface")
    mat_imp_latent, cols_imp_latent = build_matrix(records, "implicit", "latent")

    plot_routing(
        mat_exp_surface,
        cols_exp_surface,
        "Routing matrix (explicit surface labels)",
        outdir / "llm_routing_explicit_surface.png",
    )
    plot_routing(
        mat_imp_surface,
        cols_imp_surface,
        "Routing matrix (implicit surface aliases)",
        outdir / "llm_routing_implicit_surface.png",
    )
    plot_routing(
        mat_imp_latent,
        cols_imp_latent,
        "Routing matrix (implicit decoded to latent roles)",
        outdir / "llm_routing_implicit_latent.png",
    )

    deltas_exp = delta_max_per_context(records, "explicit", mode="latent")
    deltas_imp = delta_max_per_context(records, "implicit", mode="latent")
    plot_delta_max(deltas_exp, deltas_imp, outdir / "llm_delta_max_latent.png")

    surface_exp_acc = accuracy(records, "explicit", "surface")
    surface_imp_acc = accuracy(records, "implicit", "surface")
    latent_exp_acc = accuracy(records, "explicit", "latent")
    latent_imp_acc = accuracy(records, "implicit", "latent")
    plot_accuracy(
        surface_exp_acc,
        surface_imp_acc,
        latent_exp_acc,
        latent_imp_acc,
        outdir / "llm_accuracy.png",
    )

    std_exp_surface = matrix_std(mat_exp_surface)
    std_imp_surface = matrix_std(mat_imp_surface)
    std_imp_latent = matrix_std(mat_imp_latent)
    legibility_ratio = std_exp_surface / max(std_imp_surface, 1e-6)

    summary = {
        "n_parsed": n_parsed,
        "n_total": n_total,
        "routing_matrix_explicit_surface": mat_exp_surface.tolist(),
        "routing_matrix_implicit_surface": mat_imp_surface.tolist(),
        "routing_matrix_implicit_latent": mat_imp_latent.tolist(),
        "surface_accuracy_explicit": surface_exp_acc,
        "surface_accuracy_implicit": surface_imp_acc,
        "latent_accuracy_explicit": latent_exp_acc,
        "latent_accuracy_implicit": latent_imp_acc,
        "routing_std_explicit_surface": std_exp_surface,
        "routing_std_implicit_surface": std_imp_surface,
        "routing_std_implicit_latent": std_imp_latent,
        "surface_legibility_ratio": legibility_ratio,
        "delta_max_explicit_latent": deltas_exp,
        "delta_max_implicit_latent": deltas_imp,
    }
    (outdir / "llm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Routing matrix (explicit surface labels) ===")
    print(f"cols={cols_exp_surface}")
    print(mat_exp_surface.round(3))
    print(f"std (all)={std_exp_surface:.3f}")

    print("\n=== Routing matrix (implicit surface aliases) ===")
    print(f"cols={cols_imp_surface}")
    print(mat_imp_surface.round(3))
    print(f"std (all)={std_imp_surface:.3f}")

    print("\n=== Routing matrix (implicit decoded latent roles) ===")
    print(f"cols={cols_imp_latent}")
    print(mat_imp_latent.round(3))
    print(f"std (all)={std_imp_latent:.3f}")

    print("\n=== Accuracy ===")
    print(f"  Explicit surface-name accuracy: {surface_exp_acc:.3f}")
    print(f"  Implicit surface-name accuracy: {surface_imp_acc:.3f}")
    print(f"  Explicit latent-role accuracy:  {latent_exp_acc:.3f}")
    print(f"  Implicit latent-role accuracy:  {latent_imp_acc:.3f}")

    print("\n=== Delta_max (latent roles) ===")
    for role in ROLES:
        print(f"  {role:10s}  explicit={deltas_exp[role]:.3f}  implicit={deltas_imp[role]:.3f}")

    print(f"\nSurface legibility ratio (explicit/implicit std): {legibility_ratio:.2f}x")
    print(f"Wrote {outdir}/llm_summary.json and plots.")


if __name__ == "__main__":
    main()
