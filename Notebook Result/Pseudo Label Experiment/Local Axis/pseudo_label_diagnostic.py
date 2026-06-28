"""
pseudo_label_diagnostic.py — FaceRankNet
=========================================
Quality diagnostic + visualization for pseudo-label variants.

Designed to be called in Cell 5 of the Colab notebook (and from the local
bandwidth-sweep cell added for the experiment/local-beauty-axis branch).

Public API
----------
pseudo_label_quality_report(pseudo_labels, holistic_ratings, ...)
    Full breakdown: global ρ, per organ, per ethnicity, per gender, per
    rating-bucket. Returns a dict; prints a readable summary.

plot_quality_report(reports, out_path=None)
    Bar + heatmap visualization of one or more report dicts.

plot_bandwidth_sweep(sweep_reports, out_path=None)
    Line chart of ρ vs bandwidth, one line per (variant × bucket).

plot_pseudo_vs_holistic(pseudo_labels, holistic_ratings, out_path=None)
    Hexbin scatter of mean-pseudo-score vs holistic rating with bucket
    overlays — for visually inspecting tail behaviour.

build_comparison_table(reports)
    Returns a pandas DataFrame ranking methods by global ρ.

Buckets follow the plan:
    Jelek  : rating  < 2
    Avg    : 2 <= rating < 3
    Mid    : 3 <= rating < 4
    Cantik : rating >= 4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


BUCKET_EDGES: list[tuple[str, float, float]] = [
    ("Jelek",  -np.inf, 2.0),
    ("Avg",     2.0,    3.0),
    ("Mid",     3.0,    4.0),
    ("Cantik",  4.0,    np.inf),
]
BUCKET_ORDER: list[str] = [b[0] for b in BUCKET_EDGES]


# ---------------------------------------------------------------------------
# Filename → gender helper (SCUT-FBP5500 naming convention)
# ---------------------------------------------------------------------------

def gender_from_filename(fname: str) -> str:
    """
    SCUT-FBP5500 names are like ``AF1814.jpg`` / ``CM137.jpg``:
    [0] = ethnicity (A/C), [1] = gender (F/M). Returns 'Female' / 'Male' /
    'Unknown'.
    """
    if len(fname) < 2:
        return "Unknown"
    g = fname[1].upper()
    if g == "F":
        return "Female"
    if g == "M":
        return "Male"
    return "Unknown"


def _bucket_for(rating: float) -> str:
    for name, lo, hi in BUCKET_EDGES:
        if lo <= rating < hi:
            return name
    return "Unknown"


def _safe_spearman(xs: list[float], ys: list[float]) -> tuple[float, int]:
    if len(xs) < 5:
        return float("nan"), len(xs)
    rho, _ = spearmanr(xs, ys)
    return float(rho), len(xs)


# ---------------------------------------------------------------------------
# Main quality report
# ---------------------------------------------------------------------------

def pseudo_label_quality_report(
    pseudo_labels: dict[str, dict[str, float]],
    holistic_ratings: dict[str, float],
    ethnicity_map: dict[str, str] | None = None,
    method_name: str = "unnamed",
    verbose: bool = True,
) -> dict:
    """
    Compute Spearman ρ between *mean pseudo-score* (across organs) and the
    holistic rating, then break that ρ down by organ, ethnicity, gender, and
    rating-bucket.

    Returns
    -------
    dict with keys:
        method            : str
        n                 : total faces used
        global_rho        : float
        per_organ         : {organ: rho}
        per_ethnicity     : {eth: {'rho': float, 'n': int}}
        per_gender        : {gender: {'rho': float, 'n': int}}
        per_bucket        : {bucket: {'rho': float, 'n': int}}
        organ_names       : list[str] — fixed display order
        bucket_names      : list[str] — fixed display order ['Jelek', 'Avg', 'Mid', 'Cantik']
    """
    common = [f for f in pseudo_labels if f in holistic_ratings]
    if len(common) < 10:
        logger.warning("Too few common samples (%d).", len(common))
        return {
            "method": method_name,
            "n": len(common),
            "global_rho": float("nan"),
            "per_organ": {},
            "per_ethnicity": {},
            "per_gender": {},
            "per_bucket": {},
        }

    organ_names = sorted({o for f in common for o in pseudo_labels[f]})

    mean_pseudo = [float(np.mean(list(pseudo_labels[f].values()))) for f in common]
    ratings = [float(holistic_ratings[f]) for f in common]

    global_rho, _ = _safe_spearman(mean_pseudo, ratings)

    # Per organ
    per_organ = {}
    for organ in organ_names:
        xs = [float(pseudo_labels[f][organ]) for f in common]
        ys = ratings
        rho, _ = _safe_spearman(xs, ys)
        per_organ[organ] = rho

    # Per ethnicity
    per_ethnicity: dict[str, dict] = {}
    if ethnicity_map:
        groups: dict[str, list[int]] = {}
        for i, f in enumerate(common):
            groups.setdefault(ethnicity_map.get(f, "Unknown"), []).append(i)
        for eth, idxs in groups.items():
            rho, n = _safe_spearman([mean_pseudo[i] for i in idxs], [ratings[i] for i in idxs])
            per_ethnicity[eth] = {"rho": rho, "n": n}

    # Per gender (from filename prefix)
    groups_g: dict[str, list[int]] = {}
    for i, f in enumerate(common):
        groups_g.setdefault(gender_from_filename(f), []).append(i)
    per_gender = {}
    for gen, idxs in groups_g.items():
        rho, n = _safe_spearman([mean_pseudo[i] for i in idxs], [ratings[i] for i in idxs])
        per_gender[gen] = {"rho": rho, "n": n}

    # Per rating-bucket
    buckets_idx: dict[str, list[int]] = {b: [] for b in BUCKET_ORDER}
    for i, r in enumerate(ratings):
        buckets_idx[_bucket_for(r)].append(i)
    per_bucket = {}
    for b in BUCKET_ORDER:
        idxs = buckets_idx[b]
        rho, n = _safe_spearman([mean_pseudo[i] for i in idxs], [ratings[i] for i in idxs])
        per_bucket[b] = {"rho": rho, "n": n}

    report = {
        "method": method_name,
        "n": len(common),
        "global_rho": global_rho,
        "per_organ": per_organ,
        "per_ethnicity": per_ethnicity,
        "per_gender": per_gender,
        "per_bucket": per_bucket,
        "organ_names": organ_names,
        "bucket_names": BUCKET_ORDER,
    }

    if verbose:
        _print_report(report)
    return report


def _fmt_rho(rho: float) -> str:
    return "  nan " if not np.isfinite(rho) else f"{rho:+.4f}"


def _print_report(r: dict) -> None:
    print("=" * 72)
    print(f"  Pseudo-label quality — method: {r['method']}   (n={r['n']})")
    print("-" * 72)
    print(f"  Global Spearman ρ = {_fmt_rho(r['global_rho'])}")
    print("-" * 72)
    print("  Per organ:")
    for o, rho in r["per_organ"].items():
        print(f"    {o:<12} {_fmt_rho(rho)}")
    if r["per_ethnicity"]:
        print("-" * 72)
        print("  Per ethnicity:")
        for eth, d in r["per_ethnicity"].items():
            print(f"    {eth:<12} {_fmt_rho(d['rho'])}   n={d['n']}")
    print("-" * 72)
    print("  Per gender:")
    for gen, d in r["per_gender"].items():
        print(f"    {gen:<12} {_fmt_rho(d['rho'])}   n={d['n']}")
    print("-" * 72)
    print("  Per rating-bucket (the bit that matters most for the tails):")
    for b in BUCKET_ORDER:
        d = r["per_bucket"][b]
        print(f"    {b:<8} {_fmt_rho(d['rho'])}   n={d['n']}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Comparison table across methods
# ---------------------------------------------------------------------------

def build_comparison_table(reports: list[dict]) -> pd.DataFrame:
    """
    Stack reports into a DataFrame: one row per method, columns =
    [global, per-bucket, per-eth, per-gender], sorted by global ρ desc.
    """
    rows = []
    for r in reports:
        row = {"method": r["method"], "n": r["n"], "global": r["global_rho"]}
        for b in BUCKET_ORDER:
            row[f"bucket/{b}"] = r["per_bucket"].get(b, {}).get("rho", float("nan"))
        for eth, d in r["per_ethnicity"].items():
            row[f"eth/{eth}"] = d["rho"]
        for gen, d in r["per_gender"].items():
            row[f"gender/{gen}"] = d["rho"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if "global" in df.columns:
        df = df.sort_values("global", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _ensure_mpl():
    import matplotlib.pyplot as plt
    return plt


def plot_quality_report(
    reports: list[dict],
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (14.0, 9.0),
):
    """
    For each report, plot:
      - bar chart of ρ per rating-bucket (4 bars)
      - bar chart of ρ per organ
      - bar chart of ρ per ethnicity (if available)
    Methods are color-coded; layout adapts to N=1..N=many methods.
    """
    plt = _ensure_mpl()
    if not reports:
        raise ValueError("reports is empty.")

    n_methods = len(reports)
    methods = [r["method"] for r in reports]
    organ_names = reports[0]["organ_names"]
    eth_names = sorted({eth for r in reports for eth in r["per_ethnicity"]})

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Pseudo-label quality — Spearman ρ vs holistic rating",
        fontsize=13, fontweight="bold",
    )

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_methods)]
    width = 0.8 / max(1, n_methods)

    # Per bucket
    ax = axes[0, 0]
    x = np.arange(len(BUCKET_ORDER))
    for j, r in enumerate(reports):
        vals = [r["per_bucket"][b]["rho"] for b in BUCKET_ORDER]
        ax.bar(x + j * width, vals, width, label=r["method"], color=colors[j])
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(BUCKET_ORDER)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title("ρ per rating-bucket (tails matter most)")
    ax.set_ylabel("Spearman ρ")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # Per organ
    ax = axes[0, 1]
    x = np.arange(len(organ_names))
    for j, r in enumerate(reports):
        vals = [r["per_organ"].get(o, float("nan")) for o in organ_names]
        ax.bar(x + j * width, vals, width, color=colors[j])
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(organ_names, rotation=20, ha="right")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title("ρ per organ")
    ax.grid(axis="y", alpha=0.3)

    # Per ethnicity
    ax = axes[1, 0]
    if eth_names:
        x = np.arange(len(eth_names))
        for j, r in enumerate(reports):
            vals = [r["per_ethnicity"].get(e, {}).get("rho", float("nan")) for e in eth_names]
            ax.bar(x + j * width, vals, width, color=colors[j])
        ax.set_xticks(x + width * (n_methods - 1) / 2)
        ax.set_xticklabels(eth_names)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_title("ρ per ethnicity")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "ethnicity not provided", ha="center", va="center")
        ax.set_axis_off()

    # Global ρ summary
    ax = axes[1, 1]
    vals = [r["global_rho"] for r in reports]
    ax.barh(np.arange(n_methods), vals, color=colors)
    ax.set_yticks(np.arange(n_methods))
    ax.set_yticklabels(methods, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_title("Global ρ — ranking summary")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:+.3f}", va="center", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        logger.info("Saved quality plot → %s", out)
    return fig


def plot_bandwidth_sweep(
    sweep_reports: dict[str, dict[float, dict]],
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (12.0, 8.0),
):
    """
    Visualize the bandwidth sweep.

    Parameters
    ----------
    sweep_reports : dict[variant_name, dict[bandwidth, report_dict]]
        e.g. ``{"local_axis": {0.3: r1, 0.5: r2, ...}, "local_rmse": {...}}``
        The baseline (axis_eth / rmse_eth) can be passed as a flat-line
        reference by including a sentinel bandwidth (e.g. -1) — it will be
        plotted as a horizontal dashed line.

    out_path : optional path to save the figure.
    """
    plt = _ensure_mpl()

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Bandwidth sweep — ρ vs σ_kernel", fontsize=13, fontweight="bold")

    panels = [("global", "Global ρ", axes[0, 0])]
    for j, b in enumerate(BUCKET_ORDER):
        row, col = (0, 1 + j) if j < 2 else (1, j - 2)
        panels.append((f"bucket/{b}", f"ρ — bucket {b}", axes[row, col]))

    # Extra panel: per-organ mean
    panels.append(("organ_mean", "ρ — mean across organs", axes[1, 2]))

    cmap = plt.get_cmap("tab10")
    variant_colors = {name: cmap(i % 10) for i, name in enumerate(sweep_reports)}

    for variant_name, by_bw in sweep_reports.items():
        bws = sorted(b for b in by_bw if b > 0)
        for metric_key, title, ax in panels:
            ys = []
            for bw in bws:
                r = by_bw[bw]
                if metric_key == "global":
                    ys.append(r["global_rho"])
                elif metric_key == "organ_mean":
                    vals = [v for v in r["per_organ"].values() if np.isfinite(v)]
                    ys.append(float(np.mean(vals)) if vals else float("nan"))
                else:
                    bname = metric_key.split("/", 1)[1]
                    ys.append(r["per_bucket"].get(bname, {}).get("rho", float("nan")))
            ax.plot(bws, ys, "o-", color=variant_colors[variant_name],
                    label=variant_name, linewidth=2, markersize=6)
            ax.set_title(title)
            ax.set_xlabel("bandwidth σ")
            ax.set_ylabel("Spearman ρ")
            ax.grid(alpha=0.3)
            ax.axhline(0, color="k", linewidth=0.4)

        # Baselines (global ref) — flat dashed line on every panel
        for bw_sentinel, r in by_bw.items():
            if bw_sentinel > 0:
                continue
            label = f"{variant_name} baseline"
            for metric_key, _, ax in panels:
                if metric_key == "global":
                    y = r["global_rho"]
                elif metric_key == "organ_mean":
                    vals = [v for v in r["per_organ"].values() if np.isfinite(v)]
                    y = float(np.mean(vals)) if vals else float("nan")
                else:
                    bname = metric_key.split("/", 1)[1]
                    y = r["per_bucket"].get(bname, {}).get("rho", float("nan"))
                ax.axhline(y, linestyle="--", color=variant_colors[variant_name],
                           alpha=0.6, label=label)
                label = None  # only one legend entry per variant

    # Single legend
    handles, labels = panels[0][2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        logger.info("Saved bandwidth-sweep plot → %s", out)
    return fig


def plot_pseudo_vs_holistic(
    pseudo_labels: dict[str, dict[str, float]],
    holistic_ratings: dict[str, float],
    method_name: str = "method",
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (7.0, 6.0),
):
    """
    Hexbin scatter of mean-pseudo-score vs holistic rating with bucket
    boundaries overlaid. Useful for spotting the failure mode in tails.
    """
    plt = _ensure_mpl()

    common = [f for f in pseudo_labels if f in holistic_ratings]
    pseudo = np.array(
        [float(np.mean(list(pseudo_labels[f].values()))) for f in common]
    )
    rating = np.array([float(holistic_ratings[f]) for f in common])

    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(rating, pseudo, gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="count")

    for _, lo, hi in BUCKET_EDGES:
        if np.isfinite(lo):
            ax.axvline(lo, color="white", linestyle=":", alpha=0.7)

    rho, _ = spearmanr(pseudo, rating)
    ax.set_xlabel("Holistic rating (GT)")
    ax.set_ylabel("Mean pseudo-score")
    ax.set_title(f"{method_name} — mean pseudo vs holistic   (ρ = {rho:+.3f}, n={len(common)})")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        logger.info("Saved pseudo-vs-holistic scatter → %s", out)
    return fig
