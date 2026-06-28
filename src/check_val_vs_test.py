"""
check_val_vs_test.py — distribution comparison between val and test splits.

Hipotesis: gap val-test PCC ~0.05 disebabkan test set lebih sulit, BUKAN
overfit ke val. Cara cek: bandingkan distribusi rating, etnis, gender
antara val (carved dari train_csv) dan test (test_csv).

Usage (Colab atau lokal):
    from check_val_vs_test import compare_val_vs_test
    compare_val_vs_test(TRAIN_CSV, TEST_CSV)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

import config


def _bucket_of(r: float, edges: tuple[float, ...]) -> int:
    for i, e in enumerate(edges):
        if r < e:
            return i
    return len(edges)


def _stratified_val_split(df: pd.DataFrame, train_frac: float, seed: int):
    buckets = [_bucket_of(float(r), config.BUCKET_EDGES)
               for r in df[config.COL_RATING].tolist()]
    train_df, val_df = train_test_split(
        df, train_size=train_frac, random_state=seed, stratify=buckets,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _print_dist(name: str, series: pd.Series, bins: list[float] | None = None) -> dict:
    print(f"\n--- {name} ---")
    if bins is not None:
        cuts = pd.cut(series, bins=bins, include_lowest=True,
                      labels=[f"[{bins[i]:.1f},{bins[i+1]:.1f})"
                              for i in range(len(bins) - 1)])
        counts = cuts.value_counts().sort_index()
        pct = (counts / counts.sum() * 100).round(2)
        out = pd.DataFrame({"n": counts, "%": pct})
        print(out.to_string())
        return out.to_dict()
    else:
        counts = series.value_counts()
        pct = (counts / counts.sum() * 100).round(2)
        out = pd.DataFrame({"n": counts, "%": pct})
        print(out.to_string())
        return out.to_dict()


def compare_val_vs_test(
    train_csv: str,
    test_csv: str,
    train_frac: float = config.TRAIN_VAL_SPLIT,
    seed: int = config.SEED if hasattr(config, "SEED") else 42,
) -> dict:
    """Compare val (carved from train_csv) against test_csv."""
    train_full = pd.read_csv(train_csv)
    test_df    = pd.read_csv(test_csv)

    for df in (train_full, test_df):
        if "Ethnicity" not in df.columns:
            df["Ethnicity"] = df["Filename"].apply(
                lambda f: "Asian" if f[0].upper() == "A" else "Caucasian")
        if "Gender" not in df.columns:
            df["Gender"] = df["Filename"].apply(
                lambda f: "Female" if f[1].upper() == "F" else "Male")

    _, val_df = _stratified_val_split(train_full, train_frac, seed)

    print("=" * 60)
    print(f"  Val size : {len(val_df)}   |  Test size : {len(test_df)}")
    print("=" * 60)

    rating_bins = [1.0, 2.0, 3.0, 4.0, 5.01]

    print("\n### Rating distribution ###")
    _print_dist("VAL  rating buckets",  val_df[config.COL_RATING],  rating_bins)
    _print_dist("TEST rating buckets",  test_df[config.COL_RATING], rating_bins)

    print(f"\nVAL  rating  mean={val_df[config.COL_RATING].mean():.3f}  "
          f"std={val_df[config.COL_RATING].std():.3f}  "
          f"high(>4)={(val_df[config.COL_RATING] > 4).sum()} "
          f"({(val_df[config.COL_RATING] > 4).mean()*100:.2f}%)")
    print(f"TEST rating  mean={test_df[config.COL_RATING].mean():.3f}  "
          f"std={test_df[config.COL_RATING].std():.3f}  "
          f"high(>4)={(test_df[config.COL_RATING] > 4).sum()} "
          f"({(test_df[config.COL_RATING] > 4).mean()*100:.2f}%)")

    ks_stat, ks_p = stats.ks_2samp(val_df[config.COL_RATING],
                                   test_df[config.COL_RATING])
    print(f"\nKS test (rating val vs test):  D={ks_stat:.4f}  p={ks_p:.4f}")
    print("  -> p>0.05 means same distribution (gap likely from model variance, not split)")
    print("  -> p<0.05 means distributions differ (some of the gap is structural)")

    print("\n### Ethnicity distribution ###")
    _print_dist("VAL  ethnicity",  val_df["Ethnicity"])
    _print_dist("TEST ethnicity",  test_df["Ethnicity"])

    print("\n### Gender distribution ###")
    _print_dist("VAL  gender",  val_df["Gender"])
    _print_dist("TEST gender",  test_df["Gender"])

    print("\n### High-rating (>4) breakdown ###")
    for name, df in [("VAL", val_df), ("TEST", test_df)]:
        hi = df[df[config.COL_RATING] > 4]
        if len(hi) == 0:
            print(f"  {name}: 0 high-rating faces")
            continue
        eth = hi["Ethnicity"].value_counts().to_dict()
        gen = hi["Gender"].value_counts().to_dict()
        print(f"  {name}  n={len(hi)}  ethnicity={eth}  gender={gen}")

    return {
        "val_n": len(val_df),
        "test_n": len(test_df),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "val_rating_mean": float(val_df[config.COL_RATING].mean()),
        "test_rating_mean": float(test_df[config.COL_RATING].mean()),
        "val_high_frac": float((val_df[config.COL_RATING] > 4).mean()),
        "test_high_frac": float((test_df[config.COL_RATING] > 4).mean()),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python check_val_vs_test.py <train_csv> <test_csv>")
        sys.exit(1)
    compare_val_vs_test(sys.argv[1], sys.argv[2])
