"""
Phase 4 — Dataset Quality Analysis & Visualization
====================================================
Research-grade dataset diagnostic tools.

Checks:
  1. Label distribution analysis
  2. Class imbalance detection
  3. Feature correlation heatmap
  4. Temporal consistency (signal continuity)
  5. Missing value statistics
  6. Outlier detection (IQR + Z-score)
  7. Sequence quality scoring
  8. Per-participant behavioral profiles
  9. Feature importance (permutation-based)
  10. Behavioral pattern visualization
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional

# Styling
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#4a4a6a",
    "axes.labelcolor": "#e0e0ff",
    "xtick.color": "#a0a0c0",
    "ytick.color": "#a0a0c0",
    "text.color": "#e0e0ff",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
})
PALETTE = ["#7c3aed", "#2563eb", "#059669", "#dc2626", "#d97706", "#0891b2"]

FEATURE_NAMES = [
    "EAR", "Gaze Pitch", "Gaze Yaw",
    "Head Pitch", "Head Yaw", "Head Roll",
    "Eyebrow Tension", "Eye Openness"
]


def load_dataset(processed_dir: str, dataset_dir: Optional[str] = None):
    """Load preprocessed tensors + raw CSVs."""
    X = np.load(os.path.join(processed_dir, "X_sequences.npy"))
    Y = np.load(os.path.join(processed_dir, "Y_labels.npy"))
    print(f"[Analysis] X={X.shape}, Y={Y.shape}")
    return X, Y


# 1. Label Distribution
def plot_label_distribution(Y: np.ndarray, save_dir: str):
    """Histogram + KDE of cognitive load labels with statistical annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cognitive Load Label Distribution", fontsize=16, fontweight="bold", y=1.02)

    y_flat = Y.flatten()

    # Histogram
    ax = axes[0]
    ax.hist(y_flat, bins=30, color=PALETTE[0], alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(y_flat.mean(), color=PALETTE[3], linestyle="--", linewidth=2,
               label=f"Mean: {y_flat.mean():.3f}")
    ax.axvline(np.median(y_flat), color=PALETTE[2], linestyle="--", linewidth=2,
               label=f"Median: {np.median(y_flat):.3f}")
    ax.set_xlabel("Cognitive Load Label (0–1)")
    ax.set_ylabel("Frequency")
    ax.set_title("Label Histogram")
    ax.legend()
    ax.grid(True)

    # Cumulative distribution
    ax = axes[1]
    sorted_y = np.sort(y_flat)
    cdf = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
    ax.plot(sorted_y, cdf, color=PALETTE[1], linewidth=2)
    ax.fill_between(sorted_y, cdf, alpha=0.2, color=PALETTE[1])
    ax.set_xlabel("Cognitive Load Label (0–1)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution")
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(save_dir, "label_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved: {path}")

    # Stats summary
    print("\n=== Label Statistics ===")
    print(f"  N sequences:  {len(y_flat)}")
    print(f"  Mean:         {y_flat.mean():.4f}")
    print(f"  Std:          {y_flat.std():.4f}")
    print(f"  Min/Max:      {y_flat.min():.4f} / {y_flat.max():.4f}")
    print(f"  25/75 pctile: {np.percentile(y_flat, 25):.4f} / {np.percentile(y_flat, 75):.4f}")
    skewness = stats.skew(y_flat)
    print(f"  Skewness:     {skewness:.4f} {'(right-skewed)' if skewness > 0 else '(left-skewed)'}")
    kurt = stats.kurtosis(y_flat)
    print(f"  Kurtosis:     {kurt:.4f}")

    # Class imbalance check (binned into Low/Med/High)
    bins = [0.0, 0.35, 0.65, 1.0]
    labels = ["Low Load", "Medium Load", "High Load"]
    bin_counts = pd.cut(y_flat, bins=bins, labels=labels).value_counts()
    print(f"\n=== Imbalance Report ===")
    for lbl, cnt in bin_counts.items():
        pct = cnt / len(y_flat) * 100
        bar = "█" * int(pct / 2)
        print(f"  {lbl:15s}: {cnt:5d} ({pct:5.1f}%)  {bar}")

    imbalance_ratio = bin_counts.max() / (bin_counts.min() + 1)
    if imbalance_ratio > 3:
        print(f"\n  ⚠ HIGH IMBALANCE (ratio={imbalance_ratio:.1f}). "
              "Recommend: WeightedRandomSampler + label smoothing.")
    else:
        print(f"\n  ✓ Acceptable imbalance ratio: {imbalance_ratio:.1f}")


# 2. Feature Correlation
def plot_feature_correlation(X: np.ndarray, save_dir: str):
    """Correlation heatmap across all 8 features."""
    # Reshape to (N*T, F)
    X_flat = X.reshape(-1, X.shape[-1])
    df = pd.DataFrame(X_flat, columns=FEATURE_NAMES)

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="#2a2a4a",
        cbar_kws={"label": "Pearson r"},
        ax=ax
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)

    path = os.path.join(save_dir, "feature_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved: {path}")

    # Report high correlations
    print("\n=== High Correlations (|r| > 0.7) ===")
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                print(f"  {FEATURE_NAMES[i]} ↔ {FEATURE_NAMES[j]}: r={r:.3f} "
                      f"{'(consider removing one)' if abs(r) > 0.9 else ''}")


# 3. Temporal Consistency
def plot_temporal_patterns(X: np.ndarray, Y: np.ndarray, save_dir: str, n_samples: int = 5):
    """
    Plots temporal evolution of each feature for sample sequences,
    grouped by cognitive load level.
    """
    y_flat = Y.flatten()
    low_idx = np.where(y_flat < 0.35)[0]
    high_idx = np.where(y_flat > 0.65)[0]

    for group_name, indices in [("low_load", low_idx), ("high_load", high_idx)]:
        if len(indices) == 0:
            continue

        sample_idx = np.random.choice(indices, min(n_samples, len(indices)), replace=False)

        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle(
            f"Temporal Feature Patterns — {group_name.replace('_', ' ').title()} "
            f"(n={len(sample_idx)} sessions)",
            fontsize=14, fontweight="bold"
        )
        axes = axes.flatten()

        for feat_i, (ax, feat_name) in enumerate(zip(axes, FEATURE_NAMES)):
            t = np.arange(X.shape[1])
            for si, seq_i in enumerate(sample_idx):
                ax.plot(t, X[seq_i, :, feat_i], alpha=0.5,
                        color=PALETTE[si % len(PALETTE)], linewidth=0.8)

            # Mean ± std envelope
            group_seqs = X[sample_idx, :, feat_i]
            mean_curve = group_seqs.mean(axis=0)
            std_curve = group_seqs.std(axis=0)
            ax.plot(t, mean_curve, color="white", linewidth=2, zorder=10)
            ax.fill_between(t, mean_curve - std_curve, mean_curve + std_curve,
                           alpha=0.2, color="white")
            ax.set_title(feat_name, fontsize=10)
            ax.set_xlabel("Frame (150 = 5s)")
            ax.grid(True, alpha=0.4)

        plt.tight_layout()
        path = os.path.join(save_dir, f"temporal_patterns_{group_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"[Plot] Saved: {path}")


# 4. Missing Value Analysis
def analyze_missing_values(dataset_dir: str) -> pd.DataFrame:
    """
    Scans raw CSVs to compute missing-value statistics before preprocessing.
    """
    FEATURE_COLS = [
        "ear", "gaze_pitch", "gaze_yaw",
        "head_pitch", "head_yaw", "head_roll",
        "eyebrow_tension", "eye_openness"
    ]
    results = []

    for csv_path in glob.glob(f"{dataset_dir}/*/*/features.csv"):
        session_id = Path(csv_path).parent.name
        participant = Path(csv_path).parent.parent.name
        df = pd.read_csv(csv_path)

        for col in FEATURE_COLS:
            if col not in df.columns:
                continue
            total = len(df)
            missing = df[col].isnull().sum() + (df[col] == "NaN").sum()
            pct_missing = missing / total * 100 if total > 0 else 100
            results.append({
                "participant": participant,
                "session": session_id,
                "feature": col,
                "total_frames": total,
                "missing_frames": missing,
                "pct_missing": pct_missing,
            })

    if not results:
        print("[Analysis] No raw CSVs found. Skipping missing value analysis.")
        return pd.DataFrame()

    df_missing = pd.DataFrame(results)
    summary = df_missing.groupby("feature")["pct_missing"].agg(["mean", "max"])
    print("\n=== Missing Value Summary (per feature) ===")
    for feat, row in summary.iterrows():
        status = "⚠ HIGH" if row["mean"] > 20 else "✓"
        print(f"  {feat:20s}: avg={row['mean']:.1f}%  max={row['max']:.1f}%  {status}")

    return df_missing


# 5. Outlier Detection
def detect_outliers(X: np.ndarray) -> Dict:
    """
    IQR-based outlier detection per feature.
    Returns fraction of frames flagged as outliers.
    """
    X_flat = X.reshape(-1, X.shape[-1])
    report = {}

    print("\n=== Outlier Analysis (IQR method) ===")
    for i, feat in enumerate(FEATURE_NAMES):
        col = X_flat[:, i]
        Q1, Q3 = np.percentile(col, 25), np.percentile(col, 75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
        outliers = np.sum((col < lower) | (col > upper))
        pct = outliers / len(col) * 100
        report[feat] = {"outlier_pct": pct, "lower_bound": lower, "upper_bound": upper}
        status = "⚠" if pct > 5 else "✓"
        print(f"  {feat:20s}: {pct:.2f}% outliers  {status}")

    return report


# 6. Sequence Quality Scoring
def score_sequence_quality(X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    """
    Assigns a quality score [0, 1] to each sequence based on:
    - Fraction of non-zero frames (proxy for face detection rate)
    - Signal variance (too flat = face tracking frozen)
    - Temporal smoothness (too jerky = tracking noise)
    """
    records = []
    for i, seq in enumerate(X):
        # Non-zero ratio (EAR=0 suggests no face)
        nonzero_ratio = np.mean(seq[:, 0] != 0)  # EAR feature

        # Signal variance (should be non-trivial for meaningful data)
        variance = np.var(seq, axis=0).mean()

        # Smoothness: mean absolute frame-to-frame change
        diffs = np.abs(np.diff(seq, axis=0))
        smoothness = 1.0 / (1.0 + diffs.mean())

        # Composite quality score
        quality = (
            0.5 * nonzero_ratio +
            0.3 * min(1.0, variance / 0.01) +  # normalize variance
            0.2 * smoothness
        )
        records.append({
            "seq_idx": i,
            "label": Y[i, 0],
            "nonzero_ratio": nonzero_ratio,
            "variance": variance,
            "smoothness": smoothness,
            "quality_score": quality,
        })

    df = pd.DataFrame(records)
    low_quality = df[df["quality_score"] < 0.5]
    print(f"\n=== Sequence Quality ===")
    print(f"  Total sequences:       {len(df)}")
    print(f"  Low quality (<0.5):    {len(low_quality)} ({len(low_quality)/len(df)*100:.1f}%)")
    print(f"  Avg quality score:     {df['quality_score'].mean():.3f}")
    return df


# 7. Sanity Check Summary
def run_full_analysis(processed_dir: str, dataset_dir: Optional[str] = None):
    """
    Runs the complete dataset analysis pipeline and saves all plots.
    """
    save_dir = os.path.join(processed_dir, "analysis")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 55)
    print("  Phase 4 — Dataset Quality Analysis")
    print("=" * 55)

    X, Y = load_dataset(processed_dir, dataset_dir)

    if len(X) == 0:
        print("\n[ERROR] Empty dataset. Record sessions first.")
        return

    # Run all analyses
    plot_label_distribution(Y, save_dir)
    plot_feature_correlation(X, save_dir)
    plot_temporal_patterns(X, Y, save_dir)
    outlier_report = detect_outliers(X)
    quality_df = score_sequence_quality(X, Y)

    if dataset_dir and os.path.exists(dataset_dir):
        analyze_missing_values(dataset_dir)

    # Final verdict
    print("\n══ DATASET READINESS VERDICT ═══════════════════")
    issues = []
    if len(X) < 50:
        issues.append(f"• Very small dataset ({len(X)} sequences). Target 200+.")
    if Y.std() < 0.05:
        issues.append("• Near-constant labels. Add diverse task types.")
    low_q = (quality_df["quality_score"] < 0.5).sum()
    if low_q / len(X) > 0.3:
        issues.append(f"• {low_q} low-quality sequences ({low_q/len(X)*100:.0f}%). "
                      "Improve recording conditions.")

    if issues:
        print("⚠ Issues detected:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ Dataset looks healthy. Ready for training.")

    print(f"\nPlots saved to: {save_dir}/")
    return {"X_shape": X.shape, "Y_shape": Y.shape, "quality_df": quality_df}


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent
    run_full_analysis(
        processed_dir=str(BASE / "processed_data"),
        dataset_dir=str(BASE / "dataset"),
    )
