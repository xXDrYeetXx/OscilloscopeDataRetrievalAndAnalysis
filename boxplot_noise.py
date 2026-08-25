#!/usr/bin/env python3
"""
Box plot of qualifying ordered pairs from zero_pairs.csv.

Produces:
    1. Box plot of noise_power_dbm for all qualifying pairs
    2. Box plot of abs_v3_mean_volts for all qualifying pairs
    3. Combined side-by-side box plots

Usage:
    python boxplot_noise.py
    python boxplot_noise.py --input path/to/zero_pairs.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# CONFIGURATION
# =====================================================================

INPUT_CSV        = "v3_segmented_noise_data/zero_pairs.csv"
OUTPUT_DIRECTORY = "v3_segmented_noise_data/plots"


# =====================================================================
# HELPERS
# =====================================================================

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = [
        "noise_power_watts",
        "noise_power_dbm",
        "abs_v3_mean_volts",
        "global_pair_number",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}"
        )

    df = df[
        df["noise_power_watts"].apply(
            lambda w: math.isfinite(float(w)) and float(w) > 0
        )
    ].copy()

    df["noise_power_watts"] = df["noise_power_watts"].astype(float)
    df["noise_power_dbm"]   = df["noise_power_dbm"].astype(float)
    df["abs_v3_mean_volts"] = df["abs_v3_mean_volts"].astype(float)

    return df.reset_index(drop=True)


def print_statistics(df: pd.DataFrame):
    noise = df["noise_power_dbm"]
    v3    = df["abs_v3_mean_volts"] * 1e3

    print("\n--- Noise Power (dBm) ---")
    print(f"  Samples   : {len(noise)}")
    print(f"  Median    : {noise.median():.4f} dBm")
    print(f"  Mean      : {noise.mean():.4f} dBm")
    print(f"  Std       : {noise.std():.4f} dBm")
    print(f"  Min       : {noise.min():.4f} dBm")
    print(f"  Max       : {noise.max():.4f} dBm")
    print(f"  Q1        : {noise.quantile(0.25):.4f} dBm")
    print(f"  Q3        : {noise.quantile(0.75):.4f} dBm")
    print(f"  IQR       : {noise.quantile(0.75) - noise.quantile(0.25):.4f} dBm")

    print("\n--- |mean(V3)| Alignment (mV) ---")
    print(f"  Samples   : {len(v3)}")
    print(f"  Median    : {v3.median():.4f} mV")
    print(f"  Mean      : {v3.mean():.4f} mV")
    print(f"  Std       : {v3.std():.4f} mV")
    print(f"  Min       : {v3.min():.4f} mV")
    print(f"  Max       : {v3.max():.4f} mV")
    print(f"  Q1        : {v3.quantile(0.25):.4f} mV")
    print(f"  Q3        : {v3.quantile(0.75):.4f} mV")
    print(f"  IQR       : {v3.quantile(0.75) - v3.quantile(0.25):.4f} mV")


# =====================================================================
# PLOTS
# =====================================================================

def plot_noise_boxplot(df: pd.DataFrame, output_dir: Path):
    """Box plot of noise power in dBm."""
    fig, ax = plt.subplots(figsize=(6, 7))

    bp = ax.boxplot(
        df["noise_power_dbm"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="red", linewidth=2),
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(
            marker="o",
            markersize=3,
            alpha=0.4,
            color="steelblue",
        ),
    )

    median_val = df["noise_power_dbm"].median()
    mean_val   = df["noise_power_dbm"].mean()

    ax.axhline(
        mean_val,
        color="orange",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean = {mean_val:.3f} dBm",
    )

    ax.set_ylabel("Noise power (dBm)")
    ax.set_title(
        f"Noise power at 10 MHz\n"
        f"n = {len(df)} qualifying pairs  |  "
        f"Median = {median_val:.3f} dBm"
    )
    ax.set_xticks([1])
    ax.set_xticklabels(["Qualifying pairs\n|mean(V3)| ≤ threshold"])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "boxplot_noise_power.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {path}")


def plot_alignment_boxplot(df: pd.DataFrame, output_dir: Path):
    """Box plot of the alignment metric |mean(V3)|."""
    fig, ax = plt.subplots(figsize=(6, 7))

    ax.boxplot(
        df["abs_v3_mean_volts"] * 1e3,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="red", linewidth=2),
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(
            marker="o",
            markersize=3,
            alpha=0.4,
            color="steelblue",
        ),
    )

    median_val = df["abs_v3_mean_volts"].median() * 1e3

    ax.set_ylabel("|mean(V3)| (mV)")
    ax.set_title(
        f"V3 alignment level\n"
        f"n = {len(df)} qualifying pairs  |  "
        f"Median = {median_val:.3f} mV"
    )
    ax.set_xticks([1])
    ax.set_xticklabels(["Qualifying pairs\n|mean(V3)| ≤ threshold"])
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "boxplot_alignment.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_combined_boxplot(df: pd.DataFrame, output_dir: Path):
    """
    Side-by-side box plots of noise power and alignment level
    on two y-axes.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 7)
    )

    # Noise power
    ax1.boxplot(
        df["noise_power_dbm"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="red", linewidth=2),
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(
            marker="o",
            markersize=3,
            alpha=0.4,
            color="steelblue",
        ),
    )

    ax1.axhline(
        df["noise_power_dbm"].mean(),
        color="orange",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean = {df['noise_power_dbm'].mean():.3f} dBm",
    )

    ax1.set_ylabel("Noise power (dBm)")
    ax1.set_title("Noise power at 10 MHz")
    ax1.set_xticks([1])
    ax1.set_xticklabels(["Qualifying\npairs"])
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # Alignment metric
    ax2.boxplot(
        df["abs_v3_mean_volts"] * 1e3,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="red", linewidth=2),
        boxprops=dict(facecolor="coral", alpha=0.6),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(
            marker="o",
            markersize=3,
            alpha=0.4,
            color="coral",
        ),
    )

    ax2.set_ylabel("|mean(V3)| (mV)")
    ax2.set_title("V3 alignment level")
    ax2.set_xticks([1])
    ax2.set_xticklabels(["Qualifying\npairs"])
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Qualifying ordered pairs  |  n = {len(df)}",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()
    path = output_dir / "boxplot_combined.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# =====================================================================
# MAIN
# =====================================================================

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=INPUT_CSV,
        help="Path to zero_pairs.csv",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIRECTORY,
        help="Directory for saved plots",
    )
    return parser.parse_args()


def main():
    args       = parse_arguments()
    csv_path   = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {csv_path}")
    df = load_data(csv_path)
    print(f"Valid qualifying pairs loaded: {len(df)}")

    if len(df) == 0:
        print("No valid qualifying pairs found in the CSV.")
        return

    print_statistics(df)

    plot_noise_boxplot(df, output_dir)
    plot_alignment_boxplot(df, output_dir)
    plot_combined_boxplot(df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()