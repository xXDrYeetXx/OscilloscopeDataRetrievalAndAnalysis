#!/usr/bin/env python3
"""
Balance the zero_pairs.csv dataset around V3 = 0.

The signed sum of V3 mean values across all qualifying pairs should be
near zero. If it is not, one side is overrepresented, which could bias
the noise measurement.

This script:
    1. Locates the target run directory (defaults to the latest Run N).
    2. Loads zero_pairs.csv from that run folder.
    3. Calculates the signed sum of V3 means.
    4. Decides if balancing is needed.
    5. If so, randomly removes pairs from the overrepresented side
       until the signed sum is within the balance threshold.
    6. Saves the balanced dataset to zero_pairs_balanced.csv in the same run folder.
    7. Reports mean noise power before and after balancing in watts and dBm.

Usage:
    python balance_pairs.py                     # Balances latest run
    python balance_pairs.py --run 2             # Balances Run 2
    python balance_pairs.py --input path/to/zero_pairs.csv
"""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_DIRECTORY = "v3_segmented_noise_data"

# The signed sum of V3 means must fall within this range to be
# considered balanced. Expressed in volts.
BALANCE_THRESHOLD_VOLTS = 1e-3

# Random seed for reproducibility. Set to None for a different
# random removal each run.
RANDOM_SEED = 42


# =====================================================================
# RUN DIRECTORY RESOLUTION
# =====================================================================

def find_target_run_directory(base_dir: Path, run_number: int | None = None) -> Path:
    """
    Find a specific 'Run N' directory, or locate the highest numbered run directory.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Base output directory does not exist: {base_dir}")

    if run_number is not None:
        target_dir = base_dir / f"Run {run_number}"
        if not target_dir.exists():
            raise FileNotFoundError(f"Requested run directory does not exist: {target_dir}")
        return target_dir

    highest_run_number = 0
    latest_dir = None

    for path in base_dir.iterdir():
        if not path.is_dir():
            continue

        match = re.fullmatch(r"Run (\d+)", path.name)
        if match is None:
            continue

        num = int(match.group(1))
        if num > highest_run_number:
            highest_run_number = num
            latest_dir = path

    if latest_dir is None:
        raise FileNotFoundError(f"No 'Run N' directories found inside {base_dir}")

    return latest_dir


# =====================================================================
# HELPERS
# =====================================================================

def watts_to_dbm(watts: float) -> float:
    if math.isfinite(watts) and watts > 0:
        return 10.0 * math.log10(watts / 1e-3)
    return math.nan


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = [
        "noise_power_watts",
        "noise_power_dbm",
        "v3_mean_volts",
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
    df["v3_mean_volts"]     = df["v3_mean_volts"].astype(float)
    df["abs_v3_mean_volts"] = df["abs_v3_mean_volts"].astype(float)

    return df.reset_index(drop=True)


def calculate_mean_noise(df: pd.DataFrame) -> tuple[float, float]:
    """
    Calculate mean noise power in watts then convert to dBm.
    Averaging must be done in watts, not dBm.
    """
    mean_watts = float(df["noise_power_watts"].mean())
    mean_dbm   = watts_to_dbm(mean_watts)
    return mean_watts, mean_dbm


def needs_balancing(signed_sum: float) -> bool:
    return abs(signed_sum) > BALANCE_THRESHOLD_VOLTS


# =====================================================================
# BALANCING
# =====================================================================

def balance_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Randomly remove pairs from the overrepresented side of zero
    until the signed sum of V3 means falls within the balance
    threshold.

    Returns the balanced dataframe and the number of pairs removed.
    """
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    balanced_df    = df.copy()
    pairs_removed  = 0

    while True:
        signed_sum = float(balanced_df["v3_mean_volts"].sum())

        if not needs_balancing(signed_sum):
            break

        if signed_sum > 0:
            candidates = balanced_df[
                balanced_df["v3_mean_volts"] > 0
            ].index.tolist()
        else:
            candidates = balanced_df[
                balanced_df["v3_mean_volts"] < 0
            ].index.tolist()

        if len(candidates) == 0:
            print(
                "Warning: no candidates left to remove on the "
                "overrepresented side. Stopping early."
            )
            break

        drop_index    = random.choice(candidates)
        balanced_df   = balanced_df.drop(drop_index)
        pairs_removed += 1

    return balanced_df.reset_index(drop=True), pairs_removed


# =====================================================================
# REPORTING
# =====================================================================

def print_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    pairs_removed: int,
    mean_watts_before: float,
    mean_dbm_before: float,
    mean_watts_after: float,
    mean_dbm_after: float,
):
    signed_sum_before = float(df_before["v3_mean_volts"].sum())
    signed_sum_after  = float(df_after["v3_mean_volts"].sum())

    positive_before = int((df_before["v3_mean_volts"] > 0).sum())
    negative_before = int((df_before["v3_mean_volts"] < 0).sum())
    positive_after  = int((df_after["v3_mean_volts"] > 0).sum())
    negative_after  = int((df_after["v3_mean_volts"] < 0).sum())

    print("\n" + "=" * 55)
    print("DATASET BALANCE REPORT")
    print("=" * 55)

    print("\n--- Before balancing ---")
    print(f"  Total pairs       : {len(df_before)}")
    print(f"  Positive V3 pairs : {positive_before}")
    print(f"  Negative V3 pairs : {negative_before}")
    print(
        f"  Signed sum of V3  : "
        f"{signed_sum_before:.6e} V"
    )
    print(
        f"  Mean noise        : {mean_watts_before:.6e} W  "
        f"({mean_dbm_before:.4f} dBm)"
    )

    if pairs_removed == 0:
        print(
            f"\n  Dataset is already balanced within "
            f"{BALANCE_THRESHOLD_VOLTS * 1e3:.2f} mV. "
            f"No pairs removed."
        )
    else:
        print(
            f"\n  Balancing threshold : "
            f"{BALANCE_THRESHOLD_VOLTS * 1e3:.2f} mV"
        )
        print(f"  Pairs removed       : {pairs_removed}")

        print("\n--- After balancing ---")
        print(f"  Total pairs       : {len(df_after)}")
        print(f"  Positive V3 pairs : {positive_after}")
        print(f"  Negative V3 pairs : {negative_after}")
        print(
            f"  Signed sum of V3  : "
            f"{signed_sum_after:.6e} V"
        )
        print(
            f"  Mean noise        : {mean_watts_after:.6e} W  "
            f"({mean_dbm_after:.4f} dBm)"
        )

        change_dbm = mean_dbm_after - mean_dbm_before
        print(
            f"\n  Change in mean    : {change_dbm:+.4f} dBm"
        )

        if abs(change_dbm) < 0.1:
            print(
                "  Interpretation    : Balancing had negligible "
                "effect on the noise measurement."
            )
        else:
            print(
                "  Interpretation    : Balancing changed the mean "
                "noticeably. The original dataset was biased."
            )

    print("=" * 55)


# =====================================================================
# MAIN
# =====================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Balance zero_pairs.csv within run directories."
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run number to process (e.g. --run 1). Defaults to the latest run.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Direct path to zero_pairs.csv (overrides --run detection).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Direct path for output CSV (defaults to zero_pairs_balanced.csv in the same run folder).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=BALANCE_THRESHOLD_VOLTS,
        help=(
            "Balance threshold in volts. Default: "
            f"{BALANCE_THRESHOLD_VOLTS:.4f} V"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    global BALANCE_THRESHOLD_VOLTS
    BALANCE_THRESHOLD_VOLTS = args.threshold

    if args.input:
        csv_path = Path(args.input).expanduser().resolve()
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else csv_path.parent / "zero_pairs_balanced.csv"
        )
    else:
        base_dir = Path(BASE_DIRECTORY).expanduser().resolve()
        run_dir = find_target_run_directory(base_dir, args.run)
        csv_path = run_dir / "zero_pairs.csv"
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else run_dir / "zero_pairs_balanced.csv"
        )

    print(f"Target run file : {csv_path}")
    print(f"Output path     : {output_path}")

    if not csv_path.exists():
        print(f"Error: Could not find '{csv_path.name}' at {csv_path}")
        return

    df = load_data(csv_path)
    print(f"Pairs loaded: {len(df)}")

    if len(df) == 0:
        print("No valid pairs found.")
        return

    # Before balancing
    mean_watts_before, mean_dbm_before = calculate_mean_noise(df)
    signed_sum = float(df["v3_mean_volts"].sum())

    if not needs_balancing(signed_sum):
        print(
            f"\nDataset is already balanced within "
            f"{BALANCE_THRESHOLD_VOLTS * 1e3:.2f} mV."
        )
        df_balanced   = df
        pairs_removed = 0
    else:
        print(
            f"\nDataset needs balancing. "
            f"Signed sum = {signed_sum:.6e} V. "
            f"Removing pairs from overrepresented side..."
        )
        df_balanced, pairs_removed = balance_dataset(df)

    # After balancing
    mean_watts_after, mean_dbm_after = calculate_mean_noise(
        df_balanced
    )

    print_report(
        df_before         = df,
        df_after          = df_balanced,
        pairs_removed     = pairs_removed,
        mean_watts_before = mean_watts_before,
        mean_dbm_before   = mean_dbm_before,
        mean_watts_after  = mean_watts_after,
        mean_dbm_after    = mean_dbm_after,
    )

    df_balanced.to_csv(output_path, index=False)
    print(f"\nBalanced dataset saved to: {output_path}")


if __name__ == "__main__":
    main()