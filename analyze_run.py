#!/usr/bin/env python3
"""
Analyze the relationship between V3 imbalance and measured noise
for one or more acquisition runs.

IMPORTANT:
    This script does NOT balance the data.
    This script does NOT randomly delete observations.
    This script does NOT create an edited CSV dataset.

It preserves the physical distribution of V3 exactly as measured.

For each Run N directory it:
    1. Locates zero_pairs.csv.
    2. Loads all valid qualifying observations.
    3. Reports the natural signed V3 distribution.
    4. Reports noise statistics versus |V3|.
    5. Performs continuous |V3| regression.
    6. Performs signed-V3 regression.
    7. Performs a |V3| window sweep.
    8. Reports noise for positive and negative V3 separately.
    9. Reports chronological block statistics.
   10. Reports whether the apparent noise improvement at smaller |V3|
       is actually supported by the data.

No observations are removed.

Usage
-----

Analyze latest run:

    python analyze_v3.py

Analyze a specific run:

    python analyze_v3.py --run 2

Analyze every run:

    python analyze_v3.py --all

Use a custom base directory:

    python analyze_v3.py --base "C:\\path\\to\\v3_segmented_noise_data"

"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_DIRECTORY = "v3_segmented_noise_data"

# |V3| windows to investigate.
WINDOWS_MV = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
]

# Number of chronological observations per block.
BLOCK_SIZE = 25

# Minimum observations required before reporting a window.
MIN_WINDOW_OBSERVATIONS = 5

# Bootstrap repetitions for confidence intervals.
BOOTSTRAP_REPETITIONS = 5000

# Reproducible bootstrap.
RANDOM_SEED = 42


# =====================================================================
# RUN DIRECTORY RESOLUTION
# =====================================================================

def find_run_directories(base_dir: Path) -> list[Path]:
    """Return all Run N directories in numerical order."""

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base directory does not exist:\n{base_dir}"
        )

    runs = []

    for path in base_dir.iterdir():
        if not path.is_dir():
            continue

        match = re.fullmatch(r"Run (\d+)", path.name)

        if match:
            runs.append((int(match.group(1)), path))

    if not runs:
        raise FileNotFoundError(
            f"No Run N directories found in:\n{base_dir}"
        )

    runs.sort(key=lambda x: x[0])

    return [path for _, path in runs]


def find_target_run(
    base_dir: Path,
    run_number: int | None = None,
) -> Path:

    runs = find_run_directories(base_dir)

    if run_number is None:
        return runs[-1]

    for path in runs:
        if path.name == f"Run {run_number}":
            return path

    raise FileNotFoundError(
        f"Run {run_number} does not exist in:\n{base_dir}"
    )


# =====================================================================
# DATA LOADING
# =====================================================================

def load_data(csv_path: Path) -> pd.DataFrame:

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find:\n{csv_path}"
        )

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

    # Convert numeric columns.
    numeric_columns = [
        "noise_power_watts",
        "noise_power_dbm",
        "v3_mean_volts",
        "abs_v3_mean_volts",
        "global_pair_number",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Keep only physically valid noise measurements.
    valid = (
        np.isfinite(df["noise_power_watts"])
        & (df["noise_power_watts"] > 0)
        & np.isfinite(df["v3_mean_volts"])
        & np.isfinite(df["abs_v3_mean_volts"])
    )

    df = df.loc[valid].copy()

    # Recalculate |V3| from signed V3 rather than trusting the CSV.
    df["abs_v3_mean_volts"] = np.abs(
        df["v3_mean_volts"]
    )

    # Chronological order.
    if "global_pair_number" in df.columns:
        df = df.sort_values(
            "global_pair_number"
        )

    return df.reset_index(drop=True)


# =====================================================================
# UNIT CONVERSIONS
# =====================================================================

def watts_to_dbm(watts: float) -> float:

    if (
        math.isfinite(watts)
        and watts > 0
    ):
        return 10.0 * math.log10(
            watts / 1e-3
        )

    return math.nan


def mean_noise(df: pd.DataFrame) -> tuple[float, float]:

    mean_watts = float(
        df["noise_power_watts"].mean()
    )

    mean_dbm = watts_to_dbm(
        mean_watts
    )

    return mean_watts, mean_dbm


# =====================================================================
# BOOTSTRAP
# =====================================================================

def bootstrap_mean_dbm(
    df: pd.DataFrame,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> tuple[float, float]:

    values = df["noise_power_watts"].to_numpy(
        dtype=float
    )

    if len(values) < 2:
        value = watts_to_dbm(
            float(values[0])
        )

        return value, value

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    means = np.empty(
        repetitions,
        dtype=float,
    )

    n = len(values)

    for i in range(repetitions):

        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        means[i] = np.mean(sample)

    dbm_values = 10.0 * np.log10(
        means / 1e-3
    )

    return (
        float(np.percentile(dbm_values, 2.5)),
        float(np.percentile(dbm_values, 97.5)),
    )


# =====================================================================
# BASIC REPORT
# =====================================================================

def print_basic_statistics(df: pd.DataFrame):

    v3_mv = (
        df["v3_mean_volts"]
        .to_numpy()
        * 1e3
    )

    abs_v3_mv = np.abs(v3_mv)

    mean_watts, mean_dbm = mean_noise(df)

    positive = int(
        np.sum(v3_mv > 0)
    )

    negative = int(
        np.sum(v3_mv < 0)
    )

    zero = int(
        np.sum(v3_mv == 0)
    )

    print("\n--- Overall dataset ---")

    print(
        f"Total observations : {len(df)}"
    )

    print(
        f"V3 range           : "
        f"{np.min(v3_mv):.4f} to "
        f"{np.max(v3_mv):.4f} mV"
    )

    print(
        f"|V3| range         : "
        f"{np.min(abs_v3_mv):.4f} to "
        f"{np.max(abs_v3_mv):.4f} mV"
    )

    print(
        f"Mean V3            : "
        f"{np.mean(v3_mv):.6f} mV"
    )

    print(
        f"Mean |V3|          : "
        f"{np.mean(abs_v3_mv):.6f} mV"
    )

    print(
        f"Median |V3|        : "
        f"{np.median(abs_v3_mv):.6f} mV"
    )

    print(
        f"Positive V3        : {positive}"
    )

    print(
        f"Negative V3        : {negative}"
    )

    print(
        f"Zero V3            : {zero}"
    )

    print(
        f"Signed sum V3      : "
        f"{np.sum(v3_mv):.6f} mV"
    )

    print(
        f"Mean noise         : "
        f"{mean_watts:.6e} W"
    )

    print(
        f"Mean noise         : "
        f"{mean_dbm:.6f} dBm"
    )


# =====================================================================
# CONTINUOUS |V3| REGRESSION
# =====================================================================

def continuous_abs_v3_analysis(df: pd.DataFrame):

    x = (
        df["abs_v3_mean_volts"]
        .to_numpy()
        * 1e3
    )

    y = df["noise_power_watts"].to_numpy()

    print("\n" + "=" * 70)
    print("CONTINUOUS |V3| DEPENDENCE")
    print("=" * 70)

    # Linear regression.
    result = stats.linregress(
        x,
        y,
    )

    print(
        "\nLinear model:"
        "\n    noise = a + b|V3|"
    )

    print(
        f"  slope       : "
        f"{result.slope:.6e} W/mV"
    )

    print(
        f"  intercept   : "
        f"{result.intercept:.6e} W"
    )

    print(
        f"  R²          : "
        f"{result.rvalue ** 2:.8f}"
    )

    print(
        f"  correlation : "
        f"{result.rvalue:.8f}"
    )

    print(
        f"  p-value     : "
        f"{result.pvalue:.8f}"
    )

    # Quadratic regression.
    X = np.column_stack(
        [
            np.ones_like(x),
            x,
            x ** 2,
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    prediction = X @ coefficients

    ss_res = np.sum(
        (y - prediction) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else math.nan
    )

    print(
        "\nQuadratic model:"
        "\n    noise = a + b|V3| + c|V3|²"
    )

    print(
        f"  |V3| coefficient  : "
        f"{coefficients[1]:.6e} W/mV"
    )

    print(
        f"  |V3|² coefficient : "
        f"{coefficients[2]:.6e} W/mV²"
    )

    print(
        f"  R²                : "
        f"{r2:.8f}"
    )

    # Estimate noise at V3 = 0.
    estimated_zero_watts = (
        coefficients[0]
    )

    estimated_zero_dbm = watts_to_dbm(
        estimated_zero_watts
    )

    print(
        f"\nEstimated noise at |V3| = 0:"
    )

    print(
        f"  {estimated_zero_watts:.6e} W"
    )

    print(
        f"  {estimated_zero_dbm:.6f} dBm"
    )


# =====================================================================
# SIGNED V3 ANALYSIS
# =====================================================================

def signed_v3_analysis(df: pd.DataFrame):

    x = (
        df["v3_mean_volts"]
        .to_numpy()
        * 1e3
    )

    y = df["noise_power_watts"].to_numpy()

    result = stats.linregress(
        x,
        y,
    )

    print("\n" + "=" * 70)
    print("SIGNED V3 DEPENDENCE")
    print("=" * 70)

    print(
        "\nLinear model:"
        "\n    noise = a + bV3"
    )

    print(
        f"  slope       : "
        f"{result.slope:.6e} W/mV"
    )

    print(
        f"  R²          : "
        f"{result.rvalue ** 2:.8f}"
    )

    print(
        f"  correlation : "
        f"{result.rvalue:.8f}"
    )

    print(
        f"  p-value     : "
        f"{result.pvalue:.8f}"
    )


# =====================================================================
# POSITIVE / NEGATIVE COMPARISON
# =====================================================================

def signed_side_analysis(df: pd.DataFrame):

    positive = df[
        df["v3_mean_volts"] > 0
    ]

    negative = df[
        df["v3_mean_volts"] < 0
    ]

    print("\n" + "=" * 70)
    print("POSITIVE / NEGATIVE V3 COMPARISON")
    print("=" * 70)

    if len(positive) > 0:

        watts, dbm = mean_noise(
            positive
        )

        print(
            f"\nPositive V3:"
            f"\n  N           : {len(positive)}"
            f"\n  Mean |V3|   : "
            f"{positive['abs_v3_mean_volts'].mean() * 1e3:.6f} mV"
            f"\n  Mean noise  : "
            f"{watts:.6e} W"
            f"\n  Mean noise  : "
            f"{dbm:.6f} dBm"
        )

    if len(negative) > 0:

        watts, dbm = mean_noise(
            negative
        )

        print(
            f"\nNegative V3:"
            f"\n  N           : {len(negative)}"
            f"\n  Mean |V3|   : "
            f"{negative['abs_v3_mean_volts'].mean() * 1e3:.6f} mV"
            f"\n  Mean noise  : "
            f"{watts:.6e} W"
            f"\n  Mean noise  : "
            f"{dbm:.6f} dBm"
        )

    if len(positive) > 1 and len(negative) > 1:

        p_watts, _ = mean_noise(
            positive
        )

        n_watts, _ = mean_noise(
            negative
        )

        difference_db = watts_to_dbm(
            p_watts
        ) - watts_to_dbm(
            n_watts
        )

        print(
            f"\nPositive - negative:"
            f"\n  {difference_db:+.6f} dB"
        )


# =====================================================================
# WINDOW SWEEP
# =====================================================================

def window_sweep(df: pd.DataFrame):

    print("\n" + "=" * 70)
    print("|V3| WINDOW SWEEP")
    print("=" * 70)

    rows = []

    for window_mv in WINDOWS_MV:

        mask = (
            df["abs_v3_mean_volts"]
            <= window_mv * 1e-3
        )

        subset = df.loc[mask]

        n = len(subset)

        if n < MIN_WINDOW_OBSERVATIONS:
            continue

        mean_watts, mean_dbm = mean_noise(
            subset
        )

        ci_low, ci_high = (
            bootstrap_mean_dbm(
                subset
            )
        )

        rows.append(
            {
                "window_mV": window_mv,
                "n_pairs": n,
                "mean_abs_v3_mV":
                    subset[
                        "abs_v3_mean_volts"
                    ].mean() * 1e3,
                "mean_noise_watts":
                    mean_watts,
                "mean_noise_dbm":
                    mean_dbm,
                "bootstrap_ci_low_dbm":
                    ci_low,
                "bootstrap_ci_high_dbm":
                    ci_high,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        print(
            "\nNo windows contained enough observations."
        )
        return

    print()

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # Compare tightest and widest windows.
    if len(result) >= 2:

        tight = result.iloc[0]
        wide = result.iloc[-1]

        difference = (
            tight["mean_noise_dbm"]
            - wide["mean_noise_dbm"]
        )

        print(
            "\nTightest available window:"
        )

        print(
            f"  |V3| <= "
            f"{tight['window_mV']:.3f} mV"
        )

        print(
            f"  Mean noise = "
            f"{tight['mean_noise_dbm']:.6f} dBm"
        )

        print(
            "\nWidest window:"
        )

        print(
            f"  |V3| <= "
            f"{wide['window_mV']:.3f} mV"
        )

        print(
            f"  Mean noise = "
            f"{wide['mean_noise_dbm']:.6f} dBm"
        )

        print(
            f"\nTight - wide noise difference:"
            f" {difference:+.6f} dB"
        )


# =====================================================================
# CHRONOLOGICAL BLOCK ANALYSIS
# =====================================================================

def chronological_blocks(df: pd.DataFrame):

    print("\n" + "=" * 70)
    print("CHRONOLOGICAL BLOCK ANALYSIS")
    print("=" * 70)

    print(
        f"\nBlock size: {BLOCK_SIZE} observations"
    )

    rows = []

    n_blocks = math.ceil(
        len(df) / BLOCK_SIZE
    )

    for i in range(n_blocks):

        first = (
            i * BLOCK_SIZE
        )

        last = min(
            first + BLOCK_SIZE,
            len(df),
        )

        block = df.iloc[
            first:last
        ]

        if len(block) == 0:
            continue

        watts, dbm = mean_noise(
            block
        )

        rows.append(
            {
                "block": i + 1,
                "n": len(block),
                "mean_v3_mV":
                    block["v3_mean_volts"].mean()
                    * 1e3,
                "mean_abs_v3_mV":
                    block["abs_v3_mean_volts"].mean()
                    * 1e3,
                "mean_noise_dbm":
                    dbm,
            }
        )

    result = pd.DataFrame(rows)

    print()

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    if len(result) >= 3:

        correlation = np.corrcoef(
            result["mean_abs_v3_mV"],
            result["mean_noise_dbm"],
        )[0, 1]

        print(
            f"\nChronological block "
            f"correlation(|V3|, noise): "
            f"{correlation:.6f}"
        )


# =====================================================================
# FINAL INTERPRETATION
# =====================================================================

def interpretation(df: pd.DataFrame):

    x = (
        df["abs_v3_mean_volts"]
        .to_numpy()
        * 1e3
    )

    y = df["noise_power_watts"].to_numpy()

    result = stats.linregress(
        x,
        y,
    )

    r2 = result.rvalue ** 2

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if result.pvalue < 0.05:

        if result.slope > 0:

            print(
                "\nThere IS statistically significant evidence "
                "that noise increases with |V3|."
            )

            print(
                "This supports the hypothesis that reducing "
                "the detector imbalance can reduce the measured noise."
            )

        else:

            print(
                "\nThere IS statistically significant evidence "
                "that noise decreases with |V3|."
            )

            print(
                "This is opposite to the expected explanation "
                "that better balance reduces the measured noise."
            )

    else:

        print(
            "\nThere is NO statistically significant linear "
            "relationship between |V3| and noise."
        )

        print(
            "Therefore this run does not provide strong evidence "
            "that the noise improvement is caused simply by reduced |V3|."
        )

    print(
        f"\nR² = {r2:.8f}"
    )

    print(
        "\nNo positive/negative balancing was performed."
    )

    print(
        "No observations were randomly deleted."
    )

    print(
        "The natural physical V3 distribution was preserved."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "zero_pairs.csv contains only observations that passed "
        "the original acquisition threshold. Therefore this analysis "
        "describes the relationship inside the accepted region; "
        "it cannot determine behavior outside that region."
    )


# =====================================================================
# SINGLE RUN
# =====================================================================

def analyze_run(run_directory: Path):

    csv_path = (
        run_directory
        / "zero_pairs.csv"
    )

    print("\n")
    print("#" * 70)
    print(
        f"RUN ANALYSIS: {run_directory.name}"
    )
    print("#" * 70)

    print(
        f"\nInput CSV:\n  {csv_path}"
    )

    if not csv_path.exists():

        print(
            "\nWARNING: zero_pairs.csv does not exist "
            "for this run."
        )

        return

    df = load_data(
        csv_path
    )

    if len(df) == 0:

        print(
            "\nNo valid observations."
        )

        return

    print_basic_statistics(
        df
    )

    continuous_abs_v3_analysis(
        df
    )

    signed_v3_analysis(
        df
    )

    signed_side_analysis(
        df
    )

    window_sweep(
        df
    )

    chronological_blocks(
        df
    )

    interpretation(
        df
    )


# =====================================================================
# COMMAND LINE
# =====================================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze V3 imbalance and noise "
            "without modifying the original data."
        )
    )

    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help=(
            "Run number to analyze. "
            "Defaults to the latest run."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Analyze every Run N directory."
        ),
    )

    parser.add_argument(
        "--base",
        default=BASE_DIRECTORY,
        help=(
            "Base directory containing Run N folders."
        ),
    )

    return parser.parse_args()


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_arguments()

    base_directory = (
        Path(args.base)
        .expanduser()
        .resolve()
    )

    if args.all:

        runs = find_run_directories(
            base_directory
        )

        print(
            f"Found {len(runs)} run(s): "
            + ", ".join(
                path.name
                for path in runs
            )
        )

        for run in runs:

            analyze_run(
                run
            )

    else:

        run = find_target_run(
            base_directory,
            args.run,
        )

        analyze_run(
            run
        )


if __name__ == "__main__":
    main()