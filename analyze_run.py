#!/usr/bin/env python3
"""
Estimate measured noise at V3 = 0.

Scientific interpretation
-------------------------
A previous analysis found no meaningful relationship between V3 and
measured noise within the selected near-zero range. Under the resulting
flat-near-zero assumption, this script estimates the noise at V3 = 0 by
pooling qualifying near-zero observations.

To avoid treating correlated subwindows from one oscilloscope record as
independent repetitions:

    1. Noise power is averaged in linear watts within each long acquisition.
    2. The acquisition-level means are averaged with equal weight.
    3. Whole acquisitions are bootstrap-resampled for a 95% confidence
       interval.
    4. Watts are converted to dBm only after averaging.

The result is the mean measured power near V3 = 0 in the acquisition
script's effective noise bandwidth. It is not a dBm average and is not
the total broadband noise.

Usage
-----
Analyze latest run:

    python analyze_zero_noise.py

Analyze a specific run:

    python analyze_zero_noise.py --run 2

Analyze every run separately:

    python analyze_zero_noise.py --all

Use a custom base directory:

    python analyze_zero_noise.py --base "/path/to/v3_segmented_noise_data"

Override the near-zero range:

    python analyze_zero_noise.py --max-abs-v3-mv 5.0
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_DIRECTORY = "v3_segmented_noise_data"

# Analyze observations satisfying |mean(V3)| <= this value.
MAX_ABS_V3_MV = 5.0

# Number of acquisition-level bootstrap repetitions.
BOOTSTRAP_REPETITIONS = 10_000

# Reproducible bootstrap.
RANDOM_SEED = 42

# Percentile confidence interval.
CONFIDENCE_LEVEL = 0.95


# =====================================================================
# UNIT CONVERSION
# =====================================================================

def watts_to_dbm(watts: float) -> float:
    """Convert positive power in watts to dBm."""
    if not math.isfinite(watts) or watts <= 0:
        return math.nan

    return 10.0 * math.log10(watts / 1e-3)


# =====================================================================
# RUN DIRECTORY RESOLUTION
# =====================================================================

def find_run_directories(base_directory: Path) -> list[Path]:
    """Return all Run N directories in numerical order."""
    if not base_directory.exists():
        raise FileNotFoundError(
            f"Base directory does not exist:\n{base_directory}"
        )

    runs: list[tuple[int, Path]] = []

    for path in base_directory.iterdir():
        if not path.is_dir():
            continue

        match = re.fullmatch(r"Run (\d+)", path.name)

        if match:
            runs.append((int(match.group(1)), path))

    if not runs:
        raise FileNotFoundError(
            f"No Run N directories found in:\n{base_directory}"
        )

    runs.sort(key=lambda item: item[0])

    return [path for _, path in runs]


def find_target_run(
    base_directory: Path,
    run_number: int | None,
) -> Path:
    """Find a specified run or the latest run."""
    runs = find_run_directories(base_directory)

    if run_number is None:
        return runs[-1]

    expected_name = f"Run {run_number}"

    for path in runs:
        if path.name == expected_name:
            return path

    raise FileNotFoundError(
        f"{expected_name} does not exist in:\n{base_directory}"
    )


# =====================================================================
# DATA LOADING AND VALIDATION
# =====================================================================

def load_zero_data(
    csv_path: Path,
    max_abs_v3_mv: float,
) -> tuple[pd.DataFrame, int, int]:
    """
    Load valid near-zero observations.

    Returns
    -------
    data
        Valid observations inside the requested V3 range.
    original_rows
        Number of rows originally present.
    excluded_rows
        Number of rows excluded by validity and range checks.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find:\n{csv_path}"
        )

    data = pd.read_csv(csv_path)
    original_rows = len(data)

    required_columns = [
        "long_acquisition_number",
        "noise_power_watts",
        "v3_mean_volts",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}"
        )

    optional_numeric_columns = [
        "enbw_hz",
        "actual_fft_bin_frequency_hz",
        "requested_frequency_hz",
    ]

    numeric_columns = required_columns + [
        column
        for column in optional_numeric_columns
        if column in data.columns
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["abs_v3_mv"] = np.abs(
        data["v3_mean_volts"]
    ) * 1e3

    valid = (
        np.isfinite(data["long_acquisition_number"])
        & np.isfinite(data["noise_power_watts"])
        & (data["noise_power_watts"] > 0)
        & np.isfinite(data["v3_mean_volts"])
        & np.isfinite(data["abs_v3_mv"])
        & (data["abs_v3_mv"] <= max_abs_v3_mv)
    )

    data = data.loc[valid].copy()
    excluded_rows = original_rows - len(data)

    if data.empty:
        raise ValueError(
            "No valid observations remain inside the requested "
            "near-zero range."
        )

    data["long_acquisition_number"] = (
        data["long_acquisition_number"].astype(int)
    )

    return data, original_rows, excluded_rows


# =====================================================================
# ACQUISITION-LEVEL ESTIMATE
# =====================================================================

def calculate_acquisition_means(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average subwindow powers within each long acquisition.

    Each resulting row represents one oscilloscope acquisition and
    receives equal weight in the final estimate.
    """
    acquisition_means = (
        data.groupby(
            "long_acquisition_number",
            as_index=False,
        )
        .agg(
            mean_noise_watts=(
                "noise_power_watts",
                "mean",
            ),
            mean_v3_mv=(
                "v3_mean_volts",
                lambda values: float(np.mean(values) * 1e3),
            ),
            mean_abs_v3_mv=(
                "abs_v3_mv",
                "mean",
            ),
            observations=(
                "noise_power_watts",
                "size",
            ),
        )
    )

    return acquisition_means


def estimate_zero_noise(
    acquisition_means: pd.DataFrame,
) -> tuple[float, float]:
    """
    Estimate noise at V3 = 0 under the flat-near-zero assumption.

    Returns the equal-acquisition mean in watts and dBm.
    """
    mean_watts = float(
        acquisition_means["mean_noise_watts"].mean()
    )

    return mean_watts, watts_to_dbm(mean_watts)


# =====================================================================
# CLUSTER BOOTSTRAP
# =====================================================================

def bootstrap_acquisition_mean(
    acquisition_means: pd.DataFrame,
    repetitions: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, float]:
    """
    Bootstrap complete long acquisitions.

    Since the input contains one mean per long acquisition, resampling
    these rows is equivalent to resampling acquisition clusters while
    retaining equal acquisition weighting.
    """
    values = acquisition_means[
        "mean_noise_watts"
    ].to_numpy(dtype=float)

    number_of_acquisitions = len(values)

    if number_of_acquisitions < 2:
        point_watts = float(values[0])
        point_dbm = watts_to_dbm(point_watts)

        return {
            "low_watts": point_watts,
            "high_watts": point_watts,
            "low_dbm": point_dbm,
            "high_dbm": point_dbm,
        }

    rng = np.random.default_rng(random_seed)

    bootstrap_means = np.empty(
        repetitions,
        dtype=float,
    )

    for repetition in range(repetitions):
        sampled_indices = rng.integers(
            low=0,
            high=number_of_acquisitions,
            size=number_of_acquisitions,
        )

        bootstrap_means[repetition] = float(
            np.mean(values[sampled_indices])
        )

    alpha = 1.0 - confidence_level

    low_percentile = 100.0 * alpha / 2.0
    high_percentile = 100.0 * (1.0 - alpha / 2.0)

    low_watts, high_watts = np.percentile(
        bootstrap_means,
        [low_percentile, high_percentile],
    )

    return {
        "low_watts": float(low_watts),
        "high_watts": float(high_watts),
        "low_dbm": watts_to_dbm(float(low_watts)),
        "high_dbm": watts_to_dbm(float(high_watts)),
    }


# =====================================================================
# OPTIONAL MEASUREMENT METADATA
# =====================================================================

def summarize_optional_measurement_fields(
    data: pd.DataFrame,
) -> None:
    """Report frequency and ENBW metadata when available."""
    if "enbw_hz" in data.columns:
        valid_enbw = data.loc[
            np.isfinite(data["enbw_hz"])
            & (data["enbw_hz"] > 0),
            "enbw_hz",
        ]

        if not valid_enbw.empty:
            print(
                f"Median ENBW             : "
                f"{valid_enbw.median() / 1e3:.6f} kHz"
            )
            print(
                f"ENBW range              : "
                f"{valid_enbw.min() / 1e3:.6f} to "
                f"{valid_enbw.max() / 1e3:.6f} kHz"
            )

    if "actual_fft_bin_frequency_hz" in data.columns:
        valid_frequency = data.loc[
            np.isfinite(
                data["actual_fft_bin_frequency_hz"]
            ),
            "actual_fft_bin_frequency_hz",
        ]

        if not valid_frequency.empty:
            print(
                f"Median FFT frequency    : "
                f"{valid_frequency.median() / 1e6:.9f} MHz"
            )


# =====================================================================
# SINGLE-RUN ANALYSIS
# =====================================================================

def analyze_run(
    run_directory: Path,
    max_abs_v3_mv: float,
    repetitions: int,
) -> None:
    """Estimate near-zero noise for one run."""
    csv_path = run_directory / "zero_pairs.csv"

    print("\n" + "#" * 72)
    print(f"ZERO-V3 NOISE ESTIMATE: {run_directory.name}")
    print("#" * 72)
    print(f"Input: {csv_path}")

    try:
        (
            data,
            original_rows,
            excluded_rows,
        ) = load_zero_data(
            csv_path=csv_path,
            max_abs_v3_mv=max_abs_v3_mv,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"\nUnable to analyze run: {error}")
        return

    acquisition_means = calculate_acquisition_means(data)

    mean_watts, mean_dbm = estimate_zero_noise(
        acquisition_means
    )

    confidence_interval = bootstrap_acquisition_mean(
        acquisition_means=acquisition_means,
        repetitions=repetitions,
        confidence_level=CONFIDENCE_LEVEL,
        random_seed=RANDOM_SEED,
    )

    signed_v3_mv = (
        data["v3_mean_volts"].to_numpy(dtype=float)
        * 1e3
    )

    positive_count = int(np.sum(signed_v3_mv > 0))
    negative_count = int(np.sum(signed_v3_mv < 0))
    exact_zero_count = int(np.sum(signed_v3_mv == 0))

    print("\nData summary")
    print(f"Original CSV rows       : {original_rows}")
    print(f"Excluded rows           : {excluded_rows}")
    print(f"Analyzed observations   : {len(data)}")
    print(
        f"Long acquisitions       : "
        f"{len(acquisition_means)}"
    )
    print(
        f"Observations/acquisition: "
        f"{acquisition_means['observations'].min()} to "
        f"{acquisition_means['observations'].max()}"
    )
    print(
        f"Required V3 range       : "
        f"|mean(V3)| <= {max_abs_v3_mv:.6f} mV"
    )
    print(
        f"Observed signed range   : "
        f"{signed_v3_mv.min():+.6f} to "
        f"{signed_v3_mv.max():+.6f} mV"
    )
    print(
        f"Mean signed V3          : "
        f"{signed_v3_mv.mean():+.6f} mV"
    )
    print(
        f"Mean |V3|              : "
        f"{np.abs(signed_v3_mv).mean():.6f} mV"
    )
    print(f"Positive observations   : {positive_count}")
    print(f"Negative observations   : {negative_count}")
    print(f"Exact-zero observations : {exact_zero_count}")

    summarize_optional_measurement_fields(data)

    confidence_percent = 100.0 * CONFIDENCE_LEVEL

    print("\nPrimary result")
    print(
        f"Estimated noise at V3=0 : "
        f"{mean_watts:.12e} W"
    )
    print(
        f"Estimated noise at V3=0 : "
        f"{mean_dbm:.6f} dBm"
    )
    print(
        f"{confidence_percent:.1f}% cluster-bootstrap CI:"
    )
    print(
        f"  Watts                 : "
        f"{confidence_interval['low_watts']:.12e} to "
        f"{confidence_interval['high_watts']:.12e} W"
    )
    print(
        f"  dBm                   : "
        f"{confidence_interval['low_dbm']:.6f} to "
        f"{confidence_interval['high_dbm']:.6f} dBm"
    )

    if len(acquisition_means) < 20:
        print(
            "\nCaution: fewer than 20 independent long acquisitions "
            "were available, so the confidence interval may be unstable."
        )

    print(
        "\nInterpretation:"
        "\n  Under the previously established flat-near-zero assumption,"
        "\n  this acquisition-weighted linear-power mean estimates the"
        "\n  measured noise at V3 = 0."
        "\n  The dBm value applies to the reported measurement ENBW."
    )


# =====================================================================
# COMMAND LINE
# =====================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate noise at V3 = 0 using acquisition-level "
            "averaging and a cluster bootstrap."
        )
    )

    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help=(
            "Run number to analyze. Defaults to the latest run."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze every Run N directory separately.",
    )

    parser.add_argument(
        "--base",
        default=BASE_DIRECTORY,
        help="Base directory containing Run N folders.",
    )

    parser.add_argument(
        "--max-abs-v3-mv",
        type=float,
        default=MAX_ABS_V3_MV,
        help=(
            "Maximum accepted absolute mean V3 in mV. "
            f"Default: {MAX_ABS_V3_MV}"
        ),
    )

    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=BOOTSTRAP_REPETITIONS,
        help=(
            "Number of acquisition-level bootstrap repetitions. "
            f"Default: {BOOTSTRAP_REPETITIONS}"
        ),
    )

    return parser.parse_args()


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    args = parse_arguments()

    if args.max_abs_v3_mv <= 0:
        raise ValueError(
            "--max-abs-v3-mv must be greater than zero."
        )

    if args.bootstrap_repetitions < 100:
        raise ValueError(
            "--bootstrap-repetitions must be at least 100."
        )

    base_directory = (
        Path(args.base)
        .expanduser()
        .resolve()
    )

    if args.all:
        run_directories = find_run_directories(
            base_directory
        )
    else:
        run_directories = [
            find_target_run(
                base_directory,
                args.run,
            )
        ]

    for run_directory in run_directories:
        analyze_run(
            run_directory=run_directory,
            max_abs_v3_mv=args.max_abs_v3_mv,
            repetitions=args.bootstrap_repetitions,
        )


if __name__ == "__main__":
    main()