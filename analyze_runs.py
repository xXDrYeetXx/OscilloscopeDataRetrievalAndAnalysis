#!/usr/bin/env python3
"""
analyze_runs.py
===============

Fit the fixed-design calibration model

    N(P) = k P + N_dark

to independently acquired run means.

Each run contains many subwindows nested inside long acquisitions.
Long acquisitions—not individual subwindows—are treated as the
independent sampling units.

Primary analysis
----------------
1. Reconstruct each run mean from zero_pairs.csv.
2. Estimate its sampling variance using a delete-one-acquisition
   cluster jackknife.
3. Fit inverse-variance weighted least squares.
4. Use an HC3 sandwich covariance matrix for analytic uncertainty.
5. Use a fixed-design wild bootstrap for slope/intercept uncertainty
   and a bootstrap test of H0: k = 0.
6. Use a nested acquisition-cluster bootstrap as a sensitivity analysis
   for uncertainty attributable to acquisition sampling.
7. Report leave-one-run-out calibration sensitivity.
8. Plot pointwise 95% intervals for the fitted mean.

Squeezing mode
--------------
When requested, all runs are split by the squeezing_device_present
field in each run's converged_result.json. Both groups are fitted
independently, then the slope ratio and squeezing magnitude are
computed in memory using the same pipeline as calculate_squeezing.py.
Both calibration results, bootstrap sample files, and a squeezing
result JSON are saved. The combined figure shows both calibration
lines with their pointwise intervals and per-run error bars.

Important limitations
---------------------
- Beam powers are treated as fixed and measured without relevant error.
- The number of independent calibration observations is the number of
  complete runs, not the number of subwindows.
- HC3 and wild-bootstrap inference remain approximate when there are
  very few runs.
- Without repeated complete runs at the same power, within-run
  sampling uncertainty cannot be cleanly separated from between-run
  reproducibility.
- Convergence-based stopping may affect finite-sample behavior.
- Absolute noise values depend on acquisition bandwidth.
- The slope ratio removes only the fitted additive intercept. It does
  not correct power-dependent backgrounds or mismatched bandwidths.
- The squeezing comparison assumes the two calibration datasets are
  independent.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


# =====================================================================
# CONFIGURATION
# =====================================================================

REQUIRE_CONVERGED_RUNS = True

WILD_BOOTSTRAP_REPETITIONS = 4999
CLUSTER_BOOTSTRAP_REPETITIONS = 4999
RANDOM_SEED = 42

CONFIDENCE_LEVEL = 0.95

SQUEEZING_BOOTSTRAP_REPETITIONS = 100_000
SQUEEZING_RANDOM_SEED = 20260828
MINIMUM_BOOTSTRAP_SAMPLES = 999

SIGNATURE_FIELDS = (
    "target_frequency_hz",
    "actual_frequency_hz",
    "enbw_hz",
    "subwindow_duration_seconds",
    "long_record_duration_seconds",
    "actual_sample_rate_hz",
    "reference_impedance_ohms",
    "spectral_method",
)
NUMERIC_SIGNATURE_RTOL = 1e-6

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

GROUP_STYLES = {
    "reference": {
        "color": "#1f77b4",
        "marker": "o",
        "label_prefix": "Reference (no squeezing)",
    },
    "squeezed": {
        "color": "#d62728",
        "marker": "s",
        "label_prefix": "Squeezed",
    },
}


# =====================================================================
# GENERAL UTILITIES
# =====================================================================

def get_auto_unit(max_value_mw: float) -> tuple[float, str]:
    value = abs(float(max_value_mw))
    if value >= 1.0:
        return 1.0, "mW"
    if value >= 1e-3:
        return 1e3, "μW"
    if value >= 1e-6:
        return 1e6, "nW"
    if value >= 1e-9:
        return 1e9, "pW"
    return 1e12, "fW"


def parse_run_input(input_str: str) -> list[int]:
    runs: set[int] = set()
    for part in input_str.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            match = re.fullmatch(r"(\d+)-(\d+)", part)
            if not match:
                raise ValueError(f"Invalid range format: {part!r}")
            start = int(match.group(1))
            end = int(match.group(2))
            if start > end:
                raise ValueError(
                    f"Range start greater than end: {part!r}"
                )
            runs.update(range(start, end + 1))
        else:
            if not part.isdigit():
                raise ValueError(f"Invalid run number: {part!r}")
            runs.add(int(part))
    return sorted(runs)


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def percentile_interval(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan, math.nan
    alpha = 1.0 - confidence_level
    low, high = np.percentile(
        values,
        [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
    )
    return float(low), float(high)


def basic_bootstrap_interval(
    estimate: float,
    bootstrap_estimates: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    values = np.asarray(bootstrap_estimates, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan, math.nan
    alpha = 1.0 - confidence_level
    deviations = values - estimate
    lo, hi = np.percentile(
        deviations,
        [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
    )
    return float(estimate - hi), float(estimate - lo)


def interval_text(
    interval: tuple[float, float],
    decimals: int,
) -> str:
    low, high = interval
    if math.isfinite(low) and math.isfinite(high):
        return f"[{low:.{decimals}f}, {high:.{decimals}f}]"
    return "undefined"


# =====================================================================
# CLUSTER SUMMARIES AND RUN-LEVEL VARIANCE
# =====================================================================

def cluster_jackknife_variance(
    cluster_sums: np.ndarray,
    cluster_counts: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    cluster_sums = np.asarray(cluster_sums, dtype=float)
    cluster_counts = np.asarray(cluster_counts, dtype=float)

    if cluster_sums.size != cluster_counts.size:
        raise ValueError(
            "Cluster sums and counts must have equal length"
        )

    number_of_clusters = cluster_sums.size

    if number_of_clusters < 2:
        raise ValueError(
            "At least two qualifying long-acquisition clusters are required"
        )

    if (
        not np.all(np.isfinite(cluster_sums))
        or not np.all(np.isfinite(cluster_counts))
        or np.any(cluster_counts <= 0)
    ):
        raise ValueError("Invalid acquisition-cluster summaries")

    total_sum = float(np.sum(cluster_sums))
    total_count = float(np.sum(cluster_counts))

    if total_count <= 0:
        raise ValueError("Run contains no qualifying subwindows")

    pooled_mean = total_sum / total_count
    remaining_counts = total_count - cluster_counts

    if np.any(remaining_counts <= 0):
        raise ValueError(
            "Deleting an acquisition would leave no observations"
        )

    leave_one_out_means = (
        total_sum - cluster_sums
    ) / remaining_counts
    leave_one_out_center = float(np.mean(leave_one_out_means))

    variance = (
        (number_of_clusters - 1.0)
        / number_of_clusters
        * float(np.sum((leave_one_out_means - leave_one_out_center) ** 2))
    )

    return pooled_mean, variance, leave_one_out_means


def load_acquisition_clusters(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    required_columns = {
        "long_acquisition_number",
        "noise_power_watts",
    }

    pair_data = pd.read_csv(
        csv_path,
        usecols=lambda name: name in required_columns,
    )

    missing = required_columns.difference(pair_data.columns)
    if missing:
        raise ValueError(
            "zero_pairs.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    pair_data["long_acquisition_number"] = pd.to_numeric(
        pair_data["long_acquisition_number"], errors="coerce"
    )
    pair_data["noise_power_watts"] = pd.to_numeric(
        pair_data["noise_power_watts"], errors="coerce"
    )

    valid = (
        pair_data["long_acquisition_number"].notna()
        & pair_data["noise_power_watts"].notna()
        & np.isfinite(pair_data["noise_power_watts"])
        & (pair_data["noise_power_watts"] > 0)
    )

    if not bool(valid.all()):
        invalid_count = int((~valid).sum())
        raise ValueError(
            f"zero_pairs.csv contains {invalid_count} invalid rows"
        )

    pair_data = pair_data.loc[valid].copy()

    grouped = pair_data.groupby(
        "long_acquisition_number", sort=True
    )["noise_power_watts"]

    cluster_sums = grouped.sum().to_numpy(dtype=float)
    cluster_counts = grouped.size().to_numpy(dtype=int)

    return cluster_sums, cluster_counts, len(pair_data)


def read_run_signature(metadata: dict) -> dict:
    """Extract measurement-signature fields from run JSON."""
    return {
        "target_frequency_hz": metadata.get("target_frequency_hz"),
        "actual_frequency_hz": metadata.get("actual_frequency_hz"),
        "enbw_hz": metadata.get("enbw_hz"),
        "subwindow_duration_seconds": metadata.get(
            "subwindow_duration_seconds"
        ),
        "long_record_duration_seconds": metadata.get(
            "long_record_duration_seconds"
        ),
        "actual_sample_rate_hz": metadata.get("actual_sample_rate_hz"),
        "reference_impedance_ohms": metadata.get(
            "reference_impedance_ohms"
        ),
        "spectral_method": metadata.get("spectral_method"),
    }


def signatures_compatible(
    sig_a: dict,
    sig_b: dict,
) -> list[str]:
    """Return mismatch descriptions between two signatures."""
    mismatches: list[str] = []

    for field in SIGNATURE_FIELDS:
        a = sig_a.get(field)
        b = sig_b.get(field)

        if a is None or b is None:
            mismatches.append(f"{field}: one or both values missing")
            continue

        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            a_f, b_f = float(a), float(b)
            if not (
                math.isfinite(a_f)
                and math.isfinite(b_f)
                and math.isclose(
                    a_f, b_f,
                    rel_tol=NUMERIC_SIGNATURE_RTOL,
                    abs_tol=0.0,
                )
            ):
                mismatches.append(
                    f"{field}: {a_f!r} vs {b_f!r}"
                )
        else:
            if str(a) != str(b):
                mismatches.append(f"{field}: {a!r} vs {b!r}")

    return mismatches


# =====================================================================
# DATA LOADING
# =====================================================================

def extract_data_from_runs(
    run_numbers: list[int],
    base_folder: str = ".",
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Load run metadata, cluster summaries, and measurement signature.

    Returns
    -------
    dataframe
        One row per valid run, sorted by beam power.
    cluster_data
        List of dicts with sums_watts and counts arrays, in the same
        order as dataframe rows.
    signature
        Measurement-signature dict assembled from the first run and
        verified consistent across all runs.
    """
    base_directory = Path(base_folder).expanduser().resolve()

    records: list[dict] = []
    cluster_data: list[dict] = []
    group_signature: dict | None = None

    print(f"\n--- SCANNING DIRECTORY: {base_directory} ---")

    for run_number in run_numbers:
        run_directory = base_directory / f"Run {run_number}"
        json_path = run_directory / "converged_result.json"
        pair_csv_path = run_directory / "zero_pairs.csv"

        if not json_path.is_file():
            print(
                f"[WARNING] Skipping Run {run_number}: "
                f"missing {json_path}",
                file=sys.stderr,
            )
            continue

        if not pair_csv_path.is_file():
            print(
                f"[WARNING] Skipping Run {run_number}: "
                f"missing {pair_csv_path}",
                file=sys.stderr,
            )
            continue

        try:
            with json_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)

            beam_power_mw = float(metadata["beam_power_mw"])
            json_mean_watts = float(metadata["final_mean_watts"])
            converged = bool(metadata.get("converged", False))

            if REQUIRE_CONVERGED_RUNS and not converged:
                print(
                    f"[WARNING] Skipping Run {run_number}: "
                    "run did not satisfy its convergence rule",
                    file=sys.stderr,
                )
                continue

            if not math.isfinite(beam_power_mw) or beam_power_mw < 0:
                raise ValueError(
                    "beam_power_mw must be finite and nonnegative"
                )

            if (
                not math.isfinite(json_mean_watts)
                or json_mean_watts <= 0
            ):
                raise ValueError(
                    "final_mean_watts must be finite and positive"
                )

            this_signature = read_run_signature(metadata)

            if group_signature is None:
                group_signature = this_signature
            else:
                mismatches = signatures_compatible(
                    group_signature, this_signature
                )
                if mismatches:
                    mismatch_text = "; ".join(mismatches)
                    raise ValueError(
                        f"Run {run_number} measurement signature "
                        f"differs from Run {run_numbers[0]}: "
                        + mismatch_text
                    )

            (
                cluster_sums,
                cluster_counts,
                pair_count,
            ) = load_acquisition_clusters(pair_csv_path)

            (
                csv_mean_watts,
                jackknife_variance_watts2,
                _,
            ) = cluster_jackknife_variance(cluster_sums, cluster_counts)

            if (
                not math.isfinite(jackknife_variance_watts2)
                or jackknife_variance_watts2 <= 0
            ):
                raise ValueError(
                    "cluster-jackknife variance is nonpositive"
                )

            discrepancy_scale = max(
                abs(csv_mean_watts),
                abs(json_mean_watts),
                np.finfo(float).tiny,
            )
            if (
                abs(csv_mean_watts - json_mean_watts) / discrepancy_scale
                > 1e-6
            ):
                raise ValueError(
                    "mean from zero_pairs.csv disagrees with JSON"
                )

            records.append(
                {
                    "Run": run_number,
                    "Power_mW": beam_power_mw,
                    "Noise_Watts": csv_mean_watts,
                    "Run_Variance_Watts2": jackknife_variance_watts2,
                    "Run_SE_Watts": math.sqrt(jackknife_variance_watts2),
                    "Acquisition_Clusters": int(cluster_sums.size),
                    "Qualifying_Pairs": pair_count,
                    "Converged": converged,
                }
            )

            cluster_data.append(
                {
                    "run": run_number,
                    "sums_watts": cluster_sums,
                    "counts": cluster_counts,
                }
            )

            print(
                f"  • Run {run_number:3d}: "
                f"P = {beam_power_mw:.7g} mW | "
                f"N = {csv_mean_watts:.7e} W | "
                f"jackknife SE = "
                f"{math.sqrt(jackknife_variance_watts2):.3e} W | "
                f"clusters = {cluster_sums.size}"
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
            pd.errors.ParserError,
        ) as error:
            print(
                f"[ERROR] Skipping Run {run_number}: {error}",
                file=sys.stderr,
            )

    if not records:
        raise RuntimeError("No valid converged runs could be loaded")

    dataframe = pd.DataFrame(records)
    order = np.argsort(dataframe["Power_mW"].to_numpy())
    dataframe = dataframe.iloc[order].reset_index(drop=True)
    cluster_data = [cluster_data[i] for i in order]

    if len(dataframe) < 4:
        raise RuntimeError(
            "At least four complete runs are required"
        )

    if dataframe["Power_mW"].nunique() < 2:
        raise RuntimeError(
            "At least two distinct beam-power settings are required"
        )

    return dataframe, cluster_data, group_signature or {}


def peek_squeezing_flag(
    run_number: int,
    base_folder: str = ".",
) -> str | None:
    """Return the squeezing_device_present string for one run."""
    json_path = (
        Path(base_folder).expanduser().resolve()
        / f"Run {run_number}"
        / "converged_result.json"
    )

    if not json_path.is_file():
        return None

    try:
        with json_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        return str(metadata.get("squeezing_device_present", "Unknown"))
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def split_runs_by_squeezing(
    run_numbers: list[int],
    base_folder: str = ".",
) -> tuple[list[int], list[int]]:
    """Split run numbers into (reference, squeezed) groups.

    Runs where squeezing_device_present is 'False' go to reference.
    Runs where squeezing_device_present is 'True'  go to squeezed.
    Runs where the value is 'Unknown' or unreadable are skipped with
    a warning.
    """
    reference_runs: list[int] = []
    squeezed_runs: list[int] = []
    skipped: list[int] = []

    for run_number in run_numbers:
        flag = peek_squeezing_flag(run_number, base_folder)

        if flag == "False":
            reference_runs.append(run_number)
        elif flag == "True":
            squeezed_runs.append(run_number)
        else:
            skipped.append(run_number)

    if skipped:
        print(
            f"[WARNING] Skipping {len(skipped)} run(s) with "
            f"Unknown or unreadable squeezing_device_present: "
            + ", ".join(str(r) for r in skipped),
            file=sys.stderr,
        )

    return reference_runs, squeezed_runs


# =====================================================================
# WLS FITTING AND HC3 COVARIANCE
# =====================================================================

def solve_wls(
    x: np.ndarray,
    y: np.ndarray,
    variances: np.ndarray,
) -> dict:
    """Fit y = k*x + intercept by inverse-variance WLS with HC3."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    variances = np.asarray(variances, dtype=float)

    if not (x.ndim == y.ndim == variances.ndim == 1):
        raise ValueError("x, y, and variances must be one-dimensional")
    if not (len(x) == len(y) == len(variances)):
        raise ValueError("x, y, and variances must have equal length")
    if len(x) < 3:
        raise ValueError("At least three runs are required")
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(variances))
        or np.any(variances <= 0)
    ):
        raise ValueError("Fit inputs contain invalid values")

    weights = 1.0 / variances
    weight_sum = float(np.sum(weights))
    weighted_x_mean = float(np.dot(weights, x) / weight_sum)
    weighted_x_var = float(
        np.dot(weights, (x - weighted_x_mean) ** 2) / weight_sum
    )

    if weighted_x_var <= 0:
        raise ValueError("The slope is not identifiable")

    x_scale = math.sqrt(weighted_x_var)
    z = (x - weighted_x_mean) / x_scale

    n = len(x)
    design_scaled = np.column_stack((z, np.ones(n)))
    sqrt_w = np.sqrt(weights)
    transformed_design = design_scaled * sqrt_w[:, np.newaxis]
    transformed_response = y * sqrt_w

    gamma, _, rank, singular_values = np.linalg.lstsq(
        transformed_design, transformed_response, rcond=None
    )

    if rank < 2:
        raise ValueError("Regression design matrix is rank deficient")

    coefficient_transform = np.array(
        [
            [1.0 / x_scale, 0.0],
            [-weighted_x_mean / x_scale, 1.0],
        ],
        dtype=float,
    )

    beta = coefficient_transform @ gamma
    slope = float(beta[0])
    intercept = float(beta[1])

    original_design = np.column_stack((x, np.ones(n)))
    fitted = original_design @ beta
    residuals = y - fitted
    transformed_residuals = sqrt_w * residuals

    information = transformed_design.T @ transformed_design
    bread = np.linalg.inv(information)
    leverage = np.einsum(
        "ij,jk,ik->i", transformed_design, bread, transformed_design
    )
    leverage = np.clip(leverage, 0.0, 1.0 - 1e-10)

    hc3_adjusted = transformed_residuals / (1.0 - leverage)
    meat = transformed_design.T @ (
        transformed_design * hc3_adjusted[:, np.newaxis] ** 2
    )
    cov_gamma_hc3 = bread @ meat @ bread
    cov_beta_hc3 = (
        coefficient_transform
        @ cov_gamma_hc3
        @ coefficient_transform.T
    )
    cov_beta_hc3 = (cov_beta_hc3 + cov_beta_hc3.T) / 2.0

    slope_se = math.sqrt(max(float(cov_beta_hc3[0, 0]), 0.0))
    intercept_se = math.sqrt(max(float(cov_beta_hc3[1, 1]), 0.0))

    weighted_y_mean = float(np.dot(weights, y) / weight_sum)
    weighted_sse = float(np.dot(weights, residuals ** 2))
    weighted_sst = float(
        np.dot(weights, (y - weighted_y_mean) ** 2)
    )

    weighted_r_squared = (
        1.0 - weighted_sse / weighted_sst
        if weighted_sst > 0
        else math.nan
    )

    unweighted_rmse = math.sqrt(float(np.mean(residuals ** 2)))
    transformed_residual_rms = math.sqrt(
        float(np.mean(transformed_residuals ** 2))
    )

    degrees_of_freedom = n - 2
    slope_t = (
        slope / slope_se
        if slope_se > 0
        else math.copysign(math.inf, slope)
    )
    slope_p = float(
        2.0 * t_dist.sf(abs(slope_t), degrees_of_freedom)
    )
    condition_number = float(
        singular_values[0] / singular_values[-1]
    )

    return {
        "beta": beta,
        "slope": slope,
        "intercept": intercept,
        "fitted": fitted,
        "residuals": residuals,
        "weights": weights,
        "variances": variances,
        "transformed_residuals": transformed_residuals,
        "leverage": leverage,
        "covariance_hc3": cov_beta_hc3,
        "slope_se_hc3": slope_se,
        "intercept_se_hc3": intercept_se,
        "weighted_r_squared": weighted_r_squared,
        "unweighted_rmse": unweighted_rmse,
        "transformed_residual_rms": transformed_residual_rms,
        "degrees_of_freedom": degrees_of_freedom,
        "slope_t_hc3": slope_t,
        "slope_p_hc3_approx": slope_p,
        "condition_number": condition_number,
    }


# =====================================================================
# BOOTSTRAP PROCEDURES
# =====================================================================

def fit_intercept_only_wls(
    y: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = 1.0 / variances
    intercept = float(np.dot(weights, y) / np.sum(weights))
    fitted = np.full_like(y, intercept, dtype=float)
    residuals = y - fitted
    leverage = weights / np.sum(weights)
    return fitted, residuals, leverage


def fixed_design_wild_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    variances: np.ndarray,
    original_fit: dict,
    repetitions: int,
    random_seed: int,
    confidence_level: float,
) -> dict:
    """Wild bootstrap with Rademacher multipliers on HC2-adjusted residuals."""
    rng = np.random.default_rng(random_seed)
    n = len(x)

    slope_samples = np.full(repetitions, np.nan)
    intercept_samples = np.full(repetitions, np.nan)
    null_t_samples = np.full(repetitions, np.nan)

    fitted = original_fit["fitted"]
    residuals = original_fit["residuals"]
    leverage = original_fit["leverage"]
    adjusted_residuals = residuals / np.sqrt(
        np.maximum(1.0 - leverage, 1e-10)
    )

    null_fitted, null_residuals, null_leverage = (
        fit_intercept_only_wls(y, variances)
    )
    null_adjusted = null_residuals / np.sqrt(
        np.maximum(1.0 - null_leverage, 1e-10)
    )

    for i in range(repetitions):
        multipliers = rng.choice(
            np.array([-1.0, 1.0]), size=n
        )

        try:
            bf = solve_wls(
                x,
                fitted + adjusted_residuals * multipliers,
                variances,
            )
            slope_samples[i] = bf["slope"]
            intercept_samples[i] = bf["intercept"]
        except (ValueError, np.linalg.LinAlgError):
            pass

        try:
            nf = solve_wls(
                x,
                null_fitted + null_adjusted * multipliers,
                variances,
            )
            null_t_samples[i] = nf["slope_t_hc3"]
        except (ValueError, np.linalg.LinAlgError):
            pass

    valid_slope = slope_samples[np.isfinite(slope_samples)]
    valid_intercept = intercept_samples[
        np.isfinite(intercept_samples)
    ]
    valid_null_t = null_t_samples[np.isfinite(null_t_samples)]

    slope_ci = basic_bootstrap_interval(
        original_fit["slope"], valid_slope, confidence_level
    )
    intercept_ci = basic_bootstrap_interval(
        original_fit["intercept"], valid_intercept, confidence_level
    )

    observed_t = abs(original_fit["slope_t_hc3"])
    bootstrap_p = (
        (1.0 + float(np.sum(np.abs(valid_null_t) >= observed_t)))
        / (valid_null_t.size + 1.0)
        if valid_null_t.size
        else math.nan
    )

    return {
        "slope_samples": valid_slope,
        "intercept_samples": valid_intercept,
        "null_t_samples": valid_null_t,
        "slope_ci": slope_ci,
        "intercept_ci": intercept_ci,
        "slope_p_value": float(bootstrap_p),
        "valid_repetitions": int(valid_slope.size),
        "valid_null_repetitions": int(valid_null_t.size),
    }


def cluster_bootstrap_calibration(
    x: np.ndarray,
    variances: np.ndarray,
    cluster_data: list[dict],
    repetitions: int,
    random_seed: int,
    confidence_level: float,
) -> dict:
    """Resample long acquisitions within each run independently."""
    rng = np.random.default_rng(random_seed)
    n = len(cluster_data)
    slope_samples = np.full(repetitions, np.nan)
    intercept_samples = np.full(repetitions, np.nan)

    for bi in range(repetitions):
        bootstrap_means = np.empty(n, dtype=float)
        for ri, run_clusters in enumerate(cluster_data):
            sums = run_clusters["sums_watts"]
            counts = run_clusters["counts"]
            nc = len(sums)
            selected = rng.integers(0, nc, size=nc)
            s_sum = float(np.sum(sums[selected]))
            s_count = float(np.sum(counts[selected]))
            bootstrap_means[ri] = s_sum / s_count

        try:
            bf = solve_wls(x, bootstrap_means, variances)
            slope_samples[bi] = bf["slope"]
            intercept_samples[bi] = bf["intercept"]
        except (ValueError, np.linalg.LinAlgError):
            pass

    valid = np.isfinite(slope_samples) & np.isfinite(intercept_samples)
    return {
        "slope_samples": slope_samples[valid],
        "intercept_samples": intercept_samples[valid],
        "slope_ci": percentile_interval(
            slope_samples[valid], confidence_level
        ),
        "intercept_ci": percentile_interval(
            intercept_samples[valid], confidence_level
        ),
        "valid_repetitions": int(np.sum(valid)),
    }


# =====================================================================
# LEAVE-ONE-RUN-OUT SENSITIVITY
# =====================================================================

def leave_one_run_out_analysis(
    x: np.ndarray,
    y: np.ndarray,
    variances: np.ndarray,
    run_numbers: np.ndarray,
    full_fit: dict,
) -> pd.DataFrame:
    records: list[dict] = []
    full_slope = full_fit["slope"]
    full_intercept = full_fit["intercept"]

    for index, run_number in enumerate(run_numbers):
        keep = np.ones(len(x), dtype=bool)
        keep[index] = False
        try:
            rf = solve_wls(x[keep], y[keep], variances[keep])
            slope_change = rf["slope"] - full_slope
            rel_change = (
                slope_change / full_slope
                if full_slope != 0
                else math.nan
            )
            records.append(
                {
                    "Run": int(run_number),
                    "LOO_Slope": rf["slope"],
                    "Slope_Change": slope_change,
                    "Relative_Slope_Change": rel_change,
                    "LOO_Intercept": rf["intercept"],
                    "Intercept_Change": rf["intercept"] - full_intercept,
                }
            )
        except (ValueError, np.linalg.LinAlgError):
            records.append(
                {
                    "Run": int(run_number),
                    "LOO_Slope": math.nan,
                    "Slope_Change": math.nan,
                    "Relative_Slope_Change": math.nan,
                    "LOO_Intercept": math.nan,
                    "Intercept_Change": math.nan,
                }
            )

    return pd.DataFrame(records)


# =====================================================================
# POINTWISE MEAN INTERVALS
# =====================================================================

def pointwise_mean_interval(
    power: np.ndarray,
    fit: dict,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    power = np.asarray(power, dtype=float)
    design = np.column_stack((power, np.ones(power.size)))
    fitted = design @ fit["beta"]
    variances = np.einsum(
        "ij,jk,ik->i", design, fit["covariance_hc3"], design
    )
    variances = np.maximum(variances, 0.0)
    se = np.sqrt(variances)
    t_crit = float(
        t_dist.ppf(
            0.5 + confidence_level / 2.0,
            fit["degrees_of_freedom"],
        )
    )
    return fitted, fitted - t_crit * se, fitted + t_crit * se


# =====================================================================
# SQUEEZING RATIO CALCULATION
# =====================================================================

def independently_combine_bootstraps(
    reference_slope: float,
    reference_slope_samples: np.ndarray,
    squeezed_slope: float,
    squeezed_slope_samples: np.ndarray,
    repetitions: int,
    random_seed: int,
) -> dict:
    """Independently resample the two slope-error distributions."""
    rng = np.random.default_rng(random_seed)

    ref_errors = reference_slope_samples - reference_slope
    sq_errors = squeezed_slope_samples - squeezed_slope

    ref_idx = rng.integers(0, ref_errors.size, size=repetitions)
    sq_idx = rng.integers(0, sq_errors.size, size=repetitions)

    ref_draws = reference_slope + ref_errors[ref_idx]
    sq_draws = squeezed_slope + sq_errors[sq_idx]

    finite = np.isfinite(ref_draws) & np.isfinite(sq_draws)
    ref_draws = ref_draws[finite]
    sq_draws = sq_draws[finite]

    denom_nonpositive = ref_draws <= 0
    num_nonpositive = sq_draws <= 0
    usable = ref_draws != 0
    ratio_draws = sq_draws[usable] / ref_draws[usable]
    ratio_draws = ratio_draws[np.isfinite(ratio_draws)]

    if ratio_draws.size < MINIMUM_BOOTSTRAP_SAMPLES:
        raise ValueError(
            "Too few finite ratio bootstrap realizations were produced"
        )

    return {
        "ratio_draws": ratio_draws,
        "finite_joint_draws": int(ref_draws.size),
        "ratio_draws_used": int(ratio_draws.size),
        "reference_nonpositive_fraction": float(
            np.mean(denom_nonpositive)
        ),
        "squeezed_nonpositive_fraction": float(
            np.mean(num_nonpositive)
        ),
    }


def calculate_squeezing(
    reference_slope_si: float,
    squeezed_slope_si: float,
    reference_slope_samples_si: np.ndarray,
    squeezed_slope_samples_si: np.ndarray,
    repetitions: int,
    random_seed: int,
    confidence_level: float,
) -> dict:
    """Compute slope ratio, squeezing magnitude, and bootstrap intervals."""
    ratio = squeezed_slope_si / reference_slope_si

    bootstrap = independently_combine_bootstraps(
        reference_slope=reference_slope_si,
        reference_slope_samples=reference_slope_samples_si,
        squeezed_slope=squeezed_slope_si,
        squeezed_slope_samples=squeezed_slope_samples_si,
        repetitions=repetitions,
        random_seed=random_seed,
    )

    ratio_ci = basic_bootstrap_interval(
        estimate=ratio,
        bootstrap_estimates=bootstrap["ratio_draws"],
        confidence_level=confidence_level,
    )

    ratio_low, ratio_high = ratio_ci

    signed_change_db = 10.0 * math.log10(ratio)
    squeezing_magnitude_db = -signed_change_db
    reduction_percent = 100.0 * (1.0 - ratio)

    log_defined = (
        math.isfinite(ratio_low)
        and math.isfinite(ratio_high)
        and ratio_low > 0
        and ratio_high > 0
    )

    if log_defined:
        signed_db_ci = (
            10.0 * math.log10(ratio_low),
            10.0 * math.log10(ratio_high),
        )
        squeezing_ci = (-signed_db_ci[1], -signed_db_ci[0])
        reduction_ci = (
            100.0 * (1.0 - ratio_high),
            100.0 * (1.0 - ratio_low),
        )
    else:
        signed_db_ci = (math.nan, math.nan)
        squeezing_ci = (math.nan, math.nan)
        reduction_ci = (math.nan, math.nan)

    return {
        "ratio": ratio,
        "ratio_ci": ratio_ci,
        "signed_change_db": signed_change_db,
        "signed_change_db_ci": signed_db_ci,
        "squeezing_magnitude_db": squeezing_magnitude_db,
        "squeezing_magnitude_db_ci": squeezing_ci,
        "reduction_percent": reduction_percent,
        "reduction_percent_ci": reduction_ci,
        "log_interval_defined": log_defined,
        "bootstrap": bootstrap,
    }


def print_squeezing_summary(
    sq: dict,
    confidence_level: float,
) -> None:
    cp = 100.0 * confidence_level
    boot = sq["bootstrap"]

    print()
    print("=" * 72)
    print("SQUEEZING ANALYSIS")
    print("=" * 72)
    print(
        f"Noise ratio R = k_sq / k_ref:    "
        f"{sq['ratio']:.8g}"
    )
    print(
        f"Basic-bootstrap {cp:.0f}% CI:        "
        + interval_text(sq["ratio_ci"], 8)
    )
    print()
    print(
        f"Signed noise change 10 log10(R): "
        f"{sq['signed_change_db']:.5f} dB"
    )
    print(
        f"{cp:.0f}% CI:                         "
        + interval_text(sq["signed_change_db_ci"], 5)
        + " dB"
    )
    print()
    print(
        f"Squeezing magnitude:             "
        f"{sq['squeezing_magnitude_db']:.5f} dB"
    )
    print(
        f"{cp:.0f}% CI:                         "
        + interval_text(sq["squeezing_magnitude_db_ci"], 5)
        + " dB"
    )
    print()
    print(
        f"Noise reduction:                 "
        f"{sq['reduction_percent']:.3f}%"
    )
    print(
        f"{cp:.0f}% CI:                         "
        + interval_text(sq["reduction_percent_ci"], 3)
        + "%"
    )
    print("-" * 72)
    print(
        f"Ref  bootstrap draws with nonpositive slope: "
        f"{100.0 * boot['reference_nonpositive_fraction']:.3f}%"
    )
    print(
        f"Sq   bootstrap draws with nonpositive slope: "
        f"{100.0 * boot['squeezed_nonpositive_fraction']:.3f}%"
    )

    if not sq["log_interval_defined"]:
        print(
            "[WARNING] Ratio interval includes a nonpositive value; "
            "dB interval is undefined."
        )

    if (
        boot["reference_nonpositive_fraction"] > 0.01
        or boot["squeezed_nonpositive_fraction"] > 0.01
    ):
        print(
            "[WARNING] >1% of bootstrap slope draws are nonpositive. "
            "The dB interval should not be treated as reliable."
        )

    ratio_low, ratio_high = sq["ratio_ci"]
    if math.isfinite(ratio_low) and math.isfinite(ratio_high):
        if ratio_high < 1.0:
            print(
                f"Bootstrap {cp:.0f}% CI lies entirely below R = 1: "
                "squeezing is resolved at this confidence level."
            )
        elif ratio_low > 1.0:
            print(
                "Bootstrap CI lies entirely above R = 1: "
                "the squeezed slope exceeds the reference (anti-squeezing "
                "or swapped inputs)."
            )
        else:
            print(
                f"Bootstrap {cp:.0f}% CI includes R = 1: the data do "
                "not resolve squeezing from no change at this level."
            )

    print("-" * 72)
    print(
        "These results assume the two calibration datasets are independent."
    )
    print(
        "No detection-efficiency or optical-loss correction was applied."
    )
    print("=" * 72)


# =====================================================================
# OUTPUT HELPERS
# =====================================================================

def build_calibration_result(
    dataframe: pd.DataFrame,
    fit: dict,
    wild: dict,
    cluster_bootstrap: dict,
    loo: pd.DataFrame,
    power_unit: str,
    noise_unit: str,
    power_scale: float,
    noise_scale: float,
    signature: dict,
    group_label: str,
) -> dict:
    """Build the JSON-serialisable result for one calibration."""
    watts_to_display = 1e3 * noise_scale

    slope_si = fit["slope"] / watts_to_display * (1.0 / power_scale)
    intercept_si = fit["intercept"] / watts_to_display

    return {
        "group": group_label,
        "model": "noise = slope * power + dark_intercept",
        "number_of_runs": int(len(dataframe)),
        "power_unit": power_unit,
        "noise_unit": noise_unit,
        "estimation": {
            "method": (
                "inverse cluster-jackknife variance weighted "
                "least squares"
            ),
            "covariance": "HC3 sandwich",
            "slope": finite_or_none(fit["slope"]),
            "slope_hc3_standard_error": finite_or_none(
                fit["slope_se_hc3"]
            ),
            "slope_si_w_per_w": finite_or_none(slope_si),
            "dark_intercept": finite_or_none(fit["intercept"]),
            "dark_intercept_hc3_standard_error": finite_or_none(
                fit["intercept_se_hc3"]
            ),
            "dark_intercept_si_w": finite_or_none(intercept_si),
            "weighted_r_squared": finite_or_none(
                fit["weighted_r_squared"]
            ),
            "unweighted_residual_rmse": finite_or_none(
                fit["unweighted_rmse"]
            ),
            "approximate_hc3_t_statistic": finite_or_none(
                fit["slope_t_hc3"]
            ),
            "approximate_hc3_t_p_value": finite_or_none(
                fit["slope_p_hc3_approx"]
            ),
        },
        "fixed_design_wild_bootstrap": {
            "repetitions": wild["valid_repetitions"],
            "confidence_level": CONFIDENCE_LEVEL,
            "slope_basic_interval": [
                finite_or_none(wild["slope_ci"][0]),
                finite_or_none(wild["slope_ci"][1]),
            ],
            "dark_intercept_basic_interval": [
                finite_or_none(wild["intercept_ci"][0]),
                finite_or_none(wild["intercept_ci"][1]),
            ],
            "slope_null_test_p_value": finite_or_none(
                wild["slope_p_value"]
            ),
            "samples_file": (
                f"{group_label}_bootstrap_slope_samples.npz"
            ),
        },
        "within_run_cluster_bootstrap_sensitivity": {
            "repetitions": cluster_bootstrap["valid_repetitions"],
            "confidence_level": CONFIDENCE_LEVEL,
            "slope_percentile_interval": [
                finite_or_none(cluster_bootstrap["slope_ci"][0]),
                finite_or_none(cluster_bootstrap["slope_ci"][1]),
            ],
            "dark_intercept_percentile_interval": [
                finite_or_none(cluster_bootstrap["intercept_ci"][0]),
                finite_or_none(cluster_bootstrap["intercept_ci"][1]),
            ],
        },
        "leave_one_run_out": [
            {
                k: (
                    int(v)
                    if k == "Run"
                    else finite_or_none(v)
                )
                for k, v in row.items()
            }
            for row in loo.to_dict(orient="records")
        ],
        "measurement_signature": signature,
    }


def save_calibration_outputs(
    result: dict,
    wild: dict,
    fit: dict,
    power_scale: float,
    noise_scale: float,
    group_label: str,
) -> None:
    """Save the JSON result and bootstrap NPZ file for one group."""
    watts_to_display = 1e3 * noise_scale

    slope_samples_si = (
        wild["slope_samples"] / watts_to_display * (1.0 / power_scale)
    )

    npz_path = Path(
        f"{group_label}_bootstrap_slope_samples.npz"
    ).resolve()

    np.savez_compressed(
        npz_path,
        slope_samples_si_w_per_w=slope_samples_si,
    )

    json_path = Path(
        f"{group_label}_calibration_results.json"
    ).resolve()

    json_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"  Calibration JSON:  {json_path}")
    print(f"  Bootstrap samples: {npz_path}")


def print_fit_summary(
    fit: dict,
    wild: dict,
    cluster_bootstrap: dict,
    loo: pd.DataFrame,
    power_unit: str,
    noise_unit: str,
    group_label: str,
) -> None:
    slope_unit = f"{noise_unit}/{power_unit}"
    cp = 100.0 * CONFIDENCE_LEVEL

    print(f"\n{'=' * 68}")
    print(f"CALIBRATION RESULTS — {group_label.upper()}")
    print(f"{'=' * 68}")
    print(
        f" Slope k:                       "
        f"{fit['slope']:.7g} ± "
        f"{fit['slope_se_hc3']:.3g} {slope_unit}"
    )
    print(
        f" Dark intercept:                "
        f"{fit['intercept']:.7g} ± "
        f"{fit['intercept_se_hc3']:.3g} {noise_unit}"
    )
    print(
        f" Wild-bootstrap {cp:.0f}% CI, slope: "
        + interval_text(wild["slope_ci"], 6)
        + f" {slope_unit}"
    )
    print(
        f" Wild-bootstrap {cp:.0f}% CI, dark:  "
        + interval_text(wild["intercept_ci"], 6)
        + f" {noise_unit}"
    )
    print(
        f" Cluster-bootstrap {cp:.0f}% range, slope: "
        + interval_text(cluster_bootstrap["slope_ci"], 6)
        + f" {slope_unit}"
    )
    print(f" Weighted R²:                   {fit['weighted_r_squared']:.6f}")
    print(
        f" Raw residual RMSE:             "
        f"{fit['unweighted_rmse']:.6g} {noise_unit}"
    )
    print(
        f" RMS transformed residual:      "
        f"{fit['transformed_residual_rms']:.6g} (dimensionless)"
    )
    print(
        f" Approx HC3 slope t:            "
        f"{fit['slope_t_hc3']:.6g}"
    )
    print(
        f" Approx HC3 t p-value:          "
        f"{fit['slope_p_hc3_approx']:.6e}"
    )
    print(
        f" Wild-bootstrap slope p-value:  "
        f"{wild['slope_p_value']:.6e}"
    )
    print(f" {'─' * 50}")
    print(" LEAVE-ONE-RUN-OUT SLOPE SENSITIVITY")

    max_loo_change = 0.0
    for row in loo.itertuples(index=False):
        if math.isfinite(row.LOO_Slope):
            pct = 100.0 * row.Relative_Slope_Change
            print(
                f"   Omit Run {row.Run:3d}: "
                f"k = {row.LOO_Slope:.7g} {slope_unit}, "
                f"change = {pct:+.2f}%"
            )
            max_loo_change = max(
                max_loo_change, abs(row.Relative_Slope_Change)
            )
        else:
            print(f"   Omit Run {row.Run:3d}: refit undefined")

    print(
        f" Maximum absolute LOO slope change: "
        f"{max_loo_change * 100:.2f}%"
    )


# =====================================================================
# SINGLE-GROUP PIPELINE
# =====================================================================

def run_single_calibration(
    run_numbers: list[int],
    base_folder: str,
    group_label: str,
) -> tuple[pd.DataFrame, dict, dict, dict, dict, float, float, str, str, dict]:
    """Run the complete calibration pipeline for one group of runs.

    Returns
    -------
    dataframe, fit, wild, cluster_bootstrap, loo,
    power_scale, noise_scale, power_unit, noise_unit, signature
    """
    dataframe, cluster_data, signature = extract_data_from_runs(
        run_numbers, base_folder
    )

    power_scale, power_unit = get_auto_unit(
        dataframe["Power_mW"].abs().max()
    )
    max_noise_mw = dataframe["Noise_Watts"].abs().max() * 1e3
    noise_scale, noise_unit = get_auto_unit(max_noise_mw)
    watts_to_display = 1e3 * noise_scale

    power_display = (
        dataframe["Power_mW"].to_numpy(dtype=float) * power_scale
    )
    noise_display = (
        dataframe["Noise_Watts"].to_numpy(dtype=float)
        * watts_to_display
    )
    run_variance_display = (
        dataframe["Run_Variance_Watts2"].to_numpy(dtype=float)
        * watts_to_display ** 2
    )

    fit = solve_wls(power_display, noise_display, run_variance_display)

    dataframe["Power_Display"] = power_display
    dataframe["Noise_Display"] = noise_display
    dataframe["Run_SE_Display"] = np.sqrt(run_variance_display)
    dataframe["Fitted_Display"] = fit["fitted"]
    dataframe["Residual_Display"] = fit["residuals"]

    print(
        f"\nRunning {WILD_BOOTSTRAP_REPETITIONS:,} wild-bootstrap "
        f"replicates ({group_label})..."
    )
    wild = fixed_design_wild_bootstrap(
        x=power_display,
        y=noise_display,
        variances=run_variance_display,
        original_fit=fit,
        repetitions=WILD_BOOTSTRAP_REPETITIONS,
        random_seed=RANDOM_SEED,
        confidence_level=CONFIDENCE_LEVEL,
    )

    print(
        f"Running {CLUSTER_BOOTSTRAP_REPETITIONS:,} cluster-bootstrap "
        f"replicates ({group_label})..."
    )
    cluster_bootstrap = cluster_bootstrap_calibration(
        x=power_display,
        variances=run_variance_display,
        cluster_data=cluster_data,
        repetitions=CLUSTER_BOOTSTRAP_REPETITIONS,
        random_seed=RANDOM_SEED + 1,
        confidence_level=CONFIDENCE_LEVEL,
    )

    loo = leave_one_run_out_analysis(
        x=power_display,
        y=noise_display,
        variances=run_variance_display,
        run_numbers=dataframe["Run"].to_numpy(),
        full_fit=fit,
    )

    print_fit_summary(
        fit, wild, cluster_bootstrap, loo,
        power_unit, noise_unit, group_label,
    )

    return (
        dataframe, fit, wild, cluster_bootstrap, loo,
        power_scale, noise_scale, power_unit, noise_unit, signature,
    )


# =====================================================================
# PLOTTING
# =====================================================================

def make_model_grid(
    power_display: np.ndarray,
    extra_fraction: float = 0.1,
    n_points: int = 400,
) -> np.ndarray:
    max_power = float(np.max(power_display))
    right = (
        (1.0 + extra_fraction) * max_power
        if max_power > 0
        else 1.0
    )
    return np.linspace(0.0, right, n_points)


def plot_single_group(figure_path_png: str = "calibration_figure.png") -> None:
    """Placeholder: handled inline in main."""
    pass


def plot_combined(
    group_frames: dict[str, pd.DataFrame],
    group_fits: dict[str, dict],
    power_units: dict[str, str],
    noise_units: dict[str, str],
    figure_path_png: str = "calibration_figure.png",
    figure_path_pdf: str = "calibration_figure.pdf",
) -> None:
    """Plot both calibration groups on a shared figure."""
    all_powers = np.concatenate(
        [
            df["Power_Display"].to_numpy()
            for df in group_frames.values()
        ]
    )

    power_grid = make_model_grid(all_powers)

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1,
        figsize=(9.0, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    for group_key, df in group_frames.items():
        style = GROUP_STYLES[group_key]
        fit = group_fits[group_key]
        noise_unit = noise_units[group_key]
        color = style["color"]
        marker = style["marker"]
        label_prefix = style["label_prefix"]

        fitted_grid, ci_low, ci_high = pointwise_mean_interval(
            power_grid, fit, CONFIDENCE_LEVEL
        )

        r2 = fit["weighted_r_squared"]
        r2_label = (
            f"{r2:.4f}" if math.isfinite(r2) else "undef"
        )

        ax_main.fill_between(
            power_grid, ci_low, ci_high,
            color=color, alpha=0.15,
        )
        ax_main.plot(
            power_grid, fitted_grid,
            color=color, linewidth=2.0,
            label=(
                f"{label_prefix}: "
                f"k={fit['slope']:.3g}, "
                f"R²={r2_label}"
            ),
        )

        ax_main.errorbar(
            df["Power_Display"].to_numpy(),
            df["Noise_Display"].to_numpy(),
            yerr=df["Run_SE_Display"].to_numpy(),
            fmt=marker,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            markersize=6,
            zorder=4,
            label=f"{label_prefix} run means ± jackknife SE",
        )

        for run_id, x_val, y_val in zip(
            df["Run"],
            df["Power_Display"].to_numpy(),
            df["Noise_Display"].to_numpy(),
        ):
            ax_main.annotate(
                str(run_id),
                xy=(x_val, y_val),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.0,
                color=color,
            )

        ax_res.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
        ax_res.errorbar(
            df["Power_Display"].to_numpy(),
            df["Residual_Display"].to_numpy(),
            yerr=df["Run_SE_Display"].to_numpy(),
            fmt=marker,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            ecolor=color,
            elinewidth=0.8,
            capsize=3,
            markersize=5.0,
            zorder=3,
        )

    all_noise_units = list(set(noise_units.values()))
    noise_unit_label = (
        all_noise_units[0] if len(all_noise_units) == 1
        else " / ".join(all_noise_units)
    )

    all_power_units = list(set(power_units.values()))
    power_unit_label = (
        all_power_units[0] if len(all_power_units) == 1
        else " / ".join(all_power_units)
    )

    ax_main.set_xlim(0.0, float(np.max(power_grid)))
    ax_main.set_ylabel(
        f"Linear noise power ({noise_unit_label})"
    )
    ax_main.set_title(
        "Cluster-aware photodetector noise calibration — "
        "squeezed vs. reference",
        fontweight="bold",
    )
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(fontsize=8.0, framealpha=0.9)

    ax_res.set_xlabel(
        f"Optical beam power ({power_unit_label})"
    )
    ax_res.set_ylabel("Residual")
    ax_res.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()

    png_path = Path(figure_path_png).resolve()
    pdf_path = Path(figure_path_pdf).resolve()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"  Figure PNG: {png_path}")
    print(f"  Figure PDF: {pdf_path}")

    plt.show()


def plot_single(
    dataframe: pd.DataFrame,
    fit: dict,
    power_unit: str,
    noise_unit: str,
    figure_path_png: str = "calibration_figure.png",
    figure_path_pdf: str = "calibration_figure.pdf",
) -> None:
    """Plot one calibration group."""
    power_grid = make_model_grid(
        dataframe["Power_Display"].to_numpy()
    )
    fitted_grid, ci_low, ci_high = pointwise_mean_interval(
        power_grid, fit, CONFIDENCE_LEVEL
    )

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1,
        figsize=(8.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    r2 = fit["weighted_r_squared"]
    r2_label = f"{r2:.4f}" if math.isfinite(r2) else "undef"

    ax_main.fill_between(
        power_grid, ci_low, ci_high,
        color="#1f77b4", alpha=0.18,
        label="Approx. pointwise 95% interval for fitted mean",
    )
    ax_main.plot(
        power_grid, fitted_grid,
        color="#1f77b4", linewidth=2.0,
        label=(
            f"WLS fit: N = {fit['slope']:.3g}P + {fit['intercept']:.3g} "
            f"(R²={r2_label})"
        ),
    )
    ax_main.plot(
        power_grid, fit["slope"] * power_grid,
        color="#2ca02c", linestyle="--", linewidth=1.8,
        label="Fitted shot-noise component kP",
    )
    ax_main.errorbar(
        dataframe["Power_Display"].to_numpy(),
        dataframe["Noise_Display"].to_numpy(),
        yerr=dataframe["Run_SE_Display"].to_numpy(),
        fmt="o", color="#1f77b4",
        markeredgecolor="white", markeredgewidth=0.7,
        ecolor="#555555", elinewidth=1.0, capsize=3,
        markersize=6, zorder=4,
        label="Run means ± cluster-jackknife SE",
    )

    for run_id, x_val, y_val in zip(
        dataframe["Run"],
        dataframe["Power_Display"].to_numpy(),
        dataframe["Noise_Display"].to_numpy(),
    ):
        ax_main.annotate(
            str(run_id),
            xy=(x_val, y_val),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#333333",
        )

    ax_main.set_xlim(0.0, float(np.max(power_grid)))
    ax_main.set_ylabel(f"Linear noise power ({noise_unit})")
    ax_main.set_title(
        "Cluster-aware photodetector noise calibration",
        fontweight="bold",
    )
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(fontsize=8.2, framealpha=0.9)

    ax_res.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax_res.errorbar(
        dataframe["Power_Display"].to_numpy(),
        dataframe["Residual_Display"].to_numpy(),
        yerr=dataframe["Run_SE_Display"].to_numpy(),
        fmt="o", color="#1f77b4",
        markeredgecolor="white", markeredgewidth=0.6,
        ecolor="#777777", elinewidth=0.9, capsize=3,
        markersize=5.5,
    )
    ax_res.set_xlabel(f"Optical beam power ({power_unit})")
    ax_res.set_ylabel(f"Residual\n({noise_unit})")
    ax_res.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()

    png_path = Path(figure_path_png).resolve()
    pdf_path = Path(figure_path_pdf).resolve()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"  Figure PNG: {png_path}")
    print(f"  Figure PDF: {pdf_path}")

    plt.show()


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    print("=" * 68)
    print("CLUSTER-AWARE PHOTODETECTOR NOISE CALIBRATION")
    print("=" * 68)

    # ── Mode selection ─────────────────────────────────────────────────
    while True:
        mode_input = (
            input(
                "\nWould you like to do a squeezing comparison? [y/n]: "
            )
            .strip()
            .lower()
        )
        if mode_input in {"y", "n"}:
            squeezing_mode = mode_input == "y"
            break
        print("Please enter y or n.")

    # ── Run number input ───────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                "Enter runs, e.g. '1-8' or '1,3-6,9': "
            )
            all_run_numbers = parse_run_input(user_input)
            if not all_run_numbers:
                raise ValueError("No run numbers were provided")
            break
        except ValueError as error:
            print(f"Input error: {error}\n")

    # ══════════════════════════════════════════════════════════════════
    # SINGLE CALIBRATION MODE
    # ══════════════════════════════════════════════════════════════════
    if not squeezing_mode:
        try:
            (
                dataframe, fit, wild, cluster_bootstrap, loo,
                power_scale, noise_scale, power_unit, noise_unit,
                signature,
            ) = run_single_calibration(
                all_run_numbers,
                base_folder=".",
                group_label="calibration",
            )
        except RuntimeError as error:
            print(f"\n[FATAL] {error}", file=sys.stderr)
            sys.exit(1)

        watts_to_display = 1e3 * noise_scale

        result = build_calibration_result(
            dataframe=dataframe,
            fit=fit,
            wild=wild,
            cluster_bootstrap=cluster_bootstrap,
            loo=loo,
            power_unit=power_unit,
            noise_unit=noise_unit,
            power_scale=power_scale,
            noise_scale=noise_scale,
            signature=signature,
            group_label="calibration",
        )

        print("\nSaved:")
        save_calibration_outputs(
            result=result,
            wild=wild,
            fit=fit,
            power_scale=power_scale,
            noise_scale=noise_scale,
            group_label="calibration",
        )

        dataframe.to_csv(
            "calibration_diagnostics.csv", index=False
        )
        print(
            f"  Diagnostics:       "
            f"{Path('calibration_diagnostics.csv').resolve()}"
        )

        plot_single(
            dataframe=dataframe,
            fit=fit,
            power_unit=power_unit,
            noise_unit=noise_unit,
        )

        return

    # ══════════════════════════════════════════════════════════════════
    # SQUEEZING COMPARISON MODE
    # ══════════════════════════════════════════════════════════════════

    reference_runs, squeezed_runs = split_runs_by_squeezing(
        all_run_numbers, base_folder="."
    )

    print(
        f"\nReference runs (squeezing_device_present = False): "
        + (", ".join(str(r) for r in reference_runs) or "none")
    )
    print(
        f"Squeezed runs  (squeezing_device_present = True):  "
        + (", ".join(str(r) for r in squeezed_runs) or "none")
    )

    if not reference_runs:
        print(
            "[FATAL] No reference (unsqueezed) runs found.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not squeezed_runs:
        print(
            "[FATAL] No squeezed runs found.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        (
            df_ref, fit_ref, wild_ref, cb_ref, loo_ref,
            pscale_ref, nscale_ref, punit_ref, nunit_ref,
            sig_ref,
        ) = run_single_calibration(
            reference_runs, base_folder=".", group_label="reference"
        )

        (
            df_sq, fit_sq, wild_sq, cb_sq, loo_sq,
            pscale_sq, nscale_sq, punit_sq, nunit_sq,
            sig_sq,
        ) = run_single_calibration(
            squeezed_runs, base_folder=".", group_label="squeezed"
        )

    except RuntimeError as error:
        print(f"\n[FATAL] {error}", file=sys.stderr)
        sys.exit(1)

    # ── Cross-group signature check ────────────────────────────────────
    mismatches = signatures_compatible(sig_ref, sig_sq)
    if mismatches:
        mismatch_text = "\n  - ".join(mismatches)
        print(
            "\n[FATAL] Reference and squeezed measurement signatures "
            "do not match:\n  - " + mismatch_text,
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Convert bootstrap slopes to SI ────────────────────────────────
    def to_si(samples: np.ndarray, pscale: float, nscale: float) -> np.ndarray:
        return samples / (1e3 * nscale) * (1.0 / pscale)

    ref_slope_si = fit_ref["slope"] / (1e3 * nscale_ref) * (1.0 / pscale_ref)
    sq_slope_si  = fit_sq["slope"]  / (1e3 * nscale_sq)  * (1.0 / pscale_sq)

    ref_samples_si = to_si(wild_ref["slope_samples"], pscale_ref, nscale_ref)
    sq_samples_si  = to_si(wild_sq["slope_samples"],  pscale_sq,  nscale_sq)

    # ── Squeezing calculation ──────────────────────────────────────────
    try:
        sq_results = calculate_squeezing(
            reference_slope_si=ref_slope_si,
            squeezed_slope_si=sq_slope_si,
            reference_slope_samples_si=ref_samples_si,
            squeezed_slope_samples_si=sq_samples_si,
            repetitions=SQUEEZING_BOOTSTRAP_REPETITIONS,
            random_seed=SQUEEZING_RANDOM_SEED,
            confidence_level=CONFIDENCE_LEVEL,
        )
    except ValueError as error:
        print(f"\n[FATAL] Squeezing calculation failed: {error}", file=sys.stderr)
        sys.exit(1)

    print_squeezing_summary(sq_results, CONFIDENCE_LEVEL)

    # ── Save outputs ───────────────────────────────────────────────────
    result_ref = build_calibration_result(
        df_ref, fit_ref, wild_ref, cb_ref, loo_ref,
        punit_ref, nunit_ref, pscale_ref, nscale_ref,
        sig_ref, "reference",
    )
    result_sq = build_calibration_result(
        df_sq, fit_sq, wild_sq, cb_sq, loo_sq,
        punit_sq, nunit_sq, pscale_sq, nscale_sq,
        sig_sq, "squeezed",
    )

    boot = sq_results["bootstrap"]

    squeezing_json = {
        "method": (
            "independent combination of fixed-design wild-bootstrap "
            "slope-error distributions"
        ),
        "confidence_level": CONFIDENCE_LEVEL,
        "independence_assumption": (
            "Reference and squeezed calibration datasets are independent."
        ),
        "reference_calibration": {
            "results_file": "reference_calibration_results.json",
            "bootstrap_file": "reference_bootstrap_slope_samples.npz",
            "number_of_runs": int(len(df_ref)),
            "slope_si_w_per_w": finite_or_none(ref_slope_si),
        },
        "squeezed_calibration": {
            "results_file": "squeezed_calibration_results.json",
            "bootstrap_file": "squeezed_bootstrap_slope_samples.npz",
            "number_of_runs": int(len(df_sq)),
            "slope_si_w_per_w": finite_or_none(sq_slope_si),
        },
        "measurement_signature": sig_ref,
        "results": {
            "noise_ratio": finite_or_none(sq_results["ratio"]),
            "noise_ratio_interval": [
                finite_or_none(sq_results["ratio_ci"][0]),
                finite_or_none(sq_results["ratio_ci"][1]),
            ],
            "signed_noise_change_db": finite_or_none(
                sq_results["signed_change_db"]
            ),
            "signed_noise_change_db_interval": [
                finite_or_none(sq_results["signed_change_db_ci"][0]),
                finite_or_none(sq_results["signed_change_db_ci"][1]),
            ],
            "squeezing_magnitude_db": finite_or_none(
                sq_results["squeezing_magnitude_db"]
            ),
            "squeezing_magnitude_db_interval": [
                finite_or_none(sq_results["squeezing_magnitude_db_ci"][0]),
                finite_or_none(sq_results["squeezing_magnitude_db_ci"][1]),
            ],
            "noise_reduction_percent": finite_or_none(
                sq_results["reduction_percent"]
            ),
            "noise_reduction_percent_interval": [
                finite_or_none(sq_results["reduction_percent_ci"][0]),
                finite_or_none(sq_results["reduction_percent_ci"][1]),
            ],
            "logarithmic_interval_defined": (
                sq_results["log_interval_defined"]
            ),
        },
        "bootstrap_diagnostics": {
            "finite_joint_draws": boot["finite_joint_draws"],
            "finite_ratio_draws": boot["ratio_draws_used"],
            "reference_nonpositive_slope_fraction": (
                boot["reference_nonpositive_fraction"]
            ),
            "squeezed_nonpositive_slope_fraction": (
                boot["squeezed_nonpositive_fraction"]
            ),
        },
        "limitations": [
            "This analysis assumes the two datasets are independent.",
            (
                "The ratio removes only the fitted additive intercept; "
                "it does not correct power-dependent backgrounds."
            ),
            "No detection-efficiency correction was applied.",
            (
                "Upstream uncertainty is approximate when there are "
                "few calibration runs."
            ),
        ],
    }

    print("\nSaved:")
    save_calibration_outputs(
        result_ref, wild_ref, fit_ref,
        pscale_ref, nscale_ref, "reference",
    )
    save_calibration_outputs(
        result_sq, wild_sq, fit_sq,
        pscale_sq, nscale_sq, "squeezed",
    )

    sq_json_path = Path("squeezing_results.json").resolve()
    sq_json_path.write_text(
        json.dumps(squeezing_json, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"  Squeezing JSON:    {sq_json_path}")

    df_ref.to_csv("reference_calibration_diagnostics.csv", index=False)
    df_sq.to_csv("squeezed_calibration_diagnostics.csv", index=False)
    print(
        f"  Diagnostics:       "
        f"{Path('reference_calibration_diagnostics.csv').resolve()}"
    )
    print(
        f"                     "
        f"{Path('squeezed_calibration_diagnostics.csv').resolve()}"
    )

    plot_combined(
        group_frames={"reference": df_ref, "squeezed": df_sq},
        group_fits={"reference": fit_ref, "squeezed": fit_sq},
        power_units={"reference": punit_ref, "squeezed": punit_sq},
        noise_units={"reference": nunit_ref, "squeezed": nunit_sq},
    )


if __name__ == "__main__":
    main()