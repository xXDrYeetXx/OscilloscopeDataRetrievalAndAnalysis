#!/usr/bin/env python3
"""
calculate_squeezing.py
======================

Compare independently acquired squeezed and unsqueezed calibration slopes.

For the fitted models

    N_unsq(P) = k_unsq P + b_unsq
    N_sq(P)   = k_sq   P + b_sq

the observed noise ratio is

    R = k_sq / k_unsq.

The signed noise change is

    D = 10 log10(R) dB,

and the positive squeezing magnitude, when R < 1, is

    S = -10 log10(R) dB.

Uncertainty is propagated by independently sampling the fixed-design
wild-bootstrap slope-error distributions saved by the two upstream
calibration analyses. A basic bootstrap interval is constructed for R
and then transformed monotonically into dB and percent reduction.

This script is valid only when the two calibration datasets are
independent. Paired, interleaved, or otherwise correlated measurements
require a joint bootstrap that preserves their dependence.

The ratio removes a constant fitted additive intercept from the slope
comparison. It does not correct power-dependent backgrounds, changing
dark noise, detector nonlinearity, or mismatched measurement bandwidths.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_REPETITIONS = 100_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_RANDOM_SEED = 20260828
DEFAULT_OUTPUT = "squeezing_results.json"

MINIMUM_BOOTSTRAP_SAMPLES = 999

# Fields that must agree between the squeezed and reference datasets.
# Additional experimental settings should be added if they can alter
# the measured calibration slope.
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
NUMERIC_SIGNATURE_ATOL = 0.0


def finite_or_none(value: float) -> float | None:
    """Return a finite float or None for standards-compliant JSON."""
    value = float(value)
    return value if math.isfinite(value) else None


def load_json(path: Path) -> dict[str, Any]:
    """Load and validate a JSON object."""
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {path}: {error}") from error

    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")

    return value


def nested_get(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
    source: Path,
) -> Any:
    """Retrieve a required nested JSON value."""
    current: Any = mapping

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(keys)
            raise ValueError(
                f"{source} is missing required field {joined!r}"
            )
        current = current[key]

    return current


def load_calibration(path: Path) -> dict[str, Any]:
    """Load one upstream calibration result and bootstrap sample file."""
    result = load_json(path)

    slope = float(
        nested_get(
            result,
            ("estimation", "slope_si_w_per_w"),
            path,
        )
    )

    if not math.isfinite(slope) or slope <= 0:
        raise ValueError(
            f"{path}: fitted SI slope must be finite and positive"
        )

    samples_name = nested_get(
        result,
        (
            "fixed_design_wild_bootstrap",
            "samples_file",
        ),
        path,
    )

    if not isinstance(samples_name, str) or not samples_name.strip():
        raise ValueError(
            f"{path}: bootstrap samples_file must be a filename"
        )

    samples_path = Path(samples_name).expanduser()

    if not samples_path.is_absolute():
        samples_path = path.parent / samples_path

    try:
        with np.load(samples_path, allow_pickle=False) as archive:
            if "slope_samples_si_w_per_w" not in archive:
                raise ValueError(
                    f"{samples_path} is missing "
                    "'slope_samples_si_w_per_w'"
                )

            slope_samples = np.asarray(
                archive["slope_samples_si_w_per_w"],
                dtype=float,
            ).reshape(-1)

    except OSError as error:
        raise ValueError(
            f"Could not read bootstrap file {samples_path}: {error}"
        ) from error

    slope_samples = slope_samples[np.isfinite(slope_samples)]

    if slope_samples.size < MINIMUM_BOOTSTRAP_SAMPLES:
        raise ValueError(
            f"{samples_path} has only {slope_samples.size} valid slope "
            f"samples; at least {MINIMUM_BOOTSTRAP_SAMPLES} are required"
        )

    signature = nested_get(
        result,
        ("measurement_signature",),
        path,
    )

    if not isinstance(signature, dict):
        raise ValueError(
            f"{path}: measurement_signature must be a JSON object"
        )

    missing_signature_fields = [
        field
        for field in SIGNATURE_FIELDS
        if field not in signature
    ]

    if missing_signature_fields:
        raise ValueError(
            f"{path}: measurement_signature is missing "
            + ", ".join(missing_signature_fields)
        )

    return {
        "path": path.resolve(),
        "slope": slope,
        "slope_samples": slope_samples,
        "samples_path": samples_path.resolve(),
        "signature": signature,
        "number_of_runs": result.get("number_of_runs"),
    }


def compare_signatures(
    reference: dict[str, Any],
    squeezed: dict[str, Any],
) -> list[str]:
    """Return descriptions of incompatible measurement settings."""
    mismatches: list[str] = []

    for field in SIGNATURE_FIELDS:
        reference_value = reference[field]
        squeezed_value = squeezed[field]

        if isinstance(reference_value, bool) or isinstance(
            squeezed_value,
            bool,
        ):
            equal = reference_value == squeezed_value

        elif isinstance(reference_value, (int, float)) and isinstance(
            squeezed_value,
            (int, float),
        ):
            reference_number = float(reference_value)
            squeezed_number = float(squeezed_value)

            equal = (
                math.isfinite(reference_number)
                and math.isfinite(squeezed_number)
                and math.isclose(
                    reference_number,
                    squeezed_number,
                    rel_tol=NUMERIC_SIGNATURE_RTOL,
                    abs_tol=NUMERIC_SIGNATURE_ATOL,
                )
            )

        else:
            equal = str(reference_value) == str(squeezed_value)

        if not equal:
            mismatches.append(
                f"{field}: reference={reference_value!r}, "
                f"squeezed={squeezed_value!r}"
            )

    return mismatches


def basic_bootstrap_interval(
    estimate: float,
    bootstrap_estimates: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """Construct a two-sided basic bootstrap confidence interval."""
    values = np.asarray(bootstrap_estimates, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return math.nan, math.nan

    alpha = 1.0 - confidence_level
    deviations = values - estimate

    lower_deviation, upper_deviation = np.percentile(
        deviations,
        [
            100.0 * alpha / 2.0,
            100.0 * (1.0 - alpha / 2.0),
        ],
    )

    return (
        float(estimate - upper_deviation),
        float(estimate - lower_deviation),
    )


def independently_combine_bootstraps(
    reference: dict[str, Any],
    squeezed: dict[str, Any],
    repetitions: int,
    random_seed: int,
) -> dict[str, Any]:
    """Construct independent bootstrap realizations of the slope ratio.

    The stored wild-bootstrap distributions are first expressed as
    errors relative to their corresponding point estimates. Samples are
    then drawn independently from the two empirical error distributions.

    This independent random indexing is important: directly pairing two
    files generated with the same random seed could create artificial
    correlation between nominally independent calibrations.
    """
    rng = np.random.default_rng(random_seed)

    reference_errors = (
        reference["slope_samples"] - reference["slope"]
    )
    squeezed_errors = (
        squeezed["slope_samples"] - squeezed["slope"]
    )

    reference_indices = rng.integers(
        0,
        reference_errors.size,
        size=repetitions,
    )
    squeezed_indices = rng.integers(
        0,
        squeezed_errors.size,
        size=repetitions,
    )

    reference_draws = (
        reference["slope"]
        + reference_errors[reference_indices]
    )
    squeezed_draws = (
        squeezed["slope"]
        + squeezed_errors[squeezed_indices]
    )

    finite_draws = (
        np.isfinite(reference_draws)
        & np.isfinite(squeezed_draws)
    )

    reference_draws = reference_draws[finite_draws]
    squeezed_draws = squeezed_draws[finite_draws]

    denominator_nonpositive = reference_draws <= 0
    numerator_nonpositive = squeezed_draws <= 0

    # Signed ratios are retained when possible. Nonpositive ratios make
    # logarithmic squeezing undefined but are statistically important:
    # silently deleting them would understate instability.
    usable_denominator = reference_draws != 0

    ratio_draws = (
        squeezed_draws[usable_denominator]
        / reference_draws[usable_denominator]
    )
    ratio_draws = ratio_draws[np.isfinite(ratio_draws)]

    if ratio_draws.size < MINIMUM_BOOTSTRAP_SAMPLES:
        raise ValueError(
            "Too few finite ratio bootstrap realizations were produced"
        )

    return {
        "ratio_draws": ratio_draws,
        "finite_joint_draws": int(reference_draws.size),
        "ratio_draws_used": int(ratio_draws.size),
        "reference_nonpositive_fraction": float(
            np.mean(denominator_nonpositive)
        ),
        "squeezed_nonpositive_fraction": float(
            np.mean(numerator_nonpositive)
        ),
    }


def calculate_results(
    reference: dict[str, Any],
    squeezed: dict[str, Any],
    repetitions: int,
    random_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Calculate the ratio and transformed bootstrap intervals."""
    reference_slope = float(reference["slope"])
    squeezed_slope = float(squeezed["slope"])

    ratio = squeezed_slope / reference_slope

    bootstrap = independently_combine_bootstraps(
        reference=reference,
        squeezed=squeezed,
        repetitions=repetitions,
        random_seed=random_seed,
    )

    ratio_interval = basic_bootstrap_interval(
        estimate=ratio,
        bootstrap_estimates=bootstrap["ratio_draws"],
        confidence_level=confidence_level,
    )

    ratio_low, ratio_high = ratio_interval

    signed_change_db = 10.0 * math.log10(ratio)
    squeezing_magnitude_db = -signed_change_db
    reduction_percent = 100.0 * (1.0 - ratio)

    logarithmic_interval_defined = (
        math.isfinite(ratio_low)
        and math.isfinite(ratio_high)
        and ratio_low > 0
        and ratio_high > 0
    )

    if logarithmic_interval_defined:
        signed_db_interval = (
            10.0 * math.log10(ratio_low),
            10.0 * math.log10(ratio_high),
        )

        # Negation reverses interval order.
        squeezing_interval = (
            -signed_db_interval[1],
            -signed_db_interval[0],
        )

        # Reduction decreases as the ratio increases.
        reduction_interval = (
            100.0 * (1.0 - ratio_high),
            100.0 * (1.0 - ratio_low),
        )
    else:
        signed_db_interval = (math.nan, math.nan)
        squeezing_interval = (math.nan, math.nan)
        reduction_interval = (math.nan, math.nan)

    return {
        "reference_slope_si_w_per_w": reference_slope,
        "squeezed_slope_si_w_per_w": squeezed_slope,
        "noise_ratio": ratio,
        "noise_ratio_interval": ratio_interval,
        "signed_noise_change_db": signed_change_db,
        "signed_noise_change_db_interval": signed_db_interval,
        "squeezing_magnitude_db": squeezing_magnitude_db,
        "squeezing_magnitude_db_interval": squeezing_interval,
        "noise_reduction_percent": reduction_percent,
        "noise_reduction_percent_interval": reduction_interval,
        "logarithmic_interval_defined": logarithmic_interval_defined,
        "bootstrap": bootstrap,
    }


def interval_text(
    interval: tuple[float, float],
    decimals: int,
) -> str:
    """Format an interval or report it as undefined."""
    low, high = interval

    if math.isfinite(low) and math.isfinite(high):
        return f"[{low:.{decimals}f}, {high:.{decimals}f}]"

    return "undefined"


def build_output(
    reference: dict[str, Any],
    squeezed: dict[str, Any],
    results: dict[str, Any],
    repetitions: int,
    random_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Build a standards-compliant machine-readable result."""
    bootstrap = results["bootstrap"]

    return {
        "method": (
            "independent combination of fixed-design wild-bootstrap "
            "slope-error distributions"
        ),
        "confidence_interval_method": "basic bootstrap",
        "confidence_level": confidence_level,
        "bootstrap_repetitions_requested": repetitions,
        "random_seed": random_seed,
        "independence_assumption": (
            "The squeezed and reference calibration datasets are "
            "assumed independent."
        ),
        "reference_calibration": {
            "results_file": str(reference["path"]),
            "bootstrap_file": str(reference["samples_path"]),
            "number_of_runs": reference["number_of_runs"],
            "slope_si_w_per_w": finite_or_none(reference["slope"]),
        },
        "squeezed_calibration": {
            "results_file": str(squeezed["path"]),
            "bootstrap_file": str(squeezed["samples_path"]),
            "number_of_runs": squeezed["number_of_runs"],
            "slope_si_w_per_w": finite_or_none(squeezed["slope"]),
        },
        "measurement_signature": reference["signature"],
        "results": {
            "noise_ratio": finite_or_none(results["noise_ratio"]),
            "noise_ratio_interval": [
                finite_or_none(results["noise_ratio_interval"][0]),
                finite_or_none(results["noise_ratio_interval"][1]),
            ],
            "signed_noise_change_db": finite_or_none(
                results["signed_noise_change_db"]
            ),
            "signed_noise_change_db_interval": [
                finite_or_none(
                    results["signed_noise_change_db_interval"][0]
                ),
                finite_or_none(
                    results["signed_noise_change_db_interval"][1]
                ),
            ],
            "squeezing_magnitude_db": finite_or_none(
                results["squeezing_magnitude_db"]
            ),
            "squeezing_magnitude_db_interval": [
                finite_or_none(
                    results["squeezing_magnitude_db_interval"][0]
                ),
                finite_or_none(
                    results["squeezing_magnitude_db_interval"][1]
                ),
            ],
            "noise_reduction_percent": finite_or_none(
                results["noise_reduction_percent"]
            ),
            "noise_reduction_percent_interval": [
                finite_or_none(
                    results["noise_reduction_percent_interval"][0]
                ),
                finite_or_none(
                    results["noise_reduction_percent_interval"][1]
                ),
            ],
            "logarithmic_interval_defined": (
                results["logarithmic_interval_defined"]
            ),
        },
        "bootstrap_diagnostics": {
            "finite_joint_draws": bootstrap["finite_joint_draws"],
            "finite_ratio_draws": bootstrap["ratio_draws_used"],
            "reference_nonpositive_slope_fraction": (
                bootstrap["reference_nonpositive_fraction"]
            ),
            "squeezed_nonpositive_slope_fraction": (
                bootstrap["squeezed_nonpositive_fraction"]
            ),
        },
        "limitations": [
            (
                "This analysis is invalid for paired or correlated "
                "calibrations unless their joint dependence is preserved."
            ),
            (
                "The ratio removes only a constant fitted additive "
                "intercept; it does not remove power-dependent or "
                "condition-dependent background."
            ),
            (
                "Detection-loss correction is not applied."
            ),
            (
                "Upstream uncertainty remains approximate when there "
                "are few independent calibration runs."
            ),
        ],
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate a squeezed/reference slope ratio using upstream "
            "fixed-design wild-bootstrap distributions."
        )
    )

    parser.add_argument(
        "reference",
        type=Path,
        help="Unsqueezed calibration_results.json",
    )
    parser.add_argument(
        "squeezed",
        type=Path,
        help="Squeezed calibration_results.json",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=(
            "Number of independently combined bootstrap draws "
            f"(default: {DEFAULT_REPETITIONS})"
        ),
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help=(
            "Two-sided confidence level "
            f"(default: {DEFAULT_CONFIDENCE_LEVEL})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.repetitions < MINIMUM_BOOTSTRAP_SAMPLES:
        print(
            f"[FATAL] --repetitions must be at least "
            f"{MINIMUM_BOOTSTRAP_SAMPLES}",
            file=sys.stderr,
        )
        sys.exit(2)

    if not 0 < arguments.confidence_level < 1:
        print(
            "[FATAL] --confidence-level must be between 0 and 1",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        reference = load_calibration(
            arguments.reference.expanduser().resolve()
        )
        squeezed = load_calibration(
            arguments.squeezed.expanduser().resolve()
        )

        mismatches = compare_signatures(
            reference["signature"],
            squeezed["signature"],
        )

        if mismatches:
            mismatch_text = "\n  - ".join(mismatches)
            raise ValueError(
                "The calibration measurement conditions do not match:\n"
                f"  - {mismatch_text}"
            )

        results = calculate_results(
            reference=reference,
            squeezed=squeezed,
            repetitions=arguments.repetitions,
            random_seed=arguments.seed,
            confidence_level=arguments.confidence_level,
        )

    except ValueError as error:
        print(f"[FATAL] {error}", file=sys.stderr)
        sys.exit(1)

    confidence_percent = 100.0 * arguments.confidence_level
    bootstrap = results["bootstrap"]

    print()
    print("=" * 72)
    print("SQUEEZED / SHOT-NOISE SLOPE COMPARISON")
    print("=" * 72)
    print(
        f"Reference slope: {results['reference_slope_si_w_per_w']:.8g} W/W"
    )
    print(
        f"Squeezed slope:  {results['squeezed_slope_si_w_per_w']:.8g} W/W"
    )
    print("-" * 72)
    print(
        f"Noise ratio R = k_sq/k_ref: "
        f"{results['noise_ratio']:.8g}"
    )
    print(
        f"Basic-bootstrap {confidence_percent:.1f}% interval: "
        f"{interval_text(results['noise_ratio_interval'], 8)}"
    )
    print()
    print(
        f"Signed noise change 10 log10(R): "
        f"{results['signed_noise_change_db']:.5f} dB"
    )
    print(
        f"{confidence_percent:.1f}% interval: "
        f"{interval_text(results['signed_noise_change_db_interval'], 5)} dB"
    )
    print()
    print(
        f"Squeezing magnitude -10 log10(R): "
        f"{results['squeezing_magnitude_db']:.5f} dB"
    )
    print(
        f"{confidence_percent:.1f}% interval: "
        f"{interval_text(results['squeezing_magnitude_db_interval'], 5)} dB"
    )
    print()
    print(
        f"Noise reduction: "
        f"{results['noise_reduction_percent']:.3f}%"
    )
    print(
        f"{confidence_percent:.1f}% interval: "
        f"{interval_text(results['noise_reduction_percent_interval'], 3)}%"
    )
    print("-" * 72)
    print(
        "Reference bootstrap draws with nonpositive slope: "
        f"{100.0 * bootstrap['reference_nonpositive_fraction']:.3f}%"
    )
    print(
        "Squeezed bootstrap draws with nonpositive slope:  "
        f"{100.0 * bootstrap['squeezed_nonpositive_fraction']:.3f}%"
    )

    if not results["logarithmic_interval_defined"]:
        print(
            "[WARNING] The ratio interval includes a nonpositive value. "
            "A finite two-sided dB interval is therefore not defined."
        )

    if (
        bootstrap["reference_nonpositive_fraction"] > 0.01
        or bootstrap["squeezed_nonpositive_fraction"] > 0.01
    ):
        print(
            "[WARNING] More than 1% of bootstrap slope draws are "
            "nonpositive. The ratio is weakly identified and should not "
            "be summarized as a reliable finite squeezing interval."
        )

    ratio_low, ratio_high = results["noise_ratio_interval"]

    if math.isfinite(ratio_low) and math.isfinite(ratio_high):
        if ratio_high < 1.0:
            print(
                "The two-sided bootstrap interval lies below R = 1."
            )
        elif ratio_low > 1.0:
            print(
                "The two-sided bootstrap interval lies above R = 1 "
                "(increased noise)."
            )
        else:
            print(
                "The two-sided bootstrap interval includes R = 1; "
                "the data do not resolve squeezing from no change at "
                "this confidence level."
            )

    print("-" * 72)
    print(
        "Assumption: the two calibration datasets are independent."
    )
    print(
        "No detection-efficiency or optical-loss correction was applied."
    )
    print("=" * 72)

    output = build_output(
        reference=reference,
        squeezed=squeezed,
        results=results,
        repetitions=arguments.repetitions,
        random_seed=arguments.seed,
        confidence_level=arguments.confidence_level,
    )

    output_path = arguments.output.expanduser().resolve()
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nSaved result: {output_path}")


if __name__ == "__main__":
    main()