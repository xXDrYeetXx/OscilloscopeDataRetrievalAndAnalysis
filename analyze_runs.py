"""
analyze_runs.py
===============

Loads converged noise measurements from multiple acquisition runs and
fits a linear shot-noise calibration curve:

    N(P) = k * P + N_dark

STATISTICAL METHODS
-------------------
1. OLS REGRESSION

   The fit is performed with numpy.linalg.lstsq rather than explicitly
   solving the normal equations. Power is centered and scaled before
   fitting to improve numerical stability. The fitted coefficients and
   covariance matrix are then transformed back into the original display
   units.

   Under the classical fixed-design OLS assumptions,

       Cov(beta_hat) = sigma_hat^2 * (X^T X)^(-1)

   where sigma_hat^2 is the unbiased residual-variance estimate and the
   residual degrees of freedom are n - 2.

   All reported "+/- 1 SE" quantities are standard errors, not automatic
   68% confidence intervals. If the residuals are independent, Gaussian,
   and homoscedastic, then

       (estimate - true value) / SE ~ t_(n-2)

   and the coverage of estimate +/- 1 SE is

       P(-1 <= T_(n-2) <= 1).

   This coverage approaches 68.27% only as the degrees of freedom become
   large. The script reports the finite-sample value, conditional on the
   Gaussian-error model.

2. GOODNESS OF FIT

   Reports R-squared, Pearson r, a two-tailed t-test p-value for the slope
   (H0: k = 0), and the maximum absolute residual.

3. DERIVED QUANTITIES

   Linear quantities use exact covariance propagation because they are
   linear combinations of the fitted parameters:

       Var(k*P + N_dark)
           = P^2 Var(k) + Var(N_dark) + 2 P Cov(k, N_dark)

       Var(k*P) = P^2 Var(k)

   These are standard errors of the fitted mean response. They do not
   include residual/run-to-run scatter and therefore are not prediction
   intervals for individual future measurements.

   dBm quantities are nonlinear transformations. The script transforms
   the endpoints y - SE and y + SE separately instead of reporting a
   symmetric first-order dB error bar. This produces an asymmetric
   transformed interval. If y - SE <= 0, the lower dBm endpoint is
   undefined.

ASSUMPTIONS
-----------
- One mean noise value is used per run.
- Beam powers are treated as fixed and measured without relevant error.
- For the usual OLS covariance estimate, residuals should be independent
  and homoscedastic.
- Student-t p-values and the stated finite-sample coverage are exact only
  under normally distributed residuals.
- With few runs, these assumptions cannot be tested reliably. Inspect the
  residual plot and use scientific judgment.
- Absolute dBm values depend on the measurement bandwidth used during
  acquisition.
- The calibration-slope ratio used elsewhere is bandwidth-independent only
  when both datasets use identical acquisition settings.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


def get_auto_unit(max_value_mw: float) -> tuple[float, str]:
    """Return the multiplier needed to display a value supplied in mW.

    Examples
    --------
    1 mW      -> multiplier 1, unit "mW"
    0.001 mW  -> multiplier 1e3, unit "μW"
    1e-6 mW   -> multiplier 1e6, unit "nW"
    """
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
    """Parse expressions such as '1,3-5,7-20' into unique run numbers."""
    runs: set[int] = set()

    for part in input_str.strip().split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            match = re.fullmatch(r"(\d+)-(\d+)", part)

            if not match:
                raise ValueError(f"Invalid range format: '{part}'")

            start = int(match.group(1))
            end = int(match.group(2))

            if start > end:
                raise ValueError(f"Range start is greater than range end: '{part}'")

            runs.update(range(start, end + 1))
        else:
            if not part.isdigit():
                raise ValueError(f"Invalid run number: '{part}'")

            runs.add(int(part))

    return sorted(runs)


def extract_data_from_runs(
    run_numbers: list[int],
    base_folder: str = "v3_converged_noise_data",
) -> pd.DataFrame:
    """Load beam power and converged mean noise power from each selected run."""
    base_dir = Path(base_folder).expanduser().resolve()
    records: list[dict] = []

    print(f"\n--- SCANNING DIRECTORY: {base_dir} ---")

    for run_num in run_numbers:
        json_path = base_dir / f"Run {run_num}" / "converged_result.json"

        if not json_path.is_file():
            print(
                f"[WARNING] Skipping Run {run_num}: could not find '{json_path}'",
                file=sys.stderr,
            )
            continue

        try:
            with json_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            power_mw = float(data["beam_power_mw"])
            noise_watts = float(data["final_mean_watts"])

            if not np.isfinite(power_mw):
                raise ValueError("beam_power_mw is not finite")

            if not np.isfinite(noise_watts):
                raise ValueError("final_mean_watts is not finite")

            if power_mw < 0:
                raise ValueError("beam_power_mw cannot be negative")

            if noise_watts <= 0:
                raise ValueError("final_mean_watts must be strictly positive")

            records.append(
                {
                    "Run": run_num,
                    "Power_mW": power_mw,
                    "Noise_Watts": noise_watts,
                }
            )

            print(
                f"  • Run {run_num:2d}: "
                f"Power = {power_mw:.6g} mW | "
                f"Noise = {noise_watts:.6e} W"
            )

        except KeyError as exc:
            print(
                f"[ERROR] Skipping Run {run_num}: missing JSON key {exc}",
                file=sys.stderr,
            )
        except (TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
            print(
                f"[ERROR] Skipping Run {run_num}: {exc}",
                file=sys.stderr,
            )

    if not records:
        print(
            "\n[FATAL] No valid run data could be extracted.",
            file=sys.stderr,
        )
        sys.exit(1)

    return (
        pd.DataFrame(records)
        .sort_values(by="Power_mW")
        .reset_index(drop=True)
    )


def fit_ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = k*x + N_dark using centered/scaled least squares.

    Centering and scaling x avoids the unnecessary numerical instability
    associated with explicitly inverting the unscaled normal-equation
    matrix. The covariance matrix is transformed back into the original
    x and y display units.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays")

    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of values")

    n_pts = len(x)

    if n_pts < 3:
        print(
            "\n[FATAL] At least 3 valid data points are required to estimate "
            "a slope, intercept, and residual variance.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        print(
            "\n[FATAL] Fit data contain NaN or infinite values.",
            file=sys.stderr,
        )
        sys.exit(1)

    x_mean = float(np.mean(x))
    x_scale = float(np.std(x, ddof=0))

    if not np.isfinite(x_scale) or x_scale == 0.0:
        print(
            "\n[FATAL] All beam powers are identical. The slope cannot be "
            "estimated; provide runs spanning multiple power levels.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fit in a centered and scaled coordinate:
    #
    #     z = (x - x_mean) / x_scale
    #     y = gamma_slope*z + gamma_center
    #
    # Transform back afterward:
    #
    #     k      = gamma_slope / x_scale
    #     N_dark = gamma_center - k*x_mean
    z = (x - x_mean) / x_scale
    design_scaled = np.column_stack((z, np.ones(n_pts)))

    gamma, _, rank, singular_values = np.linalg.lstsq(
        design_scaled,
        y,
        rcond=None,
    )

    if rank < 2:
        print(
            "\n[FATAL] The regression design matrix is rank deficient. "
            "The slope and intercept cannot both be estimated.",
            file=sys.stderr,
        )
        sys.exit(1)

    gamma_slope, gamma_center = gamma

    transform = np.array(
        [
            [1.0 / x_scale, 0.0],
            [-x_mean / x_scale, 1.0],
        ]
    )

    beta = transform @ gamma
    k, n_dark = beta

    design_original = np.column_stack((x, np.ones(n_pts)))
    y_pred = design_original @ beta
    residuals = y - y_pred

    dof = n_pts - 2
    sse = float(residuals @ residuals)
    sigma2 = sse / dof
    residual_std = float(np.sqrt(max(sigma2, 0.0)))

    try:
        scaled_information_inv = np.linalg.inv(
            design_scaled.T @ design_scaled
        )
    except np.linalg.LinAlgError:
        print(
            "\n[FATAL] Could not construct the parameter covariance matrix.",
            file=sys.stderr,
        )
        sys.exit(1)

    cov_gamma = sigma2 * scaled_information_inv
    cov_beta = transform @ cov_gamma @ transform.T

    # Remove tiny negative diagonal values caused solely by roundoff.
    slope_variance = max(float(cov_beta[0, 0]), 0.0)
    intercept_variance = max(float(cov_beta[1, 1]), 0.0)

    slope_std_err = float(np.sqrt(slope_variance))
    intercept_std_err = float(np.sqrt(intercept_variance))
    cov_k_ndark = float(cov_beta[0, 1])

    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))

    if x_std > 0 and y_std > 0:
        r_value = float(np.corrcoef(x, y)[0, 1])
    else:
        r_value = float("nan")

    y_centered = y - np.mean(y)
    sst = float(y_centered @ y_centered)

    if sst > 0:
        r_squared = 1.0 - sse / sst
    elif np.isclose(sse, 0.0):
        # A constant response fitted exactly has no meaningful conventional
        # R-squared because the total sum of squares is zero.
        r_squared = float("nan")
    else:
        r_squared = float("nan")

    if slope_std_err > 0:
        t_stat = float(k / slope_std_err)
        p_value = float(2.0 * t_dist.sf(abs(t_stat), dof))
    elif k == 0:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = float(np.copysign(np.inf, k))
        p_value = 0.0

    # This coverage statement is exact only under the fixed-design,
    # independent Gaussian-error OLS model.
    one_se_coverage = float(
        t_dist.cdf(1.0, dof) - t_dist.cdf(-1.0, dof)
    )

    scaled_condition_number = float(
        singular_values[0] / singular_values[-1]
    )

    return {
        "k": float(k),
        "slope_std_err": slope_std_err,
        "n_dark": float(n_dark),
        "intercept_std_err": intercept_std_err,
        "cov_k_ndark": cov_k_ndark,
        "cov_beta": cov_beta,
        "r_value": r_value,
        "r_squared": float(r_squared),
        "t_stat": t_stat,
        "p_value": p_value,
        "y_pred": y_pred,
        "residuals": residuals,
        "dof": dof,
        "sigma2": sigma2,
        "residual_std": residual_std,
        "one_se_coverage": one_se_coverage,
        "scaled_condition_number": scaled_condition_number,
    }


def propagate_total_noise_std(
    power: float,
    slope_std_err: float,
    intercept_std_err: float,
    cov_k_ndark: float,
) -> float:
    """Return the exact SE of the fitted mean k*P + N_dark."""
    variance = (
        power**2 * slope_std_err**2
        + intercept_std_err**2
        + 2.0 * power * cov_k_ndark
    )

    # The expression is a covariance quadratic form and is nonnegative
    # mathematically. Clamp tiny negative values caused by floating point.
    return float(np.sqrt(max(variance, 0.0)))


def propagate_shot_noise_std(
    power: float,
    slope_std_err: float,
) -> float:
    """Return the exact SE of k*P for fixed power P."""
    return float(abs(power) * slope_std_err)


def linear_interval_to_dbm(
    y_point: float,
    y_std: float,
    unit_db_correction: float,
) -> dict:
    """Transform a linear point estimate and +/-1-SE endpoints into dBm.

    The input y_point is expressed in the selected display unit. The
    correction converts that display unit into dBm.

    This is a transformed linear-domain +/-1-SE interval, not a symmetric
    standard error expressed in dB.
    """
    if not np.isfinite(y_point) or not np.isfinite(y_std):
        return {
            "dbm_point": float("nan"),
            "dbm_low": float("nan"),
            "dbm_high": float("nan"),
            "low_undefined": True,
            "point_undefined": True,
            "high_undefined": True,
            "relative_se": float("nan"),
        }

    y_std = abs(y_std)
    y_low = y_point - y_std
    y_high = y_point + y_std

    point_undefined = y_point <= 0
    low_undefined = y_low <= 0
    high_undefined = y_high <= 0

    dbm_point = (
        10.0 * np.log10(y_point) + unit_db_correction
        if not point_undefined
        else float("nan")
    )
    dbm_low = (
        10.0 * np.log10(y_low) + unit_db_correction
        if not low_undefined
        else float("nan")
    )
    dbm_high = (
        10.0 * np.log10(y_high) + unit_db_correction
        if not high_undefined
        else float("nan")
    )

    relative_se = (
        abs(y_std / y_point)
        if y_point != 0
        else float("inf")
    )

    return {
        "dbm_point": float(dbm_point),
        "dbm_low": float(dbm_low),
        "dbm_high": float(dbm_high),
        "low_undefined": low_undefined,
        "point_undefined": point_undefined,
        "high_undefined": high_undefined,
        "relative_se": float(relative_se),
    }


def print_dbm_result(
    quantity_name: str,
    interval: dict,
) -> None:
    """Print a transformed linear-domain +/-1-SE interval in dBm."""
    if interval["point_undefined"]:
        print(
            f"  --> {quantity_name}: dBm value UNDEFINED because the "
            "fitted linear value is not strictly positive.\n"
        )
        return

    point = interval["dbm_point"]
    relative_se = interval["relative_se"]

    if interval["low_undefined"]:
        if interval["high_undefined"]:
            print(
                f"  --> {quantity_name} = {point:.5f} dBm; both transformed "
                "interval endpoints are undefined.\n"
            )
        else:
            print(
                f"  --> {quantity_name} = {point:.5f} dBm\n"
                f"      transformed upper (+1 SE) endpoint: "
                f"{interval['dbm_high']:.5f} dBm\n"
                "      transformed lower (-1 SE) endpoint: UNDEFINED "
                "because y - SE <= 0\n"
            )
    else:
        print(
            f"  --> {quantity_name} = {point:.5f} dBm\n"
            f"      transformed linear-domain ±1-SE interval: "
            f"[{interval['dbm_low']:.5f}, "
            f"{interval['dbm_high']:.5f}] dBm\n"
            f"      relative SE of underlying linear value: "
            f"{relative_se * 100:.1f}%\n"
        )

    if np.isfinite(relative_se) and relative_se > 0.5:
        print(
            "  [CAUTION] The relative linear-domain SE exceeds 50%; "
            "the corresponding dBm result is poorly determined.\n"
        )


def interactive_calculator(
    k: float,
    n_dark_display: float,
    power_unit: str,
    noise_unit: str,
    slope_std_err: float,
    intercept_std_err: float,
    cov_k_ndark: float,
    unit_db_correction: float,
    dof: int,
    one_se_coverage: float,
) -> None:
    """Interactively evaluate fitted total-noise and shot-noise equations."""
    dbm_available = k > 0 and np.isfinite(k)

    equation_names = {
        "1": f"Total Noise ({noise_unit})",
        "2": "Total Noise (dBm)",
        "3": f"Shot Noise ({noise_unit})",
        "4": "Shot Noise (dBm)",
    }

    while True:
        print("\n----------------------------------------------------------")
        print("                INTERACTIVE NOISE CALCULATOR")
        print("----------------------------------------------------------")
        print("Linear uncertainty values are standard errors of the")
        print("fitted mean response, not prediction intervals.")
        print(
            f"Under Gaussian OLS errors, dof = {dof} gives "
            f"{one_se_coverage * 100:.1f}% coverage for ±1 SE."
        )
        print("Without Gaussian residuals, that exact coverage claim")
        print("does not apply.")
        print("----------------------------------------------------------")
        print("Select an equation:")
        print("  [1] Total Noise (linear)")
        print(
            "  [2] Total Noise (dBm)"
            + ("" if dbm_available else " — unavailable because k <= 0")
        )
        print("  [3] Shot Noise (linear)")
        print(
            "  [4] Shot Noise (dBm)"
            + ("" if dbm_available else " — unavailable because k <= 0")
        )
        print("  [q] Quit calculator")

        choice = input("\nEnter choice (1-4 or q): ").strip().lower()

        if choice == "q":
            print("Exiting calculator loop...\n")
            return

        if choice not in equation_names:
            print("Invalid selection. Choose 1, 2, 3, 4, or q.")
            continue

        if choice in {"2", "4"} and not dbm_available:
            print(
                "That equation is unavailable because the fitted slope is "
                "not strictly positive."
            )
            continue

        print(f"\n>>> Selected: {equation_names[choice]}")
        print("    Type 'b' to return or 'q' to quit.")

        while True:
            power_text = input(
                f"Enter optical beam power P (in {power_unit}): "
            ).strip().lower()

            if power_text == "b":
                break

            if power_text == "q":
                print("Exiting calculator loop...\n")
                return

            try:
                power = float(power_text)
            except ValueError:
                print("Enter a finite number, 'b', or 'q'.")
                continue

            if not np.isfinite(power):
                print("Power must be finite.")
                continue

            if power < 0:
                print("Power cannot be negative.")
                continue

            if choice == "1":
                result = k * power + n_dark_display
                result_std = propagate_total_noise_std(
                    power,
                    slope_std_err,
                    intercept_std_err,
                    cov_k_ndark,
                )

                print(
                    f"  --> N_total = {result:.6f} ± "
                    f"{result_std:.6f} {noise_unit}\n"
                    "      exact SE propagation for the fitted mean; "
                    "not a prediction interval\n"
                )

            elif choice == "2":
                linear_point = k * power + n_dark_display

                if linear_point <= 0:
                    print(
                        "The fitted total-noise value is not strictly "
                        "positive, so its dBm value is undefined."
                    )
                    continue

                linear_std = propagate_total_noise_std(
                    power,
                    slope_std_err,
                    intercept_std_err,
                    cov_k_ndark,
                )

                interval = linear_interval_to_dbm(
                    linear_point,
                    linear_std,
                    unit_db_correction,
                )

                print_dbm_result("N_total", interval)

            elif choice == "3":
                result = k * power
                result_std = propagate_shot_noise_std(
                    power,
                    slope_std_err,
                )

                print(
                    f"  --> N_shot = {result:.6f} ± "
                    f"{result_std:.6f} {noise_unit}\n"
                    "      exact SE propagation for fixed P; "
                    "not a prediction interval\n"
                )

            elif choice == "4":
                if power <= 0:
                    print(
                        "Power must be strictly positive for shot noise "
                        "expressed in dBm."
                    )
                    continue

                linear_point = k * power

                if linear_point <= 0:
                    print(
                        "The fitted shot-noise value is not strictly "
                        "positive, so its dBm value is undefined."
                    )
                    continue

                linear_std = propagate_shot_noise_std(
                    power,
                    slope_std_err,
                )

                interval = linear_interval_to_dbm(
                    linear_point,
                    linear_std,
                    unit_db_correction,
                )

                print_dbm_result("N_shot", interval)


def format_statistic(value: float, decimals: int = 6) -> str:
    """Format a statistic while handling undefined values."""
    if not np.isfinite(value):
        return "undefined"
    return f"{value:.{decimals}f}"


def main() -> None:
    print("==========================================================")
    print("   QUANTUM SHOT NOISE AUTOMATED RUN CALIBRATOR (AUTO-UNIT)")
    print("==========================================================")

    while True:
        try:
            user_input = input(
                "Enter run numbers to process "
                "(e.g. '1-6' or '1,3-5,7'): "
            )
            run_list = parse_run_input(user_input)

            if not run_list:
                raise ValueError("No run numbers were provided")

            break
        except ValueError as exc:
            print(f"Input Error: {exc}. Please try again.\n")

    df = extract_data_from_runs(run_list)
    n_runs = len(df)

    power_scale, power_unit = get_auto_unit(
        df["Power_mW"].abs().max()
    )

    # Convert W to mW before selecting a display prefix.
    max_noise_mw = df["Noise_Watts"].abs().max() * 1e3
    noise_scale, noise_unit = get_auto_unit(max_noise_mw)

    df["Power_Display"] = df["Power_mW"] * power_scale
    df["Noise_Display"] = (
        df["Noise_Watts"] * 1e3 * noise_scale
    )

    fit = fit_ols(
        df["Power_Display"].to_numpy(),
        df["Noise_Display"].to_numpy(),
    )

    k = fit["k"]
    slope_std_err = fit["slope_std_err"]
    n_dark_display = fit["n_dark"]
    intercept_std_err = fit["intercept_std_err"]
    cov_k_ndark = fit["cov_k_ndark"]
    r_value = fit["r_value"]
    r_squared = fit["r_squared"]
    t_stat = fit["t_stat"]
    p_value = fit["p_value"]
    dof = fit["dof"]
    residual_std = fit["residual_std"]
    one_se_coverage = fit["one_se_coverage"]

    df["Predicted_Display"] = fit["y_pred"]
    df["Residual_Display"] = fit["residuals"]

    max_residual = float(
        np.max(np.abs(df["Residual_Display"].to_numpy()))
    )

    # A displayed noise value y corresponds to:
    #
    #     y / noise_scale mW
    #
    # Therefore:
    #
    #     dBm = 10*log10(y / noise_scale)
    #          = 10*log10(y) - 10*log10(noise_scale)
    unit_db_correction = -10.0 * np.log10(noise_scale)

    n_dark_watts = (
        n_dark_display / noise_scale
    ) * 1e-3

    print("\n==========================================================")
    print("             PHOTODETECTOR FIT STATISTICS")
    print("==========================================================")
    print(
        f" Model: Noise({noise_unit}) = k * Power({power_unit}) "
        f"+ N_dark({noise_unit})"
    )
    print(f" Valid runs:                 {n_runs}")
    print(f" Residual degrees of freedom:{dof:>8d}")
    print("----------------------------------------------------------")
    print(" Uncertainty convention: ±1 standard error (SE)")
    print(
        f" Gaussian-model coverage:    "
        f"{one_se_coverage * 100:.1f}% for ±1 SE"
    )
    print(" Coverage caveat:            exact only for independent,")
    print("                             Gaussian, homoscedastic errors")
    print("----------------------------------------------------------")
    print(
        f" Responsiveness slope k:     {k:.6g} ± "
        f"{slope_std_err:.6g} {noise_unit}/{power_unit}"
    )
    print(
        f" Dark noise N_dark:          {n_dark_display:.6g} ± "
        f"{intercept_std_err:.6g} {noise_unit}"
    )

    if n_dark_watts > 0 and np.isfinite(n_dark_watts):
        dark_dbm = 10.0 * np.log10(n_dark_watts / 1e-3)
        print(f" Predicted dark floor:       {dark_dbm:.5f} dBm")
    else:
        print(" Predicted dark floor:       undefined in dBm (<= 0)")

    print("----------------------------------------------------------")
    print(
        f" R²:                         "
        f"{format_statistic(r_squared)}"
    )
    print(
        f" Pearson r:                  "
        f"{format_statistic(r_value)}"
    )
    print(
        f" Slope t-statistic:          "
        f"{format_statistic(t_stat)}"
    )
    print(f" Two-tailed slope p-value:   {p_value:.6e}")
    print(
        f" Residual standard deviation:{residual_std:>12.6g} "
        f"{noise_unit}"
    )
    print(
        f" Maximum absolute residual:  {max_residual:.6g} "
        f"{noise_unit}"
    )
    print("==========================================================\n")

    dbm_formulas_available = k > 0 and np.isfinite(k)

    print("==========================================================")
    print("                 DERIVED NOISE FORMULAS")
    print(f"              P is expressed in {power_unit}")
    print("==========================================================")
    print(f" (1) Total Noise ({noise_unit}):")
    print(
        f"     N_total(P) = ({k:.6g} ± {slope_std_err:.6g}) P "
        f"+ ({n_dark_display:.6g} ± {intercept_std_err:.6g})"
    )
    print("     Parameter estimates are correlated; use the full")
    print("     covariance expression when evaluating uncertainty.\n")

    print(f" (2) Shot Noise ({noise_unit}):")
    print(
        f"     N_shot(P) = ({k:.6g} ± "
        f"{slope_std_err:.6g}) P\n"
    )

    if dbm_formulas_available:
        a_term = (
            10.0 * np.log10(k)
            + unit_db_correction
        )
        c_term = n_dark_display / k
        log2_coefficient = 10.0 * np.log10(2.0)

        print(" (3) Total Noise (dBm):")

        if c_term >= 0:
            print(
                f"     N_total,dBm(P) = {a_term:.6f} "
                f"+ {log2_coefficient:.6f} log₂"
                f"(P + {c_term:.6g})"
            )
        else:
            print(
                f"     N_total,dBm(P) = {a_term:.6f} "
                f"+ {log2_coefficient:.6f} log₂"
                f"(P - {abs(c_term):.6g})"
            )

        print("     Valid only where kP + N_dark > 0.\n")

        print(" (4) Shot Noise (dBm):")
        print(
            f"     N_shot,dBm(P) = {a_term:.6f} "
            f"+ {log2_coefficient:.6f} log₂(P)"
        )
        print("     Valid only for P > 0.\n")
    else:
        print(" (3) Total Noise (dBm): unavailable because k <= 0.")
        print(" (4) Shot Noise (dBm): unavailable because k <= 0.\n")
        print(
            "[WARNING] The nonpositive fitted slope is inconsistent with "
            "the expected positive shot-noise response. Inspect the data, "
            "units, residuals, and selected runs.",
            file=sys.stderr,
        )

    print("==========================================================\n")

    max_power = float(df["Power_Display"].max())
    model_right = max_power * 1.1 if max_power > 0 else 1.0
    power_model = np.linspace(0.0, model_right, 200)

    total_noise_model = k * power_model + n_dark_display
    shot_noise_model = k * power_model

    fig, (ax_main, ax_residual) = plt.subplots(
        2,
        1,
        figsize=(8.5, 7.5),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3, 1],
            "hspace": 0.08,
        },
    )

    r_squared_label = (
        f"{r_squared:.4f}"
        if np.isfinite(r_squared)
        else "undefined"
    )

    ax_main.plot(
        power_model,
        total_noise_model,
        color="#1f77b4",
        linewidth=2,
        label=(
            f"Total fit: N = {k:.3g}P + "
            f"{n_dark_display:.3g} {noise_unit} "
            f"(R²={r_squared_label})"
        ),
    )

    ax_main.plot(
        power_model,
        shot_noise_model,
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        label=f"Shot-noise component: N = {k:.3g}P",
    )

    ax_main.scatter(
        df["Power_Display"],
        df["Noise_Display"],
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.6,
        s=55,
        zorder=4,
        label="Measured run means",
    )

    ax_main.scatter(
        [0.0],
        [n_dark_display],
        color="#d62728",
        marker="D",
        s=65,
        zorder=5,
        label=(
            f"Fitted dark intercept: "
            f"{n_dark_display:.3g} {noise_unit}"
        ),
    )

    ax_main.scatter(
        [0.0],
        [0.0],
        color="#2ca02c",
        marker="o",
        s=65,
        zorder=5,
        label="Shot-noise origin",
    )

    observed_max = float(df["Noise_Display"].max())
    annotation_scale = max(
        observed_max,
        abs(n_dark_display),
        float(np.max(np.abs(total_noise_model))),
        1.0,
    )

    ax_main.annotate(
        f"(0, {n_dark_display:.3g}) {noise_unit}",
        xy=(0.0, n_dark_display),
        xytext=(
            model_right * 0.06,
            n_dark_display + 0.10 * annotation_scale,
        ),
        arrowprops={
            "arrowstyle": "->",
            "color": "#d62728",
            "lw": 1.2,
        },
        fontsize=9,
        fontweight="bold",
        color="#d62728",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "#d62728",
            "alpha": 0.85,
        },
    )

    ax_main.annotate(
        f"(0, 0) {noise_unit}",
        xy=(0.0, 0.0),
        xytext=(
            model_right * 0.08,
            0.05 * annotation_scale,
        ),
        arrowprops={
            "arrowstyle": "->",
            "color": "#2ca02c",
            "lw": 1.2,
        },
        fontsize=9,
        fontweight="bold",
        color="#2ca02c",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "#2ca02c",
            "alpha": 0.85,
        },
    )

    plotted_values = np.concatenate(
        (
            df["Noise_Display"].to_numpy(),
            total_noise_model,
            shot_noise_model,
            np.array([0.0, n_dark_display]),
        )
    )

    finite_plotted_values = plotted_values[
        np.isfinite(plotted_values)
    ]

    y_min = float(np.min(finite_plotted_values))
    y_max = float(np.max(finite_plotted_values))
    y_span = y_max - y_min

    if y_span == 0:
        y_span = max(abs(y_max), 1.0)

    ax_main.set_xlim(0.0, model_right)
    ax_main.set_ylim(
        y_min - 0.08 * y_span,
        y_max + 0.12 * y_span,
    )
    ax_main.margins(x=0)
    ax_main.set_ylabel(
        f"Linear Noise Power ({noise_unit})",
        fontsize=11,
    )
    ax_main.set_title(
        "Photodetector Noise Calibration "
        f"({power_unit} / {noise_unit})",
        fontsize=12,
        fontweight="bold",
    )
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=8.5,
    )

    ax_residual.axhline(
        0.0,
        color="gray",
        linestyle="--",
        linewidth=1,
    )
    ax_residual.scatter(
        df["Power_Display"],
        df["Residual_Display"],
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
        s=42,
        zorder=3,
    )
    ax_residual.set_xlabel(
        f"Optical Beam Power ({power_unit})",
        fontsize=11,
    )
    ax_residual.set_ylabel(
        f"Residuals\n({noise_unit})",
        fontsize=10,
    )
    ax_residual.grid(True, linestyle=":", alpha=0.6)
    ax_residual.margins(x=0)

    residual_bound = (
        max_residual * 1.5
        if max_residual > 0
        else max(residual_std, 0.1)
    )
    ax_residual.set_ylim(-residual_bound, residual_bound)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    interactive_calculator(
        k=k,
        n_dark_display=n_dark_display,
        power_unit=power_unit,
        noise_unit=noise_unit,
        slope_std_err=slope_std_err,
        intercept_std_err=intercept_std_err,
        cov_k_ndark=cov_k_ndark,
        unit_db_correction=unit_db_correction,
        dof=dof,
        one_se_coverage=one_se_coverage,
    )

    if plt.fignum_exists(fig.number):
        print("Close the plot window to exit the script completely.")
        plt.show()


if __name__ == "__main__":
    main()