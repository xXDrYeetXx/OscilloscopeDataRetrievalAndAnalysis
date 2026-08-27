import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

# Set crisp figure rendering
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


def get_auto_unit(max_value_watts_or_mw: float) -> tuple[float, str]:
    """Returns a scaling multiplier and unit symbol based on the maximum magnitude."""
    val = abs(max_value_watts_or_mw)
    if val >= 1.0:
        return 1.0, "mW"
    elif val >= 1e-3:
        return 1e3, "μW"
    elif val >= 1e-6:
        return 1e6, "nW"
    elif val >= 1e-9:
        return 1e9, "pW"
    else:
        return 1e-3, "W"


def parse_run_input(input_str: str) -> list[int]:
    """Parses expressions like '1,3-5,7-20' into a sorted list of unique run numbers."""
    runs = set()
    parts = input_str.strip().split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            match = re.match(r"^(\d+)-(\d+)$", part)
            if not match:
                raise ValueError(f"Invalid range format: '{part}'")
            start, end = int(match.group(1)), int(match.group(2))
            if start > end:
                raise ValueError(f"Range start > end: '{part}'")
            runs.update(range(start, end + 1))
        else:
            if not part.isdigit():
                raise ValueError(f"Invalid run number: '{part}'")
            runs.add(int(part))

    return sorted(list(runs))


def extract_data_from_runs(
    run_numbers: list[int], base_folder: str = "."
) -> pd.DataFrame:
    """Iterates through specified Run directories and extracts beam power (mW)
    and raw linear noise power (Watts).
    """
    base_dir = Path(base_folder).expanduser().resolve()
    records = []

    print(f"\n--- SCANNING DIRECTORY: {base_dir} ---")

    for run_num in run_numbers:
        run_dir = base_dir / f"Run {run_num}"
        json_path = run_dir / "converged_result.json"

        if not json_path.is_file():
            print(
                f"[WARNING] Skipping Run {run_num}: Could not find '{json_path}'",
                file=sys.stderr,
            )
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            power_mw = float(data["beam_power_mw"])
            noise_watts = float(data["final_mean_watts"])

            records.append(
                {
                    "Run": run_num,
                    "Power_mW": power_mw,
                    "Noise_Watts": noise_watts,
                }
            )
            print(
                f"  • Run {run_num:2d}: Loaded Power = {power_mw:.3f} mW | Noise = {noise_watts:.3e} W"
            )

        except KeyError as ke:
            print(
                f"[ERROR] Run {run_num}: Missing key in JSON ({ke})",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"[ERROR] Run {run_num}: Failed to parse JSON ({exc})",
                file=sys.stderr,
            )

    if not records:
        print(
            "\n[FATAL] No valid run data could be extracted. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.DataFrame(records).sort_values(by="Power_mW").reset_index(drop=True)
    return df


def fit_ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Ordinary least squares fit of y = k*x + N_dark, with a proper covariance-matrix
    derivation of the standard errors on BOTH the slope and the intercept.
    """
    n_pts = len(x)
    if n_pts < 3:
        print(
            "\n[FATAL] Need at least 3 data points for a fit with uncertainty estimates.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Design matrix: column 0 -> slope term, column 1 -> intercept term
    X = np.column_stack([x, np.ones(n_pts)])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    k, n_dark = beta

    y_pred = X @ beta
    residuals = y - y_pred

    dof = n_pts - 2
    sigma2 = np.sum(residuals**2) / dof  # residual variance (unbiased)
    cov_beta = sigma2 * XtX_inv  # parameter covariance matrix

    slope_std_err = np.sqrt(cov_beta[0, 0])
    intercept_std_err = np.sqrt(cov_beta[1, 1])

    # Correlation / R^2 describe how well the line explains the data
    if np.std(x) > 0:
        r_value = np.corrcoef(x, y)[0, 1]
    else:
        r_value = 0.0
    r_squared = r_value**2

    # Two-tailed p-value for the slope (H0: k == 0)
    t_stat = k / slope_std_err if slope_std_err > 0 else np.inf
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), dof))

    return {
        "k": k,
        "slope_std_err": slope_std_err,
        "n_dark": n_dark,
        "intercept_std_err": intercept_std_err,
        "r_value": r_value,
        "r_squared": r_squared,
        "p_value": p_value,
        "y_pred": y_pred,
        "residuals": residuals,
    }


def interactive_calculator(
    k: float, N_dark_disp: float, a_term: float, c_term: float, p_unit: str, n_unit: str
):
    """Provides an interactive CLI loop allowing users to evaluate any of the 4 derived equations."""
    log2_coeff = 10.0 * np.log10(2)

    eq_names = {
        "1": f"Total Noise ({n_unit})",
        "2": "Total Noise (dBm)",
        "3": f"Shot Noise ({n_unit})",
        "4": "Shot Noise (dBm)",
    }

    while True:
        print("\n----------------------------------------------------------")
        print("                INTERACTIVE NOISE CALCULATOR              ")
        print("----------------------------------------------------------")
        print("Select an equation to evaluate:")
        print("  [1] Total Noise (linear)")
        print("  [2] Total Noise (dBm)")
        print("  [3] Shot Noise (linear)")
        print("  [4] Shot Noise (dBm)")
        print("  [q] Quit Calculator")

        choice = input("\nEnter choice (1-4 or q): ").strip().lower()

        if choice == "q":
            print("Exiting calculator loop...\n")
            break
        elif choice not in eq_names:
            print("Invalid selection. Please choose 1, 2, 3, 4, or q.")
            continue

        selected_eq = eq_names[choice]
        print(f"\n>>> Selected: ({choice}) {selected_eq}")
        print("    (Type 'b' to go back to equation list, 'q' to quit)")

        # Secondary loop: query power for the selected equation
        while True:
            p_str = input(f"Enter optical beam power P (in {p_unit}): ").strip().lower()

            if p_str == "b":
                break
            if p_str == "q":
                print("Exiting calculator loop...\n")
                return

            try:
                P = float(p_str)
                if P < 0:
                    print("Error: Power cannot be negative.")
                    continue

                if choice == "1":
                    res = k * P + N_dark_disp
                    print(f"  --> N_total = {res:.6f} {n_unit}\n")

                elif choice == "2":
                    val_inside = P + c_term
                    if val_inside <= 0:
                        print("Error: (P + C) must be strictly positive for logarithmic calculation.")
                        continue
                    res = a_term + log2_coeff * np.log2(val_inside)
                    print(f"  --> N_total = {res:.5f} dBm\n")

                elif choice == "3":
                    res = k * P
                    print(f"  --> N_shot = {res:.6f} {n_unit}\n")

                elif choice == "4":
                    if P <= 0:
                        print("Error: Power P must be strictly positive (> 0) for shot noise dBm.")
                        continue
                    res = a_term + log2_coeff * np.log2(P)
                    print(f"  --> N_shot = {res:.5f} dBm\n")

            except ValueError:
                print("Invalid numerical value. Enter a valid float, 'b', or 'q'.")


def main():
    # ------------------------------------------------------------------
    # 1. User Input Handling
    # ------------------------------------------------------------------
    print("==========================================================")
    print("   QUANTUM SHOT NOISE AUTOMATED RUN CALIBRATOR (AUTO-UNIT)")
    print("==========================================================")

    while True:
        try:
            prompt = (
                "Enter run numbers to process (e.g. '1-6' or '1,3-5,7'): "
            )
            user_input = input(prompt)
            run_list = parse_run_input(user_input)
            if not run_list:
                raise ValueError("No valid run numbers provided.")
            break
        except ValueError as err:
            print(f"Input Error: {err}. Please try again.\n")

    # ------------------------------------------------------------------
    # 2. Load Raw Data
    # ------------------------------------------------------------------
    df = extract_data_from_runs(run_list)
    n = len(df)

    # ------------------------------------------------------------------
    # 3. Determine Dynamic Prefixes
    # ------------------------------------------------------------------
    p_scale, p_unit = get_auto_unit(df["Power_mW"].max())
    n_scale, n_unit = get_auto_unit(
        df["Noise_Watts"].max() * 1e3
    )  # scale check relative to mW

    df["Power_Display"] = df["Power_mW"] * p_scale
    df["Noise_Display"] = df["Noise_Watts"] * 1e3 * n_scale  # Convert W -> scaled display unit

    # ------------------------------------------------------------------
    # 4. Linear Fit (OLS via normal equations, with a full covariance matrix)
    # ------------------------------------------------------------------
    fit = fit_ols(df["Power_Display"].to_numpy(), df["Noise_Display"].to_numpy())

    k = fit["k"]
    std_err = fit["slope_std_err"]
    N_dark_disp = fit["n_dark"]
    intercept_std_err = fit["intercept_std_err"]
    r_value = fit["r_value"]
    r_squared = fit["r_squared"]
    p_value = fit["p_value"]

    df["Predicted_Display"] = fit["y_pred"]
    df["Residual_Display"] = fit["residuals"]
    max_residual = np.max(np.abs(df["Residual_Display"]))

    # Convert N_dark back to Watts for dBm check
    N_dark_watts = (N_dark_disp / n_scale) * 1e-3

    # ------------------------------------------------------------------
    # 5. Print Statistical Report with Dynamic Units
    # ------------------------------------------------------------------
    print("\n==========================================================")
    print("             PHOTODETECTOR FIT STATISTICS                 ")
    print("==========================================================")
    print(
        f" Model: Noise({n_unit}) = constant * Power({p_unit}) + Dark Noise({n_unit}) "
    )
    print(f" Extracted Data Runs:       {n} successful runs           ")
    print("----------------------------------------------------------")
    print(
        f" Responsiveness Slope (k):  {k:.4f} ± {std_err:.4f} {n_unit}/{p_unit}"
    )
    print(
        f" Dark Noise (N_dark):       {N_dark_disp:.4f} ± {intercept_std_err:.4f} {n_unit}"
    )
    if N_dark_watts > 0:
        dbm_val = 10 * np.log10(N_dark_watts / 1e-3)
        print(f" Predicted Dark Floor dBm:  {dbm_val:.3f} dBm")
    print("----------------------------------------------------------")
    print(f" Coefficient of Det. (R²):  {r_squared:.6f}")
    print(f" Correlation Coeff. (r):    {r_value:.6f}")
    print(f" Two-Tailed p-value:        {p_value:.6e}")
    print(f" Max Absolute Residual:     {max_residual:.4f} {n_unit}")
    print("==========================================================\n")

    # ------------------------------------------------------------------
    # 5b. Derived Noise Formulas (as functions of Power P, in p_unit)
    # ------------------------------------------------------------------
    unit_db_correction = -10.0 * np.log10(n_scale)
    a_term = 10.0 * np.log10(k) + unit_db_correction
    c_term = N_dark_disp / k
    log2_coeff = 10.0 * np.log10(2)  # Approximately 3.0103

    print("==========================================================")
    print("                 DERIVED NOISE FORMULAS                   ")
    print(f"              (P = optical beam power, in {p_unit})       ")
    print("==========================================================")

    print(f" (1) Total Noise ({n_unit}):")
    print(f"   N_total = {k:.4f} * P + {N_dark_disp:.4f}")
    print()

    print(" (2) Total Noise (dBm):")
    print(f"   a = {a_term:.5f}")
    print(f"   C = {c_term:.6f}")
    print("   Full equation:")
    print(f"   N ~ {a_term:.5f} + {log2_coeff:.4f} log\u2082(P + {c_term:.6f})")
    print()

    print(f" (3) Shot Noise ({n_unit}):")
    print(f"   N_shot  = {k:.4f} * P")
    print()

    print(" (4) Shot Noise (dBm):")
    print("   Full equation:")
    print(f"   N ~ {a_term:.5f} + {log2_coeff:.4f} log\u2082(P)")

    print("==========================================================\n")

    # ------------------------------------------------------------------
    # 6. Build Matplotlib Figure (Now displayed BEFORE calculator)
    # ------------------------------------------------------------------
    max_p = (
        df["Power_Display"].max() * 1.1
        if df["Power_Display"].max() > 0
        else 2.0
    )
    P_model = np.linspace(0, max_p, 100)
    N_total_model = k * P_model + N_dark_disp
    N_shot_model = k * P_model

    fig, (ax_main, ax_res) = plt.subplots(
        2,
        1,
        figsize=(8.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    formula_label = f"Total Fit: $\\text{{Total Noise({n_unit})}} = {k:.3f} \\cdot \\text{{Power({p_unit})}} + {N_dark_disp:.3f}\\text{{ {n_unit}}}$ ($R^2={r_squared:.4f}$)"

    ax_main.plot(
        P_model,
        N_total_model,
        color="#1f77b4",
        linewidth=2,
        label=formula_label,
    )

    ax_main.plot(
        P_model,
        N_shot_model,
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        label=f"Calibrated Shot-Noise Limit: $\\text{{Shot Noise({n_unit})}} = {k:.3f} \\cdot \\text{{Power({p_unit})}}$",
    )

    ax_main.scatter(
        df["Power_Display"],
        df["Noise_Display"],
        color="#1f77b4",
        s=50,
        zorder=4,
        label="Shot Noise Data (Gross)",
    )

    ax_main.scatter(
        [0],
        [N_dark_disp],
        color="#d62728",
        marker="D",
        s=65,
        zorder=5,
        label=f"Dark Noise Intersect $(0, {N_dark_disp:.3f}\\text{{ {n_unit}}})$",
    )
    ax_main.scatter(
        [0],
        [0],
        color="#2ca02c",
        marker="o",
        s=65,
        zorder=5,
        label="SNL Origin Intersect $(0, 0)$",
    )

    # Dynamic Annotations
    ax_main.annotate(
        f"$(0.000, {N_dark_disp:.3f})$ {n_unit}",
        xy=(0, N_dark_disp),
        xytext=(
            max_p * 0.05,
            N_dark_disp * 1.2
            if N_dark_disp > 0
            else df["Noise_Display"].max() * 0.1,
        ),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        color="#d62728",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="#d62728",
            alpha=0.8,
        ),
    )

    ax_main.annotate(
        f"$(0.000, 0.000)$ {n_unit}",
        xy=(0, 0),
        xytext=(max_p * 0.07, df["Noise_Display"].max() * 0.05),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        color="#2ca02c",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="#2ca02c",
            alpha=0.8,
        ),
    )

    ax_main.set_xlim(left=0, right=max_p)
    ax_main.set_ylim(bottom=-0.05 * df["Noise_Display"].max())
    ax_main.margins(x=0)
    ax_res.margins(x=0)

    ax_main.set_ylabel(f"Linear Noise Power ({n_unit})", fontsize=11)
    ax_main.set_title(
        f"Photodetector Noise Calibration (Auto-Scaled to {p_unit} / {n_unit})",
        fontsize=12,
        fontweight="bold",
    )
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(
        loc="upper left", frameon=True, facecolor="white", framealpha=0.9
    )

    # Bottom Panel: Residuals
    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax_res.scatter(
        df["Power_Display"],
        df["Residual_Display"],
        color="#1f77b4",
        s=40,
        marker="o",
        zorder=3,
    )

    ax_res.set_xlabel(f"Optical Beam Power ({p_unit})", fontsize=11)
    ax_res.set_ylabel(f"Residuals ({n_unit})", fontsize=10)
    ax_res.grid(True, linestyle=":", alpha=0.6)

    res_bound = max_residual * 1.5 if max_residual > 0 else 0.1
    ax_res.set_ylim(-res_bound, res_bound)

    plt.tight_layout()
    # Draw plot in non-blocking mode so the script can proceed to the interactive calculator
    plt.show(block=False)
    plt.pause(0.1)

    # ------------------------------------------------------------------
    # 7. Interactive Equation Evaluation Loop
    # ------------------------------------------------------------------
    interactive_calculator(k, N_dark_disp, a_term, c_term, p_unit, n_unit)

    # Keep the plot window alive after the user quits the calculator loop
    if plt.fignum_exists(fig.number):
        print("Close the plot window to exit the script completely.")
        plt.show()

if __name__ == "__main__":
    main()