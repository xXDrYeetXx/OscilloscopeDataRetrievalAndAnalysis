import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Set crisp figure rendering
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


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
    run_numbers: list[int], base_folder: str = "v3_converged_noise_data"
) -> pd.DataFrame:
    """Iterates through specified Run directories and extracts beam power (mW)

    and linear noise power standardized strictly to mW.
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
            noise_mw = noise_watts * 1e3  # Convert Watts -> mW

            records.append(
                {
                    "Run": run_num,
                    "Power_mW": power_mw,
                    "Noise_mW": noise_mw,
                }
            )
            print(
                f"  • Run {run_num:2d}: Loaded Power = {power_mw:.4f} mW | Noise = {noise_mw:.6e} mW"
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


def main():
    # ------------------------------------------------------------------
    # 1. Ask the user for run selection (supports '1,3-5,7-20')
    # ------------------------------------------------------------------
    print("==========================================================")
    print("      QUANTUM SHOT NOISE AUTOMATED RUN CALIBRATOR         ")
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
    # 2. Load JSON data for each selected run (All values in mW)
    # ------------------------------------------------------------------
    df = extract_data_from_runs(run_list)
    n = len(df)

    # ------------------------------------------------------------------
    # 3. Linear Regression & Residual Calculations
    # ------------------------------------------------------------------
    # Model: Noise(mW) = k * Power(mW) + Dark_Noise(mW)
    slope, intercept, r_value, p_value, std_err = linregress(
        df["Power_mW"], df["Noise_mW"]
    )
    k, N_dark_mw = slope, intercept
    r_squared = r_value**2

    # Intercept Standard Error calculation
    n_pts = len(df)
    intercept_std_err = (
        std_err
        * np.sqrt(np.sum(df["Power_mW"] ** 2) / n_pts)
        / np.std(df["Power_mW"], ddof=0)
    )

    df["Predicted_mW"] = k * df["Power_mW"] + N_dark_mw
    df["Residual_mW"] = df["Noise_mW"] - df["Predicted_mW"]
    max_residual = np.max(np.abs(df["Residual_mW"]))

    # ------------------------------------------------------------------
    # 4. Print Statistical Analysis Report (Standardized to mW)
    # ------------------------------------------------------------------
    print("\n==========================================================")
    print("             PHOTODETECTOR FIT STATISTICS                 ")
    print("==========================================================")
    print(" Model: Noise(mW) = constant * Power(mW) + Dark Noise(mW) ")
    print(f" Extracted Data Runs:       {n} successful runs           ")
    print("----------------------------------------------------------")
    print(f" Responsiveness Slope (k):  {k:.6e} ± {std_err:.6e} (dimensionless)")
    print(
        f" Dark Noise (N_dark):       {N_dark_mw:.6e} ± {intercept_std_err:.6e} mW"
    )
    if N_dark_mw > 0:
        dbm_val = 10 * np.log10(N_dark_mw)
        print(f" Predicted Dark Floor dBm:  {dbm_val:.3f} dBm")
    print("----------------------------------------------------------")
    print(f" Coefficient of Det. (R²):  {r_squared:.6f}")
    print(f" Correlation Coeff. (r):    {r_value:.6f}")
    print(f" Two-Tailed p-value:        {p_value:.6e}")
    print(f" Max Absolute Residual:     {max_residual:.6e} mW")
    print("==========================================================\n")

    # ------------------------------------------------------------------
    # 5. Build & Display Matplotlib Figure
    # ------------------------------------------------------------------
    max_p = df["Power_mW"].max() * 1.1 if df["Power_mW"].max() > 0 else 2.0
    P_model = np.linspace(0, max_p, 100)
    N_total_model = k * P_model + N_dark_mw
    N_shot_model = k * P_model

    fig, (ax_main, ax_res) = plt.subplots(
        2,
        1,
        figsize=(8.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # --- TOP PANEL ---
    # Formula formatted in legend key
    formula_label = f"Total Fit: $\\text{{Noise(mW)}} = {k:.3e} \\cdot \\text{{Power(mW)}} + {N_dark_mw:.3e}\\text{{ mW}}$ ($R^2={r_squared:.4f}$)"

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
        label="Calibrated Shot-Noise Limit: $\\text{Noise(mW)} = \\text{constant} \\cdot \\text{Power(mW)}$",
    )

    ax_main.scatter(
        df["Power_mW"],
        df["Noise_mW"],
        color="#1f77b4",
        s=50,
        zorder=4,
        label="Shot Noise Data (Gross)",
    )

    ax_main.scatter(
        [0],
        [N_dark_mw],
        color="#d62728",
        marker="D",
        s=65,
        zorder=5,
        label=f"Dark Noise Intersect $(0, {N_dark_mw:.3e}\\text{{ mW}})$",
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

    # Annotations on Intersects
    ax_main.annotate(
        f"$(0.000, {N_dark_mw:.3e})$ mW",
        xy=(0, N_dark_mw),
        xytext=(max_p * 0.05, N_dark_mw * 1.3 if N_dark_mw > 0 else 0.1),
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
        "$(0.000, 0.000)$ mW",
        xy=(0, 0),
        xytext=(max_p * 0.07, max(df["Noise_mW"]) * 0.05),
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
    ax_main.set_ylim(bottom=-0.05 * max(df["Noise_mW"]))
    ax_main.margins(x=0)
    ax_res.margins(x=0)

    ax_main.set_ylabel("Linear Noise Power (mW)", fontsize=11)
    ax_main.set_title(
        "Photodetector Noise Calibration: Y-Axis Intersects & SNL Baseline",
        fontsize=12,
        fontweight="bold",
    )
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(
        loc="upper left", frameon=True, facecolor="white", framealpha=0.9
    )

    # --- BOTTOM PANEL ---
    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax_res.scatter(
        df["Power_mW"],
        df["Residual_mW"],
        color="#1f77b4",
        s=40,
        marker="o",
        zorder=3,
    )

    ax_res.set_xlabel("Optical Beam Power (mW)", fontsize=11)
    ax_res.set_ylabel("Residuals (mW)", fontsize=10)
    ax_res.grid(True, linestyle=":", alpha=0.6)

    res_bound = max_residual * 1.5 if max_residual > 0 else 1e-10
    ax_res.set_ylim(-res_bound, res_bound)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()