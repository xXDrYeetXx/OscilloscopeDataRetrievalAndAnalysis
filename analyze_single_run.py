#!/usr/bin/env python3
"""
Extract summary statistics from the zero_pairs.csv file of a given run.

Usage:
    python extract_zero_pairs_stats.py
    (then type the run number when prompted)

The script assumes the default folder layout created by the acquisition
script:

    <current_working_dir>/
        v3_converged_noise_data/
            Run 1/
                zero_pairs.csv
            Run 2/
                zero_pairs.csv
            ...
"""

import csv
import os
import statistics
import sys
from pathlib import Path

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Ask the user for the run number
    # ------------------------------------------------------------------
    while True:
        try:
            run_input = input("Enter the run number (e.g. 1 for Run 1): ").strip()
            run_number = int(run_input)
            if run_number <= 0:
                raise ValueError
            break
        except ValueError:
            print("Please enter a positive integer.")

    # ------------------------------------------------------------------
    # 2. Build the expected path to zero_pairs.csv
    # ------------------------------------------------------------------
    base_dir = Path("v3_converged_noise_data").expanduser().resolve()
    run_dir = base_dir / f"Run {run_number}"
    csv_path = run_dir / "zero_pairs.csv"

    if not csv_path.is_file():
        print(f"\nError: Could not find '{csv_path}'.",
              file=sys.stderr)
        print("Make sure:")
        print("  • The folder v3_converged_noise_data exists in the current directory,")
        print("  • A sub‑folder named 'Run <number>' exists inside it,")
        print("  • zero_pairs.csv is present in that sub‑folder.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Read the CSV and collect the noise‑power columns
    # ------------------------------------------------------------------
    noise_dbm_vals = []   # column: noise_power_dbm
    noise_watts_vals = [] # column: noise_power_watts

    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Verify that the expected columns exist
            required = {"noise_power_dbm", "noise_power_watts"}
            if not required.issubset(reader.fieldnames):
                missing = required - set(reader.fieldnames or [])
                print(f"Error: CSV is missing required column(s): {missing}",
                      file=sys.stderr)
                sys.exit(1)

            for row in reader:
                try:
                    noise_dbm_vals.append(float(row["noise_power_dbm"]))
                    noise_watts_vals.append(float(row["noise_power_watts"]))
                except ValueError:
                    # Skip rows with malformed numbers – but warn the user
                    print(
                        f"Warning: Skipping row {reader.line_num} due to "
                        f"non‑numeric noise value.",
                        file=sys.stderr,
                    )
    except Exception as exc:
        print(f"Failed to read CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    if not noise_dbm_vals:
        print("Error: No valid data rows found in the CSV.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Compute statistics
    # ------------------------------------------------------------------
    n_rows = len(noise_dbm_vals)

    mean_dbm = statistics.mean(noise_dbm_vals)
    stdev_dbm = statistics.stdev(noise_dbm_vals) if n_rows > 1 else 0.0

    mean_watts = statistics.mean(noise_watts_vals)
    stdev_watts = statistics.stdev(noise_watts_vals) if n_rows > 1 else 0.0

    # ------------------------------------------------------------------
    # 5. Print a concise summary
    # ------------------------------------------------------------------
    print("\n=== Statistics for zero_pairs.csv ===")
    print(f"Run number          : {run_number}")
    print(f"File path           : {csv_path}")
    print(f"Number of rows      : {n_rows:,}")
    print()
    print("Noise power (dBm):")
    print(f"  Mean   : {mean_dbm: .4f} dBm")
    print(f"  StdDev : {stdev_dbm: .4f} dBm")
    print()
    print("Noise power (watts):")
    print(f"  Mean   : {mean_watts:.3e} W")
    print(f"  StdDev : {stdev_watts:.3e} W")
    print()
    # Also show the dBm‑to‑watts conversion sanity check (optional)
    # mean_dbm_check = 10 * math.log10(mean_watts / 1e-3) if mean_watts > 0 else float('-inf')
    # print(f"(Check) Mean dBm from watts: {mean_dbm_check: .4f} dBm")
    print("=== End of summary ===\n")

if __name__ == "__main__":
    main()