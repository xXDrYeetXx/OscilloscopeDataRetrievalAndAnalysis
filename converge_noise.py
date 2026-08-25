#!/usr/bin/env python3
"""
Acquire zero-window noise pairs until the running mean noise power
converges to within a specified tolerance.

Convergence condition
---------------------
The running mean noise power (computed in watts) is checked every
CHECK_INTERVAL_PAIRS qualifying pairs. Convergence is declared when
the relative change between consecutive checks falls below
CONVERGENCE_TOLERANCE for CONVERGENCE_STREAK_REQUIRED consecutive
checks:

    abs(mean_now - mean_prev) / mean_prev <= CONVERGENCE_TOLERANCE

Usage:
    python converge_noise.py

The final converged noise value is printed and saved to
converged_result.json.
"""

from __future__ import annotations

import csv
import json
import math
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import periodogram
import pyvisa


# =====================================================================
# USER CONFIGURATION
# =====================================================================

OSCILLOSCOPE_IP              = "192.168.1.100"
VISA_RESOURCE                = f"TCPIP0::{OSCILLOSCOPE_IP}::inst0::INSTR"

CHANNEL                      = 3
ZERO_MEAN_THRESHOLD          = 10e-3        # volts
TARGET_FREQUENCY_HZ          = 10e6         # Hz
SUBWINDOW_DURATION_SECONDS   = 15e-6        # seconds
LONG_RECORD_DURATION_SECONDS = 150e-6       # seconds
REQUESTED_SAMPLE_RATE_HZ     = 1e9          # Sa/s
REQUESTED_ACQUISITION_POINTS = round(
    LONG_RECORD_DURATION_SECONDS * REQUESTED_SAMPLE_RATE_HZ
)
REFERENCE_IMPEDANCE_OHMS     = 50.0

# How often to check convergence, in qualifying pairs.
CHECK_INTERVAL_PAIRS         = 20

# Fractional tolerance for declaring convergence.
# 0.01 means the mean must change by less than 1% between checks.
CONVERGENCE_TOLERANCE        = 0.01

# How many consecutive checks must pass before stopping.
CONVERGENCE_STREAK_REQUIRED  = 5

# Minimum qualifying pairs before convergence is even checked.
MIN_PAIRS_BEFORE_CHECK       = 50

OUTPUT_DIRECTORY             = "v3_converged_noise_data"
VISA_TIMEOUT_SECONDS         = 60.0
ACQUISITION_INTERVAL_SECONDS = 0.0
PYVISA_BACKEND               = "@py"


# =====================================================================
# PROGRAM STATE
# =====================================================================

stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


# =====================================================================
# OSCILLOSCOPE
# =====================================================================

def query_float(scope, command):
    return float(scope.query(command).strip().split()[-1])


def configure_scope(scope):
    scope.write("*CLS")
    scope.write(":SYSTem:HEADer OFF")
    scope.write(f":CHANnel{CHANNEL}:DISPlay ON")
    scope.write(f":TIMebase:RANGe {LONG_RECORD_DURATION_SECONDS}")
    scope.write(f":ACQuire:SRATe {REQUESTED_SAMPLE_RATE_HZ}")
    scope.write(
        f":ACQuire:POINts:ANALog {REQUESTED_ACQUISITION_POINTS}"
    )
    scope.query("*OPC?")


def read_channel_waveform(scope):
    scope.write(f":WAVeform:SOURce CHANnel{CHANNEL}")
    scope.write(":WAVeform:FORMat ASCii")

    sample_interval_s = query_float(scope, ":WAVeform:XINCrement?")
    time_origin_s     = query_float(scope, ":WAVeform:XORigin?")

    response      = scope.query(":WAVeform:DATA?")
    voltage_volts = np.fromstring(
        response.strip(), sep=",", dtype=np.float64
    )

    if voltage_volts.size == 0:
        raise RuntimeError("No Channel 3 data returned.")

    time_seconds = (
        time_origin_s
        + np.arange(voltage_volts.size, dtype=np.float64)
        * sample_interval_s
    )

    return time_seconds, voltage_volts, sample_interval_s


# =====================================================================
# SIGNAL PROCESSING
# =====================================================================

def remove_invalid(time_seconds, voltage_volts):
    valid = (
        np.isfinite(time_seconds)
        & np.isfinite(voltage_volts)
        & (np.abs(voltage_volts) < 1e30)
    )
    return time_seconds[valid], voltage_volts[valid]


def split_into_subwindows(time_s, voltage_v, sample_interval_s):
    sample_rate_hz      = 1.0 / sample_interval_s
    samples_per_window  = int(
        round(SUBWINDOW_DURATION_SECONDS * sample_rate_hz)
    )

    if samples_per_window < 4:
        raise RuntimeError("Subwindow too short for current sample rate.")

    n_windows = voltage_v.size // samples_per_window

    if n_windows == 0:
        raise RuntimeError("Acquisition shorter than one subwindow.")

    for i in range(n_windows):
        first = i * samples_per_window
        last  = first + samples_per_window
        yield i, time_s[first:last], voltage_v[first:last], sample_rate_hz


def calculate_noise(voltage_volts, sample_rate_hz):
    if voltage_volts.size < 4:
        raise RuntimeError("Too few samples for periodogram.")

    freq_hz, volt_psd = periodogram(
        voltage_volts,
        fs=sample_rate_hz,
        window="hann",
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )

    power_psd = volt_psd / REFERENCE_IMPEDANCE_OHMS

    w      = np.hanning(voltage_volts.size)
    enbw   = float(
        sample_rate_hz * np.sum(w ** 2) / (np.sum(w) ** 2)
    )

    index  = int(np.argmin(np.abs(freq_hz - TARGET_FREQUENCY_HZ)))
    noise_watts = float(power_psd[index]) * enbw

    if noise_watts > 0 and math.isfinite(noise_watts):
        noise_dbm = 10.0 * math.log10(noise_watts / 1e-3)
    else:
        noise_dbm = math.nan

    return float(freq_hz[index]), noise_watts, noise_dbm, enbw


# =====================================================================
# CSV
# =====================================================================

CSV_COLUMNS = [
    "global_pair_number",
    "timestamp_utc",
    "abs_v3_mean_volts",
    "noise_power_watts",
    "noise_power_dbm",
    "running_mean_watts",
    "running_mean_dbm",
]


def create_csv(path):
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def append_csv_row(path, row):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
        f.flush()


# =====================================================================
# CONVERGENCE CHECK
# =====================================================================

def check_convergence(
    noise_watts_list: list,
    prev_mean_watts: float,
) -> tuple[float, bool]:
    """
    Compute the current running mean and check whether it has changed
    by less than CONVERGENCE_TOLERANCE relative to the previous check.

    Returns the new mean and whether this check passed.
    """
    current_mean = float(np.mean(noise_watts_list))

    if math.isnan(prev_mean_watts) or prev_mean_watts == 0:
        return current_mean, False

    relative_change = abs(current_mean - prev_mean_watts) / prev_mean_watts
    passed = relative_change <= CONVERGENCE_TOLERANCE

    return current_mean, passed


# =====================================================================
# MAIN
# =====================================================================

def main():
    output_dir = Path(OUTPUT_DIRECTORY).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "zero_pairs.csv"
    create_csv(csv_path)

    signal.signal(signal.SIGINT,  request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    resource_manager = None
    scope            = None

    # State
    noise_watts_list      = []
    global_pair_number    = 0
    long_acq_number       = 0
    convergence_streak    = 0
    prev_mean_watts       = math.nan
    converged             = False

    try:
        print(f"Connecting to {OSCILLOSCOPE_IP} ...")
        resource_manager = pyvisa.ResourceManager(PYVISA_BACKEND)
        scope = resource_manager.open_resource(VISA_RESOURCE)
        scope.timeout    = int(VISA_TIMEOUT_SECONDS * 1000)
        scope.chunk_size = 10 * 1024 * 1024

        instrument_id = scope.query("*IDN?").strip()
        print(f"Connected to: {instrument_id}")

        configure_scope(scope)

        print(f"\nConvergence tolerance  : "
              f"{CONVERGENCE_TOLERANCE * 100:.1f}%")
        print(f"Streak required        : "
              f"{CONVERGENCE_STREAK_REQUIRED} consecutive checks")
        print(f"Check interval         : "
              f"every {CHECK_INTERVAL_PAIRS} qualifying pairs")
        print(f"Minimum pairs          : {MIN_PAIRS_BEFORE_CHECK}")
        print(f"Alignment threshold    : "
              f"|mean(V3)| <= {ZERO_MEAN_THRESHOLD * 1e3:.1f} mV")
        print("Press Ctrl+C to stop early.\n")

        while not stop_requested and not converged:
            try:
                long_acq_number += 1
                scope.query(":SINGle;*OPC?")

                time_s, voltage_v, sample_interval_s = (
                    read_channel_waveform(scope)
                )
                time_s, voltage_v = remove_invalid(time_s, voltage_v)

                if voltage_v.size == 0:
                    raise RuntimeError("No valid Channel 3 samples.")

                timestamp = datetime.now(timezone.utc).isoformat()

                for (
                    _,
                    window_time,
                    window_voltage,
                    sample_rate_hz,
                ) in split_into_subwindows(
                    time_s, voltage_v, sample_interval_s
                ):
                    abs_v3_mean = float(abs(np.mean(window_voltage)))
                    aligned     = abs_v3_mean <= ZERO_MEAN_THRESHOLD

                    if not aligned:
                        continue

                    global_pair_number += 1

                    _, noise_watts, noise_dbm, _ = calculate_noise(
                        window_voltage, sample_rate_hz
                    )

                    if not math.isfinite(noise_watts):
                        continue

                    noise_watts_list.append(noise_watts)

                    running_mean_w   = float(
                        np.mean(noise_watts_list)
                    )
                    running_mean_dbm = (
                        10.0 * math.log10(running_mean_w / 1e-3)
                        if running_mean_w > 0 else math.nan
                    )

                    append_csv_row(csv_path, [
                        global_pair_number,
                        timestamp,
                        f"{abs_v3_mean:.12e}",
                        f"{noise_watts:.12e}",
                        f"{noise_dbm:.9f}",
                        f"{running_mean_w:.12e}",
                        f"{running_mean_dbm:.9f}",
                    ])

                    print(
                        f"\rPairs: {global_pair_number:5d}  "
                        f"|mean(V3)|={abs_v3_mean * 1e3:.3f} mV  "
                        f"noise={noise_dbm:.3f} dBm  "
                        f"running mean={running_mean_dbm:.3f} dBm  "
                        f"streak={convergence_streak}/"
                        f"{CONVERGENCE_STREAK_REQUIRED}",
                        end="",
                        flush=True,
                    )

                    # Check convergence every CHECK_INTERVAL_PAIRS pairs.
                    if (
                        global_pair_number >= MIN_PAIRS_BEFORE_CHECK
                        and global_pair_number % CHECK_INTERVAL_PAIRS == 0
                    ):
                        new_mean, passed = check_convergence(
                            noise_watts_list, prev_mean_watts
                        )

                        if passed:
                            convergence_streak += 1
                        else:
                            convergence_streak = 0

                        prev_mean_watts = new_mean

                        if convergence_streak >= CONVERGENCE_STREAK_REQUIRED:
                            converged = True
                            break

                if ACQUISITION_INTERVAL_SECONDS > 0:
                    time.sleep(ACQUISITION_INTERVAL_SECONDS)

            except pyvisa.errors.VisaIOError as error:
                print(f"\nVISA error: {error}", file=sys.stderr)
                time.sleep(1.0)

            except (ValueError, RuntimeError) as error:
                print(f"\nError: {error}", file=sys.stderr)
                time.sleep(1.0)

        # Final result
        print("\n")

        if len(noise_watts_list) == 0:
            print("No qualifying pairs were collected.")
            return

        final_mean_watts = float(np.mean(noise_watts_list))
        final_std_watts  = float(np.std(noise_watts_list))
        final_mean_dbm   = 10.0 * math.log10(
            final_mean_watts / 1e-3
        )

        if converged:
            print("--- CONVERGED ---")
        else:
            print("--- STOPPED BEFORE CONVERGENCE ---")

        print(f"Qualifying pairs     : {global_pair_number}")
        print(
            f"Mean noise           : {final_mean_watts:.6e} W  "
            f"({final_mean_dbm:.4f} dBm)"
        )
        print(f"Std deviation        : {final_std_watts:.6e} W")
        print(f"Converged            : {converged}")

        result = {
            "converged"          : converged,
            "qualifying_pairs"   : global_pair_number,
            "final_mean_watts"   : final_mean_watts,
            "final_std_watts"    : final_std_watts,
            "final_mean_dbm"     : final_mean_dbm,
            "convergence_tolerance" : CONVERGENCE_TOLERANCE,
            "streak_required"    : CONVERGENCE_STREAK_REQUIRED,
            "check_interval"     : CHECK_INTERVAL_PAIRS,
            "completed_utc"      : datetime.now(timezone.utc).isoformat(),
        }

        result_path = output_dir / "converged_result.json"
        result_path.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(f"Result saved to      : {result_path}")
        print(f"Pair data saved to   : {csv_path}")

    finally:
        if scope is not None:
            scope.close()
        if resource_manager is not None:
            resource_manager.close()


if __name__ == "__main__":
    main()