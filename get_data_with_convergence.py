#!/usr/bin/env python3
"""
Acquire Channel 3 zero-window noise measurements until the mean noise
power converges.

Physical measurement
--------------------
Channel 3:

    V3(t) = V1(t) - V2(t)

A subwindow qualifies when:

    abs(mean(V3)) <= ZERO_MEAN_THRESHOLD

No positive/negative V3 balancing is performed.

Noise calculation
-----------------
For each qualifying subwindow:

    1. Calculate mean(V3)
    2. Calculate the periodogram using a Hann window
    3. Find the FFT bin nearest TARGET_FREQUENCY_HZ
    4. Convert voltage PSD to power PSD using 50 ohms
    5. Multiply by ENBW to obtain noise power in watts
    6. Convert that individual measurement to dBm for reporting

The running mean is ALWAYS calculated in watts:

    mean_power_watts = mean(individual_power_watts)

Only after averaging is the result converted to dBm:

    mean_dBm = 10 * log10(mean_power_watts / 1e-3)

Convergence
-----------
The cumulative mean power is checked every CHECK_INTERVAL_PAIRS
qualifying observations.

Convergence requires:

    abs(mean_now - mean_previous) / mean_previous
        <= CONVERGENCE_TOLERANCE

for CONVERGENCE_STREAK_REQUIRED consecutive checks.

A minimum number of observations must be collected before checking.

The script preserves the natural physical distribution of V3.
It does NOT randomly delete positive or negative observations.

Output
------
Each execution creates:

    v3_converged_noise_data/
        Run 1/
        Run 2/
        Run 3/
        ...

Each run contains:

    zero_pairs.csv
    convergence_history.csv
    converged_result.json

Usage
-----
    python get_data.py

Press Ctrl+C to stop early.
"""

from __future__ import annotations

import csv
import json
import math
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyvisa
from scipy.signal import periodogram


# =====================================================================
# USER CONFIGURATION
# =====================================================================

OSCILLOSCOPE_IP = "192.168.137.113"

VISA_RESOURCE = (
    f"TCPIP0::{OSCILLOSCOPE_IP}::inst0::INSTR"
)

CHANNEL = 3

# -------------------------------------------------------------
# Alignment condition
# -------------------------------------------------------------

ZERO_MEAN_THRESHOLD = 5e-3       # 5 mV

# -------------------------------------------------------------
# Noise measurement
# -------------------------------------------------------------

TARGET_FREQUENCY_HZ = 10e6       # 10 MHz

SUBWINDOW_DURATION_SECONDS = 15e-6

LONG_RECORD_DURATION_SECONDS = 150e-6

REQUESTED_SAMPLE_RATE_HZ = 1e9

REQUESTED_ACQUISITION_POINTS = round(
    LONG_RECORD_DURATION_SECONDS
    * REQUESTED_SAMPLE_RATE_HZ
)

REFERENCE_IMPEDANCE_OHMS = 50.0

# -------------------------------------------------------------
# Convergence
# -------------------------------------------------------------

# Check cumulative mean every N qualifying measurements.
CHECK_INTERVAL_PAIRS = 20

# Maximum fractional change allowed between convergence checks.
#
# 0.01 = 1%
# 0.005 = 0.5%
# 0.001 = 0.1%
#
CONVERGENCE_TOLERANCE = 0.01

# Number of consecutive successful checks required.
CONVERGENCE_STREAK_REQUIRED = 5

# Do not even begin convergence checking until this many
# qualifying measurements have been collected.
MIN_PAIRS_BEFORE_CHECK = 50

# -------------------------------------------------------------
# Acquisition
# -------------------------------------------------------------

OUTPUT_DIRECTORY = "v3_converged_noise_data"

VISA_TIMEOUT_SECONDS = 60.0

ACQUISITION_INTERVAL_SECONDS = 0.0

PYVISA_BACKEND = "@py"

# -------------------------------------------------------------
# Channel 3 vertical settings
# -------------------------------------------------------------

CHANNEL_VERTICAL_SCALE_VOLTS = 10e-3

CHANNEL_VERTICAL_OFFSET_VOLTS = 0.0


# =====================================================================
# PROGRAM STATE
# =====================================================================

stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


# =====================================================================
# RUN DIRECTORY
# =====================================================================

def get_next_run_directory(base_directory: Path) -> Path:
    """
    Create the next Run N directory.

    Example:

        Run 1
        Run 2
        Run 3

    If Run 1 and Run 2 exist, Run 3 is created.
    """

    base_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    highest_run_number = 0

    for path in base_directory.iterdir():

        if not path.is_dir():
            continue

        match = re.fullmatch(
            r"Run (\d+)",
            path.name,
        )

        if match is None:
            continue

        number = int(match.group(1))

        highest_run_number = max(
            highest_run_number,
            number,
        )

    next_number = highest_run_number + 1

    run_directory = (
        base_directory
        / f"Run {next_number}"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_directory


# =====================================================================
# UNIT CONVERSION
# =====================================================================

def watts_to_dbm(watts: float) -> float:

    if (
        math.isfinite(watts)
        and watts > 0
    ):
        return (
            10.0
            * math.log10(
                watts / 1e-3
            )
        )

    return math.nan


# =====================================================================
# OSCILLOSCOPE COMMUNICATION
# =====================================================================

def query_float(
    scope,
    command: str,
) -> float:

    response = (
        scope
        .query(command)
        .strip()
    )

    return float(
        response.split()[-1]
    )


def configure_scope(scope):

    scope.write("*CLS")

    scope.write(
        ":SYSTem:HEADer OFF"
    )

    scope.write(
        f":CHANnel{CHANNEL}:DISPlay ON"
    )

    scope.write(
        f":CHANnel{CHANNEL}:SCALe "
        f"{CHANNEL_VERTICAL_SCALE_VOLTS}"
    )

    scope.write(
        f":CHANnel{CHANNEL}:OFFSet "
        f"{CHANNEL_VERTICAL_OFFSET_VOLTS}"
    )

    scope.write(
        f":TIMebase:RANGe "
        f"{LONG_RECORD_DURATION_SECONDS}"
    )

    scope.write(
        f":ACQuire:SRATe "
        f"{REQUESTED_SAMPLE_RATE_HZ}"
    )

    scope.write(
        ":WAVeform:STReaming OFF"
    )

    scope.write(
        f":ACQuire:POINts:ANALog "
        f"{REQUESTED_ACQUISITION_POINTS}"
    )

    scope.query("*OPC?")


def read_channel_waveform(scope):

    scope.write(
        f":WAVeform:SOURce "
        f"CHANnel{CHANNEL}"
    )

    scope.write(
        ":WAVeform:FORMat ASCii"
    )

    scope.write(
        ":WAVeform:POINts:MODE RAW"
    )

    scope.write(
        f":WAVeform:POINts "
        f"{REQUESTED_ACQUISITION_POINTS}"
    )

    sample_interval_s = query_float(
        scope,
        ":WAVeform:XINCrement?",
    )

    time_origin_s = query_float(
        scope,
        ":WAVeform:XORigin?",
    )

    response = scope.query(
        ":WAVeform:DATA?"
    )

    voltage_volts = np.fromstring(
        response.strip(),
        sep=",",
        dtype=np.float64,
    )

    if voltage_volts.size == 0:
        raise RuntimeError(
            "No Channel 3 waveform data returned."
        )

    time_seconds = (
        time_origin_s
        + np.arange(
            voltage_volts.size,
            dtype=np.float64,
        )
        * sample_interval_s
    )

    return (
        time_seconds,
        voltage_volts,
        sample_interval_s,
    )


# =====================================================================
# DATA CLEANING
# =====================================================================

def remove_invalid(
    time_seconds,
    voltage_volts,
):

    valid = (
        np.isfinite(time_seconds)
        & np.isfinite(voltage_volts)
        & (
            np.abs(voltage_volts)
            < 1e30
        )
    )

    return (
        time_seconds[valid],
        voltage_volts[valid],
    )


# =====================================================================
# SUBWINDOW SEGMENTATION
# =====================================================================

def split_into_subwindows(
    time_seconds,
    voltage_volts,
    sample_interval_s,
):

    sample_rate_hz = (
        1.0
        / sample_interval_s
    )

    samples_per_window = int(
        round(
            SUBWINDOW_DURATION_SECONDS
            * sample_rate_hz
        )
    )

    if samples_per_window < 4:
        raise RuntimeError(
            "Subwindow is too short for "
            "the current sample rate."
        )

    n_windows = (
        voltage_volts.size
        // samples_per_window
    )

    if n_windows == 0:
        raise RuntimeError(
            "Acquisition is shorter "
            "than one subwindow."
        )

    for i in range(n_windows):

        first = (
            i
            * samples_per_window
        )

        last = (
            first
            + samples_per_window
        )

        yield (
            i,
            time_seconds[first:last],
            voltage_volts[first:last],
            sample_rate_hz,
        )


# =====================================================================
# NOISE MEASUREMENT
# =====================================================================

def calculate_noise(
    voltage_volts,
    sample_rate_hz,
):
    """
    Calculate noise power at the target frequency.

    Returns:

        actual_frequency_hz
        noise_psd_w_per_hz
        noise_power_watts
        noise_power_dbm
        enbw_hz
    """

    if voltage_volts.size < 4:
        raise RuntimeError(
            "Too few samples for periodogram."
        )

    frequency_hz, voltage_psd = (
        periodogram(
            voltage_volts,
            fs=sample_rate_hz,
            window="hann",
            detrend="constant",
            return_onesided=True,
            scaling="density",
        )
    )

    # V^2/Hz -> W/Hz
    power_psd_w_per_hz = (
        voltage_psd
        / REFERENCE_IMPEDANCE_OHMS
    )

    n = voltage_volts.size

    window = np.hanning(n)

    enbw_hz = float(
        sample_rate_hz
        * np.sum(window ** 2)
        / np.sum(window) ** 2
    )

    index = int(
        np.argmin(
            np.abs(
                frequency_hz
                - TARGET_FREQUENCY_HZ
            )
        )
    )

    actual_frequency_hz = float(
        frequency_hz[index]
    )

    noise_psd_w_per_hz = float(
        power_psd_w_per_hz[index]
    )

    # PSD * ENBW = power in the measurement bandwidth.
    noise_power_watts = (
        noise_psd_w_per_hz
        * enbw_hz
    )

    noise_power_dbm = watts_to_dbm(
        noise_power_watts
    )

    return (
        actual_frequency_hz,
        noise_psd_w_per_hz,
        noise_power_watts,
        noise_power_dbm,
        enbw_hz,
    )


# =====================================================================
# CSV
# =====================================================================

PAIR_COLUMNS = [
    "global_pair_number",
    "long_acquisition_number",
    "subwindow_number",
    "timestamp_utc",
    "subwindow_start_seconds",
    "subwindow_stop_seconds",
    "samples_in_subwindow",
    "sample_rate_hz",
    "v3_mean_volts",
    "abs_v3_mean_volts",
    "noise_psd_w_per_hz",
    "noise_power_watts",
    "noise_power_dbm",
    "actual_frequency_hz",
    "frequency_error_hz",
    "enbw_hz",
    "running_mean_watts",
    "running_mean_dbm",
]


CONVERGENCE_COLUMNS = [
    "check_number",
    "qualifying_pairs",
    "mean_watts",
    "mean_dbm",
    "std_watts",
    "sem_watts",
    "relative_change",
    "relative_change_percent",
    "passed",
    "streak",
    "timestamp_utc",
]


def create_csv(
    path: Path,
    columns,
):

    if path.exists():
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        csv.writer(f).writerow(
            columns
        )


def append_csv_row(
    path: Path,
    row,
):

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        csv.writer(f).writerow(
            row
        )

        f.flush()


# =====================================================================
# CONVERGENCE
# =====================================================================

def calculate_statistics(
    noise_watts: list[float],
):
    """
    Calculate statistics in watts.

    The mean is calculated in watts first.
    dBm is calculated only from the mean power.
    """

    values = np.asarray(
        noise_watts,
        dtype=np.float64,
    )

    mean_watts = float(
        np.mean(values)
    )

    std_watts = float(
        np.std(
            values,
            ddof=1,
        )
    ) if len(values) > 1 else math.nan

    sem_watts = (
        std_watts
        / math.sqrt(len(values))
        if len(values) > 1
        else math.nan
    )

    mean_dbm = watts_to_dbm(
        mean_watts
    )

    return (
        mean_watts,
        mean_dbm,
        std_watts,
        sem_watts,
    )


def check_convergence(
    noise_watts: list[float],
    previous_mean_watts: float,
):
    """
    Compare the new cumulative mean against
    the previous convergence-check mean.
    """

    (
        current_mean,
        current_dbm,
        std_watts,
        sem_watts,
    ) = calculate_statistics(
        noise_watts
    )

    if (
        not math.isfinite(
            previous_mean_watts
        )
        or previous_mean_watts <= 0
    ):

        return (
            current_mean,
            current_dbm,
            std_watts,
            sem_watts,
            math.nan,
            False,
        )

    relative_change = (
        abs(
            current_mean
            - previous_mean_watts
        )
        / previous_mean_watts
    )

    passed = (
        relative_change
        <= CONVERGENCE_TOLERANCE
    )

    return (
        current_mean,
        current_dbm,
        std_watts,
        sem_watts,
        relative_change,
        passed,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    global stop_requested

    base_directory = (
        Path(
            OUTPUT_DIRECTORY
        )
        .expanduser()
        .resolve()
    )

    run_directory = (
        get_next_run_directory(
            base_directory
        )
    )

    print()
    print("=" * 70)
    print("V3 ZERO-WINDOW NOISE ACQUISITION")
    print("=" * 70)
    print()
    print(
        f"Run directory         : "
        f"{run_directory}"
    )

    pair_csv = (
        run_directory
        / "zero_pairs.csv"
    )

    convergence_csv = (
        run_directory
        / "convergence_history.csv"
    )

    result_json = (
        run_directory
        / "converged_result.json"
    )

    create_csv(
        pair_csv,
        PAIR_COLUMNS,
    )

    create_csv(
        convergence_csv,
        CONVERGENCE_COLUMNS,
    )

    signal.signal(
        signal.SIGINT,
        request_stop,
    )

    signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    resource_manager = None
    scope = None

    # -------------------------------------------------------------
    # Measurement state
    # -------------------------------------------------------------

    noise_watts = []

    global_pair_number = 0

    long_acquisition_number = 0

    convergence_check_number = 0

    convergence_streak = 0

    previous_mean_watts = math.nan

    converged = False

    zero_positive_count = 0
    zero_negative_count = 0

    try:

        # ---------------------------------------------------------
        # Connect
        # ---------------------------------------------------------

        print()
        print(
            f"Connecting to oscilloscope "
            f"at {OSCILLOSCOPE_IP} ..."
        )

        resource_manager = (
            pyvisa.ResourceManager(
                PYVISA_BACKEND
            )
        )

        scope = (
            resource_manager
            .open_resource(
                VISA_RESOURCE
            )
        )

        scope.timeout = int(
            VISA_TIMEOUT_SECONDS
            * 1000
        )

        scope.chunk_size = (
            10 * 1024 * 1024
        )

        instrument_id = (
            scope
            .query("*IDN?")
            .strip()
        )

        print(
            f"Connected to: "
            f"{instrument_id}"
        )

        configure_scope(scope)

        # ---------------------------------------------------------
        # Configuration report
        # ---------------------------------------------------------

        print()
        print(
            f"Channel                : "
            f"{CHANNEL} = V1 - V2"
        )

        print(
            f"Alignment condition    : "
            f"|mean(V3)| <= "
            f"{ZERO_MEAN_THRESHOLD * 1e3:.3f} mV"
        )

        print(
            f"Target frequency       : "
            f"{TARGET_FREQUENCY_HZ / 1e6:.3f} MHz"
        )

        print(
            f"Subwindow duration     : "
            f"{SUBWINDOW_DURATION_SECONDS * 1e6:.3f} us"
        )

        print(
            f"Convergence tolerance  : "
            f"{CONVERGENCE_TOLERANCE * 100:.3f}%"
        )

        print(
            f"Check interval         : "
            f"{CHECK_INTERVAL_PAIRS} pairs"
        )

        print(
            f"Minimum pairs          : "
            f"{MIN_PAIRS_BEFORE_CHECK}"
        )

        print(
            f"Required streak        : "
            f"{CONVERGENCE_STREAK_REQUIRED}"
        )

        print()
        print(
            "Mean noise is averaged "
            "in W before conversion to dBm."
        )

        print(
            "No positive/negative V3 "
            "balancing is performed."
        )

        print()
        print(
            "Press Ctrl+C to stop early."
        )
        print()

        # =========================================================
        # ACQUISITION LOOP
        # =========================================================

        while (
            not stop_requested
            and not converged
        ):

            try:

                long_acquisition_number += 1

                # -------------------------------------------------
                # Acquire one long waveform
                # -------------------------------------------------

                scope.query(
                    ":SINGle;*OPC?"
                )

                (
                    time_seconds,
                    voltage_volts,
                    sample_interval_s,
                ) = read_channel_waveform(
                    scope
                )

                (
                    time_seconds,
                    voltage_volts,
                ) = remove_invalid(
                    time_seconds,
                    voltage_volts,
                )

                if voltage_volts.size == 0:
                    raise RuntimeError(
                        "No valid Channel 3 samples."
                    )

                acquisition_timestamp = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

                # -------------------------------------------------
                # Process subwindows
                # -------------------------------------------------

                for (
                    subwindow_index,
                    window_time,
                    window_voltage,
                    sample_rate_hz,
                ) in split_into_subwindows(
                    time_seconds,
                    voltage_volts,
                    sample_interval_s,
                ):

                    # ---------------------------------------------
                    # Alignment metric
                    # ---------------------------------------------

                    v3_mean = float(
                        np.mean(
                            window_voltage
                        )
                    )

                    abs_v3_mean = abs(
                        v3_mean
                    )

                    # ---------------------------------------------
                    # Alignment decision
                    # ---------------------------------------------

                    if (
                        not math.isfinite(
                            abs_v3_mean
                        )
                        or
                        abs_v3_mean
                        > ZERO_MEAN_THRESHOLD
                    ):
                        continue

                    # ---------------------------------------------
                    # Count natural sign distribution
                    # ---------------------------------------------

                    if v3_mean > 0:
                        zero_positive_count += 1

                    elif v3_mean < 0:
                        zero_negative_count += 1

                    # ---------------------------------------------
                    # Noise measurement
                    # ---------------------------------------------

                    (
                        actual_frequency_hz,
                        noise_psd_w_per_hz,
                        noise_power_watts,
                        noise_power_dbm,
                        enbw_hz,
                    ) = calculate_noise(
                        window_voltage,
                        sample_rate_hz,
                    )

                    if not math.isfinite(
                        noise_power_watts
                    ):
                        continue

                    # ---------------------------------------------
                    # Add measurement
                    # ---------------------------------------------

                    global_pair_number += 1

                    noise_watts.append(
                        noise_power_watts
                    )

                    (
                        running_mean_watts,
                        running_mean_dbm,
                        running_std_watts,
                        running_sem_watts,
                    ) = calculate_statistics(
                        noise_watts
                    )

                    # ---------------------------------------------
                    # Save measurement
                    # ---------------------------------------------

                    append_csv_row(
                        pair_csv,
                        [
                            global_pair_number,
                            long_acquisition_number,
                            subwindow_index + 1,
                            acquisition_timestamp,
                            f"{window_time[0]:.12e}",
                            f"{window_time[-1]:.12e}",
                            window_voltage.size,
                            f"{sample_rate_hz:.12e}",
                            f"{v3_mean:.12e}",
                            f"{abs_v3_mean:.12e}",
                            f"{noise_psd_w_per_hz:.12e}",
                            f"{noise_power_watts:.12e}",
                            f"{noise_power_dbm:.9f}",
                            f"{actual_frequency_hz:.12e}",
                            f"{actual_frequency_hz - TARGET_FREQUENCY_HZ:.12e}",
                            f"{enbw_hz:.12e}",
                            f"{running_mean_watts:.12e}",
                            f"{running_mean_dbm:.9f}",
                        ],
                    )

                    # ---------------------------------------------
                    # Display
                    # ---------------------------------------------

                    print(
                        f"\rPairs={global_pair_number:5d} | "
                        f"|V3|={abs_v3_mean * 1e3:7.3f} mV | "
                        f"noise={noise_power_dbm:8.3f} dBm | "
                        f"mean={running_mean_dbm:8.3f} dBm | "
                        f"streak="
                        f"{convergence_streak}/"
                        f"{CONVERGENCE_STREAK_REQUIRED}",
                        end="",
                        flush=True,
                    )

                    # =============================================
                    # CONVERGENCE CHECK
                    # =============================================

                    if (
                        global_pair_number
                        >= MIN_PAIRS_BEFORE_CHECK
                        and
                        global_pair_number
                        % CHECK_INTERVAL_PAIRS
                        == 0
                    ):

                        (
                            current_mean_watts,
                            current_mean_dbm,
                            std_watts,
                            sem_watts,
                            relative_change,
                            passed,
                        ) = check_convergence(
                            noise_watts,
                            previous_mean_watts,
                        )

                        convergence_check_number += 1

                        if passed:

                            convergence_streak += 1

                        else:

                            convergence_streak = 0

                        previous_mean_watts = (
                            current_mean_watts
                        )

                        # -----------------------------------------
                        # Save convergence history
                        # -----------------------------------------

                        append_csv_row(
                            convergence_csv,
                            [
                                convergence_check_number,
                                global_pair_number,
                                f"{current_mean_watts:.12e}",
                                f"{current_mean_dbm:.9f}",
                                f"{std_watts:.12e}",
                                f"{sem_watts:.12e}",
                                (
                                    f"{relative_change:.12e}"
                                    if math.isfinite(
                                        relative_change
                                    )
                                    else ""
                                ),
                                (
                                    f"{relative_change * 100:.6f}"
                                    if math.isfinite(
                                        relative_change
                                    )
                                    else ""
                                ),
                                passed,
                                convergence_streak,
                                datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            ],
                        )

                        # -----------------------------------------
                        # Print check
                        # -----------------------------------------

                        print()

                        if math.isfinite(
                            relative_change
                        ):

                            print(
                                f"  CHECK "
                                f"#{convergence_check_number}: "
                                f"{global_pair_number} pairs | "
                                f"mean="
                                f"{current_mean_dbm:.5f} dBm | "
                                f"change="
                                f"{relative_change * 100:.4f}% | "
                                f"{'PASS' if passed else 'FAIL'} | "
                                f"streak="
                                f"{convergence_streak}/"
                                f"{CONVERGENCE_STREAK_REQUIRED}"
                            )

                        else:

                            print(
                                f"  CHECK "
                                f"#{convergence_check_number}: "
                                f"{global_pair_number} pairs | "
                                f"first convergence reference"
                            )

                        # -----------------------------------------
                        # Stop if converged
                        # -----------------------------------------

                        if (
                            convergence_streak
                            >=
                            CONVERGENCE_STREAK_REQUIRED
                        ):

                            converged = True

                            print()
                            print(
                                "  >>> CONVERGENCE "
                                "CRITERION SATISFIED <<<"
                            )

                            break

                if (
                    ACQUISITION_INTERVAL_SECONDS
                    > 0
                ):

                    time.sleep(
                        ACQUISITION_INTERVAL_SECONDS
                    )

            except pyvisa.errors.VisaIOError as error:

                print(
                    f"\nVISA error: {error}",
                    file=sys.stderr,
                )

                time.sleep(1.0)

            except (
                ValueError,
                RuntimeError,
            ) as error:

                print(
                    f"\nProcessing error: "
                    f"{error}",
                    file=sys.stderr,
                )

                time.sleep(1.0)

        # =========================================================
        # FINAL RESULT
        # =========================================================

        print()
        print()
        print("=" * 70)

        if converged:
            print("CONVERGED")
        else:
            print("STOPPED BEFORE CONVERGENCE")

        print("=" * 70)

        if len(noise_watts) == 0:

            print(
                "No qualifying measurements "
                "were collected."
            )

            return

        (
            final_mean_watts,
            final_mean_dbm,
            final_std_watts,
            final_sem_watts,
        ) = calculate_statistics(
            noise_watts
        )

        # ---------------------------------------------------------
        # 95% CI for mean power
        # ---------------------------------------------------------

        if len(noise_watts) > 1:

            ci_low_watts = (
                final_mean_watts
                - 1.96 * final_sem_watts
            )

            ci_high_watts = (
                final_mean_watts
                + 1.96 * final_sem_watts
            )

            ci_low_watts = max(
                0.0,
                ci_low_watts,
            )

            ci_low_dbm = watts_to_dbm(
                ci_low_watts
            )

            ci_high_dbm = watts_to_dbm(
                ci_high_watts
            )

        else:

            ci_low_watts = math.nan
            ci_high_watts = math.nan
            ci_low_dbm = math.nan
            ci_high_dbm = math.nan

        # ---------------------------------------------------------
        # Final statistics
        # ---------------------------------------------------------

        print()
        print(
            f"Qualifying pairs       : "
            f"{len(noise_watts)}"
        )

        print(
            f"Positive V3 pairs      : "
            f"{zero_positive_count}"
        )

        print(
            f"Negative V3 pairs      : "
            f"{zero_negative_count}"
        )

        if len(noise_watts) > 0:

            print(
                f"Positive fraction      : "
                f"{zero_positive_count / len(noise_watts) * 100:.2f}%"
            )

            print(
                f"Negative fraction      : "
                f"{zero_negative_count / len(noise_watts) * 100:.2f}%"
            )

        print()
        print(
            f"Mean noise power       : "
            f"{final_mean_watts:.8e} W"
        )

        print(
            f"Mean noise             : "
            f"{final_mean_dbm:.6f} dBm"
        )

        print(
            f"Std deviation          : "
            f"{final_std_watts:.8e} W"
        )

        print(
            f"Standard error         : "
            f"{final_sem_watts:.8e} W"
        )

        if math.isfinite(
            ci_low_dbm
        ):

            print(
                f"95% CI of mean         : "
                f"{ci_low_dbm:.6f} to "
                f"{ci_high_dbm:.6f} dBm"
            )

        print()
        print(
            f"Convergence checks     : "
            f"{convergence_check_number}"
        )

        print(
            f"Convergence streak     : "
            f"{convergence_streak}/"
            f"{CONVERGENCE_STREAK_REQUIRED}"
        )

        print(
            f"Convergence tolerance  : "
            f"{CONVERGENCE_TOLERANCE * 100:.3f}%"
        )

        print(
            f"Converged              : "
            f"{converged}"
        )

        print()
        print(
            f"Pair data              : "
            f"{pair_csv}"
        )

        print(
            f"Convergence history    : "
            f"{convergence_csv}"
        )

        # ---------------------------------------------------------
        # JSON result
        # ---------------------------------------------------------

        result = {

            "converged":
                converged,

            "qualifying_pairs":
                len(noise_watts),

            "positive_v3_pairs":
                zero_positive_count,

            "negative_v3_pairs":
                zero_negative_count,

            "positive_fraction":
                (
                    zero_positive_count
                    / len(noise_watts)
                    if len(noise_watts) > 0
                    else math.nan
                ),

            "negative_fraction":
                (
                    zero_negative_count
                    / len(noise_watts)
                    if len(noise_watts) > 0
                    else math.nan
                ),

            "final_mean_watts":
                final_mean_watts,

            "final_mean_dbm":
                final_mean_dbm,

            "final_std_watts":
                final_std_watts,

            "final_sem_watts":
                final_sem_watts,

            "ci95_low_watts":
                ci_low_watts,

            "ci95_high_watts":
                ci_high_watts,

            "ci95_low_dbm":
                ci_low_dbm,

            "ci95_high_dbm":
                ci_high_dbm,

            "zero_mean_threshold_volts":
                ZERO_MEAN_THRESHOLD,

            "target_frequency_hz":
                TARGET_FREQUENCY_HZ,

            "subwindow_duration_seconds":
                SUBWINDOW_DURATION_SECONDS,

            "reference_impedance_ohms":
                REFERENCE_IMPEDANCE_OHMS,

            "convergence_tolerance":
                CONVERGENCE_TOLERANCE,

            "convergence_streak_required":
                CONVERGENCE_STREAK_REQUIRED,

            "check_interval_pairs":
                CHECK_INTERVAL_PAIRS,

            "minimum_pairs_before_check":
                MIN_PAIRS_BEFORE_CHECK,

            "averaging_method":
                "arithmetic mean of individual noise powers in watts",

            "dbm_conversion":
                "10*log10(mean_power_watts / 1e-3)",

            "positive_negative_balancing":
                False,

            "completed_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        result_json.write_text(
            json.dumps(
                result,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"Result JSON             : "
            f"{result_json}"
        )

        print("=" * 70)

    finally:

        if scope is not None:
            scope.close()

        if resource_manager is not None:
            resource_manager.close()


if __name__ == "__main__":
    main()