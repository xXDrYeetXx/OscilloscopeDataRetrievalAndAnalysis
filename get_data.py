#!/usr/bin/env python3
"""
Segmented Channel 3 alignment and noise acquisition.

Physical setup
--------------
Channel 3 is the hardware difference signal from the balanced photodetector:

    V3(t) = V1(t) - V2(t)

When the two laser signals are balanced and cancel, the arithmetic mean
of V3 over a short subwindow approaches zero. The residual FFT power at
the target frequency is the noise we want to collect.

Alignment metric
----------------
For each subwindow:

    alignment = abs(mean(V3))

This is independent of the noise fluctuations at the target frequency.
Using total RMS would mix the alignment condition with the noise being
measured and bias the result downward.

Processing
----------
1. Acquire one long Channel 3 waveform over LAN.
2. Divide it into non-overlapping subwindows.
3. For each subwindow:
       a. Calculate abs(mean(V3)) as the alignment metric.
       b. Calculate the PSD from the same samples using scipy.signal.periodogram.
       c. Extract noise power at the target frequency.
       d. Save the ordered pair:
              (abs(mean(V3)), noise power in dBm)
4. Copy every qualifying pair into zero_pairs.csv when:

       abs(mean(V3)) <= ZERO_MEAN_THRESHOLD

Requirements
------------
    pip install numpy scipy pyvisa pyvisa-py

Run:
    python main.py

Press Ctrl+C to stop.
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

# Replace with the actual IP address shown on the DSA-X 91304A under:
#   Utilities > Remote Interface > LAN
OSCILLOSCOPE_IP = "192.168.1.100"

VISA_RESOURCE = f"TCPIP0::{OSCILLOSCOPE_IP}::inst0::INSTR"

# Channel 3 is the physical V1 - V2 difference signal.
CHANNEL = 3

# Alignment threshold.
# A subwindow is accepted when abs(mean(V3)) is at or below this value.
# Start with 10 mV and tighten once you observe the physical stability.
ZERO_MEAN_THRESHOLD = 10e-3          # volts

# Frequency where noise power is extracted.
TARGET_FREQUENCY_HZ = 10e6          # 10 MHz

# Duration of each short analysis subwindow.
# For a Hann window:
#     effective noise bandwidth approximately = 1.5 / window duration
# Therefore 1.5 / 15 us = 100 kHz effective noise bandwidth.
SUBWINDOW_DURATION_SECONDS = 15e-6

# Duration of the long record downloaded each acquisition.
# At 150 us this gives approximately ten 15-us subwindows per acquisition.
LONG_RECORD_DURATION_SECONDS = 150e-6

# Sample rate.
# 1 GSa/s gives 100 samples per cycle at 10 MHz and avoids aliasing
# of signals below the analog bandwidth limit.
REQUESTED_SAMPLE_RATE_HZ = 1e9

# Number of waveform points in the long record.
REQUESTED_ACQUISITION_POINTS = round(
    LONG_RECORD_DURATION_SECONDS * REQUESTED_SAMPLE_RATE_HZ
)

# Reference impedance for converting voltage PSD to power PSD.
REFERENCE_IMPEDANCE_OHMS = 50.0

# Output folder.
OUTPUT_DIRECTORY = "v3_segmented_noise_data"

# Save complete FFT spectra for qualifying subwindows.
SAVE_ZERO_SPECTRA = True

# VISA timeout in seconds.
VISA_TIMEOUT_SECONDS = 60.0

# Optional pause between long acquisitions.
ACQUISITION_INTERVAL_SECONDS = 0.0

# Use the pure-Python VISA backend.
# This resolves the VI_ERROR_LIBRARY_NFOUND error when no
# Keysight or NI-VISA library is installed.
PYVISA_BACKEND = "@py"


# =====================================================================
# FIXED CONSTANTS
# =====================================================================

FFT_FUNCTION = 1   # Not used on the scope; FFT is computed in Python.

# =====================================================================
# PROGRAM STATE
# =====================================================================

stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


# =====================================================================
# OSCILLOSCOPE COMMUNICATION
# =====================================================================

def query_float(scope, command: str) -> float:
    """Send a SCPI query and return a single float."""
    response = scope.query(command).strip()
    return float(response.split()[-1])


def configure_scope(scope):
    """
    Prepare the oscilloscope for repeated Channel 3 acquisitions.

    The FFT is not configured on the scope because all spectral
    analysis is performed locally in Python.
    """
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
    """
    Download the completed Channel 3 waveform in ASCII (volts).

    Returns
    -------
    time_seconds         : time coordinate of each sample
    voltage_volts        : Channel 3 voltage samples
    sample_interval_s    : time between adjacent samples
    """
    scope.write(f":WAVeform:SOURce CHANnel{CHANNEL}")
    scope.write(":WAVeform:FORMat ASCii")

    sample_interval_s = query_float(scope, ":WAVeform:XINCrement?")
    time_origin_s = query_float(scope, ":WAVeform:XORigin?")

    response = scope.query(":WAVeform:DATA?")
    voltage_volts = np.fromstring(
        response.strip(), sep=",", dtype=np.float64
    )

    if voltage_volts.size == 0:
        raise RuntimeError(
            "No Channel 3 waveform data were returned."
        )

    time_seconds = (
        time_origin_s
        + np.arange(voltage_volts.size, dtype=np.float64)
        * sample_interval_s
    )

    return time_seconds, voltage_volts, sample_interval_s


# =====================================================================
# DATA CLEANING
# =====================================================================

def remove_invalid(time_seconds, voltage_volts):
    """Remove NaN, infinity, and Infiniium hole values."""
    valid = (
        np.isfinite(time_seconds)
        & np.isfinite(voltage_volts)
        & (np.abs(voltage_volts) < 1e30)
    )
    return time_seconds[valid], voltage_volts[valid]


# =====================================================================
# ALIGNMENT METRIC
# =====================================================================

def calculate_alignment_mean(voltage_volts):
    """
    Calculate the alignment metric for one subwindow.

    alignment = abs(mean(V3))

    A small value means the two laser signals are balanced and their
    average difference is near zero. This metric is independent of
    the noise fluctuations at the target frequency.
    """
    if voltage_volts.size == 0:
        return math.nan

    return float(abs(np.mean(voltage_volts)))


# =====================================================================
# NOISE MEASUREMENT
# =====================================================================

def calculate_noise_at_frequency(
    voltage_volts,
    sample_rate_hz,
    target_frequency_hz,
    impedance_ohms,
):
    """
    Estimate the noise power at the target frequency using scipy.

    Uses scipy.signal.periodogram with a Hann window, detrended to
    remove DC before the FFT. The PSD is in V^2/Hz and is divided by
    the impedance to give W/Hz. The bin power is obtained by multiplying
    the selected PSD value by the Hann window equivalent noise bandwidth:

        ENBW = 1.5 * fs / N   (approximately)

    Returns
    -------
    frequency_hz             : complete one-sided frequency axis
    power_psd_w_per_hz       : complete one-sided PSD in W/Hz
    actual_frequency_hz      : frequency of the bin nearest target
    noise_psd_w_per_hz       : PSD at the selected bin
    noise_power_watts        : noise power in the ENBW
    noise_power_dbm          : noise power in dBm
    enbw_hz                  : equivalent noise bandwidth
    """
    if voltage_volts.size < 4:
        raise RuntimeError(
            "Subwindow contains too few samples for a periodogram."
        )

    # scipy.signal.periodogram handles:
    #   - Hann windowing
    #   - DC detrending (detrend='constant' subtracts the mean)
    #   - one-sided normalization
    #   - density scaling (V^2/Hz)
    frequency_hz, voltage_psd = periodogram(
        voltage_volts,
        fs=sample_rate_hz,
        window="hann",
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )

    # Convert voltage PSD to power PSD.
    power_psd_w_per_hz = voltage_psd / impedance_ohms

    # Hann window equivalent noise bandwidth.
    # For a length-N Hann window:
    #     ENBW = fs * sum(w^2) / sum(w)^2
    # which is approximately 1.5 * fs / N.
    n = voltage_volts.size
    w = np.hanning(n)
    enbw_hz = float(
        sample_rate_hz * np.sum(w ** 2) / (np.sum(w) ** 2)
    )

    # Select the bin nearest the target frequency.
    index = int(np.argmin(np.abs(frequency_hz - target_frequency_hz)))
    actual_frequency_hz = float(frequency_hz[index])
    noise_psd_w_per_hz = float(power_psd_w_per_hz[index])

    # Noise power in the effective bandwidth of one bin.
    noise_power_watts = noise_psd_w_per_hz * enbw_hz

    if noise_power_watts > 0 and math.isfinite(noise_power_watts):
        noise_power_dbm = 10.0 * math.log10(
            noise_power_watts / 1e-3
        )
    else:
        noise_power_dbm = math.nan

    return (
        frequency_hz,
        power_psd_w_per_hz,
        actual_frequency_hz,
        noise_psd_w_per_hz,
        noise_power_watts,
        noise_power_dbm,
        enbw_hz,
    )


# =====================================================================
# SUBWINDOW SEGMENTATION
# =====================================================================

def split_into_subwindows(
    time_seconds,
    voltage_volts,
    sample_interval_s,
):
    """
    Yield non-overlapping subwindows from one long acquisition.

    Any incomplete samples at the end of the record are discarded.
    """
    sample_rate_hz = 1.0 / sample_interval_s
    samples_per_window = int(
        round(SUBWINDOW_DURATION_SECONDS * sample_rate_hz)
    )

    if samples_per_window < 4:
        raise RuntimeError(
            "Subwindow is too short for the current sample rate."
        )

    n_windows = voltage_volts.size // samples_per_window

    if n_windows == 0:
        raise RuntimeError(
            "Acquisition is shorter than one subwindow."
        )

    for i in range(n_windows):
        first = i * samples_per_window
        last = first + samples_per_window
        yield (
            i,
            time_seconds[first:last],
            voltage_volts[first:last],
            sample_rate_hz,
        )


# =====================================================================
# DATA STORAGE
# =====================================================================

CSV_COLUMNS = [
    "long_acquisition_number",
    "subwindow_number",
    "global_pair_number",
    "zero_event_number",
    "timestamp_utc",
    "subwindow_start_seconds",
    "subwindow_stop_seconds",
    "subwindow_duration_seconds",
    "samples_in_subwindow",
    "sample_rate_hz",
    "v3_mean_volts",
    "abs_v3_mean_volts",
    "zero_mean_threshold_volts",
    "within_zero_window",
    "requested_frequency_hz",
    "actual_fft_bin_frequency_hz",
    "frequency_error_hz",
    "noise_psd_w_per_hz",
    "noise_power_watts",
    "noise_power_dbm",
    "enbw_hz",
    "spectrum_file",
]


def create_csv(path: Path):
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def append_csv_row(path: Path, row):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
        f.flush()


def save_zero_spectrum(
    spectra_directory,
    long_acquisition_number,
    subwindow_number,
    global_pair_number,
    zero_event_number,
    timestamp,
    v3_mean_volts,
    frequency_hz,
    power_psd_w_per_hz,
    enbw_hz,
):
    safe_ts = (
        timestamp
        .replace(":", "-")
        .replace(".", "-")
        .replace("+", "_")
    )
    filename = (
        f"zero_{zero_event_number:08d}_"
        f"pair_{global_pair_number:08d}_"
        f"acq_{long_acquisition_number:06d}_"
        f"window_{subwindow_number:04d}_"
        f"{safe_ts}.npz"
    )

    psd_dbm_per_hz = np.full(
        power_psd_w_per_hz.shape, np.nan, dtype=np.float64
    )
    valid = np.isfinite(power_psd_w_per_hz) & (power_psd_w_per_hz > 0)
    psd_dbm_per_hz[valid] = 10.0 * np.log10(
        power_psd_w_per_hz[valid] / 1e-3
    )

    np.savez_compressed(
        spectra_directory / filename,
        timestamp_utc=np.array(timestamp),
        long_acquisition_number=np.array(long_acquisition_number),
        subwindow_number=np.array(subwindow_number),
        global_pair_number=np.array(global_pair_number),
        zero_event_number=np.array(zero_event_number),
        v3_mean_volts=np.array(v3_mean_volts),
        zero_mean_threshold_volts=np.array(ZERO_MEAN_THRESHOLD),
        frequency_hz=frequency_hz,
        power_psd_w_per_hz=power_psd_w_per_hz,
        power_psd_dbm_per_hz=psd_dbm_per_hz,
        enbw_hz=np.array(enbw_hz),
        reference_impedance_ohms=np.array(REFERENCE_IMPEDANCE_OHMS),
    )

    return filename


def save_settings(
    output_directory,
    instrument_id,
    actual_sample_rate_hz,
    samples_per_subwindow,
    actual_subwindow_duration_s,
    enbw_hz,
):
    settings = {
        "instrument": instrument_id,
        "oscilloscope_ip": OSCILLOSCOPE_IP,
        "visa_resource": VISA_RESOURCE,
        "channel": CHANNEL,
        "physical_channel_definition": "V3(t) = V1(t) - V2(t)",
        "alignment_metric": "abs(arithmetic mean of V3 in each subwindow)",
        "alignment_metric_rationale": (
            "Mean captures DC/slow imbalance and is independent of "
            "the noise fluctuations at the target frequency. "
            "Using total RMS would mix the alignment condition with "
            "the noise being measured and bias the result downward."
        ),
        "zero_condition":
            "abs(mean(V3)) <= zero_mean_threshold_volts",
        "zero_mean_threshold_volts": ZERO_MEAN_THRESHOLD,
        "target_frequency_hz": TARGET_FREQUENCY_HZ,
        "requested_subwindow_duration_seconds":
            SUBWINDOW_DURATION_SECONDS,
        "actual_subwindow_duration_seconds":
            actual_subwindow_duration_s,
        "samples_per_subwindow": samples_per_subwindow,
        "actual_sample_rate_hz": actual_sample_rate_hz,
        "requested_long_record_duration_seconds":
            LONG_RECORD_DURATION_SECONDS,
        "fft_method": "scipy.signal.periodogram",
        "fft_window": "Hann",
        "fft_detrend": "constant (subtracts arithmetic mean)",
        "fft_scaling": "density (V^2/Hz)",
        "enbw_hz": enbw_hz,
        "reference_impedance_ohms": REFERENCE_IMPEDANCE_OHMS,
        "save_zero_spectra": SAVE_ZERO_SPECTRA,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_directory / "capture_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )


# =====================================================================
# MAIN LOOP
# =====================================================================

def main():
    output_directory = Path(OUTPUT_DIRECTORY).expanduser().resolve()
    spectra_directory = output_directory / "zero_spectra"

    output_directory.mkdir(parents=True, exist_ok=True)
    if SAVE_ZERO_SPECTRA:
        spectra_directory.mkdir(parents=True, exist_ok=True)

    all_pairs_path = output_directory / "all_pairs.csv"
    zero_pairs_path = output_directory / "zero_pairs.csv"

    create_csv(all_pairs_path)
    create_csv(zero_pairs_path)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    resource_manager = None
    scope = None

    try:
        print(f"Connecting to oscilloscope at {OSCILLOSCOPE_IP} ...")
        resource_manager = pyvisa.ResourceManager(PYVISA_BACKEND)
        scope = resource_manager.open_resource(VISA_RESOURCE)
        scope.timeout = int(VISA_TIMEOUT_SECONDS * 1000)
        scope.chunk_size = 10 * 1024 * 1024

        instrument_id = scope.query("*IDN?").strip()
        print(f"Connected to: {instrument_id}")

        configure_scope(scope)

        print(f"\nChannel {CHANNEL} = V3(t) = V1(t) - V2(t)")
        print(f"Alignment metric    : abs(mean(V3))")
        print(f"Alignment threshold : abs(mean(V3)) <= "
              f"{ZERO_MEAN_THRESHOLD * 1e3:.1f} mV")
        print(f"Target frequency    : {TARGET_FREQUENCY_HZ / 1e6:.3f} MHz")
        print(f"Subwindow duration  : "
              f"{SUBWINDOW_DURATION_SECONDS * 1e6:.1f} us")
        print(f"Output directory    : {output_directory}")
        print("Press Ctrl+C to stop.\n")

        long_acquisition_number = 0
        global_pair_number = 0
        zero_event_number = 0
        settings_saved = False

        while not stop_requested:
            try:
                long_acquisition_number += 1

                # Freeze one complete acquisition before reading data.
                scope.query(":SINGle;*OPC?")

                (
                    time_seconds,
                    voltage_volts,
                    sample_interval_s,
                ) = read_channel_waveform(scope)

                time_seconds, voltage_volts = remove_invalid(
                    time_seconds, voltage_volts
                )

                if voltage_volts.size == 0:
                    raise RuntimeError(
                        "Channel 3 contains no valid samples."
                    )

                acquisition_timestamp = (
                    datetime.now(timezone.utc).isoformat()
                )

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
                    global_pair_number += 1

                    # --- Alignment metric (independent of noise) ---
                    v3_mean = float(np.mean(window_voltage))
                    abs_v3_mean = abs(v3_mean)

                    # --- Noise measurement ---
                    (
                        frequency_hz,
                        power_psd_w_per_hz,
                        actual_frequency_hz,
                        noise_psd_w_per_hz,
                        noise_power_watts,
                        noise_power_dbm,
                        enbw_hz,
                    ) = calculate_noise_at_frequency(
                        voltage_volts=window_voltage,
                        sample_rate_hz=sample_rate_hz,
                        target_frequency_hz=TARGET_FREQUENCY_HZ,
                        impedance_ohms=REFERENCE_IMPEDANCE_OHMS,
                    )

                    actual_subwindow_duration = (
                        window_voltage.size / sample_rate_hz
                    )

                    if not settings_saved:
                        save_settings(
                            output_directory=output_directory,
                            instrument_id=instrument_id,
                            actual_sample_rate_hz=sample_rate_hz,
                            samples_per_subwindow=window_voltage.size,
                            actual_subwindow_duration_s=
                                actual_subwindow_duration,
                            enbw_hz=enbw_hz,
                        )
                        print(
                            f"Actual sample rate      : "
                            f"{sample_rate_hz / 1e6:.0f} MSa/s"
                        )
                        print(
                            f"Samples per subwindow   : "
                            f"{window_voltage.size}"
                        )
                        print(
                            f"Actual subwindow        : "
                            f"{actual_subwindow_duration * 1e6:.3f} us"
                        )
                        print(
                            f"Effective noise BW      : "
                            f"{enbw_hz / 1e3:.2f} kHz\n"
                        )
                        settings_saved = True

                    # --- Alignment decision ---
                    aligned = (
                        math.isfinite(abs_v3_mean)
                        and abs_v3_mean <= ZERO_MEAN_THRESHOLD
                    )

                    spectrum_filename = ""

                    if aligned:
                        zero_event_number += 1

                        if SAVE_ZERO_SPECTRA:
                            spectrum_filename = save_zero_spectrum(
                                spectra_directory=spectra_directory,
                                long_acquisition_number=
                                    long_acquisition_number,
                                subwindow_number=subwindow_index + 1,
                                global_pair_number=global_pair_number,
                                zero_event_number=zero_event_number,
                                timestamp=acquisition_timestamp,
                                v3_mean_volts=v3_mean,
                                frequency_hz=frequency_hz,
                                power_psd_w_per_hz=power_psd_w_per_hz,
                                enbw_hz=enbw_hz,
                            )

                    row = [
                        long_acquisition_number,
                        subwindow_index + 1,
                        global_pair_number,
                        zero_event_number if aligned else "",
                        acquisition_timestamp,
                        f"{window_time[0]:.12e}",
                        f"{window_time[-1]:.12e}",
                        f"{actual_subwindow_duration:.12e}",
                        window_voltage.size,
                        f"{sample_rate_hz:.12e}",
                        f"{v3_mean:.12e}",
                        f"{abs_v3_mean:.12e}",
                        f"{ZERO_MEAN_THRESHOLD:.12e}",
                        aligned,
                        f"{TARGET_FREQUENCY_HZ:.12e}",
                        f"{actual_frequency_hz:.12e}",
                        f"{actual_frequency_hz - TARGET_FREQUENCY_HZ:.12e}",
                        f"{noise_psd_w_per_hz:.12e}",
                        f"{noise_power_watts:.12e}",
                        f"{noise_power_dbm:.9f}",
                        f"{enbw_hz:.12e}",
                        spectrum_filename,
                    ]

                    append_csv_row(all_pairs_path, row)

                    if aligned:
                        append_csv_row(zero_pairs_path, row)
                        print(
                            f"ALIGNED #{zero_event_number}: "
                            f"pair={global_pair_number}, "
                            f"|mean(V3)|={abs_v3_mean * 1e3:.4f} mV, "
                            f"noise={noise_power_dbm:.3f} dBm at "
                            f"{actual_frequency_hz / 1e6:.6f} MHz"
                        )
                    else:
                        print(
                            f"\rPair {global_pair_number}: "
                            f"|mean(V3)|={abs_v3_mean * 1e3:.4f} mV, "
                            f"noise={noise_power_dbm:.3f} dBm  "
                            f"[aligned pairs: {zero_event_number}]",
                            end="",
                            flush=True,
                        )

                if ACQUISITION_INTERVAL_SECONDS > 0:
                    time.sleep(ACQUISITION_INTERVAL_SECONDS)

            except pyvisa.errors.VisaIOError as error:
                print(
                    f"\nVISA error: {error}",
                    file=sys.stderr,
                )
                time.sleep(1.0)

            except (ValueError, RuntimeError) as error:
                print(
                    f"\nProcessing error: {error}",
                    file=sys.stderr,
                )
                time.sleep(1.0)

        print(
            f"\nStopped after {long_acquisition_number} acquisitions."
        )
        print(f"Total ordered pairs : {global_pair_number}")
        print(f"Aligned pairs saved : {zero_event_number}")
        print(f"Results written to  : {output_directory}")

    finally:
        if scope is not None:
            scope.close()
        if resource_manager is not None:
            resource_manager.close()


if __name__ == "__main__":
    main()