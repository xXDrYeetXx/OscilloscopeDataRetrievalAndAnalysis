#!/usr/bin/env python3
"""
Fast acquisition of Channel 3 near-zero noise measurements.

Speed improvements
------------------
1. Transfers waveform data as 16-bit binary rather than ASCII.
2. Processes all subwindows from an acquisition as a matrix.
3. Calculates only the Fourier bin nearest 10 MHz.
4. Maintains the running mean using a cumulative sum.
5. Writes CSV rows in batches.
6. Uses a vectorized acquisition-cluster bootstrap.
7. Prints one status line per long acquisition.

The measurement itself is unchanged:

    Channel 3 = V1 - V2

    qualify when:
        abs(mean(V3)) <= ZERO_MEAN_THRESHOLD

    noise PSD:
        Hann-windowed, mean-detrended, one-sided periodogram ordinate

    noise power:
        PSD at target bin * Hann ENBW

All noise powers are averaged in linear watts before conversion to dBm.

Convergence requires both:
1. A sufficiently narrow acquisition-cluster bootstrap interval.
2. Agreement between two recent non-overlapping acquisition blocks.

No positive/negative V3 balancing is performed.

Fixes applied (v2)
------------------
- Split :SINGle and *OPC? to eliminate the compound-command race
  condition that caused most empty-waveform errors.
- Added a hardware-status poll loop after :SINGle to confirm the
  scope has actually stopped before reading waveform data.
- Moved waveform format/points configuration to AFTER the acquisition
  completes, so the scope honours the settings at readout time.
- Removed redundant waveform format commands from
  read_channel_waveform_binary(); they are now set once in the main
  loop before each DATA? query.
- Added a per-acquisition sample-count debug line to stderr so
  transfer problems are immediately visible.

Changes (v3)
------------
- Added console input prompts for laser amplifier current (A) and
  beam power (mW) before acquisition begins.
- Both values are saved into converged_result.json.
- A summary.json file is written containing just the two experimental
  parameters and the final mean dBm.

Change (v4)
-----------
- Added verification that the vertical scale is set to the expected
  value (10 mV/div) and printed to console for user confirmation.
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
from scipy.signal import get_window


# =====================================================================
# USER CONFIGURATION
# =====================================================================

OSCILLOSCOPE_IP = "192.168.137.113"
VISA_RESOURCE = f"TCPIP0::{OSCILLOSCOPE_IP}::inst0::INSTR"

CHANNEL = 3

# Alignment.
ZERO_MEAN_THRESHOLD = 5e-3          # 5 mV

# Spectral measurement.
TARGET_FREQUENCY_HZ = 10e6
SUBWINDOW_DURATION_SECONDS = 1.5e-6
LONG_RECORD_DURATION_SECONDS = 150e-6
REQUESTED_SAMPLE_RATE_HZ = 1e9
REFERENCE_IMPEDANCE_OHMS = 50.0

REQUESTED_ACQUISITION_POINTS = round(
    LONG_RECORD_DURATION_SECONDS
    * REQUESTED_SAMPLE_RATE_HZ
)

# Scope settings.
CHANNEL_VERTICAL_SCALE_VOLTS = 10e-3   # 10 mV/div
CHANNEL_VERTICAL_OFFSET_VOLTS = 0.0

# Communication.
VISA_TIMEOUT_SECONDS = 60.0
PYVISA_BACKEND = "@py"
ACQUISITION_INTERVAL_SECONDS = 0.0

# How many times to poll the scope's operating-register before giving
# up waiting for the acquisition to stop.  Each poll sleeps
# ACQ_POLL_SLEEP_SECONDS, so the total wait ceiling is
# ACQ_POLL_MAX_TRIES * ACQ_POLL_SLEEP_SECONDS seconds.
ACQ_POLL_MAX_TRIES = 200
ACQ_POLL_SLEEP_SECONDS = 0.05        # 50 ms per poll → 10 s ceiling

# Output.
OUTPUT_DIRECTORY = "v3_converged_noise_data"

# =====================================================================
# CONVERGENCE CONFIGURATION
# =====================================================================

MIN_QUALIFYING_ACQUISITIONS = 100
CHECK_INTERVAL_ACQUISITIONS = 20

TARGET_RELATIVE_CI_HALF_WIDTH = 0.05    # 5 %

STABILITY_BLOCK_ACQUISITIONS = 50
STABILITY_TOLERANCE = 0.05              # 5 %

CONVERGENCE_STREAK_REQUIRED = 3

BOOTSTRAP_REPETITIONS = 2000
FINAL_BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_CHUNK_SIZE = 256

RANDOM_SEED = 42

MAX_LONG_ACQUISITIONS = 2000


# =====================================================================
# PROGRAM STATE
# =====================================================================

stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


# =====================================================================
# OUTPUT COLUMNS
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
    "qualifying_acquisitions",
    "qualifying_pairs",
    "mean_watts",
    "mean_dbm",
    "ci_low_watts",
    "ci_high_watts",
    "ci_low_dbm",
    "ci_high_dbm",
    "relative_ci_half_width",
    "relative_ci_half_width_percent",
    "preceding_block_mean_watts",
    "recent_block_mean_watts",
    "block_relative_change",
    "block_relative_change_percent",
    "precision_passed",
    "stability_passed",
    "passed",
    "streak",
    "timestamp_utc",
]


# =====================================================================
# GENERAL UTILITIES
# =====================================================================

def watts_to_dbm(watts: float) -> float:
    """Convert positive power in watts to dBm."""
    if math.isfinite(watts) and watts > 0:
        return 10.0 * math.log10(watts / 1e-3)
    return math.nan


def get_next_run_directory(base_directory: Path) -> Path:
    """Create and return the next Run N directory."""
    base_directory.mkdir(parents=True, exist_ok=True)

    highest_run_number = 0

    for path in base_directory.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"Run (\d+)", path.name)
        if match:
            highest_run_number = max(
                highest_run_number,
                int(match.group(1)),
            )

    run_directory = base_directory / f"Run {highest_run_number + 1}"
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def create_csv(path: Path, columns: list[str]) -> None:
    """Create a CSV with a header if it does not exist."""
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(columns)


def append_csv_rows(path: Path, rows: list[list]) -> None:
    """Append several rows with one file-open operation."""
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(rows)
        file.flush()


def safe_percent(value: float) -> str:
    if math.isfinite(value):
        return f"{value * 100:.3f}%"
    return "unavailable"


def prompt_float(prompt: str) -> float:
    """Prompt the user for a float value, retrying on invalid input."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  Please enter a numeric value.")


def prompt_squeezing_device_present(prompt: str) -> str:
    """Prompt for y/n/blank and return True, False, or Unknown."""
    while True:
        response = input(prompt).strip().lower()

        if response == "y":
            return "True"
        if response == "n":
            return "False"
        if response == "":
            return "Unknown"

        print("  Please enter y, n, or press Enter for Unknown.")


# =====================================================================
# OSCILLOSCOPE COMMUNICATION
# =====================================================================

def query_float(scope, command: str) -> float:
    response = scope.query(command).strip()
    return float(response.split()[-1])


def configure_scope(scope) -> None:
    """Configure the scope for repeated Channel 3 acquisitions."""
    scope.write("*CLS")
    scope.write(":SYSTem:HEADer OFF")

    scope.write(f":CHANnel{CHANNEL}:DISPlay ON")
    scope.write(f":CHANnel{CHANNEL}:SCALe {CHANNEL_VERTICAL_SCALE_VOLTS}")
    # Verify the scale was actually set
    scale_v = query_float(scope, f":CHANnel{CHANNEL}:SCALe?")
    print(
        f"Channel {CHANNEL} vertical scale set to {scale_v*1e3:.3f} mV/div "
        f"(requested {CHANNEL_VERTICAL_SCALE_VOLTS*1e3:.3f} mV/div)"
    )

    scope.write(f":CHANnel{CHANNEL}:OFFSet {CHANNEL_VERTICAL_OFFSET_VOLTS}")
    # Verify the offset
    offset_v = query_float(scope, f":CHANnel{CHANNEL}:OFFSet?")
    print(
        f"Channel {CHANNEL} vertical offset set to {offset_v*1e3:.3f} mV "
        f"(requested {CHANNEL_VERTICAL_OFFSET_VOLTS*1e3:.3f} mV)"
    )

    scope.write(f":TIMebase:RANGe {LONG_RECORD_DURATION_SECONDS}")
    scope.write(f":ACQuire:SRATe {REQUESTED_SAMPLE_RATE_HZ}")
    scope.write(":WAVeform:STReaming OFF")
    scope.write(f":ACQuire:POINts:ANALog {REQUESTED_ACQUISITION_POINTS}")

    # Waveform source is fixed for the whole run.
    scope.write(f":WAVeform:SOURce CHANnel{CHANNEL}")

    scope.query("*OPC?")


def wait_for_acquisition_complete(scope) -> None:
    """
    Poll the scope's Operating Condition Register until bit 3
    (Measuring / Running) clears, confirming the acquisition has
    stopped and waveform memory is readable.

    Falls back gracefully: if the register query is not supported the
    function simply waits for *OPC? instead.
    """
    for _ in range(ACQ_POLL_MAX_TRIES):
        try:
            raw = scope.query(":OPERegister:CONDition?").strip()
            condition = int(raw)
            # Bit 3 (value 8) means the scope is still running/measuring.
            if not (condition & 0x0008):
                return
        except Exception:
            break
        time.sleep(ACQ_POLL_SLEEP_SECONDS)

    # Final safety net.
    scope.query("*OPC?")


def configure_waveform_readout(scope) -> None:
    """
    Assert the waveform format/points settings immediately before each
    DATA? query.  Calling this AFTER the acquisition has stopped
    ensures the scope honours the settings at readout time.
    """
    scope.write(f":WAVeform:SOURce CHANnel{CHANNEL}")
    scope.write(":WAVeform:FORMat WORD")
    scope.write(":WAVeform:BYTeorder LSBF")
    scope.write(":WAVeform:POINts:MODE RAW")
    scope.write(f":WAVeform:POINts {REQUESTED_ACQUISITION_POINTS}")
    scope.query("*OPC?")


def read_channel_waveform_binary(scope):
    """
    Download Channel 3 using signed 16-bit binary values.

    Waveform format/points settings are assumed to have been asserted
    by the caller (configure_waveform_readout) immediately before this
    function is called.  The calibrated voltage is reconstructed as:

        voltage = (raw - y_reference) * y_increment + y_origin
    """
    sample_interval_s = query_float(scope, ":WAVeform:XINCrement?")
    if sample_interval_s <= 0.0:
        sample_interval_s = 1.0 / REQUESTED_SAMPLE_RATE_HZ

    time_origin_s = query_float(scope, ":WAVeform:XORigin?")
    y_increment   = query_float(scope, ":WAVeform:YINCrement?")
    y_origin      = query_float(scope, ":WAVeform:YORigin?")
    y_reference   = query_float(scope, ":WAVeform:YREFerence?")

    raw_values = scope.query_binary_values(
        ":WAVeform:DATA?",
        datatype="h",
        is_big_endian=False,
        container=np.array,
        expect_termination=False,
    )

    raw_values = np.asarray(raw_values, dtype=np.float64)

    # ---- DEBUG: report sample count so transfer problems are visible ----
    print(
        f"  [DEBUG] raw samples={raw_values.size}  "
        f"y_inc={y_increment:.6e}  y_orig={y_origin:.6e}",
        file=sys.stderr,
        flush=True,
    )
    # ---------------------------------------------------------------------

    if raw_values.size == 0:
        raise RuntimeError(
            "No binary waveform samples were returned."
        )

    voltage_volts = (raw_values - y_reference) * y_increment + y_origin

    valid = np.isfinite(voltage_volts) & (np.abs(voltage_volts) < 1e30)

    if not np.all(valid):
        raise RuntimeError(
            "Waveform contains invalid samples. "
            "The acquisition was rejected rather than joining "
            "samples across a data hole."
        )

    return {
        "voltage_volts":     voltage_volts,
        "sample_interval_s": sample_interval_s,
        "sample_rate_hz":    1.0 / sample_interval_s,
        "time_origin_s":     time_origin_s,
        "y_increment":       y_increment,
        "y_origin":          y_origin,
        "y_reference":       y_reference,
    }


# =====================================================================
# FAST VECTORIZED SPECTRAL ANALYSIS
# =====================================================================

class SingleBinAnalyzer:
    """
    Reusable Hann-window single-frequency periodogram analyzer.

    This computes only the FFT bin nearest the target frequency. Its
    periodogram normalization matches a one-sided Hann-windowed
    periodogram with scaling="density".
    """

    def __init__(
        self,
        samples_per_window: int,
        sample_rate_hz: float,
        target_frequency_hz: float,
        impedance_ohms: float,
    ):
        self.samples_per_window = samples_per_window
        self.sample_rate_hz     = sample_rate_hz
        self.impedance_ohms     = impedance_ohms

        self.window = get_window(
            "hann",
            samples_per_window,
            fftbins=True,
        ).astype(np.float64)

        self.window_sum        = float(np.sum(self.window))
        self.window_square_sum = float(np.sum(self.window ** 2))

        self.bin_spacing_hz = sample_rate_hz / samples_per_window

        self.bin_index = int(
            round(target_frequency_hz / self.bin_spacing_hz)
        )

        if (
            self.bin_index <= 0
            or self.bin_index >= samples_per_window // 2
        ):
            raise ValueError(
                "Target frequency is not a valid interior "
                "one-sided Fourier bin."
            )

        self.actual_frequency_hz = self.bin_index * self.bin_spacing_hz

        sample_indices = np.arange(samples_per_window, dtype=np.float64)

        # exp(-j·2π·k·n/N) including the Hann window.
        self.kernel = self.window * np.exp(
            -2j * np.pi * self.bin_index * sample_indices / samples_per_window
        )

        self.enbw_hz = float(
            sample_rate_hz
            * self.window_square_sum
            / self.window_sum ** 2
        )

        # One-sided density normalization for a non-DC, non-Nyquist bin:
        #   PSD = 2·|DFT|² / (fs·Σ(window²))
        self.psd_scale = 2.0 / (sample_rate_hz * self.window_square_sum)

    def analyze(
        self,
        windows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Analyze a matrix shaped:

            number_of_windows × samples_per_window

        Returns:
            power PSD in W/Hz
            ENBW-integrated power in W
        """
        if windows.ndim != 2:
            raise ValueError("Windows must be a two-dimensional matrix.")

        if windows.shape[1] != self.samples_per_window:
            raise ValueError("Unexpected samples per subwindow.")

        # Match detrend="constant": subtract each row's mean.
        centered = windows - np.mean(windows, axis=1, keepdims=True)

        # Matrix–vector product: one DFT coefficient per window.
        coefficients = centered @ self.kernel

        voltage_psd        = self.psd_scale * np.abs(coefficients) ** 2
        power_psd_w_per_hz = voltage_psd / self.impedance_ohms
        noise_power_watts  = power_psd_w_per_hz * self.enbw_hz

        return (
            np.asarray(power_psd_w_per_hz, dtype=np.float64),
            np.asarray(noise_power_watts,   dtype=np.float64),
        )


def prepare_window_matrix(
    voltage_volts: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, int]:
    """Convert one long waveform into a subwindow matrix."""
    samples_per_window = int(
        round(SUBWINDOW_DURATION_SECONDS * sample_rate_hz)
    )

    if samples_per_window < 4:
        raise RuntimeError("Subwindow contains too few samples.")

    number_of_windows = voltage_volts.size // samples_per_window

    if number_of_windows == 0:
        raise RuntimeError(
            "Long acquisition contains no complete subwindow."
        )

    samples_to_use = number_of_windows * samples_per_window
    windows = voltage_volts[:samples_to_use].reshape(
        number_of_windows, samples_per_window
    )

    return windows, samples_per_window


# =====================================================================
# CLUSTER STATISTICS
# =====================================================================

def cluster_summaries(
    acquisition_clusters: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return power sums and observation counts for each cluster."""
    sums = np.fromiter(
        (float(np.sum(c)) for c in acquisition_clusters),
        dtype=np.float64,
        count=len(acquisition_clusters),
    )
    counts = np.fromiter(
        (int(c.size) for c in acquisition_clusters),
        dtype=np.int64,
        count=len(acquisition_clusters),
    )
    return sums, counts


def pooled_mean_from_summaries(
    cluster_sums: np.ndarray,
    cluster_counts: np.ndarray,
) -> float:
    total_count = int(np.sum(cluster_counts))
    if total_count <= 0:
        return math.nan
    return float(np.sum(cluster_sums) / total_count)


def pooled_mean_watts(
    acquisition_clusters: list[np.ndarray],
) -> float:
    if not acquisition_clusters:
        return math.nan
    sums, counts = cluster_summaries(acquisition_clusters)
    return pooled_mean_from_summaries(sums, counts)


def cluster_bootstrap_ci(
    acquisition_clusters: list[np.ndarray],
    repetitions: int,
    random_seed: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Vectorized cluster bootstrap.

    Whole acquisitions are resampled.  Calculations use only each
    cluster's sum and count, avoiding repeated array concatenation.
    """
    number_of_clusters = len(acquisition_clusters)

    if number_of_clusters == 0:
        return math.nan, math.nan

    cluster_sums, cluster_counts = cluster_summaries(acquisition_clusters)

    if number_of_clusters == 1:
        value = pooled_mean_from_summaries(cluster_sums, cluster_counts)
        return value, value

    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(repetitions, dtype=np.float64)
    completed = 0

    while completed < repetitions:
        batch_size = min(BOOTSTRAP_CHUNK_SIZE, repetitions - completed)

        selected_indices = rng.integers(
            low=0,
            high=number_of_clusters,
            size=(batch_size, number_of_clusters),
        )

        sampled_sums   = np.sum(cluster_sums[selected_indices],   axis=1)
        sampled_counts = np.sum(cluster_counts[selected_indices], axis=1)

        bootstrap_means[completed:completed + batch_size] = (
            sampled_sums / sampled_counts
        )
        completed += batch_size

    alpha = 1.0 - confidence_level
    low, high = np.percentile(
        bootstrap_means,
        [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
    )
    return float(low), float(high)


def compare_recent_blocks(
    acquisition_clusters: list[np.ndarray],
    block_size: int,
) -> tuple[float, float, float]:
    """Compare two adjacent, non-overlapping acquisition blocks."""
    required_clusters = 2 * block_size

    if len(acquisition_clusters) < required_clusters:
        return math.nan, math.nan, math.nan

    preceding_clusters = acquisition_clusters[-required_clusters:-block_size]
    recent_clusters    = acquisition_clusters[-block_size:]

    preceding_mean = pooled_mean_watts(preceding_clusters)
    recent_mean    = pooled_mean_watts(recent_clusters)

    if not math.isfinite(preceding_mean) or preceding_mean <= 0:
        return preceding_mean, recent_mean, math.nan

    relative_change = abs(recent_mean - preceding_mean) / preceding_mean
    return preceding_mean, recent_mean, relative_change


def check_convergence(
    acquisition_clusters: list[np.ndarray],
    check_number: int,
) -> dict[str, float | bool]:
    current_mean_watts = pooled_mean_watts(acquisition_clusters)

    ci_low_watts, ci_high_watts = cluster_bootstrap_ci(
        acquisition_clusters,
        repetitions=BOOTSTRAP_REPETITIONS,
        random_seed=RANDOM_SEED + check_number,
    )

    relative_ci_half_width = (
        (ci_high_watts - ci_low_watts) / (2.0 * current_mean_watts)
        if current_mean_watts > 0
        else math.inf
    )

    (
        preceding_mean_watts,
        recent_mean_watts,
        block_relative_change,
    ) = compare_recent_blocks(acquisition_clusters, STABILITY_BLOCK_ACQUISITIONS)

    precision_passed = (
        math.isfinite(relative_ci_half_width)
        and relative_ci_half_width <= TARGET_RELATIVE_CI_HALF_WIDTH
    )

    stability_passed = (
        math.isfinite(block_relative_change)
        and block_relative_change <= STABILITY_TOLERANCE
    )

    return {
        "mean_watts":             current_mean_watts,
        "mean_dbm":               watts_to_dbm(current_mean_watts),
        "ci_low_watts":           ci_low_watts,
        "ci_high_watts":          ci_high_watts,
        "ci_low_dbm":             watts_to_dbm(ci_low_watts),
        "ci_high_dbm":            watts_to_dbm(ci_high_watts),
        "relative_ci_half_width": relative_ci_half_width,
        "preceding_mean_watts":   preceding_mean_watts,
        "recent_mean_watts":      recent_mean_watts,
        "block_relative_change":  block_relative_change,
        "precision_passed":       precision_passed,
        "stability_passed":       stability_passed,
        "passed":                 precision_passed and stability_passed,
    }


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    global stop_requested

    base_directory = Path(OUTPUT_DIRECTORY).expanduser().resolve()
    run_directory  = get_next_run_directory(base_directory)

    pair_csv        = run_directory / "zero_pairs.csv"
    convergence_csv = run_directory / "convergence_history.csv"
    result_json     = run_directory / "converged_result.json"
    summary_json    = run_directory / "summary.json"

    create_csv(pair_csv,        PAIR_COLUMNS)
    create_csv(convergence_csv, CONVERGENCE_COLUMNS)

    signal.signal(signal.SIGINT,  request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    resource_manager = None
    scope            = None

    # Statistical state.
    acquisition_clusters: list[np.ndarray] = []
    cumulative_power_watts       = 0.0
    qualifying_pair_count        = 0
    long_acquisition_number      = 0
    qualifying_acquisition_count = 0
    positive_count               = 0
    negative_count               = 0

    convergence_check_number       = 0
    last_checked_acquisition_count = 0
    convergence_streak             = 0
    converged                      = False

    analyzer: SingleBinAnalyzer | None = None
    analyzer_sample_rate_hz    = math.nan
    analyzer_samples_per_window = 0

    instrument_id            = ""
    final_convergence_result = None

    started_utc = datetime.now(timezone.utc).isoformat()

    try:
        print("=" * 72)
        print("FAST V3 NEAR-ZERO NOISE ACQUISITION")
        print("=" * 72)
        print(f"Run directory          : {run_directory}")
        print(
            f"Qualification          : "
            f"|mean(V3)| <= {ZERO_MEAN_THRESHOLD * 1e3:.3f} mV"
        )
        print(f"Target frequency       : {TARGET_FREQUENCY_HZ / 1e6:.3f} MHz")
        print(
            f"Subwindow duration     : "
            f"{SUBWINDOW_DURATION_SECONDS * 1e6:.3f} us"
        )
        print(
            f"Long record duration   : "
            f"{LONG_RECORD_DURATION_SECONDS * 1e6:.3f} us"
        )
        print("Waveform transfer      : 16-bit binary WORD")
        print("Spectral calculation   : vectorized single-bin Hann DFT")
        print("Mean calculation       : running sum in watts")
        print(
            f"Precision target       : "
            f"{TARGET_RELATIVE_CI_HALF_WIDTH * 100:.2f}%"
        )
        print(f"Stability target       : {STABILITY_TOLERANCE * 100:.2f}%")
        print()

        # ------------------------------------------------------------------
        # USER INPUT: experimental parameters
        # ------------------------------------------------------------------
        print("Enter experimental parameters:")
        laser_amplifier_amps = prompt_float(
            "  Laser amplifier current (A) : "
        )
        beam_power_mw = prompt_float(
            "  Beam power (mW)             : "
        )
        squeezing_device_present = prompt_squeezing_device_present(
            "  Is the squeezing device present? [y/n] : "
        )
        additional_notes = input("  Additional Notes: ")
        print()

        print("Press Ctrl+C to stop.")
        print()

        resource_manager = pyvisa.ResourceManager(PYVISA_BACKEND)
        scope = resource_manager.open_resource(VISA_RESOURCE)
        scope.timeout    = int(VISA_TIMEOUT_SECONDS * 1000)
        scope.chunk_size = 16 * 1024 * 1024

        instrument_id = scope.query("*IDN?").strip()
        print(f"Connected to           : {instrument_id}")

        configure_scope(scope)

        while (
            not stop_requested
            and not converged
            and long_acquisition_number < MAX_LONG_ACQUISITIONS
        ):
            acquisition_started = time.perf_counter()

            try:
                long_acquisition_number += 1
                # --------------------------------------------------------------
                # ENSURE THE VERTICAL SCALE IS LOCKED TO 10 mV/div EVERY LOOP
                # --------------------------------------------------------------
                # 1. Make sure the channel display is ON (required for scale changes)
                scope.write(f":CHANnel{CHANNEL}:DISPlay ON")
                scope.query("*OPC?")

                # 2. Turn off any autoscaling / autoset that could overwrite our value
                scope.write(f":CHANnel{CHANNEL}:SCALe:AUTO OFF")
                scope.write(":AUToset OFF")
                scope.query("*OPC?")

                # 3. Set the scale (in volts/div – the SCPI unit expects volts)
                scope.write(f":CHANnel{CHANNEL}:SCALe {CHANNEL_VERTICAL_SCALE_VOLTS}")  # 10 mV = 0.01 V
                scope.write(f":CHANnel{CHANNEL}:OFFSet {CHANNEL_VERTICAL_OFFSET_VOLTS}")  # normally 0 V
                scope.query("*OPC?")

                # --------------------------------------------------------------
                # Now proceed with the normal acquisition sequence
                # --------------------------------------------------------------

                scope.write(":SINGle")

                # ----------------------------------------------------------
                # FIX 2: Poll the Operating Condition Register until the
                # scope has actually stopped, then call *OPC? as a safety
                # net.
                # ----------------------------------------------------------
                wait_for_acquisition_complete(scope)
                configure_waveform_readout(scope)

                waveform = read_channel_waveform_binary(scope)

                voltage_volts     = waveform["voltage_volts"]
                sample_rate_hz    = waveform["sample_rate_hz"]
                sample_interval_s = waveform["sample_interval_s"]
                time_origin_s     = waveform["time_origin_s"]

                windows, samples_per_window = prepare_window_matrix(
                    voltage_volts, sample_rate_hz
                )

                # Build or rebuild analyzer only if sampling parameters
                # change.
                if (
                    analyzer is None
                    or samples_per_window != analyzer_samples_per_window
                    or not math.isclose(
                        sample_rate_hz,
                        analyzer_sample_rate_hz,
                        rel_tol=1e-12,
                        abs_tol=0.0,
                    )
                ):
                    analyzer = SingleBinAnalyzer(
                        samples_per_window=samples_per_window,
                        sample_rate_hz=sample_rate_hz,
                        target_frequency_hz=TARGET_FREQUENCY_HZ,
                        impedance_ohms=REFERENCE_IMPEDANCE_OHMS,
                    )
                    analyzer_sample_rate_hz     = sample_rate_hz
                    analyzer_samples_per_window = samples_per_window

                    print()
                    print(
                        f"Actual sample rate     : "
                        f"{sample_rate_hz / 1e6:.6f} MSa/s"
                    )
                    print(f"Samples/subwindow      : {samples_per_window}")
                    print(
                        f"FFT bin spacing        : "
                        f"{analyzer.bin_spacing_hz / 1e3:.6f} kHz"
                    )
                    print(
                        f"Actual FFT frequency   : "
                        f"{analyzer.actual_frequency_hz / 1e6:.9f} MHz"
                    )
                    print(
                        f"Hann ENBW              : "
                        f"{analyzer.enbw_hz / 1e3:.6f} kHz"
                    )
                    print()

                # Calculate all subwindow means simultaneously.
                v3_means     = np.mean(windows, axis=1)
                abs_v3_means = np.abs(v3_means)

                qualifying_mask    = (
                    np.isfinite(v3_means)
                    & (abs_v3_means <= ZERO_MEAN_THRESHOLD)
                )
                qualifying_indices = np.flatnonzero(qualifying_mask)

                timestamp = datetime.now(timezone.utc).isoformat()

                rows_to_write:   list[list]  = []
                accepted_powers: list[float] = []

                if qualifying_indices.size > 0:
                    qualifying_windows = windows[qualifying_indices]

                    power_psd_w_per_hz, noise_power_watts = analyzer.analyze(
                        qualifying_windows
                    )

                    valid_power = (
                        np.isfinite(power_psd_w_per_hz)
                        & (power_psd_w_per_hz > 0)
                        & np.isfinite(noise_power_watts)
                        & (noise_power_watts > 0)
                    )

                    qualifying_indices  = qualifying_indices[valid_power]
                    power_psd_w_per_hz  = power_psd_w_per_hz[valid_power]
                    noise_power_watts   = noise_power_watts[valid_power]

                    for local_index, window_index in enumerate(
                        qualifying_indices
                    ):
                        power_watts  = float(noise_power_watts[local_index])
                        psd_w_per_hz = float(power_psd_w_per_hz[local_index])
                        v3_mean      = float(v3_means[window_index])
                        abs_v3_mean  = abs(v3_mean)

                        if v3_mean > 0:
                            positive_count += 1
                        elif v3_mean < 0:
                            negative_count += 1

                        qualifying_pair_count  += 1
                        cumulative_power_watts += power_watts
                        running_mean_watts      = (
                            cumulative_power_watts / qualifying_pair_count
                        )
                        running_mean_dbm = watts_to_dbm(running_mean_watts)

                        first_sample = int(window_index) * samples_per_window
                        last_sample  = first_sample + samples_per_window - 1

                        start_time_s = (
                            time_origin_s + first_sample * sample_interval_s
                        )
                        stop_time_s = (
                            time_origin_s + last_sample * sample_interval_s
                        )

                        rows_to_write.append([
                            qualifying_pair_count,
                            long_acquisition_number,
                            int(window_index) + 1,
                            timestamp,
                            f"{start_time_s:.12e}",
                            f"{stop_time_s:.12e}",
                            samples_per_window,
                            f"{sample_rate_hz:.12e}",
                            f"{v3_mean:.12e}",
                            f"{abs_v3_mean:.12e}",
                            f"{psd_w_per_hz:.12e}",
                            f"{power_watts:.12e}",
                            f"{watts_to_dbm(power_watts):.9f}",
                            f"{analyzer.actual_frequency_hz:.12e}",
                            f"{analyzer.actual_frequency_hz - TARGET_FREQUENCY_HZ:.12e}",
                            f"{analyzer.enbw_hz:.12e}",
                            f"{running_mean_watts:.12e}",
                            f"{running_mean_dbm:.9f}",
                        ])

                        accepted_powers.append(power_watts)

                # One file-open operation per long acquisition.
                append_csv_rows(pair_csv, rows_to_write)

                if accepted_powers:
                    acquisition_clusters.append(
                        np.asarray(accepted_powers, dtype=np.float64)
                    )
                    qualifying_acquisition_count += 1

                elapsed_seconds = time.perf_counter() - acquisition_started

                current_mean_dbm = (
                    watts_to_dbm(
                        cumulative_power_watts / qualifying_pair_count
                    )
                    if qualifying_pair_count
                    else math.nan
                )

                print(
                    f"\rAcq={long_acquisition_number:5d} | "
                    f"clusters={qualifying_acquisition_count:5d} | "
                    f"pairs={qualifying_pair_count:6d} | "
                    f"accepted={len(accepted_powers):2d}/"
                    f"{windows.shape[0]:2d} | "
                    f"mean={current_mean_dbm:9.4f} dBm | "
                    f"time={elapsed_seconds:6.3f} s | "
                    f"streak={convergence_streak}/"
                    f"{CONVERGENCE_STREAK_REQUIRED}",
                    end="",
                    flush=True,
                )

                enough_new = (
                    qualifying_acquisition_count
                    - last_checked_acquisition_count
                    >= CHECK_INTERVAL_ACQUISITIONS
                )

                if (
                    qualifying_acquisition_count
                    >= MIN_QUALIFYING_ACQUISITIONS
                    and enough_new
                ):
                    convergence_check_number += 1
                    last_checked_acquisition_count = (
                        qualifying_acquisition_count
                    )

                    convergence_result = check_convergence(
                        acquisition_clusters,
                        convergence_check_number,
                    )
                    final_convergence_result = convergence_result

                    if convergence_result["passed"]:
                        convergence_streak += 1
                    else:
                        convergence_streak = 0

                    convergence_row = [
                        convergence_check_number,
                        qualifying_acquisition_count,
                        qualifying_pair_count,
                        f"{convergence_result['mean_watts']:.12e}",
                        f"{convergence_result['mean_dbm']:.9f}",
                        f"{convergence_result['ci_low_watts']:.12e}",
                        f"{convergence_result['ci_high_watts']:.12e}",
                        f"{convergence_result['ci_low_dbm']:.9f}",
                        f"{convergence_result['ci_high_dbm']:.9f}",
                        f"{convergence_result['relative_ci_half_width']:.12e}",
                        f"{convergence_result['relative_ci_half_width'] * 100:.6f}",
                        f"{convergence_result['preceding_mean_watts']:.12e}",
                        f"{convergence_result['recent_mean_watts']:.12e}",
                        (
                            f"{convergence_result['block_relative_change']:.12e}"
                            if math.isfinite(
                                convergence_result["block_relative_change"]
                            )
                            else ""
                        ),
                        (
                            f"{convergence_result['block_relative_change'] * 100:.6f}"
                            if math.isfinite(
                                convergence_result["block_relative_change"]
                            )
                            else ""
                        ),
                        convergence_result["precision_passed"],
                        convergence_result["stability_passed"],
                        convergence_result["passed"],
                        convergence_streak,
                        datetime.now(timezone.utc).isoformat(),
                    ]

                    append_csv_rows(convergence_csv, [convergence_row])

                    print()
                    print(f"\nCHECK #{convergence_check_number}")
                    print(
                        f"  Qualifying clusters : "
                        f"{qualifying_acquisition_count}"
                    )
                    print(f"  Qualifying pairs    : {qualifying_pair_count}")
                    print(
                        f"  Mean                : "
                        f"{convergence_result['mean_dbm']:.5f} dBm"
                    )
                    print(
                        f"  Bootstrap 95% CI    : "
                        f"{convergence_result['ci_low_dbm']:.5f} to "
                        f"{convergence_result['ci_high_dbm']:.5f} dBm"
                    )
                    print(
                        f"  CI half-width       : "
                        f"{safe_percent(convergence_result['relative_ci_half_width'])} "
                        f"[{'PASS' if convergence_result['precision_passed'] else 'FAIL'}]"
                    )
                    print(
                        f"  Recent-block change : "
                        f"{safe_percent(convergence_result['block_relative_change'])} "
                        f"[{'PASS' if convergence_result['stability_passed'] else 'FAIL'}]"
                    )
                    print(
                        f"  Combined streak     : "
                        f"{convergence_streak}/{CONVERGENCE_STREAK_REQUIRED}"
                    )

                    if convergence_streak >= CONVERGENCE_STREAK_REQUIRED:
                        converged = True
                        print(
                            "\n>>> PRECISION AND STABILITY "
                            "CRITERIA SATISFIED <<<"
                        )

                if ACQUISITION_INTERVAL_SECONDS > 0:
                    time.sleep(ACQUISITION_INTERVAL_SECONDS)

            except pyvisa.errors.VisaIOError as error:
                print(f"\nVISA error: {error}", file=sys.stderr)
                time.sleep(1.0)

            except (ValueError, RuntimeError) as error:
                print(f"\nProcessing error: {error}", file=sys.stderr)
                time.sleep(1.0)

        # =================================================================
        # FINAL RESULT
        # =================================================================

        print()
        print("\n" + "=" * 72)

        if converged:
            termination_reason = "converged"
            print("CONVERGED")
        elif long_acquisition_number >= MAX_LONG_ACQUISITIONS:
            termination_reason = "maximum acquisitions reached"
            print("MAXIMUM ACQUISITIONS REACHED")
        else:
            termination_reason = "user stop"
            print("STOPPED BY USER")

        print("=" * 72)

        if not acquisition_clusters:
            print("No qualifying measurements were collected.")
            return

        final_mean_watts = cumulative_power_watts / qualifying_pair_count
        final_mean_dbm   = watts_to_dbm(final_mean_watts)

        final_ci_low_watts, final_ci_high_watts = cluster_bootstrap_ci(
            acquisition_clusters,
            repetitions=FINAL_BOOTSTRAP_REPETITIONS,
            random_seed=RANDOM_SEED + 1_000_000,
        )

        final_ci_low_dbm  = watts_to_dbm(final_ci_low_watts)
        final_ci_high_dbm = watts_to_dbm(final_ci_high_watts)

        all_powers = np.concatenate(acquisition_clusters)
        pair_standard_deviation = (
            float(np.std(all_powers, ddof=1))
            if all_powers.size > 1
            else math.nan
        )

        relative_ci_half_width = (
            (final_ci_high_watts - final_ci_low_watts)
            / (2.0 * final_mean_watts)
        )

        print(f"Laser amplifier current : {laser_amplifier_amps:.6f} A")
        print(f"Beam power              : {beam_power_mw:.6f} mW")
        print()
        print(f"Long acquisitions tried : {long_acquisition_number}")
        print(f"Qualifying acquisitions : {qualifying_acquisition_count}")
        print(f"Qualifying pairs        : {qualifying_pair_count}")
        print(f"Positive V3 pairs       : {positive_count}")
        print(f"Negative V3 pairs       : {negative_count}")
        print()
        print(f"Mean noise power        : {final_mean_watts:.10e} W")
        print(f"Mean noise              : {final_mean_dbm:.6f} dBm")
        print(
            f"Pair standard deviation : "
            f"{pair_standard_deviation:.10e} W"
        )
        print(
            f"Cluster-bootstrap 95% CI: "
            f"{final_ci_low_dbm:.6f} to {final_ci_high_dbm:.6f} dBm"
        )
        print(f"Relative CI half-width  : {relative_ci_half_width * 100:.3f}%")

        if analyzer is not None:
            print(
                f"Actual FFT frequency    : "
                f"{analyzer.actual_frequency_hz / 1e6:.9f} MHz"
            )
            print(f"Hann ENBW               : {analyzer.enbw_hz / 1e3:.6f} kHz")

        result = {
            "instrument":                     instrument_id,
            "laser_amplifier_amps":           laser_amplifier_amps,
            "beam_power_mw":                  beam_power_mw,
            "squeezing_device_present":       squeezing_device_present,
            "additional_notes":               additional_notes,
            "termination_reason":             termination_reason,
            "converged":                      converged,
            "started_utc":                    started_utc,
            "completed_utc":                  datetime.now(timezone.utc).isoformat(),
            "long_acquisitions_attempted":    long_acquisition_number,
            "qualifying_acquisitions":        qualifying_acquisition_count,
            "qualifying_pairs":               qualifying_pair_count,
            "positive_v3_pairs":              positive_count,
            "negative_v3_pairs":              negative_count,
            "positive_fraction": (
                positive_count / qualifying_pair_count
            ),
            "negative_fraction": (
                negative_count / qualifying_pair_count
            ),
            "final_mean_watts":               final_mean_watts,
            "final_mean_dbm":                 final_mean_dbm,
            "final_ci95_low_watts":           final_ci_low_watts,
            "final_ci95_high_watts":          final_ci_high_watts,
            "final_ci95_low_dbm":             final_ci_low_dbm,
            "final_ci95_high_dbm":            final_ci_high_dbm,
            "final_relative_ci_half_width":   relative_ci_half_width,
            "pair_standard_deviation_watts":  pair_standard_deviation,
            "zero_mean_threshold_volts":      ZERO_MEAN_THRESHOLD,
            "target_frequency_hz":            TARGET_FREQUENCY_HZ,
            "actual_frequency_hz": (
                analyzer.actual_frequency_hz
                if analyzer is not None
                else math.nan
            ),
            "frequency_error_hz": (
                analyzer.actual_frequency_hz - TARGET_FREQUENCY_HZ
                if analyzer is not None
                else math.nan
            ),
            "enbw_hz": (
                analyzer.enbw_hz if analyzer is not None else math.nan
            ),
            "subwindow_duration_seconds":     SUBWINDOW_DURATION_SECONDS,
            "long_record_duration_seconds":   LONG_RECORD_DURATION_SECONDS,
            "requested_sample_rate_hz":       REQUESTED_SAMPLE_RATE_HZ,
            "actual_sample_rate_hz":          analyzer_sample_rate_hz,
            "reference_impedance_ohms":       REFERENCE_IMPEDANCE_OHMS,
            "waveform_transfer": (
                "signed 16-bit binary WORD, little-endian"
            ),
            "spectral_method": (
                "single selected DFT bin with periodic "
                "Hann window and one-sided periodogram "
                "density normalization"
            ),
            "averaging_method": (
                "pooled arithmetic mean of qualifying "
                "subwindow powers in watts"
            ),
            "uncertainty_method": (
                "bootstrap resampling of complete "
                "long acquisitions"
            ),
            "convergence_method": (
                "cluster-bootstrap precision plus "
                "adjacent non-overlapping block stability"
            ),
            "target_relative_ci_half_width":  TARGET_RELATIVE_CI_HALF_WIDTH,
            "stability_block_acquisitions":   STABILITY_BLOCK_ACQUISITIONS,
            "stability_tolerance":            STABILITY_TOLERANCE,
            "convergence_streak_required":    CONVERGENCE_STREAK_REQUIRED,
            "convergence_checks":             convergence_check_number,
            "final_convergence_streak":       convergence_streak,
            "positive_negative_balancing":    False,
        }

        result_json.write_text(
            json.dumps(result, indent=2, allow_nan=True),
            encoding="utf-8",
        )

        # ------------------------------------------------------------------
        # SUMMARY FILE
        # ------------------------------------------------------------------
        summary = {
            "laser_amplifier_amps":      laser_amplifier_amps,
            "beam_power_mw":             beam_power_mw,
            "squeezing_device_present":  squeezing_device_present,
            "additional_notes":          additional_notes,
            "final_mean_watts":          final_mean_watts,
        }
        summary_json.write_text(
            json.dumps(summary, indent=2, allow_nan=True),
            encoding="utf-8",
        )

        print()
        print(f"Pair data               : {pair_csv}")
        print(f"Convergence history     : {convergence_csv}")
        print(f"Result JSON             : {result_json}")
        print(f"Summary                 : {summary_json}")
        print("=" * 72)

    finally:
        if scope is not None:
            scope.close()
        if resource_manager is not None:
            resource_manager.close()


if __name__ == "__main__":
    main()