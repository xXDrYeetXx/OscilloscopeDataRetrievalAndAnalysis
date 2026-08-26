# Oscilloscope Data Retrieval and Analysis

Automated acquisition and statistical analysis of balanced-photodetector noise measurements using a **Keysight DSA91304A / DSA-X 91304A oscilloscope**.

The project acquires Channel 3, where

$$
V_3(t) = V_1(t) - V_2(t)
$$

and uses the average value of \(V_3\) as a measure of how well the two photodetector signals are balanced. When the detector is sufficiently balanced, the program extracts the noise power around **10 MHz** from the same waveform segment.

The goal is to replace a manual oscilloscope/FFT measurement process with a reproducible, automated acquisition and analysis pipeline.

---

## Overview

The measurement process is:

1. Connect to the oscilloscope over LAN using PyVISA.

2. Acquire a long Channel 3 waveform.

3. Divide the waveform into short, non-overlapping subwindows.

4. Calculate the mean of \(V_3\) for each subwindow.

5. Use `abs(mean(V3))` as the detector-balance metric.

6. Calculate the frequency-domain power spectral density from those same samples.

7. Extract the noise near **10 MHz**.

8. Accept the measurement when

   $$
   |\operatorname{mean}(V_3)| \leq 5\text{ mV}
   $$

9. Save the measurement and, optionally, its complete spectrum.

10. Analyze the resulting measurements statistically.

Using the same samples for both the balance measurement and the noise measurement is important: it ensures that the reported noise corresponds to the detector's actual balance at that exact time.

---

## Repository Structure

| File                                           | Purpose                                                      |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `get_data.py`                                  | Basic automated data acquisition                             |
| `get_data_with_convergence.py`                 | Acquisition with automatic statistical convergence detection |
| `analyze_run.py`                               | Statistical analysis of an acquired run                      |
| `AI Generated Report.md`                       | Background, motivation, methodology, and development notes   |
| `KeysightInfiniiumOscilloscopesGuideForAI.txt` | Keysight Infiniium automation/SCPI reference material        |
| `.gitignore`                                   | Git ignore rules                                             |

> **Note:** The current repository no longer contains the older `converge_noise.py`, `balance_pairs.py`, or `boxplot_noise.py` scripts referenced by previous versions of the README. The current workflow uses `get_data_with_convergence.py` and `analyze_run.py`.

---

# Hardware

The acquisition software is designed around a **Keysight DSA91304A / DSA-X 91304A** oscilloscope.

The physical signal of interest is Channel 3:

$$
V_3 = V_1 - V_2
$$

where \(V_1\) and \(V_2\) are the two signals being compared by the balanced photodetector.

The current default configuration uses:

* **Channel:** 3
* **Vertical scale:** 10 mV/div
* **Vertical offset:** 0 V
* **Requested sample rate:** 1 GSa/s
* **Long acquisition:** 150 µs
* **Analysis subwindow:** 15 µs
* **Target frequency:** 10 MHz
* **Reference impedance:** 50 Ω
* **Balance threshold:** 5 mV

These values are configurable near the top of the acquisition scripts.

---

# Requirements

## Python

A recent Python 3 installation is recommended.

The acquisition scripts use:

```text
numpy
scipy
pyvisa
pyvisa-py
```

Install them with:

```bash
pip install numpy scipy pyvisa pyvisa-py
```

A virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it and install the dependencies:

### Windows

```powershell
.venv\Scripts\Activate.ps1
pip install numpy scipy pyvisa pyvisa-py
```

### Linux/macOS

```bash
source .venv/bin/activate
pip install numpy scipy pyvisa pyvisa-py
```

The project is configured to use the pure-Python PyVISA backend:

```text
@py
```

so a separate NI-VISA or Keysight VISA installation is not necessarily required.

---

# Connecting to the Oscilloscope

The acquisition scripts communicate with the oscilloscope using VISA over TCP/IP.

Open `get_data.py` or `get_data_with_convergence.py` and change:

```python
OSCILLOSCOPE_IP = "192.168.137.113"
```

to the IP address of your oscilloscope.

The VISA resource is then constructed as:

```text
TCPIP0::<IP_ADDRESS>::inst0::INSTR
```

For example:

```text
TCPIP0::192.168.137.113::inst0::INSTR
```

Make sure the computer running the Python program can communicate with the oscilloscope over the network before starting an acquisition.

---

# Acquisition Workflow

## 1. Long waveform acquisition

The program requests approximately:

```text
1 GSa/s
150 µs
```

which corresponds to approximately:

```text
150,000 samples
```

per long acquisition.

The oscilloscope waveform is downloaded to Python for processing.

---

## 2. Subwindow segmentation

Each 150 µs acquisition is divided into non-overlapping:

```text
15 µs
```

subwindows.

Therefore, a typical acquisition contains approximately:

```text
10 subwindows
```

The exact number depends on the number of valid samples returned by the oscilloscope.

Incomplete samples at the end of a record are discarded.

---

## 3. Balance measurement

For every subwindow, the program calculates:

$$
V_{3,\mathrm{mean}} = \frac{1}{N}\sum_{i=1}^{N}V_3[i]
$$

and defines the balance metric as:

$$
\boxed{\left|V_{3,\mathrm{mean}}\right|}
$$

A smaller value indicates that the two detector signals are more closely balanced.

The current default acceptance criterion is:

$$
\boxed{\left|\operatorname{mean}(V_3)\right| \leq 5\text{ mV}}
$$

The mean is used instead of total RMS because RMS would include the high-frequency noise being measured and therefore mix the balance criterion with the noise measurement itself.

---

# Noise Calculation

For every subwindow, the program calculates a power spectral density using:

```python
scipy.signal.periodogram(...)
```

with:

* Hann window
* constant detrending
* one-sided spectrum
* density scaling

The resulting voltage PSD has units of:

$$
\mathrm{V^2/Hz}
$$

It is converted to power spectral density using the 50 Ω reference impedance:

$$
S_P(f)=\frac{S_V(f)}{50\ \Omega}
$$

giving:

$$
\mathrm{W/Hz}
$$

The program then finds the FFT bin nearest the target frequency:

$$
f_\mathrm{target}=10\text{ MHz}
$$

and calculates the equivalent noise bandwidth of the Hann window:

$$
\mathrm{ENBW}
=
f_s
\frac{\sum w_n^2}{(\sum w_n)^2}
$$

For a sufficiently long Hann-windowed record this is approximately:

$$
\mathrm{ENBW}\approx\frac{1.5}{T}
$$

For a 15 µs subwindow:

$$
\mathrm{ENBW}\approx100\text{ kHz}
$$

The estimated noise power is then:

$$
P_\mathrm{noise}
=
S_P(f_\mathrm{target})\times\mathrm{ENBW}
$$

Finally, the power is converted to dBm:

$$
P_{\mathrm{dBm}}
=
10\log_{10}\left(\frac{P_\mathrm{noise}}{1\text{ mW}}\right)
$$

The complete calculation is implemented in `calculate_noise_at_frequency()` in `get_data.py`.

---

# `get_data.py`

`get_data.py` is the basic continuous acquisition program.

Run it with:

```bash
python get_data.py
```

The program repeatedly:

1. Acquires a long waveform.
2. Splits it into subwindows.
3. Calculates the balance metric.
4. Calculates the 10 MHz noise measurement.
5. Records every subwindow.
6. Records qualifying measurements separately.
7. Optionally saves complete spectra.

Press:

```text
Ctrl+C
```

to request a clean stop.

---

# `get_data_with_convergence.py`

This is the preferred acquisition program when the goal is to automatically determine when enough measurements have been collected.

It performs the same fundamental measurement as `get_data.py`, but additionally tracks the accumulated qualifying measurements and evaluates whether the estimated mean noise power has stabilized.

The convergence analysis is performed using **linear power**, rather than averaging dBm values.

This is important because dBm is logarithmic:

$$
P_{\mathrm{dBm}}=10\log_{10}(P/1\mathrm{mW})
$$

Therefore, averaging dBm values does not produce the arithmetic mean power.

The convergence workflow instead operates on power in watts and converts the final result to dBm.

Run it with:

```bash
python get_data_with_convergence.py
```

The convergence implementation records its history and produces a machine-readable final result when convergence criteria are satisfied.

---

# Output Data

Acquisitions are organized into numbered run directories.

A typical output structure is:

```text
v3_segmented_noise_data/
└── Run 1/
    ├── all_pairs.csv
    ├── zero_pairs.csv
    ├── capture_settings.json
    └── zero_spectra/
        ├── zero_00000001_....npz
        ├── zero_00000002_....npz
        └── ...
```

A subsequent acquisition creates:

```text
Run 2/
```

and so on.

The program automatically finds the highest existing `Run N` directory and creates the next run number.

---

## `all_pairs.csv`

This file contains measurements from all analyzed subwindows.

Important columns include:

* `v3_mean_volts`
* `abs_v3_mean_volts`
* `zero_mean_threshold_volts`
* `within_zero_window`
* `requested_frequency_hz`
* `actual_fft_bin_frequency_hz`
* `frequency_error_hz`
* `noise_psd_w_per_hz`
* `noise_power_watts`
* `noise_power_dbm`
* `enbw_hz`

The file therefore contains both the balance measurement and the corresponding noise measurement for each subwindow.

---

## `zero_pairs.csv`

This contains only measurements satisfying:

$$
|\operatorname{mean}(V_3)|\leq5\text{ mV}
$$

These are the measurements considered sufficiently balanced for the primary noise analysis.

---

## `capture_settings.json`

The acquisition settings are saved alongside the data.

This includes information such as:

* oscilloscope identity
* IP address
* channel
* balance metric
* balance threshold
* target frequency
* subwindow duration
* sample rate
* FFT method
* FFT window
* FFT scaling
* ENBW
* reference impedance
* acquisition start time

This makes each run more reproducible and provides a record of the conditions under which the data were collected.

---

## `.npz` Spectra

When spectrum saving is enabled, qualifying measurements receive their complete power spectrum in compressed NumPy `.npz` files.

These contain information including:

* frequency axis
* power PSD in W/Hz
* power PSD in dBm/Hz
* ENBW
* detector balance value
* acquisition identifiers
* timestamp
* reference impedance

This makes it possible to perform additional frequency-domain analysis after acquisition without returning to the oscilloscope.

---

# `analyze_run.py`

`analyze_run.py` performs statistical analysis on an acquired run.

Run it according to the command-line options defined in the script, for example:

```bash
python analyze_run.py
```

The analysis is designed to determine whether the measured noise depends on detector balance and to estimate the noise behavior near perfect balance.

The analysis includes:

* descriptive statistics
* noise versus \(|V_3|\) analysis
* linear regression
* quadratic regression
* extrapolation toward \(V_3=0\)
* signed-\(V_3\) analysis
* positive/negative imbalance comparisons
* analysis over multiple balance windows
* bootstrap confidence intervals
* chronological/block analysis
* statistical interpretation

The script operates on the acquired measurements rather than artificially correcting the distribution of positive and negative imbalance values.

---

# Important Statistical Consideration

`zero_pairs.csv` is **not a random sample of all possible detector states**.

It is already filtered by:

$$
|\operatorname{mean}(V_3)|\leq5\text{ mV}
$$

Therefore, analysis of this file primarily tells us about noise **within the accepted balance region**.

It should not be interpreted as demonstrating the behavior of the detector at arbitrarily large imbalance values.

This distinction is important when interpreting regression results or attempting to infer the noise at exactly:

$$
V_3=0
$$

---

# Why Measurements Are Not Forced to Be Symmetric

The acquisition process does not artificially remove measurements simply because they occur on one side of zero.

If the experiment naturally produces more positive than negative imbalance values, those measurements are retained.

Artificially balancing the dataset by randomly deleting observations could change the statistical properties of the experiment and discard information.

The current analysis instead allows the measured distribution to speak for itself.

---

# Recommended Workflow

For a new experiment, the recommended workflow is:

### 1. Configure the oscilloscope

Connect the balanced photodetector to the appropriate oscilloscope channel and verify that Channel 3 represents:

$$
V_3=V_1-V_2
$$

### 2. Configure the IP address

Edit:

```python
OSCILLOSCOPE_IP = "..."
```

in the acquisition script.

### 3. Verify communication

Make sure the computer can communicate with the oscilloscope over the LAN connection.

### 4. Perform an acquisition

For a continuous collection:

```bash
python get_data.py
```

For automated convergence:

```bash
python get_data_with_convergence.py
```

### 5. Inspect the generated run

Look inside:

```text
v3_segmented_noise_data/Run N/
```

and verify that the CSV and settings files were created correctly.

### 6. Analyze the run

Use:

```bash
python analyze_run.py
```

with the appropriate run/input arguments.

### 7. Evaluate the results

Pay particular attention to:

* number of accepted measurements
* distribution of `abs(V3 mean)`
* noise-power distribution
* noise versus balance
* uncertainty/confidence intervals
* chronological stability
* convergence behavior

---

# Measurement Philosophy

The central idea behind this project is that **detector balance and detector noise should be measured from the same physical time interval**.

Instead of:

```text
Observe balance
        ↓
Manually adjust detector
        ↓
Look at FFT
        ↓
Record noise
```

the automated workflow is:

```text
Acquire waveform
        ↓
Split into short windows
        ↓
 ┌───────────────┐
 │ Same samples  │
 └───────────────┘
        ↓
 ┌─────────────────────┬──────────────────────┐
 │ mean(V3)            │ frequency-domain PSD │
 │                     │                      │
 │ detector balance    │ 10 MHz noise         │
 └─────────────────────┴──────────────────────┘
        ↓
Accept if |mean(V3)| ≤ threshold
        ↓
Store (balance, noise) pair
```

This synchronization is the key feature of the measurement method.

---

# Development Notes

The repository also contains:

## `AI Generated Report.md`

This document describes the motivation for the automated measurement system, the limitations of the previous manual procedure, and the reasoning behind the current approach.

It should be read alongside the code when modifying the measurement methodology.

## `KeysightInfiniiumOscilloscopesGuideForAI.txt`

This is reference material for automating Keysight Infiniium oscilloscopes and contains information relevant to SCPI commands, VISA communication, and instrument control.

---

# Known Limitations and Things to Validate

This software automates the measurement process, but automated data collection does not by itself establish that the physical measurement is correct.

Before relying on the results experimentally, the following should be validated:

1. **Oscilloscope communication**
   Confirm that the requested acquisition settings are actually applied by the instrument.

2. **Sample rate and record length**
   Verify the actual values reported by the oscilloscope rather than assuming the requested values were accepted.

3. **Frequency-bin alignment**
   The code selects the FFT bin nearest 10 MHz. The actual bin frequency is stored so the frequency error can be inspected.

4. **50 Ω assumption**
   Confirm that the voltage measurement and power conversion are appropriate for the actual measurement configuration.

5. **ENBW calculation**
   Verify that interpreting the selected periodogram bin as noise power over the Hann ENBW is appropriate for the experimental signal and measurement objective.

6. **Instrument noise floor**
   Determine whether the oscilloscope's own noise contributes significantly to the measured 10 MHz noise.

7. **Physical balance criterion**
   The current 5 mV threshold is a configurable experimental criterion, not a universal definition of a balanced detector.

8. **Comparison with the previous manual measurement**
   Run both methods under the same physical conditions and verify that they produce consistent results.

---

# Reproducibility

Each acquisition should be treated as a separate experimental run.

Do not rely solely on the CSV files. Preserve:

```text
Run N/
├── all_pairs.csv
├── zero_pairs.csv
├── capture_settings.json
└── zero_spectra/
```

together.

The settings file and raw spectra are particularly valuable when investigating unexpected results later.

---

# License

No license is currently specified for this repository.

Unless a license is added, assume that the repository's contents are **not licensed for unrestricted reuse or redistribution**.

---

# Disclaimer

This project was developed with substantial AI assistance.

The code and analysis should therefore be independently reviewed and experimentally validated before being used for quantitative scientific conclusions.

In particular, agreement between the software's output and the intended physical quantity should be established experimentally rather than assumed from the implementation alone.
