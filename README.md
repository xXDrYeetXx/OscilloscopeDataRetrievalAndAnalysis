disclaimer: lots of ai was used
# Oscilloscope Data Retrieval and Analysis

Python tools for acquiring balanced-photodetector noise from a Keysight oscilloscope, fitting shot-noise calibration curves, and calculating optical squeezing with propagated uncertainty.

## Programs

The repository contains three main programs, plus one diagnostic script:

| Program | Purpose |
|---|---|
| `get_data_with_convergence.py` | Acquires and processes oscilloscope waveforms |
| `analyze_runs.py` | Fits noise power as a function of optical beam power |
| `calculate_squeezing.py` | Compares squeezed and unsqueezed slopes |
| `is_v3_in_range_correlated_with_noise.py` | Checks whether the balance threshold on \|V3\| is correlated with the measured noise |

## Workflow

1. Collect unsqueezed (reference) calibration runs at several beam powers.
2. Collect squeezed runs using the same acquisition settings.
3. Run `analyze_runs.py` and choose squeezing-comparison mode to fit both groups and compute the squeezing result in one pass — or run it twice in single-calibration mode, once per dataset.
4. If using single-calibration mode, pass the two saved `*_calibration_results.json` files to `calculate_squeezing.py` to compute the squeezing result.

## Requirements

Python 3.10 or newer is recommended.

Install the required packages:

```bash
python -m pip install numpy scipy pandas matplotlib pyvisa pyvisa-py
```

A compatible VISA installation or PyVISA backend must be available for oscilloscope communication.

## 1. Data acquisition

Run:

```bash
python get_data_with_convergence.py
```

The program connects to the configured oscilloscope over LAN and acquires Channel 3, representing the balanced detector difference signal:

$$
V_3(t)=V_1(t)-V_2(t).
$$

Each long acquisition is divided into non-overlapping 1.5 µs subwindows. A subwindow is accepted when:

$$
\left|\operatorname{mean}(V_3)\right|\leq5\ \mathrm{mV}.
$$

For each accepted subwindow, the program calculates the noise near 10 MHz using a mean-detrended, periodic-Hann, single-bin DFT.

Noise powers are averaged in linear watts before conversion to dBm.

### Important configuration

Before collecting data, inspect the configuration constants near the top of `get_data_with_convergence.py`:

```python
OSCILLOSCOPE_IP = "192.168.137.113"
CHANNEL = 3

ZERO_MEAN_THRESHOLD = 5e-3
TARGET_FREQUENCY_HZ = 10e6
SUBWINDOW_DURATION_SECONDS = 1.5e-6
LONG_RECORD_DURATION_SECONDS = 150e-6
REQUESTED_SAMPLE_RATE_HZ = 1e9
REFERENCE_IMPEDANCE_OHMS = 50.0

CHANNEL_VERTICAL_SCALE_VOLTS = 10e-3
CHANNEL_VERTICAL_OFFSET_VOLTS = 0.0
```

Verify that the following match the physical experiment:

- oscilloscope IP address,
- channel number,
- input impedance,
- coupling and bandwidth,
- detector gain and bandwidth,
- sample rate,
- record duration,
- target frequency,
- balance threshold,
- vertical scale,
- convergence requirements.

### Convergence

The acquisition program uses two convergence conditions:

1. A cluster-bootstrap confidence interval must meet the configured precision requirement.
2. Two recent, non-overlapping acquisition blocks must agree within the configured stability tolerance.

Three consecutive successful checks are required by default.

Press `Ctrl+C` to stop acquisition manually.

## 2. Run analysis

After collecting runs at several optical powers, run:

```bash
python analyze_runs.py
```

You'll first be asked whether to do a squeezing comparison (`y`/`n`). Choosing `n` fits one calibration group; choosing `y` splits the selected runs into reference and squeezed groups automatically using each run's `squeezing_device_present` field, fits both, and computes the squeezing result in the same pass.

Then enter run numbers using individual values, ranges, or both:

```text
1,3-5,8
```

or:

```text
26-31
```

The program reads `converged_result.json` and `zero_pairs.csv` from each selected run and fits, by inverse-variance weighted least squares:

$$
N(P)=kP+N_{\mathrm{dark}},
$$

where:

- $P$ is optical beam power,
- $k$ is the beam-power-dependent noise slope,
- $N_{\mathrm{dark}}$ is the fitted power-independent intercept.

Each run's weight comes from a delete-one-acquisition cluster jackknife variance, so correlated subwindows within one run are never treated as independent samples.

The program reports:

- slope and intercept, each with an HC3 (heteroskedasticity-robust) standard error,
- a wild-bootstrap confidence interval for the slope and intercept, plus a bootstrap p-value for the slope,
- a cluster-bootstrap sensitivity range for the slope,
- weighted R² and residual RMSE,
- leave-one-run-out slope sensitivity for every run.

It also saves a calibration figure with a pointwise 95% interval and a residual panel.

An example result is:

```text
Slope k: 1.7891 ± 0.0575 nW/mW
```

Single-calibration mode saves `calibration_calibration_results.json` and `calibration_bootstrap_slope_samples.npz`, which `calculate_squeezing.py` reads directly — there's no manual entry of slopes or standard errors.

## 3. Squeezing calculation

After separately analyzing squeezed and unsqueezed datasets in single-calibration mode, run:

```bash
python calculate_squeezing.py reference_calibration_results.json squeezed_calibration_results.json
```

The matching `*_bootstrap_slope_samples.npz` file must sit alongside each JSON file, since the script reads the saved wild-bootstrap slope samples rather than a single standard error. Optional flags: `--repetitions`, `--confidence-level`, `--seed`, `--output`.

The program calculates the normalized noise ratio:

$$
R=
\frac{k_{\mathrm{sq}}}
{k_{\mathrm{unsq}}}.
$$

The measured squeezing level is:

$$
L_{\mathrm{dB}}=10\log_{10}(R).
$$

The percentage reduction below shot noise is:

$$
Q=(1-R)\times100\%.
$$

The interpretation is:

- $R<1$: squeezing,
- $R=1$: equal to the unsqueezed reference,
- $R>1$: noise above the unsqueezed reference.

Uncertainty is not propagated analytically. Instead, the script draws independently from each dataset's saved wild-bootstrap slope distribution, forms the ratio for each pair of draws, and builds a basic bootstrap confidence interval on $R$. That interval is then transformed into dB and percent-reduction intervals. Bootstrap draws with a nonpositive reference slope are excluded from the ratio, but their fraction is reported as a diagnostic — a large fraction means $R$ is poorly identified.

Example output:

```text
Noise ratio R = k_sq/k_ref: 0.666667
Basic-bootstrap 95% interval: [R_low, R_high]

Signed noise change 10 log10(R): -1.7609 dB
95% interval: [low, high] dB

Squeezing magnitude -10 log10(R): 1.7609 dB
95% interval: [low, high] dB

Noise reduction: 33.333%
95% interval: [low, high]%
```

## Output structure

Acquisition results are stored under `v3_converged_noise_data`:

```text
v3_converged_noise_data/
├── Run 1/
│   ├── zero_pairs.csv
│   ├── convergence_history.csv
│   ├── converged_result.json
│   └── summary.json
├── Run 2/
│   ├── zero_pairs.csv
│   ├── convergence_history.csv
│   ├── converged_result.json
│   └── summary.json
└── ...
```

### `zero_pairs.csv`

Contains accepted subwindow measurements, balance values, spectral results, and running means.

### `convergence_history.csv`

Contains the results of every precision and stability check.

### `converged_result.json`

Contains the full final result, instrument identification, acquisition settings, uncertainty estimates, and convergence information.

### `summary.json`

Contains a shorter summary of the experimental parameters and final mean noise power.

## Experimental requirements

Squeezed and unsqueezed measurements must use identical settings, including:

- oscilloscope channel configuration,
- detector configuration,
- sample rate,
- subwindow duration,
- target frequency,
- ENBW,
- reference impedance,
- beam-power definition,
- balance threshold,
- convergence criteria,
- analysis code.

Common multiplicative scaling factors cancel in the slope ratio only when the two datasets are measured and processed consistently.

## Statistical notes

`analyze_runs.py` uses inverse-variance weighted least squares, weighting each run by a delete-one-acquisition cluster jackknife variance rather than treating subwindows as independent samples. Slope/intercept uncertainty is reported three ways: an HC3 sandwich standard error, a fixed-design wild-bootstrap confidence interval and p-value, and a cluster-bootstrap sensitivity range. Leave-one-run-out sensitivity is also reported.

`calculate_squeezing.py` does not use first-order (delta-method) error propagation. It draws independently from the two calibrations' saved wild-bootstrap slope distributions and builds a basic bootstrap interval directly on the ratio, which is then transformed into dB and percent-reduction intervals. It assumes the squeezed and unsqueezed slope estimates are independent.

`get_data_with_convergence.py` uses a cluster bootstrap in which complete long acquisitions are resampled. This avoids treating all subwindows from one acquisition as statistically independent.

Repeated runs at selected beam powers are recommended for evaluating run-to-run reproducibility.

## Observed versus loss-corrected squeezing

The current calculation reports directly observed squeezing.

It does not correct for optical loss or imperfect detection efficiency. If source squeezing is inferred, the loss-correction model, total detection efficiency, and uncertainty in that efficiency should be reported separately.

## Reproducibility

Preserve the following with the final dataset:

- complete run directories,
- exact run selections,
- source-code version or Git commit,
- Python version,
- package versions,
- oscilloscope model and firmware,
- detector settings,
- channel impedance, coupling, and bandwidth,
- beam-power calibration,
- acquisition configuration,
- experimental notes.

To save installed package versions:

```bash
python -m pip freeze > requirements.txt
```

To install those exact versions later:

```bash
python -m pip install -r requirements.txt
```

## Documentation

See [`AI Generated Report.md`](AI%20Generated%20Report.md) for a detailed description of the acquisition, spectral analysis, convergence procedure, calibration model, and squeezing uncertainty calculation.
