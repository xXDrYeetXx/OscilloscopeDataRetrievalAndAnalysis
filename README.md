# Oscilloscope Data Retrieval and Analysis

Automated acquisition and analysis of balanced-photodetector noise measurements using a **Keysight DSA91304A oscilloscope**.

The system measures the difference signal

$$
V_3(t) = V_1(t) - V_2(t)
$$

and uses the average value of \(V_3\) to determine detector balance. When the detector is sufficiently balanced, the program measures noise near **10 MHz** from the same waveform segment.

## Features

* Automated oscilloscope control over LAN using PyVISA
* 1 GSa/s waveform acquisition
* 15 µs analysis windows
* Automatic detector-balance detection
* 10 MHz noise-power calculation using a Hann-windowed periodogram
* Conversion from voltage PSD to power using a 50 Ω reference
* Storage of raw measurements and spectra
* Automatic statistical convergence detection
* Bootstrap and regression analysis

## Repository

| File                                           | Description                                      |
| ---------------------------------------------- | ------------------------------------------------ |
| `get_data.py`                                  | Basic continuous data acquisition                |
| `get_data_with_convergence.py`                 | Acquisition with automatic convergence detection |
| `analyze_run.py`                               | Statistical analysis of an acquired run          |
| `AI Generated Report.md`                       | Project background and methodology               |
| `KeysightInfiniiumOscilloscopesGuideForAI.txt` | Keysight oscilloscope reference material         |

## Requirements

Python 3 with:

```bash
pip install numpy scipy pyvisa pyvisa-py
```

The acquisition scripts use the PyVISA backend:

```text
@py
```

## Oscilloscope Setup

The software is configured for a Keysight DSA91304A.

Before running an acquisition, set the oscilloscope IP address in the acquisition script:

```python
OSCILLOSCOPE_IP = "192.168.137.113"
```

The default measurement configuration is:

* Channel: 3
* Vertical scale: 10 mV/div
* Sample rate: 1 GSa/s
* Acquisition length: 150 µs
* Analysis window: 15 µs
* Target frequency: 10 MHz
* Reference impedance: 50 Ω
* Balance threshold: 5 mV

## Measurement Method

Each 150 µs waveform is divided into 15 µs windows.

For each window, the program calculates

$$
V_{3,\mathrm{mean}} = \frac{1}{N}\sum_{i=1}^{N}V_3[i]
$$

A measurement is considered balanced when

$$
|V_{3,\mathrm{mean}}| \leq 5\ \mathrm{mV}
$$

The same samples are then used to calculate the frequency-domain noise.

A Hann-windowed periodogram produces the voltage power spectral density:

$$
S_V(f)
$$

which is converted to power spectral density using a 50 Ω reference:

$$
S_P(f) = \frac{S_V(f)}{50\ \Omega}
$$

The bin closest to 10 MHz is selected and multiplied by the Hann equivalent noise bandwidth:

$$
P_{\mathrm{noise}} = S_P(f)\,\mathrm{ENBW}
$$

The result is converted to dBm:

$$
P_{\mathrm{dBm}} =
10\log_{10}\left(\frac{P_{\mathrm{noise}}}{1\ \mathrm{mW}}\right)
$$

Balance and noise are therefore measured from the **same physical time interval**.

## Running the Acquisition

For continuous acquisition:

```bash
python get_data.py
```

For acquisition with automatic convergence:

```bash
python get_data_with_convergence.py
```

Stop either program with `Ctrl+C`.

## Output

Runs are stored in numbered directories:

```text
v3_segmented_noise_data/
└── Run 1/
    ├── all_pairs.csv
    ├── zero_pairs.csv
    ├── capture_settings.json
    └── zero_spectra/
        └── *.npz
```

### `all_pairs.csv`

Contains measurements from all analyzed waveform windows, including:

* Mean \(V_3\)
* Absolute mean \(V_3\)
* Noise PSD
* Noise power
* Noise in dBm
* FFT frequency
* ENBW
* Acceptance status

### `zero_pairs.csv`

Contains only measurements satisfying the balance threshold.

### `zero_spectra/`

Contains complete frequency-domain spectra for accepted measurements when spectrum saving is enabled.

### `capture_settings.json`

Stores the acquisition and analysis settings used for the run.

## Statistical Analysis

Run:

```bash
python analyze_run.py
```

The analysis examines the relationship between detector imbalance and measured noise. It includes:

* Descriptive statistics
* Linear and quadratic regression
* Noise versus \(|V_3|\)
* Positive/negative imbalance comparisons
* Multiple balance windows
* Bootstrap confidence intervals
* Chronological stability analysis

The analysis does **not** artificially remove positive or negative measurements to force a symmetric dataset.

## Important Limitations

The 5 mV balance threshold is an experimental choice, not a universal definition of detector balance.

Also, `zero_pairs.csv` contains only measurements that already passed the balance threshold. Therefore, analysis of this file should not be interpreted as describing detector behavior at arbitrary levels of imbalance.
