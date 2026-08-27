# Technical Report: Automated Noise Acquisition and Squeezing Analysis

---

## 1. Purpose

This project automates noise measurements for a quantum-optics squeezing experiment. It replaces manual oscilloscope readings with a synchronized, automated pipeline covering waveform acquisition, spectral analysis, statistical convergence, linear calibration, and squeezing calculation with propagated uncertainty.

## 2. Physical setup

The experiment uses a Thorlabs PDB230C balanced amplified photodetector connected to a Keysight DSA91304A 13 GHz oscilloscope over LAN. The balanced detector produces two photocurrent outputs from two input optical beams. Channel 3 of the oscilloscope receives the hardware difference signal produced by the detector:

```math
V_3(t) = V_1(t) - V_2(t).
```

When the two input beams carry equal optical power, the detector is balanced and $V_3$ carries only noise. The power spectral density of $V_3$ near 10 MHz is used as the noise observable.

Squeezing is measured by comparing the power-dependent noise slope of squeezed light against an unsqueezed shot-noise reference collected under otherwise identical conditions. The result is independent of common multiplicative factors such as detector gain and measurement bandwidth, provided both datasets use the same acquisition and analysis settings.

## 3. Software components

| Program | Purpose |
|---|---|
| `get_data_with_convergence.py` | Acquires waveforms and computes converged noise power |
| `analyze_runs.py` | Fits noise power as a function of optical beam power |
| `calculate_squeezing.py` | Computes squeezing from the ratio of two calibration slopes |

## 4. Data acquisition

The oscilloscope is controlled over LAN using PyVISA. Channel 3 is transferred as signed, little-endian, 16-bit binary waveform data and converted to voltage using the waveform scaling parameters returned by the oscilloscope:

```math
V = (Y_{\mathrm{raw}} - Y_{\mathrm{reference}})\, Y_{\mathrm{increment}} + Y_{\mathrm{origin}}.
```

The configured acquisition parameters are:

| Parameter | Value |
|---|---:|
| Oscilloscope channel | 3 |
| Requested sample rate | 1 GSa/s |
| Long-record duration | 150 µs |
| Subwindow duration | 1.5 µs |
| Target frequency | 10 MHz |
| Reference impedance | 50 Ω |
| Vertical scale | 10 mV/div |
| Vertical offset | 0 V |
| Balance threshold | 5 mV |

The actual sample interval and rate are read back from the oscilloscope preamble for every acquisition. All spectral calculations use measured rather than requested values.

## 5. Balance selection

Each long waveform is divided into non-overlapping 1.5 µs subwindows. For each subwindow the mean detector-difference voltage is calculated:

```math
\overline{V_3} = \frac{1}{N}\sum_{n=0}^{N-1} V_3[n].
```

A subwindow is accepted when:

```math
\left|\overline{V_3}\right| \leq 5\ \mathrm{mV}.
```

The balance metric and spectral noise estimate are derived from the same samples, so each accepted result associates the detector balance with the measured noise at the same physical moment.

The mean voltage is used as the balance criterion rather than the total RMS voltage so that the selection does not directly depend on the noise being measured.

## 6. Spectral analysis

Each accepted subwindow is mean-detrended and multiplied by a periodic Hann window. A single Fourier coefficient is evaluated at the bin nearest 10 MHz.

The one-sided voltage power spectral density at that bin is:

```math
S_{VV}(f_k) = \frac{2\left|X_k\right|^2}{f_s \sum_n w_n^2},
```

where $X_k$ is the Hann-windowed Fourier coefficient, $f_s$ is the measured sample rate, and $w_n$ is the Hann window.

The voltage PSD is converted to power PSD using the reference impedance:

```math
S_{PP}(f_k) = \frac{S_{VV}(f_k)}{R},\quad R = 50\ \Omega.
```

The Hann equivalent noise bandwidth is:

```math
\mathrm{ENBW} = f_s \frac{\sum_n w_n^2}{\left(\sum_n w_n\right)^2}.
```

The noise power for the selected bin is:

```math
P_{\mathrm{noise}} = S_{PP}(f_k)\,\mathrm{ENBW}.
```

For a 1.5 µs periodic Hann window at 1 GSa/s, the nominal ENBW is approximately 1 MHz. The program records the actual ENBW, selected Fourier frequency, and frequency error for every acquisition.

All averages are performed in linear watts. Conversion to dBm occurs only after averaging:

```math
P_{\mathrm{dBm}} = 10\log_{10}\!\left(\frac{P_{\mathrm{W}}}{1\ \mathrm{mW}}\right).
```

## 7. Acquisition-level statistics

Subwindows from the same long acquisition may be correlated due to detector dynamics, laser noise, and mechanical drift. Complete acquisitions are therefore treated as statistical clusters.

The pooled mean is:

```math
\overline{P} = \frac{\sum_i \sum_j P_{ij}}{\sum_i n_i},
```

where $i$ identifies a long acquisition and $j$ identifies an accepted subwindow within it.

Uncertainty is estimated with a cluster bootstrap in which complete acquisitions are resampled with replacement, preserving within-acquisition dependence.

## 8. Convergence criteria

Convergence requires both a precision condition and a stability condition.

### 8.1 Precision condition

The cluster-bootstrap 95% confidence interval relative half-width must satisfy:

```math
h_{\mathrm{rel}} = \frac{P_{\mathrm{high}} - P_{\mathrm{low}}}{2\,\overline{P}} \leq 0.05.
```

### 8.2 Stability condition

Two adjacent, non-overlapping blocks of qualifying acquisitions are compared. The relative change must satisfy:

```math
\Delta_{\mathrm{rel}} = \frac{\left|\overline{P}_{\mathrm{recent}} - \overline{P}_{\mathrm{preceding}}\right|}{\overline{P}_{\mathrm{preceding}}} \leq 0.05.
```

### 8.3 Convergence configuration

| Parameter | Value |
|---|---:|
| Minimum qualifying acquisitions | 100 |
| Check interval | every 20 acquisitions |
| Stability block size | 50 acquisitions |
| Precision target | 5% relative CI half-width |
| Stability tolerance | 5% |
| Required consecutive passes | 3 |

The termination reason, number of attempted and qualifying acquisitions, accepted subwindows, final confidence interval, and convergence streak are saved with every result.

## 9. Shot-noise calibration

Multiple runs at different optical powers are fitted to the linear model:

```math
N(P) = kP + N_{\mathrm{dark}},
```

where $P$ is optical beam power, $k$ is the beam-power-dependent noise slope, and $N_{\mathrm{dark}}$ is the fitted power-independent intercept.

Parameters are estimated by ordinary least squares:

```math
\hat{\beta} = (X^\mathsf{T}X)^{-1}X^\mathsf{T}y,\quad
\hat{\beta} = \begin{bmatrix}k\\N_{\mathrm{dark}}\end{bmatrix}.
```

The parameter covariance matrix is:

```math
\operatorname{Cov}(\hat{\beta}) = \hat{\sigma}^2(X^\mathsf{T}X)^{-1},
```

where the residual variance uses $n - 2$ degrees of freedom. Standard errors are the square roots of the diagonal elements.

The program also reports $R^2$, Pearson $r$, a two-tailed $t$-test p-value for the slope, the maximum absolute residual, and a residual plot.

The fitted slope $k$ and its standard error are passed to the squeezing calculation.

## 10. Squeezing calculation

Independent calibration curves are obtained for squeezed and unsqueezed light with slopes $k_{\mathrm{sq}}$ and $k_{\mathrm{unsq}}$.

The normalized measured noise variance is:

```math
R = \frac{k_{\mathrm{sq}}}{k_{\mathrm{unsq}}}.
```

The measured noise level relative to shot noise is:

```math
L_{\mathrm{dB}} = 10\log_{10}(R).
```

The percentage reduction below shot noise is:

```math
Q = (1 - R) \times 100\%.
```

Because the result uses a slope ratio, common multiplicative factors cancel when both datasets are acquired and analyzed with identical settings.

## 11. Squeezing uncertainty

Slope standard errors are propagated by the first-order delta method, assuming independent slope estimates:

```math
\left(\frac{\sigma_R}{R}\right)^2 =
\left(\frac{\sigma_{\mathrm{sq}}}{k_{\mathrm{sq}}}\right)^2 +
\left(\frac{\sigma_{\mathrm{unsq}}}{k_{\mathrm{unsq}}}\right)^2.
```

The propagated dB uncertainty is:

```math
\sigma_{\mathrm{dB}} = \frac{10}{\ln(10)}\,\frac{\sigma_R}{R}.
```

Reported $\pm$ values represent one standard error. The one-standard-error percentage range is evaluated at $R \pm \sigma_R$.

## 12. Output files

Each run is stored in a numbered directory under `v3_converged_noise_data`.

| File | Contents |
|---|---|
| `zero_pairs.csv` | Accepted subwindow measurements, balance values, spectral results, running means |
| `convergence_history.csv` | Every convergence check with precision and stability results |
| `converged_result.json` | Full final result, metadata, and acquisition configuration |
| `summary.json` | Abbreviated experimental parameters and final mean noise power |

## 13. Assumptions and limitations

1. Squeezed and unsqueezed datasets use identical acquisition and analysis settings.
2. The 50 Ω conversion represents the actual electrical measurement configuration.
3. The noise response is approximately linear over the selected optical-power range.
4. OLS residuals are approximately independent with constant variance.
5. Squeezed and unsqueezed slope estimates are statistically independent.
6. First-order uncertainty propagation is adequate for the measured relative uncertainties.
7. The balance-selection threshold does not introduce a material difference between the two datasets.

The reported result is directly observed squeezing at the measurement system. No correction for optical loss or detection efficiency is currently applied. If source squeezing is inferred, the correction method, detection efficiency, and its uncertainty must be reported separately.

## 14. Reproducibility requirements

For final experimental results, preserve:

- complete run directories,
- exact run selections used in each fit,
- source-code version or Git commit,
- Python and dependency versions,
- oscilloscope model and firmware,
- detector gain and bandwidth settings,
- oscilloscope input impedance, coupling, and bandwidth,
- optical-power calibration,
- all acquisition and convergence configuration,
- experimental notes including whether the squeezing device was present.
