# Technical Report: Automated Noise Acquisition and Squeezing Analysis

## 1. Purpose

This project automates noise measurements for a quantum-optics squeezing experiment. It replaces manual oscilloscope readings with a synchronized, automated pipeline covering waveform acquisition, spectral analysis, statistical convergence, linear calibration, and squeezing calculation with propagated uncertainty.

## 2. Physical Setup

The experiment uses a Thorlabs PDB230C balanced amplified photodetector connected to a Keysight DSA91304A 13 GHz oscilloscope over LAN. The balanced detector produces two photocurrent outputs from two input optical beams. Channel 3 of the oscilloscope receives the hardware difference signal produced by the detector:

$$V_3(t) = V_1(t) - V_2(t).$$

When the two input beams carry equal optical power, the detector is balanced and $V_3$ carries only noise. The power spectral density of $V_3$ near 10 MHz is used as the noise observable.

Squeezing is measured by comparing the power-dependent noise slope of squeezed light against an unsqueezed shot-noise reference collected under otherwise identical conditions. The result is independent of common multiplicative factors such as detector gain and measurement bandwidth, provided both datasets use the same acquisition and analysis settings.

## 3. Software Components

| Program | Purpose |
|---|---|
| `get_data_with_convergence.py` | Acquires waveforms and computes converged noise power |
| `analyze_runs.py` | Fits noise power as a function of optical beam power |
| `calculate_squeezing.py` | Computes squeezing from the ratio of two calibration slopes |

## 4. Data Acquisition

The oscilloscope is controlled over LAN using PyVISA. Channel 3 is transferred as signed, little-endian, 16-bit binary waveform data and converted to voltage using the waveform scaling parameters returned by the oscilloscope:

$$V = (Y_{\mathrm{raw}} - Y_{\mathrm{reference}})\, Y_{\mathrm{increment}} + Y_{\mathrm{origin}}.$$

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

## 5. Balance Selection

Each long waveform is divided into non-overlapping 1.5 µs subwindows. For each subwindow the mean detector-difference voltage is calculated:

$$\overline{V_3} = \frac{1}{N}\sum_{n=0}^{N-1} V_3[n].$$

A subwindow is accepted when:

$$\left|\overline{V_3}\right| \leq 5\ \mathrm{mV}.$$

The balance metric and spectral noise estimate are derived from the same samples, so each accepted result associates the detector balance with the measured noise at the same physical moment.

The mean voltage is used as the balance criterion rather than the total RMS voltage so that the selection does not directly depend on the noise being measured.

## 6. Spectral Analysis

Each accepted subwindow is mean-detrended and multiplied by a periodic Hann window. A single Fourier coefficient is evaluated at the bin nearest 10 MHz.

The one-sided voltage power spectral density at that bin is:

$$S_{VV}(f_k) = \frac{2\left|X_k\right|^2}{f_s \sum_n w_n^2},$$

where $X_k$ is the Hann-windowed Fourier coefficient, $f_s$ is the measured sample rate, and $w_n$ is the Hann window.

The voltage PSD is converted to power PSD using the reference impedance:

$$S_{PP}(f_k) = \frac{S_{VV}(f_k)}{R},\quad R = 50\ \Omega.$$

The Hann equivalent noise bandwidth is:

$$\mathrm{ENBW} = f_s \frac{\sum_n w_n^2}{\left(\sum_n w_n\right)^2}.$$

The noise power for the selected bin is:

$$P_{\mathrm{noise}} = S_{PP}(f_k)\,\mathrm{ENBW}.$$

For a 1.5 µs periodic Hann window at 1 GSa/s, the nominal ENBW is approximately 1 MHz. The program records the actual ENBW, selected Fourier frequency, and frequency error for every acquisition.

All averages are performed in linear watts. Conversion to dBm occurs only after averaging:

$$P_{\mathrm{dBm}} = 10\log_{10}\!\left(\frac{P_{\mathrm{W}}}{1\ \mathrm{mW}}\right).$$

## 7. Acquisition-Level Statistics

Subwindows from the same long acquisition may be correlated due to detector dynamics, laser noise, and mechanical drift. Complete acquisitions are therefore treated as statistical clusters.

The pooled mean is:

$$\overline{P} = \frac{\sum_i \sum_j P_{ij}}{\sum_i n_i},$$

where $i$ identifies a long acquisition and $j$ identifies an accepted subwindow within it.

Uncertainty is estimated with a cluster bootstrap in which complete acquisitions are resampled with replacement, preserving within-acquisition dependence.

## 8. Convergence Criteria

Convergence requires both a precision condition and a stability condition.

### 8.1 Precision Condition

The cluster-bootstrap 95% confidence interval relative half-width must satisfy:

$$h_{\mathrm{rel}} = \frac{P_{\mathrm{high}} - P_{\mathrm{low}}}{2\,\overline{P}} \leq 0.05.$$

### 8.2 Stability Condition

Two adjacent, non-overlapping blocks of qualifying acquisitions are compared. The relative change must satisfy:

$$\Delta_{\mathrm{rel}} = \frac{\left|\overline{P}_{\mathrm{recent}} - \overline{P}_{\mathrm{preceding}}\right|}{\overline{P}_{\mathrm{preceding}}} \leq 0.05.$$

### 8.3 Convergence Configuration

| Parameter | Value |
|---|---:|
| Minimum qualifying acquisitions | 100 |
| Check interval | every 20 acquisitions |
| Stability block size | 50 acquisitions |
| Precision target | 5% relative CI half-width |
| Stability tolerance | 5% |
| Required consecutive passes | 3 |

The termination reason, number of attempted and qualifying acquisitions, accepted subwindows, final confidence interval, and convergence streak are saved with every result.

## 9. Shot-Noise Calibration

Multiple runs at different optical powers are fitted to the linear model:

$$N(P) = kP + N_{\mathrm{dark}},$$

where $P$ is optical beam power, $k$ is the beam-power-dependent noise slope, and $N_{\mathrm{dark}}$ is the fitted power-independent dark-noise intercept.

The fit is performed using `numpy.linalg.lstsq` on a centered and scaled design matrix to avoid the numerical instability associated with directly inverting the unscaled normal-equation matrix. The fitted coefficients and covariance matrix are then transformed back into the original display units. In the original units the OLS solution satisfies:

$$\hat{\beta} = (X^\mathsf{T}X)^{-1}X^\mathsf{T}y,\quad \hat{\beta} = \begin{bmatrix}k\\N_{\mathrm{dark}}\end{bmatrix}.$$

The parameter covariance matrix is:

$$\operatorname{Cov}(\hat{\beta}) = \hat{\sigma}^2(X^\mathsf{T}X)^{-1},$$

where $\hat{\sigma}^2$ is the unbiased residual variance with $n - 2$ degrees of freedom. Standard errors are the square roots of the diagonal elements.

Reported $\pm$ values are **standard errors**, not automatically 68% confidence intervals. Under the classical fixed-design OLS model with independent, Gaussian, homoscedastic residuals, the ratio $(\hat{\beta}_i - \beta_i)/\mathrm{SE}(\hat{\beta}_i)$ follows a $t_{n-2}$ distribution, and the exact coverage of a $\pm 1\ \mathrm{SE}$ interval is:

$$c = P(-1 \leq T_{n-2} \leq 1),$$

which converges to 68.27% only as $n \to \infty$. At $n = 6$ runs the coverage is approximately 62.6%. The script computes and reports the exact finite-sample value.

The program also reports $R^2$, Pearson $r$, a two-tailed $t$-test $p$-value for the slope, the residual standard deviation, the maximum absolute residual, and a residual plot.

The fitted slope $k$ and its standard error are passed to the squeezing calculation.

## 10. Squeezing Calculation

Independent calibration curves are obtained for squeezed and unsqueezed light with slopes $k_{\mathrm{sq}}$ and $k_{\mathrm{unsq}}$.

The normalized measured noise variance is:

$$R = \frac{k_{\mathrm{sq}}}{k_{\mathrm{unsq}}}.$$

The measured noise level relative to shot noise is:

$$L_{\mathrm{dB}} = 10\log_{10}(R).$$

The percentage reduction below shot noise is:

$$Q = (1 - R) \times 100\%.$$

Because the result uses a slope ratio, common multiplicative factors cancel when both datasets are acquired and analyzed with identical settings.

## 11. Squeezing Uncertainty

Slope standard errors are propagated by the first-order delta method, assuming independent slope estimates:

$$\left(\frac{\sigma_R}{R}\right)^2 = \left(\frac{\sigma_{\mathrm{sq}}}{k_{\mathrm{sq}}}\right)^2 + \left(\frac{\sigma_{\mathrm{unsq}}}{k_{\mathrm{unsq}}}\right)^2.$$

The dB uncertainty interval is obtained by transforming the linear-domain endpoints $R \pm \sigma_R$ separately through the log transform rather than applying the symmetric first-order approximation $\sigma_{\mathrm{dB}} = (10/\ln 10)\,(\sigma_R/R)$. This produces an asymmetric interval:

$$\left[\ 10\log_{10}(R - \sigma_R),\quad 10\log_{10}(R + \sigma_R)\ \right],$$

which correctly reflects the concavity of the logarithm. If $R - \sigma_R \leq 0$, the lower bound is undefined and reported as such. The symmetric approximation remains valid when the relative SE is small (below roughly 15–20%) but the asymmetric form is used throughout for consistency.

Reported $\pm$ values represent one standard error. The one-standard-error percentage range is evaluated at $R \pm \sigma_R$:

$$Q_{\mathrm{low}} = (1 - R - \sigma_R) \times 100\%, \quad Q_{\mathrm{high}} = (1 - R + \sigma_R) \times 100\%.$$

## 12. Output Files

Each run is stored in a numbered directory under `v3_converged_noise_data`.

| File | Contents |
|---|---|
| `zero_pairs.csv` | Accepted subwindow measurements, balance values, spectral results, running means |
| `convergence_history.csv` | Every convergence check with precision and stability results |
| `converged_result.json` | Full final result, metadata, and acquisition configuration |
| `summary.json` | Abbreviated experimental parameters and final mean noise power |

## 13. Assumptions and Limitations

1. Squeezed and unsqueezed datasets use identical acquisition and analysis settings.
2. The 50 Ω reference impedance represents the actual electrical measurement configuration.
3. The noise response is approximately linear over the selected optical-power range.
4. OLS residuals are approximately independent with constant variance. With few runs this cannot be formally tested; the residual plot should be inspected.
5. Student-$t$ $p$-values and the finite-sample $\pm 1\ \mathrm{SE}$ coverage are exact only under normally distributed residuals.
6. Squeezed and unsqueezed slope estimates are statistically independent.
7. First-order uncertainty propagation through the slope ratio is adequate when the relative SE on $R$ is small. The asymmetric transformed-endpoint dB interval is used regardless.
8. The balance-selection threshold does not introduce a material difference between the two datasets.

The reported result is directly observed squeezing at the measurement system. No correction for optical loss or detection efficiency is applied. If source squeezing is to be inferred, the correction method, detection efficiency, and its uncertainty must be reported separately.

## 14. Reproducibility Requirements

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
