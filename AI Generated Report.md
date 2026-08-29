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
| `is_v3_in_range_correlated_with_noise.py` | Diagnostic check of whether the \|V3\| balance threshold is correlated with measured noise |

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

Each run contributes one point to this fit: its pooled mean noise power, and a variance from a delete-one-acquisition cluster jackknife (Section 7 applied at the run level). The fit is **inverse-variance weighted least squares**, so runs with a tighter jackknife variance are weighted more heavily. The regression is solved on a centered and scaled design matrix via `numpy.linalg.lstsq` for numerical stability, then the coefficients and covariance are transformed back into the original display units.

Parameter uncertainty is not taken from the classical OLS formula $\mathrm{Cov}(\hat{\beta}) = \hat{\sigma}^2(X^\mathsf{T}X)^{-1}$. Instead, the covariance matrix is an **HC3 heteroskedasticity-consistent sandwich estimator**, which does not assume constant residual variance across runs and inflates the estimated variance more for high-leverage points — important when only a handful of runs are available. Standard errors are the square roots of its diagonal.

Two further, independent uncertainty estimates are obtained by resampling rather than by formula:

- **Fixed-design wild bootstrap.** Leverage-adjusted residuals are multiplied by random $\pm1$ (Rademacher) weights and the model is refit thousands of times, giving a basic-bootstrap confidence interval for the slope and intercept. The same procedure is repeated under an intercept-only null model to obtain a bootstrap $p$-value for the slope.
- **Cluster-bootstrap sensitivity.** The raw acquisition clusters within each run are resampled (rather than the fit residuals) and the model is refit, isolating how much slope uncertainty is attributable to acquisition-level sampling rather than run-to-run variation.

Leave-one-run-out refits (each run excluded in turn) are also reported, to flag any single run that disproportionately drives the fit.

The program reports weighted $R^2$, unweighted and transformed residual RMSE, the approximate HC3 $t$-statistic and $p$-value, the wild-bootstrap $p$-value, and a calibration figure with a pointwise confidence band and residual panel.

The fitted slope $k$ and its saved wild-bootstrap slope samples are passed to the squeezing calculation.

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

Uncertainty on $R$ is not propagated by a delta-method formula. Instead, the two calibrations' fixed-design wild-bootstrap slope samples (Section 9) are reused directly:

1. Each slope's bootstrap sample set is converted to an empirical error distribution, $\delta = k^{*} - \hat{k}$.
2. A large number of draws are taken **independently** from the reference and squeezed error distributions — independently, so that pairing samples by array index (which could inject spurious correlation if both calibrations happened to share bootstrap structure) is avoided.
3. Each pair of draws forms one realization of the ratio:

$$
R_{\mathrm{}} = \frac{\hat{k}_{\mathrm{sq}} + \delta_{\mathrm{sq}}}{\hat{k}_{\mathrm{unsq}} + \delta^{*}_{\mathrm{unsq}}}.
$$

The spread of $R^{*}$ across all realizations gives a basic bootstrap confidence interval on $R$ directly, with no assumption that the underlying slope errors are Gaussian or that their relative size is small. Because $10\log_{10}(\cdot)$ is monotonic, the confidence interval's endpoints are transformed into dB and percent-reduction endpoints after the fact, rather than propagating a single error estimate through the log:

$$\left[\ 10\log_{10}(R_{\mathrm{low}}),\quad 10\log_{10}(R_{\mathrm{high}})\ \right].$$

A realization is excluded from $R^{*}$ if its reference-slope draw is nonpositive, since a negative calibration slope cannot meaningfully appear in the denominator. The fraction of excluded draws is reported as a diagnostic: a large fraction indicates $R$ is only weakly identified, and the interval should not be treated as reliable. If the resulting interval on $R$ itself includes a nonpositive value, the dB interval is undefined and reported as such rather than silently omitted.

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
4. Run-level uncertainty from the weighted fit is summarized with an HC3 sandwich covariance and cross-checked with a wild bootstrap and a cluster bootstrap; with few runs, all three should be compared rather than any one trusted alone.
5. The approximate HC3 $t$-test $p$-value assumes near-normal residuals; the wild-bootstrap $p$-value does not, and is the preferred figure when there are few runs.
6. Squeezed and unsqueezed slope estimates are statistically independent.
7. Squeezing uncertainty is obtained by independently resampling the two calibrations' bootstrap slope distributions rather than by first-order error propagation, so no small-relative-error assumption is required. The confidence interval is transformed into dB after resampling, since the transform is monotonic.
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
