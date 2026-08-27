# Technical Report: Automated Noise Acquisition and Squeezing Analysis

## 1. Purpose

This project automates noise measurements for a quantum-optics experiment using a Thorlabs PDB230C balanced photodetector and a Keysight DSA91304A oscilloscope. It replaces manual oscilloscope readings with synchronized waveform acquisition, spectral analysis, statistical convergence testing, linear calibration, and squeezing calculations with propagated uncertainty.

The oscilloscope measures the balanced detector difference output:

$$
V_3(t)=V_1(t)-V_2(t).
$$

The primary result is obtained by comparing the beam-power-dependent noise slopes measured with squeezed and unsqueezed light under otherwise identical experimental conditions.

## 2. Software components

The analysis consists of three Python programs:

- `get_data_with_convergence.py` acquires and processes oscilloscope waveforms.
- `analyze_runs.py` fits noise power as a function of optical beam power.
- `calculate_squeezing.py` compares squeezed and unsqueezed calibration slopes.

## 3. Data acquisition

The oscilloscope is controlled over LAN using PyVISA. Channel 3 is transferred as signed, little-endian, 16-bit binary waveform data. Raw instrument values are converted to voltage using the waveform scaling parameters returned by the oscilloscope:

$$
V=(Y_{\mathrm{raw}}-Y_{\mathrm{reference}})
Y_{\mathrm{increment}}+Y_{\mathrm{origin}}.
$$

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

The actual sample interval and sample rate are obtained from the oscilloscope for every waveform. The analysis uses the measured values rather than assuming that the requested sample rate was achieved exactly.

## 4. Balance selection

Each long waveform is divided into non-overlapping 1.5 µs subwindows. For each subwindow, the mean detector-difference voltage is calculated:

$$
\overline{V_3}=\frac{1}{N}\sum_{n=0}^{N-1}V_3[n].
$$

A subwindow is accepted when:

$$
\left|\overline{V_3}\right|\leq 5\ \mathrm{mV}.
$$

The balance metric and spectral noise estimate are calculated from the same samples. Each accepted result therefore associates the detector balance and measured noise with the same physical time interval.

The mean voltage is used as the balance criterion instead of the total RMS voltage so that the selection criterion does not directly select observations according to their measured noise magnitude.

## 5. Spectral analysis

Each accepted subwindow is mean-detrended and multiplied by a periodic Hann window. The program evaluates the discrete Fourier coefficient at the Fourier bin nearest 10 MHz.

For an interior positive-frequency bin, the one-sided voltage power spectral density is calculated as:

$$
S_{VV}(f_k)=
\frac{2|X_k|^2}
{f_s\sum_n w_n^2},
$$

where:

- $X_k$ is the Hann-windowed Fourier coefficient,
- $f_s$ is the measured sample rate,
- $w_n$ is the Hann window.

The voltage PSD is converted to power PSD using the configured reference impedance:

$$
S_{PP}(f_k)=\frac{S_{VV}(f_k)}{R},
$$

where $R=50\ \Omega$.

The equivalent noise bandwidth of the Hann window is:

$$
\mathrm{ENBW}
=
f_s
\frac{\sum_n w_n^2}
{\left(\sum_n w_n\right)^2}.
$$

The selected-bin noise power is then:

$$
P_{\mathrm{noise}}
=
S_{PP}(f_k)\,\mathrm{ENBW}.
$$

For a 1.5 µs periodic Hann window, the nominal ENBW is approximately 1 MHz. The program records the actual ENBW, Fourier frequency, and frequency error calculated from the measured sample rate.

All averages are calculated in linear watts. Conversion to dBm occurs only after averaging:

$$
P_{\mathrm{dBm}}
=
10\log_{10}
\left(
\frac{P_{\mathrm{W}}}{1\ \mathrm{mW}}
\right).
$$

## 6. Acquisition-level statistics

Subwindows from the same long oscilloscope acquisition may be correlated. The program therefore groups accepted subwindows according to their parent long acquisition.

The overall reported mean is the pooled arithmetic mean of all accepted subwindow powers:

$$
\overline{P}
=
\frac{\sum_i\sum_j P_{ij}}
{\sum_i n_i},
$$

where $i$ identifies a long acquisition and $j$ identifies an accepted subwindow within that acquisition.

Uncertainty is estimated with a cluster bootstrap. Complete long-acquisition clusters are sampled with replacement so that dependence among subwindows from the same acquisition is preserved.

## 7. Convergence criteria

Convergence requires both a precision condition and a recent-stability condition.

### 7.1 Precision condition

A cluster-bootstrap 95% confidence interval is calculated for the pooled mean. The relative confidence-interval half-width is:

$$
h_{\mathrm{rel}}
=
\frac{P_{\mathrm{high}}-P_{\mathrm{low}}}
{2\overline{P}}.
$$

The precision condition passes when:

$$
h_{\mathrm{rel}}\leq 0.05.
$$

### 7.2 Stability condition

The pooled means of two adjacent, non-overlapping blocks of qualifying acquisitions are compared:

$$
\Delta_{\mathrm{rel}}
=
\frac{
\left|
\overline{P}_{\mathrm{recent}}
-
\overline{P}_{\mathrm{preceding}}
\right|
}
{\overline{P}_{\mathrm{preceding}}}.
$$

The stability condition passes when:

$$
\Delta_{\mathrm{rel}}\leq 0.05.
$$

The current convergence configuration requires:

- at least 100 qualifying acquisitions,
- convergence checks every 20 new qualifying acquisitions,
- two 50-acquisition stability blocks,
- a 5% relative confidence-interval half-width,
- a 5% recent-block stability tolerance,
- three consecutive successful checks.

These are predefined engineering convergence criteria. The final number of attempted acquisitions, qualifying acquisitions, accepted subwindows, confidence interval, and termination reason are saved with each result.

## 8. Shot-noise calibration

Each completed run records the optical beam power and final mean noise power. Multiple runs at different optical powers are fitted to:

$$
N(P)=kP+N_{\mathrm{dark}},
$$

where:

- $P$ is optical beam power,
- $k$ is the beam-power-dependent noise slope,
- $N_{\mathrm{dark}}$ is the fitted power-independent intercept.

The parameters are estimated by ordinary least squares. The model matrix is:

$$
X=
\begin{bmatrix}
P_1 & 1\\
P_2 & 1\\
\vdots & \vdots\\
P_n & 1
\end{bmatrix}.
$$

The parameter estimate is:

$$
\hat{\beta}
=
(X^\mathsf{T}X)^{-1}X^\mathsf{T}y,
$$

where:

$$
\hat{\beta}
=
\begin{bmatrix}
k\\
N_{\mathrm{dark}}
\end{bmatrix}.
$$

The residual variance is estimated using $n-2$ degrees of freedom:

$$
\hat{\sigma}^2
=
\frac{\sum_i(y_i-\hat{y}_i)^2}{n-2}.
$$

The parameter covariance matrix is:

$$
\operatorname{Cov}(\hat{\beta})
=
\hat{\sigma}^2(X^\mathsf{T}X)^{-1}.
$$

The square roots of the covariance-matrix diagonal elements provide the standard errors of the slope and intercept.

The program also reports:

- Pearson correlation coefficient $r$,
- coefficient of determination $R^2$,
- a two-tailed $t$-test p-value for the slope,
- maximum absolute residual,
- a residual plot.

The fitted slope and its standard error are passed to the squeezing calculation.

## 9. Squeezing calculation

Independent calibration curves are obtained for squeezed and unsqueezed light. Their slopes are denoted by $k_{\mathrm{sq}}$ and $k_{\mathrm{unsq}}$.

The normalized measured noise variance is:

$$
R=
\frac{k_{\mathrm{sq}}}
{k_{\mathrm{unsq}}}.
$$

The measured noise level relative to shot noise is:

$$
L_{\mathrm{dB}}
=
10\log_{10}(R).
$$

The interpretation is:

- $L_{\mathrm{dB}}<0$: squeezing,
- $L_{\mathrm{dB}}=0$: equal to the unsqueezed reference,
- $L_{\mathrm{dB}}>0$: noise above the unsqueezed reference.

The percentage reduction below shot noise is:

$$
Q=(1-R)\times100\%.
$$

Because the result is based on a slope ratio, common multiplicative factors such as ENBW and impedance conversion cancel when squeezed and unsqueezed datasets are acquired and processed using identical settings.

## 10. Squeezing uncertainty

The standard errors of the two slopes are propagated with a first-order delta method. Assuming statistically independent slope estimates:

$$
\left(\frac{\sigma_R}{R}\right)^2
=
\left(
\frac{\sigma_{\mathrm{sq}}}{k_{\mathrm{sq}}}
\right)^2
+
\left(
\frac{\sigma_{\mathrm{unsq}}}{k_{\mathrm{unsq}}}
\right)^2.
$$

Therefore:

$$
\sigma_R
=
R
\sqrt{
\left(
\frac{\sigma_{\mathrm{sq}}}{k_{\mathrm{sq}}}
\right)^2
+
\left(
\frac{\sigma_{\mathrm{unsq}}}{k_{\mathrm{unsq}}}
\right)^2
}.
$$

The propagated dB uncertainty is:

$$
\sigma_{\mathrm{dB}}
=
\frac{10}{\ln(10)}
\frac{\sigma_R}{R}.
$$

Unless otherwise stated, reported `±` values represent one standard error, approximately corresponding to a 68% interval when the linear approximation and normality assumptions are appropriate.

The one-standard-error percentage range is calculated from:

$$
Q_{\mathrm{low}}
=
\left[1-(R+\sigma_R)\right]100\%
$$

and:

$$
Q_{\mathrm{high}}
=
\left[1-(R-\sigma_R)\right]100\%.
$$

## 11. Output files

Each acquisition run is stored in a separate numbered directory.

### `zero_pairs.csv`

Contains one row for each accepted subwindow, including:

- parent acquisition number,
- subwindow number,
- timestamp,
- sample rate,
- mean detector-difference voltage,
- PSD,
- noise power in watts and dBm,
- selected Fourier frequency,
- ENBW,
- running mean.

### `convergence_history.csv`

Contains every convergence check, including:

- pooled mean,
- bootstrap confidence interval,
- relative interval half-width,
- adjacent-block means,
- block-relative change,
- pass/fail results,
- convergence streak.

### `converged_result.json`

Contains the complete final result and acquisition metadata.

### `summary.json`

Contains a reduced summary of the experimental parameters and final mean noise power.

## 12. Assumptions and limitations

The analysis relies on the following assumptions:

1. Squeezed and unsqueezed datasets use identical acquisition and analysis settings.
2. The 50 Ω conversion represents the actual electrical measurement configuration.
3. The fitted noise response is approximately linear over the selected optical-power range.
4. The OLS residuals are approximately independent and have constant variance.
5. Squeezed and unsqueezed slope estimates are independent.
6. First-order uncertainty propagation is adequate for the measured relative uncertainties.
7. The balance-selection threshold does not introduce a material difference between squeezed and unsqueezed datasets.

The directly reported result is observed squeezing at the measurement system. It is not corrected for optical loss or imperfect detection efficiency.

If loss-corrected source squeezing is reported, the correction method, total detection efficiency, and uncertainty in that efficiency must be included separately.

## 13. Reproducibility requirements

For final experimental results, preserve:

- the complete run directories,
- the exact run numbers used in each fit,
- the source-code version or Git commit,
- Python and dependency versions,
- oscilloscope model and firmware,
- detector gain and bandwidth settings,
- oscilloscope input impedance, coupling, and bandwidth settings,
- optical-power calibration information,
- all acquisition and convergence constants,
- whether the squeezing device was present for each run.

These records allow the acquisition, calibration, and squeezing calculation to be independently reconstructed.
