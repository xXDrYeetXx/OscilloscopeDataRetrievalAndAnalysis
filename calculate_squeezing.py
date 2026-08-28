"""
calculate_squeezing.py
======================
Computes squeezing in dB from the ratio of two OLS regression slopes
(squeezed vs. unsqueezed noise power vs. beam power). The intercept
is dark noise and does not affect the slopes, so the ratio R = k_sq /
k_unsq is automatically dark-noise-corrected.

Uncertainty is propagated through R via the first-order delta method,
assuming the two datasets are independent:

    (σ_R / R)² = (σ_sq / k_sq)² + (σ_unsq / k_unsq)²

The dB uncertainty interval is obtained by transforming the linear
endpoints R ± σ_R through the log separately, giving an asymmetric
interval that correctly reflects the concavity of the log transform.
If R - σ_R <= 0, the lower dB bound is undefined.

All uncertainties are ±1 standard error. Coverage depends on the
degrees of freedom of the input fits and is exact only under
independent, Gaussian, homoscedastic OLS residuals.

Both datasets must use identical measurement conditions so that any
systematic offset in absolute noise power cancels in the ratio.

EFFICIENCY CORRECTION
---------------------
get_efficiency() is defined below but intentionally not applied.
If you want to report loss-corrected (source) squeezing rather than
directly observed squeezing, collect the efficiency eta and replace:

    measured_noise_ratio = k_sq / k_unsq

with:

    measured_noise_ratio = (k_sq / k_unsq - (1 - eta)) / eta

Note that error propagation through that expression is more involved
and is not currently implemented.
"""

import math


def get_positive_float(prompt: str) -> float:
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
        except ValueError:
            pass
        print("Please enter a number greater than zero.")


def get_slope_with_uncertainty(label: str) -> tuple[float, float]:
    """Prompt for a slope and its standard error. Returns (slope, std_err)."""
    print(f"\n{label}")
    slope = get_positive_float("  Slope k: ")
    while True:
        try:
            std_err = float(input("  Standard error on k (±): "))
            if std_err >= 0:
                return slope, std_err
        except ValueError:
            pass
        print("Please enter a non-negative number.")


# DISABLED — enable and apply below if reporting loss-corrected squeezing.
def get_efficiency() -> float:
    while True:
        response = input(
            "\nTotal experimental detection efficiency (0-1, Enter for 1): "
        ).strip()

        if response == "":
            return 1.0

        try:
            efficiency = float(response)
            if 0 < efficiency <= 1:
                return efficiency
        except ValueError:
            pass

        print("Please enter a value greater than 0 and no greater than 1.")


# ------------------------------------------------------------------
# Input
# ------------------------------------------------------------------
k_unsq, se_unsq = get_slope_with_uncertainty(
    "Unsqueezed (shot-noise reference) regression:"
)
k_sq, se_sq = get_slope_with_uncertainty(
    "Squeezed light regression:"
)

if k_sq >= k_unsq:
    print(
        "\n[WARNING] k_sq >= k_unsq: the squeezed slope is not below the "
        "shot-noise reference. This indicates anti-squeezing, a data "
        "entry error, or swapped inputs. Proceeding with the calculation."
    )

# ------------------------------------------------------------------
# Point estimate
# ------------------------------------------------------------------
R = k_sq / k_unsq
squeezing_db = 10.0 * math.log10(R)
reduction_percent = (1.0 - R) * 100.0

# ------------------------------------------------------------------
# Uncertainty propagation (first-order delta method)
#
# R = k_sq / k_unsq  (independent datasets)
#
# (σ_R / R)² = (σ_sq / k_sq)² + (σ_unsq / k_unsq)²
#
# dB interval: transform R - σ_R and R + σ_R through log10 separately.
# This is asymmetric and avoids the symmetric first-order dB
# approximation, which is unreliable when σ_R / R is not small.
# ------------------------------------------------------------------
relative_variance = (se_sq / k_sq) ** 2 + (se_unsq / k_unsq) ** 2
sigma_ratio = R * math.sqrt(relative_variance)
relative_se = math.sqrt(relative_variance)

R_high = R + sigma_ratio
R_low = R - sigma_ratio

db_high = 10.0 * math.log10(R_high) if R_high > 0 else float("nan")
db_low = 10.0 * math.log10(R_low) if R_low > 0 else float("nan")

low_db_defined = R_low > 0

# Percent-reduction interval: higher R -> less reduction, lower R -> more.
reduction_at_R_low = (1.0 - R_low) * 100.0   # upper bound on reduction
reduction_at_R_high = (1.0 - R_high) * 100.0  # lower bound on reduction

# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------
print()
print("=" * 58)
print(f"  Noise ratio  R  =  {R:.6f} ± {sigma_ratio:.6f}")
print(f"  Relative SE      =  {relative_se * 100:.2f}%")

if low_db_defined:
    print(
        f"  Squeezing       =  {squeezing_db:.4f} dB "
        f"(asymmetric ±1 SE interval: [{db_low:.4f}, {db_high:.4f}] dB)"
    )
else:
    print(
        f"  Squeezing       =  {squeezing_db:.4f} dB "
        f"(upper +1 SE endpoint: {db_high:.4f} dB; "
        f"lower -1 SE endpoint: UNDEFINED because R - σ_R <= 0)"
    )

print("=" * 58)
print()
print(
    "Propagated SE assumes the two regression slopes are independent.\n"
    "Coverage depends on the degrees of freedom of each input fit\n"
    "and is exact only under Gaussian OLS errors."
)
print()

if not math.isfinite(sigma_ratio):
    print("[WARNING] Propagated uncertainty is not finite. Check inputs.")
elif squeezing_db < 0:
    if low_db_defined and math.isfinite(reduction_at_R_low) and math.isfinite(reduction_at_R_high):
        print(
            f"Observed squeezing: {abs(squeezing_db):.4f} dB below shot noise.\n"
            f"Noise reduction: {reduction_percent:.2f}% "
            f"(±1 SE range: {reduction_at_R_high:.2f}% to {reduction_at_R_low:.2f}%)"
        )
    else:
        print(
            f"Observed squeezing: {abs(squeezing_db):.4f} dB below shot noise.\n"
            f"Noise reduction: {reduction_percent:.2f}% "
            f"(lower dB bound undefined; reduction upper bound not computable)"
        )
elif squeezing_db > 0:
    print(
        f"Result is {squeezing_db:.4f} dB ABOVE shot noise "
        f"(anti-squeezing or swapped inputs)."
    )
else:
    print("The measured noise equals the shot-noise reference exactly.")