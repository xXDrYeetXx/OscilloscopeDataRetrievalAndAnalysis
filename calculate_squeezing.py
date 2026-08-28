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
    σ_dB = (10 / ln10) * (σ_R / R)

All uncertainties are ±1 standard error (~68% confidence). Both
datasets must use identical measurement conditions so that any
systematic offset in absolute noise power cancels in the ratio.

Note: get_efficiency() is defined but not applied. Enable it if
reporting loss-corrected source squeezing rather than directly
observed squeezing.
"""

import math
def get_positive_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
        except ValueError:
            pass
        print("Please enter a number greater than zero.")


def get_slope_with_uncertainty(label):
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


def get_efficiency():
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
    "Unsqueezed light regression:"
)
k_sq, se_sq = get_slope_with_uncertainty(
    "Squeezed light regression:"
)

# ------------------------------------------------------------------
# Point estimate
# ------------------------------------------------------------------
measured_noise_ratio = k_sq / k_unsq
measured_level_db = 10.0 * math.log10(measured_noise_ratio)
measured_reduction_percent = (1.0 - measured_noise_ratio) * 100.0

# ------------------------------------------------------------------
# Uncertainty propagation (first-order delta method)
#
# R = k_sq / k_unsq
#
# (σ_R / R)² = (σ_sq / k_sq)² + (σ_unsq / k_unsq)²
#
# dB = 10 * log10(R)  =>  σ_dB = (10 / ln10) * (σ_R / R)
# ------------------------------------------------------------------
if k_sq > 0 and k_unsq > 0:
    relative_variance = (se_sq / k_sq) ** 2 + (se_unsq / k_unsq) ** 2
    sigma_ratio = measured_noise_ratio * math.sqrt(relative_variance)
    sigma_db = (10.0 / math.log(10.0)) * math.sqrt(relative_variance)
else:
    sigma_ratio = float("nan")
    sigma_db = float("nan")

# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------
ratio_low = measured_noise_ratio - sigma_ratio
ratio_high = measured_noise_ratio + sigma_ratio
reduction_low = (1.0 - ratio_high) * 100.0
reduction_high = (1.0 - ratio_low) * 100.0

print()
print("=" * 50)
print(f"  Noise ratio  R = {measured_noise_ratio:.6f} ± {sigma_ratio:.6f}")
print(f"  Squeezing      = {measured_level_db:.4f} ± {sigma_db:.4f} dB")
print("=" * 50)
print("")
print("The uncertainty here is the same as of the numbers inputted.")
if measured_level_db < 0:
    print(
        f"This is a {abs(measured_level_db):.4f} ± {sigma_db:.4f} dB squeezing reduction "
        f"({reduction_low:.2f}% to {reduction_high:.2f}% lower noise than shot noise)."
    )
elif measured_level_db > 0:
    print(
        f"\nThis is {measured_level_db:.4f} ± {sigma_db:.4f} dB above shot noise "
        "(anti-squeezing rather than squeezing)."
    )
else:
    print("\nThe measured noise equals the shot-noise reference.")