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


def get_efficiency():
    while True:
        response = input(
            "Total experimental detection efficiency (0-1, Enter for 1): "
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


unsqueezed_slope = get_positive_float(
    "For the Linear Noise (nW) regression of the unsqueezed light, "
    "what was slope? "
)

squeezed_slope = get_positive_float(
    "For the Linear Noise (nW) regression of the squeezed light, "
    "what was slope? "
)

# Dark-noise-corrected normalized quadrature-noise variance.
measured_noise_ratio = squeezed_slope / unsqueezed_slope

# Signed noise level relative to shot noise:
# negative = squeezing, zero = no squeezing, positive = anti-squeezing.
measured_level_db = 10.0 * math.log10(measured_noise_ratio)
measured_reduction_percent = (1.0 - measured_noise_ratio) * 100.0

print()
print(f"The calculated squeezing is {measured_level_db:.4f} dB.")

if measured_level_db < 0:
    print(
        f"This is a {abs(measured_level_db):.4f} dB squeezing reduction "
        f"({measured_reduction_percent:.2f}% lower noise than shot noise)."
    )
elif measured_level_db > 0:
    print(
        f"This is {measured_level_db:.4f} dB above shot noise "
        "(anti-squeezing rather than squeezing)."
    )
else:
    print("The measured noise equals the shot-noise reference.")