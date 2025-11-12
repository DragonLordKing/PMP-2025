"""
Lab 6 – Exercise 2 solution, written simply and imperatively.

Model:
- Over T hours, total calls k observed, with X ~ Poisson(T * λ).
- Prior for λ: Gamma(α0, β0) in rate parameterization.
- Posterior: Gamma(α0 + k, β0 + T).
"""

import numpy as np

print("Ex. 2 – Poisson–Gamma")

# Data and prior
k = 180
T = 10.0
alpha0 = 1.0
beta0 = 1e-6  # rate parameterization

# Conjugate posterior parameters
alpha = alpha0 + k
beta = beta0 + T

# Posterior summaries
post_mean = alpha / beta
post_mode = max(alpha - 1.0, 0.0) / beta

# 94% HDI via sampling from Gamma(alpha, beta)
draws = 200_000
rng = np.random.default_rng(2025)
samples = rng.gamma(shape=alpha, scale=1.0 / beta, size=draws)
s = np.sort(samples)
m = int(0.94 * draws)
widths = s[m:] - s[: draws - m]
j = int(np.argmin(widths))
hdi_low, hdi_high = float(s[j]), float(s[j + m])

print(f"Data: k={k} calls over T={T:g} hours")
print(f"Prior: Gamma(alpha0={alpha0}, beta0={beta0}) [rate parametrization]")
print(f"Posterior: Gamma(alpha={alpha}, beta={beta})")
print(f"(a) Posterior mean of λ = {post_mean:.6f}")
print(f"(b) 94% HDI for λ = [{hdi_low:.6f}, {hdi_high:.6f}]")
print(f"(c) Posterior mode of λ = {post_mode:.6f}")

