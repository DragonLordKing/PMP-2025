"""
Here we model exam score as a function of study hours, then:
- estimate intercept/slope posteriors
- get HDIs for the coefficients
- make predictions for new study hours
"""

import numpy as np
import pymc as pm
import arviz as az


# Toy data: hours studied vs exam score (made-up, noisy)
rng = np.random.default_rng(0)
hours = np.linspace(1, 12, 15)
scores = 40 + 5 * hours + rng.normal(0, 3, size=hours.size)

# New inputs to forecast
new_hours = np.array([2.0, 6.0, 10.0])

with pm.Model() as model:
    x = pm.MutableData("hours", hours)

    alpha = pm.Normal("alpha", mu=50, sigma=20)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = alpha + beta * x
    pm.Normal("score", mu=mu, sigma=sigma, observed=scores)

    idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=1)

    # 94% HDIs for coefficients
    coef_hdi = az.hdi(idata, var_names=["alpha", "beta"], hdi_prob=0.94)
    print("\nCoefficient HDIs (94%):")
    print(coef_hdi)

    # Predict on new inputs
    pm.set_data({"hours": new_hours})
    ppc = pm.sample_posterior_predictive(idata, var_names=["score"], random_seed=1)

# 90% predictive intervals for each forecasted score
pred_int = np.percentile(ppc.posterior_predictive["score"], [5, 95], axis=(0, 1))
print("\nPredictive intervals (5th, 95th) for new_hours:")
for h, lo, hi in zip(new_hours, pred_int[0], pred_int[1]):
    print(f"hours={h:>4.1f} -> score ~ {lo:5.2f} to {hi:5.2f}")
