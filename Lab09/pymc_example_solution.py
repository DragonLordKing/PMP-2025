"""
Lab 9 - PyMC worked example.

Scenario:
- A/B test comparing click-through rates of two email subject lines.
- Observed conversions: A = 240 out of 4,000 sends, B = 300 out of 4,000.
Goal:
- Infer posterior CTRs, the probability that B beats A, and the lift of B over A.

Model (Beta-Binomial for each arm):
- p_a, p_b ~ Beta(1, 1) (uniform on CTR).
- y_a ~ Binomial(n_a, p_a), y_b ~ Binomial(n_b, p_b) with observed counts.
- Derived quantities: diff = p_b - p_a, lift = diff / p_a.

How to run:
- python Lab09/pymc_example_solution.py
- This keeps draws modest so it runs quickly on a laptop while showing the
  essential PyMC workflow: model, sample, summarize, and simple comparisons.
"""

import numpy as np
import arviz as az
import pymc as pm

# Observed A/B outcomes
n_a, k_a = 4000, 240
n_b, k_b = 4000, 300
obs_rate_a, obs_rate_b = k_a / n_a, k_b / n_b

print("PyMC Bayesian A/B test example (different problem than the lab task)")
print(f"Observed CTRs: A = {obs_rate_a:.3%} ({k_a}/{n_a}), B = {obs_rate_b:.3%} ({k_b}/{n_b})")

with pm.Model() as model:
    p_a = pm.Beta("p_a", alpha=1.0, beta=1.0)
    p_b = pm.Beta("p_b", alpha=1.0, beta=1.0)

    pm.Binomial("y_a", n=n_a, p=p_a, observed=k_a)
    pm.Binomial("y_b", n=n_b, p=p_b, observed=k_b)

    diff = pm.Deterministic("diff", p_b - p_a)
    lift = pm.Deterministic("lift", diff / p_a)

    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=2025,
        progressbar=False,
    )

summary = az.summary(trace, var_names=["p_a", "p_b", "diff", "lift"], hdi_prob=0.94)
print("\nPosterior summary (94% HDIs):")
print(summary)

posterior = trace.posterior
diff_samples = np.ravel(posterior["diff"].values)
lift_samples = np.ravel(posterior["lift"].values)

prob_b_better = float((diff_samples > 0).mean())
diff_hdi = az.hdi(diff_samples, hdi_prob=0.94)
lift_hdi = az.hdi(lift_samples, hdi_prob=0.94)

print("\nPosterior comparisons:")
print(f"P(B has higher CTR than A) ~= {prob_b_better:.3f}")
print(f"Median lift ((p_b - p_a) / p_a): {float(np.median(lift_samples)):.3%}")
print(f"94% HDI for difference p_b - p_a: [{diff_hdi[0]:.4f}, {diff_hdi[1]:.4f}]")
print(f"94% HDI for lift: [{lift_hdi[0]:.3%}, {lift_hdi[1]:.3%}]")
