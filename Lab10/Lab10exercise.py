"""
Lab 10 – Exercise 1 (Bayesian linear regression)

(A) Estimate the regression coefficients (alpha, beta) and noise sigma.
    - We write down a simple linear model:
          sales ≈ alpha + beta * publicity + noise
      where alpha is the intercept, beta is the slope, and the noise is just assumed to be Normal with std dev sigma.
    - We choose weakly-informative priors for alpha, beta, and sigma (Normal and HalfNormal, with scales copied from the example file).
    - Using PyMC, we run MCMC to approximate the posterior distributions of these three parameters given the data of the publicity and sales.

(B) Compute credible intervals (HDIs) for alpha and beta.
    - From the posterior samples that we got from the part A, we summarise alpha and beta: we print means, standard deviations, and convergence diagnostics.
    - We also compute 94% HDI (Highest Density Interval) for alpha and beta, which is what gives us a Bayesian confidence interval describing which values
      from either the intercept and slope are most compatible with the data.

(C) Predict future sales for new publicity levels.
    - We pick a few new publicity values in which we chpse 1.5, 4.0, 7.5, 10.5 for which we want the predictions of.
    - * take posterior samples of alpha, beta, sigma;
      * for each new publicity value, compute the mean prediction mu = alpha + beta * new_publicity;
      * add Gaussian noise using sigma to simulate future observations.
    - From these simulated future sales, we compute 90% predictive intervals for each new publicity level, which tells us a kind of plausible range of sales in which we might expect given the model and the observed data.
"""

import os
import warnings
import logging

os.environ["PYTENSOR_FLAGS"] = "cxx="
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar subtract")
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("arviz").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)
logging.getLogger("pytensor.configdefaults").setLevel(logging.ERROR)

import numpy as np
import pymc as pm
import arviz as az

# Data
# publicity: weekly advertising expenses (in thousands of dollars)
# sales: weekly sales revenue (in thousands of dollars)

publicity = np.array([
    1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0
])

sales = np.array([
    5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0, 15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0,
])

# New publicity levels (in thousands) for which we want predictions.
# This is used to solve part (C).
new_publicity = np.array([1.5, 4.0, 7.5, 10.5])

# Ex. 1 - Main question: build the Bayesian LR model
with pm.Model() as model:
    # Use pm.Data for the input feature
    x = pm.Data("publicity", publicity)

    # A - priors for regression coefficients and noise level, to estimate them.
    # These scales (20, 10, 5) are copied from the example file; we only changed the mean of alpha
    # from 50 to 0 because our sales are around 5–22, not around 50.
    alpha = pm.Normal("alpha", mu=0.0, sigma=20.0)   # intercept
    beta = pm.Normal("beta", mu=0.0, sigma=10.0)     # slope
    sigma = pm.HalfNormal("sigma", sigma=5.0)        # noise std dev

    # Linear mean of sales given publicity
    mu = alpha + beta * x

    # Likelihood: observed sales (length 20, fixed)
    pm.Normal("sales", mu=mu, sigma=sigma, observed=sales)

    # Sample from the posterior (A: estimate alpha, beta, sigma)
    idata = pm.sample(
        draws=2000,
        tune=1000,
        target_accept=0.9,
        random_seed=1,
        cores=1,
    )

# A - estimate regression coefficients (posterior summaries for alpha, beta, sigma)
print("\n=== Ex. 1(a): Posterior summaries for regression coefficients ===")
summary_coefs = az.summary(idata, var_names=["alpha", "beta", "sigma"])
print(summary_coefs)

# B - credible intervals (HDI) for coefficients
print("\n=== Ex. 1(b): HDIs for regression coefficients (94% HDI) ===")
coef_hdi = az.hdi(idata, var_names=["alpha", "beta"], hdi_prob=0.94)
print(coef_hdi)

# C - predict future revenues for new publicity levels

# Flatten posterior samples from all chains
# Computation using NumPy for the posterior predictive samples
alpha_samp = idata.posterior["alpha"].values.reshape(-1)  # shape: (S,)
beta_samp  = idata.posterior["beta"].values.reshape(-1)   # shape: (S,)
sigma_samp = idata.posterior["sigma"].values.reshape(-1)  # shape: (S,)

S = alpha_samp.shape[0]
J = new_publicity.shape[0]

# Compute mean predictions for each posterior sample and each new x
# mu_pred[s, j] = alpha_samp[s] + beta_samp[s] * new_publicity[j]
mu_pred = alpha_samp[:, None] + beta_samp[:, None] * new_publicity[None, :]

# Add observation noise to get full posterior predictive samples
rng = np.random.default_rng(1)
eps = rng.normal(loc=0.0, scale=1.0, size=(S, J))
y_pred = mu_pred + sigma_samp[:, None] * eps

# Compute 90% predictive intervals for each new publicity level
pred_int = np.percentile(y_pred, [5, 95], axis=0)

print("\n=== Ex. 1(c): Predictive intervals for future sales ===")
for x_val, lo, hi in zip(new_publicity, pred_int[0], pred_int[1]):
    print(f"publicity = {x_val:4.1f}k -> sales ~ {lo:5.2f}k to {hi:5.2f}k (90% PI)")