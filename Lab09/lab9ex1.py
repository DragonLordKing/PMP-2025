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
import matplotlib.pyplot as plt


def run_scenario(y_obs, theta):
    # label of the current (Y, θ) scenario
    label = f"Y={y_obs}, theta={theta}"
    print(f"\n=== {label} ===")

    # define Bayesian model for this scenario
    with pm.Model() as model:
        # prior for number of customers: n ~ Poisson(10)
        n = pm.Poisson("n", mu=10.0)
        # likelihood for observed buyers: Y | n,θ ~ Binomial(n, θ), observed = y_obs
        pm.Binomial("y_obs", n=n, p=theta, observed=y_obs)
        # future buyers: Y* | n,θ ~ Binomial(n, θ) (unobserved, used for prediction)
        pm.Binomial("y_future", n=n, p=theta)

        # sample from posterior p(n, y_future | data)
        trace = pm.sample()
        # sample posterior predictive for Y* given posterior samples of n
        ppc = pm.sample_posterior_predictive(trace, var_names=["y_future"])

    # print summary for posterior of n
    print(az.summary(trace, var_names=["n"]))

    # flatten predictive samples for Y*
    y_future_samples = ppc.posterior_predictive["y_future"].values.flatten()
    print("Predictive mean Y*:", float(np.mean(y_future_samples)))

    return label, trace, y_future_samples


def main():
    # values of observed Y to analyze
    y_values = [0, 5, 10]
    # values of θ (purchase probability) to analyze
    theta_values = [0.2, 0.5]

    # store posterior (for n) and predictive (for Y*) per scenario
    posterior_results = []
    predictive_results = []

    # loop all (Y, θ) scenarios and run inference
    for y_obs in y_values:
        for theta in theta_values:
            # run model, get posterior for n and predictive samples for Y*
            label, trace, y_future_samples = run_scenario(y_obs, theta)
            posterior_results.append((label, trace))
            predictive_results.append((label, y_future_samples))

    # 3x2 grid for posterior p(n | Y, θ)
    fig_post, axes_post = plt.subplots(len(y_values), len(theta_values), figsize=(10, 8))
    axes_post = np.atleast_2d(axes_post)

    for idx, (label, trace) in enumerate(posterior_results):
        i = idx // len(theta_values)
        j = idx % len(theta_values)
        az.plot_posterior(trace, var_names=["n"], ax=axes_post[i, j])
        axes_post[i, j].set_title(label)

    fig_post.suptitle("Posterior distributions of n", fontsize=14)
    fig_post.tight_layout(rect=[0, 0, 1, 0.96])

    # 3x2 grid for posterior predictive p(Y* | Y, θ)
    fig_pred, axes_pred = plt.subplots(len(y_values), len(theta_values), figsize=(10, 8))
    axes_pred = np.atleast_2d(axes_pred)

    for idx, (label, y_future_samples) in enumerate(predictive_results):
        i = idx // len(theta_values)
        j = idx % len(theta_values)
        az.plot_dist(y_future_samples, ax=axes_pred[i, j])
        axes_pred[i, j].set_title(label)

    fig_pred.suptitle("Posterior predictive distributions of Y*", fontsize=14)
    fig_pred.tight_layout(rect=[0, 0, 1, 0.96])

    # actually show the figures when running as a script
    plt.show()


if __name__ == "__main__":
    main()