import numpy as np
import pymc as pm
import arviz as az

def main():
    data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

    x_bar = data.mean()
    s_hat = data.std(ddof=1)

    print(f"Sample mean (x̄) = {x_bar:.2f}")
    print(f"Sample std (s)   = {s_hat:.2f}")

    # (a) Weak priors
    print("(a) Weak prior model: μ~N(x̄,10²), σ~HalfNormal(10).")
    print("    Expectation: data dominate → μ near x̄, σ near s.")
    with pm.Model() as weak_model:
        mu = pm.Normal("mu", mu=x_bar, sigma=10.0)
        sigma = pm.HalfNormal("sigma", sigma=10.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=data)
        trace_weak = pm.sample()

    # (b) Posterior for weak model
    summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
    print("\n(b) Posterior (weak prior) — 95% HDI (credible intervals):")
    print(summary_weak)
    print(f"    Quick read: should see μ ≈ {x_bar:.2f} and σ ≈ {s_hat:.2f} (data-driven).")

    # (c) Frequentist baseline
    print("\n(c) Frequentist estimates (MLE baselines):")
    print(f"    Mean: {x_bar:.2f} | SD: {s_hat:.2f}")

    # (d) Strong prior on μ
    print("\n(d) Strong prior model: μ~N(50,1²), σ~HalfNormal(10).")
    print("    Expectation: prior shrinks μ toward 50; σ may inflate to cover mismatch.")
    with pm.Model() as strong_model:
        mu_s = pm.Normal("mu", mu=50.0, sigma=1.0)
        sigma_s = pm.HalfNormal("sigma", sigma=10.0)
        pm.Normal("y", mu=mu_s, sigma=sigma_s, observed=data)
        trace_strong = pm.sample()

    summary_strong = az.summary(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)
    print("\n(d) Posterior (strong prior) — 95% HDI:")
    print(summary_strong)
    print("    Quick read: μ pulled toward 50; larger σ often appears to reconcile prior vs data.")

if __name__ == "__main__":
    main()