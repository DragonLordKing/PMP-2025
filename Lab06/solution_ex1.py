"""
Lab 6 – Exercise 1 solutions (both closed-form and empirical),
written simply and imperatively.

Problem data:
- Prevalence p = P(B) = 0.01
- Sensitivity tpr = P(T+ | B) = 0.95
- Specificity sp = P(T- | ~B) = 0.90

Closed-form derivation (sketch):
- Let B = "has disease", ~B = "no disease", and T+ = "positive test".
- Bayes: P(B|T+) = P(T+|B) P(B) / [ P(T+|B) P(B) + P(T+|~B) P(~B) ].
- Define tpr = P(T+|B) (sensitivity) and sp = P(T-|~B) (specificity), so
  P(T+|~B) = 1 - sp and P(~B) = 1 - p.
- Then PPV = [p * tpr] / [p * tpr + (1 - p) * (1 - sp)].

Minimum specificity for PPV ≥ target (algebra):
- Start with target ≤ [p tpr] / [p tpr + (1-p)(1-s)]. Cross-multiply (denominator > 0):
  target [p tpr + (1-p)(1-s)] ≤ p tpr.
- Expand: target p tpr + target (1-p) (1-s) ≤ p tpr.
- Rearrange: target (1-p) (1-s) ≤ p tpr (1 - target).
- Divide by target (1-p) (> 0): (1 - s) ≤ [p tpr (1 - target)] / [target (1 - p)].
- Hence s ≥ 1 - [p tpr (1 - target)] / [target (1 - p)]. The minimal s is the RHS.
"""

import numpy as np

print("Ex. 1 – Bayes' Theorem")

# Given parameters
p = 0.01       # prevalence P(B)
tpr = 0.95     # sensitivity P(T+ | B)
sp = 0.90      # specificity P(T- | ~B)
target = 0.50  # target PPV threshold

# ----------------------- Closed-form (derivation) ----------------------- #

# PPV = P(B | T+) = [p * tpr] / [p * tpr + (1 - p) * (1 - sp)]
numerator = p * tpr
denominator = numerator + (1.0 - p) * (1.0 - sp)
ppv_closed = numerator / denominator

# Minimum specificity for PPV >= target (solve equality, clamp to [0,1])
# sp_min = 1 - [p * tpr * (1 - target)] / [target * (1 - p)]
sp_min_closed = 1.0 - (p * tpr * (1.0 - target)) / (target * (1.0 - p))
sp_min_closed = max(0.0, min(1.0, sp_min_closed))

print("Closed-form solution:")
print(f"(a) PPV = {ppv_closed:.6f} ({ppv_closed*100:.2f}%)")
print(f"(b) min specificity for PPV ≥ {int(target*100)}%: {sp_min_closed:.6f} ({sp_min_closed*100:.2f}%)")
print("Derivation (PPV): P(B|T+) = tpr·p / [tpr·p + (1-sp)·(1-p)]")
print("Derivation (min sp): s ≥ 1 - [p·tpr·(1-target)] / [target·(1-p)]")
print()

# ---------------------------- Empirical (MC) ---------------------------- #

print("Empirical (Monte Carlo) solution:")

# Monte Carlo estimate for PPV = P(B | T+)
n = 1_000_000
rng = np.random.default_rng(2025)

diseased = rng.random(n) < p
u = rng.random(n)
test_pos = np.where(diseased, u < tpr, u < (1.0 - sp))
tp = np.count_nonzero(test_pos & diseased)
fp = np.count_nonzero(test_pos & ~diseased)
ppv_emp = tp / (tp + fp)

print(f"(a) Estimated PPV ≈ {ppv_emp:.6f} ({ppv_emp*100:.2f}%) [Monte Carlo, n=1e6]")

# Empirical bisection to find minimum specificity for PPV >= target
lo, hi = 0.0, 1.0
tol = 2e-4
max_iter = 28
n_per_iter = 300_000

for _ in range(max_iter):
    mid = 0.5 * (lo + hi)
    diseased_bis = rng.random(n_per_iter) < p
    u_bis = rng.random(n_per_iter)
    test_pos_bis = np.where(diseased_bis, u_bis < tpr, u_bis < (1.0 - mid))
    tp_bis = np.count_nonzero(test_pos_bis & diseased_bis)
    fp_bis = np.count_nonzero(test_pos_bis & ~diseased_bis)
    ppv_mid = tp_bis / (tp_bis + fp_bis) if (tp_bis + fp_bis) > 0 else np.nan
    if ppv_mid >= target:
        hi = mid
    else:
        lo = mid
    if hi - lo < tol:
        break

sp_min_emp = hi
print(f"(b) Estimated min specificity ≈ {sp_min_emp:.6f} ({sp_min_emp*100:.2f}%) [empirical bisection]")
