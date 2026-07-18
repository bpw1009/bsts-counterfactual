"""
Bayesian Structural Time Series (BSTS) Counterfactual Analysis
==============================================================

This is the fully Bayesian version of the counterfactual analysis,
using PyMC with MCMC (NUTS sampler) for posterior inference.

Same structural decomposition as the OLS version:
  - Linear trend
  - Fourier seasonality (52-week cycle)
  - Control series regression

The difference: instead of OLS point estimates with frequentist prediction
intervals, this version produces full posterior distributions over all model
parameters, yielding genuine Bayesian credible intervals on the counterfactual.

Dependencies: numpy, pandas, matplotlib, pymc, arviz, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy import stats

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# SECTION 1: DATA SIMULATION
# ═══════════════════════════════════════════════════════════════
# Replace this section with your actual data.
# Requirements:
#   - weekly_sales: the treated time series (advertiser sales)
#   - category_sales: a control series sharing market dynamics
#   - intervention_week: index where the intervention occurred

n_weeks = 104  # 2 years
t = np.arange(n_weeks)

# Trend
trend = 1000 + 3 * t

# Seasonality (52-week cycle)
seasonality = (80 * np.sin(2 * np.pi * t / 52)
               + 30 * np.cos(2 * np.pi * t / 52)
               + 20 * np.sin(4 * np.pi * t / 52))

# Control series: category-level sales (correlated but independent of intervention)
category_base = 5000 + 8 * t + 200 * np.sin(2 * np.pi * t / 52)
category_noise = np.random.normal(0, 100, n_weeks)
category_sales = category_base + category_noise

# Advertiser sales = trend + seasonality + regression on category + noise
beta_category = 0.15
noise = np.random.normal(0, 40, n_weeks)
advertiser_sales = trend + seasonality + beta_category * category_sales + noise

# Intervention: advertiser stops spending at week 78
intervention_week = 78
spend_effect = np.zeros(n_weeks)
spend_effect[intervention_week:] = -(80 + 2 * np.arange(n_weeks - intervention_week))
advertiser_sales_observed = advertiser_sales + spend_effect

# What we observe
weekly_sales = advertiser_sales_observed.copy()

print("=" * 65)
print("BSTS COUNTERFACTUAL ANALYSIS (BAYESIAN / PyMC)")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════
# SECTION 2: FEATURE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def build_features(t, n_harmonics=3, period=52):
    """Build the design matrix: trend + Fourier harmonics."""
    features = {"trend": t.astype(float)}
    for k in range(1, n_harmonics + 1):
        features[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        features[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features)

X = build_features(t)
X["category"] = category_sales

# Split pre/post
pre_idx = t < intervention_week
post_idx = t >= intervention_week

X_pre = X[pre_idx].values
y_pre = weekly_sales[pre_idx]
X_post = X[post_idx].values
y_post = weekly_sales[post_idx]

feature_names = list(X.columns)
n_features = len(feature_names)

print(f"\nPre-period:  weeks 0-{intervention_week - 1} ({pre_idx.sum()} weeks)")
print(f"Post-period: weeks {intervention_week}-{n_weeks - 1} ({post_idx.sum()} weeks)")
print(f"Features: {feature_names}")

# ═══════════════════════════════════════════════════════════════
# SECTION 3: BAYESIAN MODEL (PyMC + MCMC)
# ═══════════════════════════════════════════════════════════════

print("\nFitting Bayesian model (MCMC via NUTS)...")

with pm.Model() as bsts_model:
    # Priors on regression coefficients
    # Weakly informative: centered at 0 with moderate spread
    beta = pm.Normal("beta", mu=0, sigma=100, shape=n_features)

    # Intercept
    alpha = pm.Normal("alpha", mu=np.mean(y_pre), sigma=500)

    # Observation noise
    sigma = pm.HalfNormal("sigma", sigma=100)

    # Mean function: intercept + X @ beta
    mu = alpha + pm.math.dot(X_pre, beta)

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_pre)

    # Sample posterior
    trace = pm.sample(
        2000,
        tune=1000,
        chains=4,
        cores=1,
        random_seed=42,
        progressbar=True,
        return_inferencedata=True,
    )

print("\nSampling complete.")

# ═══════════════════════════════════════════════════════════════
# SECTION 4: POSTERIOR PREDICTIVE — COUNTERFACTUAL
# ═══════════════════════════════════════════════════════════════

# Extract posterior samples
alpha_samples = trace.posterior["alpha"].values.reshape(-1)
beta_samples = trace.posterior["beta"].values.reshape(-1, n_features)
sigma_samples = trace.posterior["sigma"].values.reshape(-1)
n_samples = len(alpha_samples)

print(f"\nPosterior samples: {n_samples}")

# Generate counterfactual predictions for the FULL time series
X_full = X.values

# Counterfactual: predicted sales at each week from each posterior sample
cf_samples = np.zeros((n_samples, n_weeks))
for i in range(n_samples):
    mu_i = alpha_samples[i] + X_full @ beta_samples[i]
    cf_samples[i] = mu_i + np.random.normal(0, sigma_samples[i], n_weeks)

cf_mean = cf_samples.mean(axis=0)
cf_lower = np.percentile(cf_samples, 2.5, axis=0)
cf_upper = np.percentile(cf_samples, 97.5, axis=0)

# Pre-period fit
pre_fitted = cf_mean[pre_idx]
pre_residuals = y_pre - pre_fitted

# Causal impact in post-period
post_cf = cf_mean[post_idx]
post_actual = y_post
point_impact = post_actual - post_cf
cumulative_impact = np.cumsum(point_impact)

print(f"\n--- COUNTERFACTUAL RESULTS (100% Spend Scenario) ---")
print(f"Mean weekly impact:       {point_impact.mean():+.1f}")
print(f"Cumulative impact:        {cumulative_impact[-1]:+.1f}")
print(f"Relative effect:          {(point_impact.mean() / post_cf.mean()) * 100:+.1f}%")

# ═══════════════════════════════════════════════════════════════
# SECTION 5: THREE SPEND SCENARIOS
# ═══════════════════════════════════════════════════════════════

scenarios = {
    "A: 50% Spend":  0.50,
    "B: 100% Spend": 1.00,
    "C: 150% Spend": 1.50,
}

# The counterfactual at 100% is what we already have.
# Scenarios scale the estimated post-period gap.
# Baseline gap per week = counterfactual - actual (the revenue left on the table)
baseline_gap = post_cf - post_actual

scenario_results = {}
scenario_samples = {}

for name, multiplier in scenarios.items():
    # Projected sales = actual + (gap * multiplier)
    projected = post_actual + baseline_gap * multiplier

    # For credible intervals, use the posterior samples
    scenario_cf = np.zeros((n_samples, post_idx.sum()))
    for i in range(n_samples):
        mu_i = alpha_samples[i] + X_post @ beta_samples[i]
        noise_i = np.random.normal(0, sigma_samples[i], post_idx.sum())
        full_cf_i = mu_i + noise_i
        gap_i = full_cf_i - post_actual
        scenario_cf[i] = post_actual + gap_i * multiplier

    scenario_mean = scenario_cf.mean(axis=0)
    scenario_lower = np.percentile(scenario_cf, 2.5, axis=0)
    scenario_upper = np.percentile(scenario_cf, 97.5, axis=0)

    uplift = ((scenario_mean.mean() - post_actual.mean()) / post_actual.mean()) * 100

    scenario_results[name] = {
        "mean": scenario_mean,
        "lower": scenario_lower,
        "upper": scenario_upper,
        "uplift": uplift,
    }
    print(f"{name}: {uplift:+.1f}% weekly uplift")

# ═══════════════════════════════════════════════════════════════
# SECTION 6: VALIDATION
# ═══════════════════════════════════════════════════════════════

print("\n--- VALIDATION ---")

# Validation 1: Pre-period MAPE
mape = np.mean(np.abs(pre_residuals) / np.abs(y_pre)) * 100
print(f"1. Pre-period MAPE: {mape:.2f}%")

# Validation 2: Prediction interval calibration
pre_cf_samples = cf_samples[:, pre_idx]
pre_lower = np.percentile(pre_cf_samples, 2.5, axis=0)
pre_upper = np.percentile(pre_cf_samples, 97.5, axis=0)
coverage = np.mean((y_pre >= pre_lower) & (y_pre <= pre_upper)) * 100
print(f"2. 95% credible interval coverage: {coverage:.1f}%")

# Validation 3: Residual diagnostics
shapiro_stat, shapiro_p = stats.shapiro(pre_residuals[:50])  # Shapiro limited to 5000
dw_stat = np.sum(np.diff(pre_residuals) ** 2) / np.sum(pre_residuals ** 2)
print(f"3. Residual diagnostics:")
print(f"   Shapiro-Wilk p-value: {shapiro_p:.4f} ({'normal' if shapiro_p > 0.05 else 'non-normal'})")
print(f"   Durbin-Watson: {dw_stat:.2f} (target ~2.0)")

# Validation 4: Placebo test on control series
print(f"4. Placebo test on control series...")

with pm.Model() as placebo_model:
# Use same features MINUS category (since category is the target in the placebo)
    X_pre_placebo = X_pre[:, :-1]  # drop last column (category)
    X_post_placebo = X_post[:, :-1]
    n_features_placebo = n_features - 1

    beta_p = pm.Normal("beta_p", mu=0, sigma=100, shape=n_features_placebo)
    alpha_p = pm.Normal("alpha_p", mu=np.mean(category_sales[pre_idx]), sigma=500)
    sigma_p = pm.HalfNormal("sigma_p", sigma=100)

    mu_p = alpha_p + pm.math.dot(X_pre_placebo, beta_p)
    y_obs_p = pm.Normal("y_obs_p", mu=mu_p, sigma=sigma_p,
                        observed=category_sales[pre_idx])

    trace_p = pm.sample(
        1000,
        tune=500,
        chains=4,
        cores=1,
        target_accept=0.95,
        random_seed=43,
        progressbar=True,
        return_inferencedata=True,
    )

alpha_p_samples = trace_p.posterior["alpha_p"].values.reshape(-1)
beta_p_samples = trace_p.posterior["beta_p"].values.reshape(-1, n_features_placebo)
sigma_p_samples = trace_p.posterior["sigma_p"].values.reshape(-1)

placebo_cf = np.zeros((len(alpha_p_samples), post_idx.sum()))
for i in range(len(alpha_p_samples)):
    mu_i = alpha_p_samples[i] + X_post_placebo @ beta_p_samples[i]
    placebo_cf[i] = mu_i + np.random.normal(0, sigma_p_samples[i], post_idx.sum())

placebo_mean = placebo_cf.mean(axis=0)
placebo_impact = category_sales[post_idx] - placebo_mean
placebo_relative = (placebo_impact.mean() / placebo_mean.mean()) * 100

print(f"   Placebo relative effect: {placebo_relative:+.2f}% (should be ~0%)")
if abs(placebo_relative) < 3:
    print(f"   ✓ No spurious effect detected — methodology is sound")
else:
    print(f"   ⚠ Possible spurious effect — review control series")

# ═══════════════════════════════════════════════════════════════
# SECTION 7: POSTERIOR DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

print("\n--- POSTERIOR DIAGNOSTICS ---")
summary = az.summary(trace, var_names=["alpha", "beta", "sigma"])
print(summary.to_string())

# Check convergence
rhat_max = float(summary["r_hat"].max())
print(f"\nMax R-hat: {rhat_max:.3f} ({'converged' if rhat_max < 1.05 else 'WARNING: not converged'})")
ess_min = float(summary["ess_bulk"].min())
print(f"Min ESS (bulk): {ess_min:.0f} ({'adequate' if ess_min > 400 else 'WARNING: low ESS'})")

# ═══════════════════════════════════════════════════════════════
# SECTION 8: PLOTS
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("BSTS Counterfactual Analysis (Bayesian / PyMC)", fontsize=14, fontweight="bold")

# Plot 1: Counterfactual vs Actual (100% scenario)
ax = axes[0, 0]
ax.plot(t, weekly_sales, "k-", linewidth=1.2, label="Observed", alpha=0.8)
ax.plot(t[post_idx], cf_mean[post_idx], "b--", linewidth=1.2, label="Counterfactual (100%)")
ax.fill_between(t[post_idx], cf_lower[post_idx], cf_upper[post_idx],
                alpha=0.2, color="blue", label="95% credible interval")
ax.axvline(intervention_week, color="red", linestyle=":", alpha=0.7, label="Intervention")
ax.set_title("Counterfactual vs. Observed")
ax.set_xlabel("Week")
ax.set_ylabel("Sales")
ax.legend(fontsize=8)

# Plot 2: Three Scenarios
ax = axes[0, 1]
ax.plot(t[post_idx], post_actual, "k-", linewidth=1.5, label="Actual (no spend)")
colors = ["#2ecc71", "#3498db", "#e74c3c"]
for (name, result), color in zip(scenario_results.items(), colors):
    ax.plot(t[post_idx], result["mean"], "--", color=color, linewidth=1.2,
            label=f'{name} ({result["uplift"]:+.0f}%)')
    ax.fill_between(t[post_idx], result["lower"], result["upper"],
                    alpha=0.1, color=color)
ax.set_title("Spend Scenarios with Credible Intervals")
ax.set_xlabel("Week")
ax.set_ylabel("Sales")
ax.legend(fontsize=8)

# Plot 3: Pre-period fit
ax = axes[1, 0]
ax.plot(t[pre_idx], y_pre, "k-", linewidth=1, label="Observed", alpha=0.8)
ax.plot(t[pre_idx], pre_fitted, "b-", linewidth=1, label="Fitted", alpha=0.8)
ax.fill_between(t[pre_idx], pre_lower, pre_upper,
                alpha=0.15, color="blue", label="95% credible interval")
ax.set_title(f"Pre-period Fit (MAPE: {mape:.1f}%)")
ax.set_xlabel("Week")
ax.set_ylabel("Sales")
ax.legend(fontsize=8)

# Plot 4: Placebo test
ax = axes[1, 1]
ax.plot(t[post_idx], category_sales[post_idx], "k-", linewidth=1,
        label="Category (observed)", alpha=0.8)
ax.plot(t[post_idx], placebo_mean, "b--", linewidth=1,
        label="Category (counterfactual)")
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.set_title(f"Placebo Test ({placebo_relative:+.1f}% — expect ~0%)")
ax.set_xlabel("Week")
ax.set_ylabel("Sales")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("bsts_bayesian_main.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nPlots saved: bsts_bayesian_main.png")

# ═══════════════════════════════════════════════════════════════
# SECTION 9: POSTERIOR DISTRIBUTION PLOTS
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Posterior Distributions & Convergence", fontsize=14, fontweight="bold")

# Plot 1: Posterior of causal impact (post-period)
post_impact_samples = cf_samples[:, post_idx].mean(axis=1) - post_actual.mean()
ax = axes[0, 0]
ax.hist(post_impact_samples, bins=50, density=True, alpha=0.7, color="#3498db")
ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="Zero effect")
ax.axvline(post_impact_samples.mean(), color="black", linestyle="-", linewidth=2,
           label=f"Mean: {post_impact_samples.mean():+.1f}")
ci_low, ci_high = np.percentile(post_impact_samples, [2.5, 97.5])
ax.axvline(ci_low, color="gray", linestyle=":", alpha=0.7)
ax.axvline(ci_high, color="gray", linestyle=":", alpha=0.7,
           label=f"95% CI: [{ci_low:+.0f}, {ci_high:+.0f}]")
ax.set_title("Posterior: Mean Weekly Causal Impact")
ax.set_xlabel("Impact (sales units)")
ax.legend(fontsize=8)

# Plot 2: Posterior of sigma (observation noise)
ax = axes[0, 1]
ax.hist(sigma_samples, bins=50, density=True, alpha=0.7, color="#2ecc71")
ax.set_title("Posterior: Observation Noise (σ)")
ax.set_xlabel("σ")

# Plot 3: Trace plot for a key coefficient (category beta)
category_idx = feature_names.index("category")
ax = axes[1, 0]
for chain in range(trace.posterior.dims["chain"]):
    chain_samples = trace.posterior["beta"].values[chain, :, category_idx]
    ax.plot(chain_samples, alpha=0.5, linewidth=0.5)
ax.set_title(f"Trace: β_category (convergence check)")
ax.set_xlabel("Sample")
ax.set_ylabel("Value")

# Plot 4: Residual diagnostics
ax = axes[1, 1]
ax.scatter(pre_fitted, pre_residuals, alpha=0.5, s=20, color="#e74c3c")
ax.axhline(0, color="black", linestyle="-", alpha=0.3)
ax.set_title("Residuals vs. Fitted (Pre-period)")
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")

plt.tight_layout()
plt.savefig("bsts_bayesian_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plots saved: bsts_bayesian_diagnostics.png")

# Probability that the causal impact is negative
prob_negative = np.mean(post_impact_samples < 0) * 100
print(f"\n--- BAYESIAN INFERENCE ---")
print(f"Posterior probability that stopping spend REDUCED sales: {prob_negative:.1f}%")
print(f"95% credible interval for weekly impact: [{ci_low:+.1f}, {ci_high:+.1f}]")
print(f"\nThis is a direct probability statement: there is a {prob_negative:.0f}% probability")
print(f"that the true causal effect of stopping spend is negative.")
print("=" * 65)
