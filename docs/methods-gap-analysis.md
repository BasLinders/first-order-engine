# Methods Gap Analysis

How FOE's methods compare with current experimentation platforms (Eppo, Statsig, Optimizely, GrowthBook, Spotify Confidence) and published practice at Microsoft ExP, Booking and Netflix. Written November 2026.

## Current coverage

| Area | FOE | Compared with platforms |
|---|---|---|
| Fixed-horizon frequentist | z-test, Šidák, Newcombe CI, Boschloo exact, non-inferiority, overdispersion factor φ | On par; Boschloo is rare |
| Decision risk | False-positive risk (incl. at observed p), false-negative discovery rate | Ahead; rarely offered |
| Variance reduction | CUPED, Lin's adjustment with HC3 | On par |
| Bayesian | Beta-Binomial, log-lift prior, P(best), expected loss, monetary projection | On par for conversions |
| Sequential | mSPRT (τ from MDE), group sequential with Lan-DeMets spending | On par for conversions |
| Continuous metrics | Welch ANOVA, Tukey, Games-Howell, Dunn, Kruskal-Wallis, Mann-Whitney, Gamma hurdle, negative binomial, winsorizing | Ahead |
| Planning | Sample size, MDE, power (conversion, continuous, RPV), Holm, forecast-driven MDE | On par or ahead |
| Diagnostics | SRM with segment breakdown, interactions between concurrent tests | Ahead; interaction checks are rare |
| Validation | Checked against independent R implementations | Ahead |

FOE is strong on single-test inference. The gaps are in metric definitions, many-metric analysis, and designs beyond a randomized A/B test.

## Gaps

### High priority

1. **Ratio metrics (delta method).** AOV, revenue per session and CTR are ratios where the analysis unit differs from the randomization unit. `AnalysisUnit.PER_TRANSACTION` treats orders as independent; with user-level randomization and repeat buyers this understates standard errors and inflates false positives. The delta method is currently only used in monetary projections.
2. **Correction across metrics, and guardrails.** Correction exists across variants (Šidák) but not across metrics or segments. Missing: Benjamini-Hochberg FDR, and a primary / secondary / guardrail metric framework with decision rules built on the existing non-inferiority test.
3. **Winner's curse.** Significant lifts are biased upward, and monetary projections use the observed lift. Platforms shrink estimates with empirical Bayes or historical priors.
4. **Sequential testing for continuous metrics.** mSPRT and group sequential designs cover conversions only. Missing: always-valid confidence sequences, ideally combined with CUPED, as in Eppo and Statsig.
5. **Quasi-experiments.** Difference-in-differences, synthetic control, and forecast-based pre/post analysis (CausalImpact-style) for changes that cannot be A/B tested. The Prophet forecasting engine is a starting point.

### Medium priority

6. **Bayesian analysis for continuous metrics**, e.g. a normal approximation on the delta-method estimate.
7. **Priors fitted to experiment history** instead of fixed widths. This also addresses item 3.
8. **Novelty and primacy detection** by testing for a trend in lift over time; the sequential trajectory data already exists.
9. **CUPAC**, or adjustment with several covariates, for users without pre-period data.
10. **Quantile treatment effects** (median, p90) with bootstrap intervals.
11. **Segment analysis** of treatment effects with multiple-testing correction.

### Low priority

12. **Bandits** (Thompson sampling). Mainly useful for short campaigns.
13. **Interference designs** (switchback, cluster randomization). Rarely needed for websites.
14. **ML-based heterogeneous effects** (causal forests, meta-learners). Heavy dependencies and little gain at typical CRO sample sizes.

## Suggested order

Items 1, 2, 3 and 8 are small native implementations (NumPy plus `scipy.special`) that fit the [dependency-reduction](dependency-reduction.md) plan. Each can be validated in `r_validation/` against `msm::deltamethod`, `p.adjust`, or a normal-normal empirical Bayes model. Item 4 is closed-form for the normal mixture. Item 5 is the largest piece of work and has the most client value.
