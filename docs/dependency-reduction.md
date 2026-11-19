# Dependency Reduction

## Reason

FOE uses a small number of functions from large statistical libraries. Their breadth brings many open issues and behavior changes between versions that FOE doesn't control. To keep control over FOE's correctness, all statistical logic is implemented natively and validated against R. Third-party code is kept only for rigorously tested numerical primitives.

## Plan

### Target state

| Dependency | Status |
|---|---|
| numpy, pandas, pydantic | Keep |
| scipy | Keep, restricted to `scipy.special` and `scipy.optimize` |
| statsmodels, patsy | Remove |
| prophet | Move to optional `forecasting` extra |

Pin upper bounds on the remaining dependencies so upgrades are deliberate and validated.

### Structure

All native statistics live in `foe/stats/`. It is the only package that imports scipy, and only `scipy.special` and `scipy.optimize`. Engines import their statistics from `foe.stats`. `tests/test_dependency_boundary.py` enforces this. Its allowlist of remaining third-party imports must shrink with each step.

### Steps

One PR per step, each validated against `r_validation/`.

1. **Distributions module** (`foe/stats/distributions.py`): native `norm`, `chi2`, `f`, `beta` and `gamma` built on `scipy.special`. Route all existing calls through it. `t` follows in step 6, where `ttest_ind` first needs it.
2. **Newcombe CI and one-way OLS/ANOVA**: replace `confint_proportions_2indep`, `smf.ols` + `anova_lm`, and the studentized residuals in `behavioral`.
3. **OLS with HC3**: replace the covariate-adjusted model in `frequentist/operations.py`.
4. **NB2 fit**: group means are the MLE; fit α by 1-D profile likelihood (`brentq` + `digamma`).
5. **Binomial GLM for interactions**: native design matrix and IRLS with explicit separation detection. Removes statsmodels and patsy.
6. **Rank and variance tests**: native `rankdata`, `mannwhitneyu`, `kruskal`, `levene`, `normaltest`, `ttest_ind`, `chisquare`.
7. **Boschloo exact test**: native implementation.
8. **Studentized range**: port R's `ptukey`/`qtukey` (Copenhaver & Holland, AS 190).
9. **prophet**: move to an optional extra with a lazy import and a clear `ImportError`.
