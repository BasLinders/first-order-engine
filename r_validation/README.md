# R validation of First Order Engine

These notebooks check the statistics in **First Order Engine (FOE)** against independent R
implementations. Each notebook calls a FOE engine, computes the same quantity in R through
`rpy2`'s `%%R` cells, and compares the two in a pass/fail table. The last cell raises if any
comparison fails, so each notebook also works as a test.

Scope so far: the engines that [Hexkit](https://github.com/BasLinders/hexkit), the Streamlit
toolkit built on FOE, uses in production. The "Used in Hexkit" column shows where each one
appears there. Other engines are listed under [Not covered yet](#not-covered-yet).

## Notebooks

| Notebook | FOE engine | Used in Hexkit | R reference | Checks |
|---|---|---|---|---|
| [01_frequentist_ztest](01_frequentist_ztest.ipynb) | `FrequentistEngine.run_synthesis`, `compute_interval_difference`, `apply_sidak`, `estimate_monetary_impact_per_variant`, `run_lin_adjustment` | experiment analysis, small-sample z-test fallback, automation (`run_lin_adjustment` is not used in Hexkit) | identity-link binomial `glm` (Wald SE = unpooled SE, `dispersion = φ`), `DescTools::BinomDiffCI`, `msm::deltamethod`, `lm` + `sandwich::vcovHC(type = "HC3")` | 384 |
| [02_boschloo_exact](02_boschloo_exact.ipynb) | `BoschlooEngine` | small sample analysis | `exact2x2::boschloo`, `DescTools::BinomDiffCI(method = "score")` | 60 |
| [03_bayesian_beta_binomial](03_bayesian_beta_binomial.ipynb) | `BayesianEngine.run_probability_analysis`, `run_monetary_projection`, `get_lift_prior` | sequential (revenue overlay), experiment analysis, automation | exact numerical integration of the Beta posteriors with the lift-prior weight (product prior for 3 arms); per-arm-AOV revenue difference at equal traffic, with mean-AOV uncertainty in closed form (Margrabe) | 47 |
| [04_continuous_metrics](04_continuous_metrics.ipynb) | `ContinuousMetricEngine.run_comparison_suite` (heuristic incl. statistics and effect sizes, NB, Gamma, hurdle), native `welch_anova` / `tukey_hsd` / `games_howell` / `dunn_test`, `fit_gamma`, `fit_negbin`, `winsorize_series`, `detect_outliers`, `run_business_case` | automation, user-level analysis, pre-test | `fBasics::dagoTest`, `car::leveneTest`, `anova`, `oneway.test`, `kruskal.test`, `rstatix::kruskal_effsize`, `wilcox.test`, `TukeyHSD`, `rstatix::games_howell_test`, `dunn.test`, `MASS::glm.nb`, `MASS::theta.ml` (exact NB2 profile, incl. an all-zero variant), Poisson GLM LRT (underdispersed counts), `MASS::fitdistr`, exact Gamma MLE, `rstandard`, `msm::deltamethod` | 288 |
| [05_pretest_planning](05_pretest_planning.ipynb) | `PretestEngine` sample size, power, MDE tables (static and forecast), RPV moments | pre-test analysis, automation | `TrialSize::TwoSampleProportion.Equality`, `power.prop.test`, `TrialSize::TwoSampleMean.Equality`, `pwr::pwr.norm.test`, simulation of the unpooled z-test, root-found CV-constant sample size | 121 |
| [06_sequential_msprt](06_sequential_msprt.ipynb) | `SequentialEngine` LLR, boundaries, power, `process_test_trajectory` | sequential analysis | numerical integration of the mixture likelihood ratio, `pwr::pwr.norm.test` | 131 |
| [07_interaction_glm](07_interaction_glm.ipynb) | `InteractionEngine.run_interaction_analysis` | interaction analysis | `glm(cbind(conv, visitors - conv) ~ t1 * t2, binomial)` | 93 |

All 1100 checks pass on the committed runs (R 4.6.2, rpy2 3.6.9, scipy 1.18.1, statsmodels 0.15.0,
FOE at commit `904170a`; notebook 03 re-run after the mean-AOV uncertainty change; notebook 04
re-run on Windows with R 4.6.1 for the native post-hocs and outlier helpers, and again (R 4.6.1,
scipy 1.18.1, statsmodels 0.15.0) for the underdispersed NB case, and again (R 4.6.1, rpy2 3.6.9, scipy 1.16.3, statsmodels
0.14.6) for the studentized-residual outlier rule and its 90/10 case; notebook 01 re-run (R 4.6.2,
rpy2 3.6.9, sandwich 3.1.3) after Lin's adjustment moved to native OLS with HC3; notebook 04
re-run (R 4.6.2, MASS 7.3.66) after the NB2 fit moved to the exact native profile likelihood).

### Tolerances

Most checks agree to 1e-6 relative or better. The looser ones are deliberate:

- **Boschloo p-values** (abs 5e-4): scipy and exact2x2 each search for the nuisance-parameter
  maximum numerically, on different grids.
- **Bayesian quantities** (abs 3e-3 on probabilities, 2–3% relative on expectations): FOE uses
  Monte Carlo and R uses exact integration, so the gap is Monte Carlo error. With 1M draws the
  observed differences are about 0.1%.
- **RPV moments** (1%): checked against a simulated sample.
- **Two-sided MDE** (1e-4 relative): FOE's closed form leaves out the far rejection tail.
- **Games-Howell adjusted p** (abs 5e-4): `rstatix::games_howell_test` rounds `p.adj` to 3
  decimals. The unrounded p is also recomputed with `ptukey` and checked to 1e-5 relative.
- **NB LR on underdispersed counts vs `glm.nb`** (1e-3 relative): `glm.nb` stops at its iteration
  limit with θ ≈ 7×10⁴ instead of ∞. The tight reference there is the Poisson GLM LRT (1e-5).

## What the validation turned up

None of these is a bug in FOE (the ones that were, `fit_gamma`'s precision and the NB test on
underdispersed counts, are fixed), but each one matters for reading or extending the results:

- **Underdispersed counts put the NB fit at the Poisson boundary.** Items per order (always ≥ 1)
  usually has variance below its mean. NB2's α then goes to its lower bound, and statsmodels
  returns NaN standard errors because the Hessian is singular in α alone. FOE used to read that
  as an unfittable model and report p = 1. It now checks identification on a Poisson fit with
  the same design, reports `dispersion_alpha = 0.0`, and matches the Poisson GLM LRT
  (notebook 04, §2b). `MASS::glm.nb` warns "iteration limit reached" on the same data.

- **`run_comparison_suite` with `unit = per_transaction` (the config default) drops zero rows
  before any test**, including the Negative Binomial count test. For "items per order" that is
  correct. For a per-visitor count KPI with real zeros, use `unit = per_visitor`. With the
  default, the NB is fit to positive counts only and gives a different answer (notebook 04, §2).
- **`ContinuousMetricEngine.fit_gamma` solves the exact Gamma MLE; `MASS::fitdistr` does not.**
  `fitdistr` is off by about 0.1% in the shape parameter, enough to move Gamma LRT p-values, so
  notebook 04 uses the exact MLE as the reference. `fit_gamma` used to run a general-purpose
  optimiser that stopped short of the maximum on about 18% of random samples (shape off by up to
  1.5%, log-likelihood up to 0.7 low). It now solves the profile equation
  `log k − ψ(k) = log x̄ − mean(log x)` with Brent's method and matches the R reference to 1e-10.
  On 200 random two-arm Gamma tests, that moved LRT p-values by up to 0.04 and changed no
  verdict at α = 0.05.
- **rstatix's Games-Howell `se` is on the studentized-range scale**, √((s₁²/n₁ + s₂²/n₂) / 2),
  so its `estimate / se` is q, not Welch's t. FOE reports t (= q / √2).
- **η²_H is clamped to [0, 1]**, as `rstatix::kruskal_effsize` does. The raw formula
  (H − k + 1) / (n − k) goes negative when H < k − 1, that is, when there is no effect at all.
- **Mann-Whitney exact vs asymptotic:** scipy's `method="auto"` switches to the exact test when
  either arm has ≤ 8 rows and there are no ties. R's `wilcox.test` uses a different rule, so the
  R side reproduces scipy's.
- **`pwr.norm.test` solved for the effect size** uses a loose internal tolerance (~1e-4). Notebook
  05 inverts its power function with a tight `uniroot` instead.

## Setup

Run these from the repository root.

1. **Install R** (≥ 4.3). The CRAN installer from <https://cran.r-project.org> ships binary
   packages and is the easiest route. With Homebrew R (`brew install r`) packages compile from
   source and also need `brew install openssl@3 nlopt pkgconf`.
2. **Install the R packages:**
   ```bash
   Rscript r_validation/install_r_packages.R
   ```
3. **Install FOE with the `validation` extra.** This installs the working tree in editable
   mode, so the notebooks always test the code you have checked out, plus Jupyter and `rpy2`:
   ```bash
   pip install -e ".[validation]"
   ```
   `rpy2` finds R through `R_HOME` or `R` on your `PATH`.

**Windows:** with rpy2 3.6.9 and no Rtools, `R CMD config --ldflags` prints nothing (it needs
`make`), and rpy2 fails with an `IndexError` in `rpy2.situation._get_r_cmd_config` instead of
falling back to its built-in DLL search. Installing Rtools should avoid it. Without Rtools,
a `sitecustomize.py` in your virtualenv that turns that `IndexError` into
`subprocess.CalledProcessError` lets rpy2's fallback run. Also, one-line `%R` magics must
quote R strings with single quotes: on Windows, IPython strips double quotes from magic
arguments.

## Running

Interactively:

```bash
cd r_validation && jupyter lab
```

Headless, as a test run (fails on the first notebook with a mismatch):

```bash
cd r_validation
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb" || break; done
```

Run from inside `r_validation/` so the notebooks can import `validation_helpers.py`.

## Not covered yet

- `foe.pretest.forecasting.TrafficForecastingEngine` (Prophet traffic forecast). This could be
  checked against R's `prophet` package, but fitted Prophet models are only loosely
  reproducible across languages.
- The CV-constant (unequal-variance) branch of the continuous sample-size formula has no
  off-the-shelf R equivalent.
- Engines outside the current scope: `foe.srm`, `foe.behavioral`, `foe.forecasting` and
  `foe.sequential.gsd_engine`.
- Helpers in the validated engines that aren't checked yet: non-inferiority, CUPED, aggregate
  φ estimation and bootstrap power.
