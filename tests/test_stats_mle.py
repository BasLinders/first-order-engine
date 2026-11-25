"""
foe.stats.mle (NB2 one-way fit) and ContinuousMetricEngine.fit_negbin /
_run_negbin_lrt against R 4.6.2, pinned so the tests don't depend on
statsmodels or R:

- alpha: 1 / MASS::theta.ml(y, mu) at the group means (MASS 7.3.66). Rows
  with mu = 0 (an all-zero variant) add exactly 0 to the log-likelihood and
  score, so they are left out of theta.ml, which evaluates y / mu.
- log-likelihoods: sum(dnbinom(y, size = theta, mu = mu, log = TRUE)).
- Poisson-boundary case: anova(glm(y ~ 1, poisson), glm(y ~ g, poisson)).

theta.ml stops at |step| < eps^(1/4), so alpha is compared at 1e-8.
"""

import math

import numpy as np
import pandas as pd
import pytest

import foe.stats.mle as mle_mod
from foe.continuous.operations import ContinuousMetricEngine
from foe.stats.mle import nb2_oneway


def nb_cases():
    rng = np.random.default_rng(2026)
    out = {}
    g = np.repeat(["A", "B", "C"], 300)
    mu = np.select([g == "A", g == "B"], [1.5, 2.2], 3.0)
    out["overdispersed"] = pd.DataFrame({"g": g, "y": rng.negative_binomial(1.2, 1.2 / (1.2 + mu))})
    g = np.repeat(["A", "B", "C"], 200)
    y = rng.negative_binomial(0.8, 0.8 / (0.8 + np.where(g == "A", 0.4, 0.6)))
    y[g == "C"] = 0
    out["zero variant"] = pd.DataFrame({"g": g, "y": y})
    g = np.repeat(["A", "B"], 150)
    out["sparse"] = pd.DataFrame({"g": g, "y": rng.poisson(np.where(g == "A", 0.05, 0.12))})
    out["intercept only"] = pd.DataFrame({"g": "A", "y": rng.negative_binomial(3.0, 3.0 / (3.0 + 4.0), 500)})
    return {k: v.assign(y=v["y"].astype(float)) for k, v in out.items()}


CASES = nb_cases()

# case: (alpha_alt, ll_alt, lr, p)
R_LRT = {
    "overdispersed": (0.82042254452950147, -1743.1619622080857, 38.257967340908181, 4.9247926218174076e-09),
    "zero variant": (0.89647335680923823, -430.08172298196092, 154.7161931086232, 2.5339937298452285e-34),
    "sparse": (0.0, None, 3.6515131419222939, 0.056018296352552843),
}


@pytest.mark.parametrize("case", list(R_LRT))
def test_negbin_lrt_matches_r(case):
    alpha, ll_alt, lr, p = R_LRT[case]
    d = CASES[case]
    fit = nb2_oneway(d["y"], d["g"])
    assert fit.alpha == pytest.approx(alpha, rel=1e-8, abs=0)
    if ll_alt is not None:
        assert fit.log_lik == pytest.approx(ll_alt, rel=1e-12)
    p_value, lr_stat, disp_alpha = ContinuousMetricEngine._run_negbin_lrt(d, "y", "g")
    assert lr_stat == pytest.approx(lr, rel=1e-9)
    assert p_value == pytest.approx(p, rel=1e-8)
    assert disp_alpha == pytest.approx(alpha, rel=1e-8, abs=0)


def test_fit_negbin_intercept_only_matches_r_and_keeps_hexkit_contract():
    # Hexkit's pre-test page reads (log_lik, alpha, res) and exp(res.params[0]).
    y = CASES["intercept only"]["y"].to_numpy()
    log_lik, alpha, res = ContinuousMetricEngine.fit_negbin(y)
    assert alpha == pytest.approx(0.32369785135054796, rel=1e-8)
    assert log_lik == pytest.approx(-1203.0948933378249, rel=1e-12)
    assert math.exp(res.params[0]) == pytest.approx(4.1, rel=1e-14)
    assert res.params[-1] == alpha and res.llf == log_lik


def test_fit_negbin_dummy_design_params_and_lrt_consistency():
    d = CASES["overdispersed"]
    y = d["y"].to_numpy()
    X = np.column_stack([np.ones(len(y)), d["g"] == "B", d["g"] == "C"]).astype(float)
    ll, alpha, res = ContinuousMetricEngine.fit_negbin(y, X)
    means = d.groupby("g")["y"].mean()
    np.testing.assert_allclose(
        res.params[:3],
        [math.log(means["A"]), math.log(means["B"] / means["A"]), math.log(means["C"] / means["A"])],
        rtol=1e-12,
    )
    assert res.params[3] == alpha
    assert ll == pytest.approx(R_LRT["overdispersed"][1], rel=1e-12)


@pytest.mark.parametrize("exog", [
    np.column_stack([np.ones(6), np.arange(6.0)]),          # continuous covariate
    np.column_stack([np.ones(6), [0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0]]),
    np.ones(6),                                               # not 2-D
])
def test_fit_negbin_rejects_designs_that_are_not_one_way(exog):
    with pytest.raises(ValueError):
        ContinuousMetricEngine.fit_negbin(np.array([0.0, 1, 2, 3, 1, 0]), exog)


@pytest.mark.parametrize("y", [[1.0, -1.0], [1.5, 2.0], [1.0, np.nan], []])
def test_nb2_rejects_non_counts(y):
    with pytest.raises(ValueError):
        nb2_oneway(y)


def test_nb2_poisson_boundary_is_exact_poisson():
    from scipy.stats import poisson

    rng = np.random.default_rng(5)
    y = rng.poisson(4.6, 2249).astype(float)
    fit = nb2_oneway(y)
    assert fit.alpha == 0.0
    assert fit.log_lik == pytest.approx(float(poisson.logpmf(y, y.mean()).sum()), rel=1e-14)


def test_nb2_grid_maximum_is_global():
    # Dense brute-force check of the profile likelihood over alpha.
    d = CASES["overdispersed"]
    fit = nb2_oneway(d["y"], d["g"])
    prof = mle_mod._NB2Profile(d["y"].to_numpy(), fit.group_means[fit.codes])
    dense = max(prof.loglik(a) for a in np.concatenate([[0.0], np.logspace(-8, 4, 4001)]))
    assert fit.log_lik >= dense - 1e-9


def test_nb2_large_count_path_agrees_with_exact_sums(monkeypatch):
    d = CASES["overdispersed"]
    exact = nb2_oneway(d["y"], d["g"])
    monkeypatch.setattr(mle_mod, "_CUMSUM_MAX_COUNT", -1)
    lgamma_path = nb2_oneway(d["y"], d["g"])
    assert lgamma_path.alpha == pytest.approx(exact.alpha, rel=1e-10)
    assert lgamma_path.log_lik == pytest.approx(exact.log_lik, rel=1e-12)


def test_nb2_all_zero_sample_has_alpha_zero():
    fit = nb2_oneway([0.0, 0.0, 0.0])
    assert fit.alpha == 0.0 and fit.log_lik == 0.0 and fit.group_totals[0] == 0.0


def test_nb2_extends_grid_when_maximum_lies_beyond_it(monkeypatch):
    d = CASES["overdispersed"]
    expected = nb2_oneway(d["y"], d["g"])
    monkeypatch.setattr(mle_mod, "_ALPHA_GRID", np.logspace(-10.0, -2.0, 9))  # max (0.82) beyond 1e-2
    got = nb2_oneway(d["y"], d["g"])
    assert got.alpha == pytest.approx(expected.alpha, rel=1e-12)


@pytest.mark.parametrize("groups", [["A", "B"], ["A", None, "B"]])
def test_nb2_rejects_bad_groups(groups):
    with pytest.raises(ValueError):
        nb2_oneway([1.0, 2.0, 3.0], groups)
