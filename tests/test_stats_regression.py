"""
foe.stats.regression.ols_hc3 and FrequentistEngine.run_lin_adjustment against
R: lm(y ~ variant * cc) with sandwich::vcovHC(type = "HC3"), R 4.6.2,
sandwich 3.1.3. Reference values are pinned so the tests don't depend on
statsmodels.
"""

import math

import numpy as np
import pandas as pd
import pytest

from foe.frequentist.operations import FrequentistEngine
from foe.stats.regression import ols_hc3


def lin_frame(seed=3, n=60, const_pre=False, with_nan=False):
    """Three arms, heteroskedastic noise, heterogeneous effect in C."""
    rng = np.random.default_rng(seed)
    v = np.array(["control", "B", "C"] * (n // 3))
    pre = np.zeros(n) if const_pre else np.round(rng.gamma(2.0, 3.0, n), 3)
    y = np.round(5 + 0.4 * pre + (v == "B") * 1.0 + (v == "C") * (0.5 + 0.3 * pre)
                 + rng.normal(0, 1 + pre / 4, n), 3)
    df = pd.DataFrame({"variant": v, "pre": pre, "y": y})
    if with_nan:
        df.loc[[4, 17], "y"] = np.nan
        df.loc[[9], "pre"] = np.nan
    return df


# case -> {variant: (ate, se, two-sided p)}, intercept
R_REF = {
    "heteroskedastic": (dict(lin_frame=dict()), {
        "B": (0.64382053660080396, 0.92498607624869178, 0.48640836468846232),
        "C": (2.4060522206785557, 0.81915860744686786, 0.0033116493690831506),
    }, 8.2573730503687646),
    "missing values": (dict(lin_frame=dict(with_nan=True)), {
        "B": (0.59920298779218062, 0.97990179795763177, 0.54087332024533796),
        "C": (2.5033995412281786, 0.89302845907930517, 0.0050587360667875263),
    }, 8.1540000567492363),
    # Covariate constant: centered column is all zeros, design is rank-deficient.
    "constant covariate": (dict(lin_frame=dict(const_pre=True)), {
        "B": (1.2528999999999992, 0.39626215300709339, 0.0015679949134287453),
        "C": (0.89484999999999959, 0.37978114250840078, 0.018461708322182132),
    }, 4.7706500000000016),
}


@pytest.mark.parametrize("case", list(R_REF))
def test_lin_adjustment_matches_r(case):
    kwargs, ref, intercept = R_REF[case]
    df = lin_frame(**kwargs["lin_frame"])
    res = FrequentistEngine.run_lin_adjustment(df, "y", "pre", control_label="control")
    assert [r["variant"] for r in res] == ["B", "C"]
    for r in res:
        ate, se, p = ref[r["variant"]]
        assert r["absolute_lift"] == pytest.approx(ate, rel=1e-10)
        assert r["std_err"] == pytest.approx(se, rel=1e-10)
        assert r["p_value"] == pytest.approx(p, rel=1e-9)
        assert r["relative_lift"] == pytest.approx(ate / intercept, rel=1e-10)


# --- engine fixes (points of interest in run_lin_adjustment's docstring) -----

def test_lin_accepts_awkward_column_names_and_labels():
    df = lin_frame().rename(
        columns={"variant": "test variant", "pre": "pre value", "y": "order value"})
    df["test variant"] = df["test variant"].replace({"B": "B's arm"})
    res = FrequentistEngine.run_lin_adjustment(
        df, "order value", "pre value", variant_col="test variant", control_label="control"
    )
    by_label = {r["variant"]: r for r in res}
    ate, se, _ = R_REF["heteroskedastic"][1]["B"]
    assert by_label["B's arm"]["absolute_lift"] == pytest.approx(ate, rel=1e-10)
    assert by_label["B's arm"]["std_err"] == pytest.approx(se, rel=1e-10)


def test_lin_ignores_missing_variant_labels():
    df = lin_frame()
    df["variant"] = df["variant"].astype(object)
    df.loc[[2, 5], "variant"] = None
    res = FrequentistEngine.run_lin_adjustment(df, "y", "pre", control_label="control")
    assert [r["variant"] for r in res] == ["B", "C"]
    assert all(math.isfinite(r["std_err"]) for r in res)


def test_lin_exactly_fit_variant_is_undefined_and_isolated():
    base = FrequentistEngine.run_lin_adjustment(
        lin_frame(), "y", "pre", control_label="control")
    # Two stray rows placed symmetrically around the covariate mean, so the
    # centering (and with it every other arm's estimate) is unchanged.
    m = lin_frame()["pre"].mean()
    stray = pd.DataFrame({"variant": ["Z", "Z"], "pre": [
                         m - 1.5, m + 1.5], "y": [3.0, 9.0]})
    df = pd.concat([lin_frame(), stray], ignore_index=True)
    res = {r["variant"]: r for r in FrequentistEngine.run_lin_adjustment(
        df, "y", "pre", control_label="control")}
    z = res["Z"]
    assert math.isnan(z["std_err"]) and math.isnan(z["p_value"])
    assert all(math.isnan(b) for b in z["ci"])
    assert z["is_significant"] is False
    for r in base:
        assert math.isfinite(res[r["variant"]]["std_err"])
        assert res[r["variant"]]["std_err"] == pytest.approx(r["std_err"], rel=1e-9)


# --- ols_hc3 ------------------------------------------------------------------

def test_ols_hc3_known_small_case():
    # y = 1 + 2x exactly plus one perturbation; HC3 by hand.
    X = np.column_stack([np.ones(5), [0.0, 1, 2, 3, 4]])
    y = np.array([1.0, 3, 5, 7, 10])
    fit = ols_hc3(X, y)
    pinv = np.linalg.pinv(X)
    h = np.diag(X @ pinv)
    e = y - X @ (pinv @ y)
    cov = pinv @ np.diag((e / (1 - h)) ** 2) @ pinv.T
    np.testing.assert_allclose(fit.params, pinv @ y, rtol=1e-12)
    np.testing.assert_allclose(fit.bse, np.sqrt(np.diag(cov)), rtol=1e-12)
    np.testing.assert_allclose(fit.leverage, h, rtol=1e-12)
    assert fit.rank == 2


def test_ols_hc3_rank_deficient_reports_rank():
    X = np.column_stack([np.ones(6), np.zeros(6), [0.0, 0, 0, 1, 1, 1]])
    fit = ols_hc3(X, np.array([1.0, 2, 3, 4, 5, 7]))
    assert fit.rank == 2
    assert fit.params[1] == 0.0
    assert fit.params[2] == pytest.approx(16 / 3 - 2, rel=1e-12)


@pytest.mark.parametrize("X,y", [
    (np.ones((3, 1)), np.array([1.0, np.nan, 2.0])),
    (np.ones((3, 1)), np.ones(4)),
    (np.ones(3), np.ones(3)),
])
def test_ols_hc3_rejects_bad_input(X, y):
    with pytest.raises(ValueError):
        ols_hc3(X, y)
