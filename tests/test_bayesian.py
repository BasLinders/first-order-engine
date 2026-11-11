import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import beta as beta_dist, norm

from foe.bayesian.operations import (
    BayesianEngine,
    get_beta_prior,
    get_lift_prior,
    BetaPrior,
    prior_prob_positive_lift,
    mc_margin,
)
from foe.core.models import ExperimentInput, BusinessCaseInput, BayesianResult


# --- Fixtures ---

@pytest.fixture
def engine():
    # Fixed seed for deterministic results across runs.
    return BayesianEngine(seed=42)


@pytest.fixture
def uninformative_priors():
    return get_beta_prior(), get_lift_prior(0.0, "uninformative")


@pytest.fixture
def skeptical_priors():
    return get_beta_prior(), get_lift_prior(0.0, "skeptical")


def make_experiment(visitors, conversions, labels=None):
    if labels is None:
        labels = [chr(65 + i) for i in range(len(visitors))]
    return ExperimentInput(visitors=visitors, conversions=conversions, labels=labels)


# --- run_probability_analysis ---

def test_probability_analysis_returns_bayesian_results(engine, uninformative_priors):
    """
    Return type should be a list of BayesianResult, one per challenger.
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment([1000, 1000], [50, 100]),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], BayesianResult)


def test_strong_challenger_dominates(engine, uninformative_priors):
    """
    A variant with double the conversion rate should have PBB near 100%.
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment([1000, 1000], [50, 100]),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    challenger = results[0]
    assert challenger.prob_being_best > 0.99
    assert challenger.prob_beat_control > 0.99


def test_flat_test_produces_even_split(engine, uninformative_priors):
    """
    Identical conversion rates should produce roughly 50/50 prob_beat_control.
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment([1000, 1000], [100, 100]),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    assert 0.40 <= results[0].prob_beat_control <= 0.60


def test_three_variants_strongest_dominates(engine, uninformative_priors):
    """
    With three variants, the strongest should have PBB > 0.90 and
    all PBBs (including control) should sum to ~1.
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment(
            [1000, 1000, 1000], [50, 75, 120],
            labels=["Control", "B", "C"],
        ),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    assert len(results) == 2  # one per challenger

    pbb_c = results[1].prob_being_best  # Variant C
    assert pbb_c > 0.90

    # All PBBs including control should sum to ~1.
    all_pbb = (
        [1.0 - sum(r.prob_being_best for r in results)]
        + [r.prob_being_best for r in results]
    )
    assert abs(sum(all_pbb) - 1.0) < 1e-6


def test_empty_input_rejected_by_model():
    """
    ExperimentInput rejects empty visitor/conversion lists at the model boundary.
    The engine layer never receives invalid input — validation happens in Pydantic
    via min_length=2 on the visitors and conversions fields.
    """
    with pytest.raises(ValidationError, match="at least 2 items"):
        make_experiment([], [])


def test_skeptical_prior_dampens_extreme_result(engine):
    """
    A skeptical lift prior should reduce prob_beat_control relative to
    an uninformative prior when the observed lift is large.
    """
    data = make_experiment([1000, 1000], [50, 100])

    result_uninformative = engine.run_probability_analysis(
        data=data,
        beta_prior=get_beta_prior(),
        lift_prior=get_lift_prior(0.0, "uninformative"),
    )
    result_skeptical = engine.run_probability_analysis(
        data=data,
        beta_prior=get_beta_prior(),
        lift_prior=get_lift_prior(0.0, "skeptical"),
    )
    # Skeptical prior should penalise a large lift.
    assert result_skeptical[0].prob_beat_control < result_uninformative[0].prob_beat_control


def test_result_fields_are_present(engine, uninformative_priors):
    """
    Each BayesianResult should carry all expected fields with sensible values.
    Covers both the original fields and the two added in the engine refactor
    (prob_beat_control, expected_uplift).
    """
    beta_prior, lift_prior = uninformative_priors
    results = engine.run_probability_analysis(
        data=make_experiment(
            [1000, 1000], [50, 100],
            labels=["Control", "Challenger"],
        ),
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )
    r = results[0]
    assert r.variant_label == "Challenger"
    assert r.control_label == "Control"
    assert 0.0 <= r.prob_being_best <= 1.0
    assert 0.0 <= r.prob_beat_control <= 1.0
    assert r.expected_uplift >= 0.0   # np.maximum(..., 0) guarantees non-negative
    assert r.expected_loss >= 0.0
    assert isinstance(r.conclusion, str) and len(r.conclusion) > 0


# --- run_monetary_projection ---

def test_monetary_positive_uplift_for_better_variant(engine, uninformative_priors):
    """
    A variant with higher CR and higher AOV should produce positive uplift
    and a net positive total contribution.
    """
    beta_prior, lift_prior = uninformative_priors
    visitors = [1000, 1000]
    conversions = [100, 110]
    labels = ["Control", "Challenger"]

    results = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=BusinessCaseInput(
            aovs={"Control": 50.0, "Challenger": 60.0},
            runtime_days=30,
            projection_period=180,
        ),
        variant_labels=labels,
        beta_prior=beta_prior,
        lift_prior=lift_prior,
        aov_cv=0.5,
    )

    assert len(results) == 1
    r = results[0]
    assert r["variant_label"] == "Challenger"
    assert r["variant_index"] == 1
    assert r["expected_uplift"] > 0
    assert r["expected_total_contribution"] > 0
    assert 0.0 <= r["prob_beat_control"] <= 1.0
    assert 0.0 <= r["prob_best_overall"] <= 1.0
    assert "conclusion" in r


def test_monetary_three_variants_returns_two_results(engine, uninformative_priors):
    """
    Three variants should produce two result dicts, one per challenger.
    """
    beta_prior, lift_prior = uninformative_priors
    visitors = [1000, 1000, 1000]
    conversions = [50, 75, 120]
    labels = ["Control", "Variant B", "Variant C"]

    results = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=BusinessCaseInput(
            aovs={"Control": 50.0, "Variant B": 50.0, "Variant C": 50.0},
            runtime_days=30,
            projection_period=90,
        ),
        variant_labels=labels,
        beta_prior=beta_prior,
        lift_prior=lift_prior,
        aov_cv=0.5,
    )

    assert len(results) == 2
    assert results[0]["variant_label"] == "Variant B"
    assert results[1]["variant_label"] == "Variant C"


def test_monetary_invalid_runtime_rejected_by_model():
    """
    BusinessCaseInput rejects runtime_days=0 at the model boundary via
    Field(..., gt=0). The engine layer never receives this invalid input.
    """
    with pytest.raises(ValidationError):
        BusinessCaseInput(
            aovs={"Control": 50.0, "Challenger": 60.0},
            runtime_days=0,
            projection_period=180,
        )


def test_monetary_label_mismatch_raises_value_error(engine, uninformative_priors):
    """
    Passing a variant_labels list whose length doesn't match visitors raises
    ValueError. Unlike runtime_days, this mismatch is not guaranteed to be
    caught by Pydantic (variant_labels is a plain function parameter), so
    run_monetary_projection enforces it explicitly.
    """
    beta_prior, lift_prior = uninformative_priors
    visitors = [1000, 1000]
    conversions = [100, 110]

    with pytest.raises(ValueError, match="variant_labels has 1 entries but visitors has 2"):
        engine.run_monetary_projection(
            visitors=visitors,
            conversions=conversions,
            biz_case=BusinessCaseInput(
                aovs={"Control": 50.0, "Challenger": 60.0},
                runtime_days=30,
                projection_period=90,
            ),
            variant_labels=["OnlyOneLabel"],
            beta_prior=beta_prior,
            lift_prior=lift_prior,
        )


def test_monetary_risk_is_always_non_negative(engine, uninformative_priors):
    """
    expected_risk must be a non-negative magnitude regardless of whether the
    variant wins or loses. A losing variant should produce a positive risk
    number, not a negative one.
    """
    beta_prior, lift_prior = uninformative_priors
    visitors = [1000, 1000]
    conversions = [200, 100]  # Control is clearly better — variant is a loser
    labels = ["Control", "Loser"]

    results = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=BusinessCaseInput(
            aovs={"Control": 50.0, "Loser": 50.0},
            runtime_days=30,
            projection_period=90,
        ),
        variant_labels=labels,
        beta_prior=beta_prior,
        lift_prior=lift_prior,
    )

    r = results[0]
    assert r["expected_risk"] >= 0.0
    # Net contribution for a loser should be negative (uplift near zero, risk positive).
    assert r["expected_total_contribution"] < 0.0


def test_alpha_beta_prior_in_biz_case_influences_projection(engine):
    """
    BusinessCaseInput.alpha_prior and beta_prior are the user-facing controls
    for the Beta prior used in run_monetary_projection. A strongly informative
    prior (large alpha+beta) should pull the posterior toward the prior mean and
    produce a different expected_uplift than the default uninformative Beta(1,1).
    """
    visitors = [1000, 1000]
    conversions = [100, 150]
    labels = ["Control", "Challenger"]

    # Default uninformative prior: Beta(1, 1)
    result_flat = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=BusinessCaseInput(
            aovs={"Control": 50.0, "Challenger": 50.0},
            runtime_days=30,
            projection_period=90,
            alpha_prior=1.0,
            beta_prior=1.0,
        ),
        variant_labels=labels,
        beta_prior=BetaPrior(alpha=1.0, beta=1.0),
    )

    # Strong prior centred near 50% conversion — heavily regularises the posterior.
    result_strong = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=BusinessCaseInput(
            aovs={"Control": 50.0, "Challenger": 50.0},
            runtime_days=30,
            projection_period=90,
            alpha_prior=500.0,
            beta_prior=500.0,
        ),
        variant_labels=labels,
        beta_prior=BetaPrior(alpha=500.0, beta=500.0),
    )

    # The strong prior shrinks the observed lift; uplift should be meaningfully lower.
    assert result_strong[0]["expected_uplift"] < result_flat[0]["expected_uplift"]


def test_biz_case_accepted_via_experiment_input():
    """
    ExperimentInput now carries an optional biz_case field so the Bayesian
    handler can receive a business case through a single JSON payload.
    Verifies the field is parsed and accessible.
    """
    biz = BusinessCaseInput(
        aovs={"Control": 50.0, "Challenger": 60.0},
        runtime_days=14,
        projection_period=90,
    )
    data = ExperimentInput(
        visitors=[1000, 1000],
        conversions=[100, 120],
        labels=["Control", "Challenger"],
        biz_case=biz,
    )
    assert data.biz_case is not None
    assert data.biz_case.aovs["Control"] == 50.0


def test_aov_uncertainty_keeps_net_contribution_unbiased(engine, uninformative_priors):
    """
    AOV draws are mean-preserving, so the net contribution E[revenue diff]
    is the same at any aov_cv; only its spread (and with it uplift and risk
    separately) grows.
    """
    beta_prior, lift_prior = uninformative_priors
    visitors = [5000, 5000]
    conversions = [250, 300]
    labels = ["Control", "Challenger"]
    biz_case = BusinessCaseInput(
        aovs={"Control": 50.0, "Challenger": 50.0},
        runtime_days=30,
        projection_period=180,
    )

    def run(cv):
        return engine.run_monetary_projection(
            visitors=visitors, conversions=conversions,
            biz_case=biz_case,
            variant_labels=labels,
            beta_prior=beta_prior, lift_prior=lift_prior,
            aov_cv=cv, n_simulations=400_000,
        )[0]

    low, high = run(0.0), run(1.5)
    assert high["expected_total_contribution"] == pytest.approx(
        low["expected_total_contribution"], rel=0.03
    )
    assert high["expected_risk"] > low["expected_risk"]


# --- Lift prior correctness ---

def _exact_prob_beat_control(visitors, conversions, lift_prior, grid=1500):
    """
    Grid-integrated P(cr_B > cr_A) under Beta(1,1) x Beta(1,1) x lift prior.
    Independent of the engine's Monte Carlo/importance-sampling path.
    """
    post = [beta_dist(1 + c, 1 + v - c) for v, c in zip(visitors, conversions)]
    grids = [np.linspace(p.ppf(1e-7), p.ppf(1 - 1e-7), grid) for p in post]
    pa, pb = np.meshgrid(*grids, indexing="ij")
    log_f = post[0].logpdf(pa) + post[1].logpdf(pb)
    if lift_prior is not None:
        log_f += norm.logpdf(np.log(pb / pa),
                             lift_prior.mean_log_lift, lift_prior.std_log_lift)
    f = np.exp(log_f - log_f.max())
    return float((f * (pb > pa)).sum() / f.sum())


@pytest.mark.parametrize("skepticism", ["skeptical", "moderate", "uninformative"])
def test_prob_beat_control_matches_exact_integration(skepticism):
    """
    Importance-weighted Monte Carlo must agree with direct numerical
    integration of the same posterior, including on small samples where
    the skeptical prior moves the answer by ~20 percentage points.
    """
    visitors, conversions = [1000, 1000], [30, 42]
    lift_prior = get_lift_prior(0.0, skepticism)
    result = BayesianEngine(seed=1).run_probability_analysis(
        data=make_experiment(visitors, conversions),
        lift_prior=lift_prior,
    )[0]
    exact = _exact_prob_beat_control(visitors, conversions, lift_prior)
    assert abs(result.prob_beat_control - exact) < 0.01


def test_no_lift_prior_is_plain_beta_binomial():
    """
    lift_prior=None must apply no weighting at all: full effective sample
    size and agreement with the exact Beta-Binomial answer.
    """
    visitors, conversions = [1000, 1000], [30, 42]
    result = BayesianEngine(seed=1).run_probability_analysis(
        data=make_experiment(visitors, conversions),
        lift_prior=None,
    )[0]
    assert result.effective_sample_size == pytest.approx(result.n_samples)
    exact = _exact_prob_beat_control(visitors, conversions, None)
    assert abs(result.prob_beat_control - exact) < 0.005


def test_multi_variant_uses_product_of_lift_priors():
    """
    With several challengers the joint prior is the product of the
    per-challenger lift priors. Check P(best) against a reference computed
    with explicit product weights from an independent draw.
    """
    visitors, conversions = [5000, 5000, 5000], [150, 150, 195]
    lift_prior = get_lift_prior(0.0, "skeptical")
    results = BayesianEngine(seed=1).run_probability_analysis(
        data=make_experiment(visitors, conversions),
        lift_prior=lift_prior,
        n_samples=200_000,
    )

    rng = np.random.default_rng(7)
    s = np.array([rng.beta(1 + c, 1 + v - c, 400_000)
                 for v, c in zip(visitors, conversions)])
    log_w = sum(
        norm.logpdf(np.log(s[i] / s[0]), lift_prior.mean_log_lift,
                    lift_prior.std_log_lift)
        for i in (1, 2)
    )
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    ref_best = np.bincount(np.argmax(s, axis=0), weights=w, minlength=3)

    assert results[0].prob_being_best == pytest.approx(ref_best[1], abs=0.005)
    assert results[1].prob_being_best == pytest.approx(ref_best[2], abs=0.005)


def test_effective_sample_size_drops_under_skeptical_prior():
    data = make_experiment([1000, 1000], [30, 42])
    flat = BayesianEngine(seed=1).run_probability_analysis(data, lift_prior=None)[0]
    skeptical = BayesianEngine(seed=1).run_probability_analysis(
        data, lift_prior=get_lift_prior(0.0, "skeptical")
    )[0]
    assert skeptical.effective_sample_size < 0.5 * flat.effective_sample_size


def test_prior_prob_positive_lift():
    assert prior_prob_positive_lift(None) == 0.5
    assert prior_prob_positive_lift(get_lift_prior(
        0.0, "skeptical")) == pytest.approx(0.5)
    # +10% / skeptical: Phi(log(1.1) / 0.1) ~= 0.83 -- a strong directional belief.
    assert prior_prob_positive_lift(get_lift_prior(
        10.0, "skeptical")) == pytest.approx(0.8295, abs=1e-3)
    assert prior_prob_positive_lift(get_lift_prior(-10.0, "moderate")) < 0.5


def test_simulate_uplift_distributions_shape_and_location():
    engine = BayesianEngine(seed=1)
    dists = engine.simulate_uplift_distributions(
        data=make_experiment([20000, 20000, 20000], [600, 660, 600]),
        lift_prior=get_lift_prior(0.0, "moderate"),
        n_samples=20_000,
    )
    assert len(dists) == 2
    assert all(d.shape == (20_000,) for d in dists)
    assert 0.05 < np.median(dists[0]) < 0.15   # observed +10%
    assert abs(np.median(dists[1])) < 0.03     # observed 0%


# --- Monetary projection: rate vs volume ---

def test_monetary_compares_rates_not_volumes(engine):
    """
    Equal conversions on uneven traffic: the variant's conversion *rate* is
    ~9% lower, so it must not get a ~50% chance to beat control just
    because it received more visitors.
    """
    visitors, conversions = [10000, 11000], [300, 300]
    labels = ["Control", "Challenger"]
    r = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=BusinessCaseInput(
            aovs={"Control": 50.0, "Challenger": 50.0},
            runtime_days=14,
            projection_period=90,
        ),
        variant_labels=labels,
        lift_prior=None,
    )[0]
    assert r["prob_beat_control"] < 0.2
    assert r["expected_total_contribution"] < 0


def test_monetary_zero_aov_does_not_produce_nan(engine):
    visitors, conversions = [1000, 1000], [100, 110]
    labels = ["Control", "Challenger"]
    r = engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=BusinessCaseInput(
            aovs={"Control": 0.0, "Challenger": 0.0},
            runtime_days=14,
            projection_period=90,
        ),
        variant_labels=labels,
    )[0]
    assert r["expected_uplift"] == 0.0
    assert r["expected_risk"] == 0.0


# --- Monetary projection: per-arm AOV ---

def _monetary(engine, visitors, conversions, aovs, **kwargs):
    labels = list(aovs)
    return engine.run_monetary_projection(
        visitors=visitors,
        conversions=conversions,
        biz_case=BusinessCaseInput(aovs=aovs, runtime_days=14, projection_period=90),
        variant_labels=labels,
        **kwargs,
    )


def test_monetary_equal_aov_matches_rate_formula(engine):
    """
    With equal, exactly known AOVs (aov_cv=0) the revenue difference is
    AOV * (b - a) * traffic, so the projection must equal
    period * traffic * AOV * E[(b - a)+] from run_probability_analysis,
    within MC error.
    """
    visitors, conversions = [9000, 9100], [270, 300]
    prior = get_lift_prior(0.0, "moderate")
    r = _monetary(engine, visitors, conversions, {"Control": 50.0, "Challenger": 50.0},
                  lift_prior=prior, aov_cv=0.0, n_simulations=400_000)[0]
    p = BayesianEngine(seed=7).run_probability_analysis(
        make_experiment(visitors, conversions), lift_prior=prior, n_samples=400_000
    )[0]
    scale = 90 * (np.mean(visitors) / 14) * 50.0
    assert r["expected_uplift"] == pytest.approx(scale * p.expected_uplift, rel=0.03)
    assert r["expected_risk"] == pytest.approx(scale * p.expected_loss, rel=0.05)
    assert r["prob_revenue_better"] == pytest.approx(r["prob_beat_control"], abs=1e-12)


@pytest.mark.parametrize("variant_aov, sign", [(60.0, 1), (40.0, -1)])
def test_monetary_counts_aov_difference_at_equal_rates(engine, variant_aov, sign):
    """
    Same conversion rate, challenger AOV 20% higher (lower): the projection
    must show a revenue gain (loss) of about period * traffic * cr * dAOV,
    while the rate-based prob_beat_control stays near 0.5.
    """
    visitors, conversions = [10000, 10000], [300, 300]
    r = _monetary(engine, visitors, conversions,
                  {"Control": 50.0, "Challenger": variant_aov}, lift_prior=None)[0]

    cr = 301 / 10002  # posterior mean under Beta(1, 1)
    expected_net = 90 * (10000 / 14) * cr * (variant_aov - 50.0)
    assert r["expected_total_contribution"] == pytest.approx(expected_net, rel=0.05)
    assert abs(r["prob_beat_control"] - 0.5) < 0.02
    if sign > 0:
        assert r["prob_revenue_better"] > 0.95
        assert r["expected_uplift"] > 0 and r["expected_uplift"] > 20 * r["expected_risk"]
    else:
        assert r["prob_revenue_better"] < 0.05
        assert r["expected_risk"] > 0 and r["expected_risk"] > 20 * r["expected_uplift"]


# --- Lift prior width names ---

@pytest.mark.parametrize("width, alias, std", [
    ("narrow", "skeptical", 0.10),
    ("medium", "moderate", 0.25),
    ("wide", "uninformative", 1.00),
])
@pytest.mark.parametrize("lift", [0.0, 7.5])
def test_lift_prior_width_names_and_aliases(width, alias, std, lift):
    new, old = get_lift_prior(lift, width), get_lift_prior(lift, alias)
    assert new == old
    assert new.std_log_lift == std
    assert new.mean_log_lift == pytest.approx(np.log1p(lift / 100))


def test_lift_prior_unknown_width_raises():
    with pytest.raises(ValueError, match="narrow.*skeptical"):
        get_lift_prior(0.0, "tight")


# --- BusinessCaseInput Beta prior ---

def _biz_prior_run(alpha, beta, **kwargs):
    return BayesianEngine(seed=11).run_monetary_projection(
        visitors=[1000, 1000],
        conversions=[100, 150],
        biz_case=BusinessCaseInput(
            aovs={"Control": 50.0, "Challenger": 50.0},
            runtime_days=30, projection_period=90,
            alpha_prior=alpha, beta_prior=beta,
        ),
        variant_labels=["Control", "Challenger"],
        **kwargs,
    )[0]


def test_biz_case_beta_prior_used_when_no_beta_prior_passed():
    """
    Without a beta_prior argument the projection uses Beta(alpha_prior,
    beta_prior) from the business case, so a strong prior shrinks the lift.
    """
    flat = _biz_prior_run(1.0, 1.0)
    strong = _biz_prior_run(500.0, 500.0)
    assert strong["expected_uplift"] < 0.8 * flat["expected_uplift"]
    # Same as passing the prior explicitly (same seed, same draws).
    explicit = _biz_prior_run(1.0, 1.0, beta_prior=BetaPrior(500.0, 500.0))
    assert strong["expected_uplift"] == explicit["expected_uplift"]


def test_explicit_beta_prior_overrides_biz_case():
    a = _biz_prior_run(500.0, 500.0, beta_prior=BetaPrior(1.0, 1.0))
    b = _biz_prior_run(1.0, 1.0)
    assert a["expected_uplift"] == b["expected_uplift"]


# --- Monte Carlo margin ---

def test_mc_margin_formula_and_validation():
    assert mc_margin(0.5, 10_000) == pytest.approx(1.96 * 0.005)
    assert mc_margin(0.9, 2_500, z=1.0) == pytest.approx(np.sqrt(0.09 / 2_500))
    assert mc_margin(0.0, 100) == 0.0
    for bad in [dict(p=1.2, ess=10), dict(p=0.5, ess=0), dict(p=0.5, ess=10, z=0)]:
        with pytest.raises(ValueError):
            mc_margin(**bad)


def test_mc_margin_matches_seed_to_seed_spread():
    """
    Under a skeptical lift prior (ESS well below n_samples), the spread of
    prob_beat_control across seeds should match mc_margin(p, ess) / z.
    """
    data = make_experiment([3000, 3000], [90, 110])
    prior = get_lift_prior(0.0, "skeptical")
    runs = [
        BayesianEngine(seed=s).run_probability_analysis(
            data, lift_prior=prior, n_samples=4_000
        )[0]
        for s in range(200)
    ]
    probs = np.array([r.prob_beat_control for r in runs])
    ess = np.mean([r.effective_sample_size for r in runs])
    assert ess < 0.8 * 4_000
    predicted_sd = mc_margin(probs.mean(), ess, z=1.0)
    assert probs.std(ddof=1) == pytest.approx(predicted_sd, rel=0.25)


# --- prob_best_overall computed inside run_monetary_projection ---

def test_monetary_prob_best_overall_computed_from_own_draws():
    """
    prob_best_overall comes from the projection's own weighted draws: it
    matches run_probability_analysis within MC error, and a passed value is
    ignored with a DeprecationWarning.
    """
    visitors, conversions = [15000, 15000, 15000], [450, 495, 480]
    labels = ["Control", "B", "C"]
    prior = get_lift_prior(0.0, "skeptical")
    biz = BusinessCaseInput(aovs=dict.fromkeys(labels, 50.0), runtime_days=14)

    expected = BayesianEngine(seed=21).run_probability_analysis(
        make_experiment(visitors, conversions, labels),
        lift_prior=prior, n_samples=200_000,
    )
    with pytest.warns(DeprecationWarning, match="prob_best_overall"):
        got = BayesianEngine(seed=22).run_monetary_projection(
            visitors, conversions, biz, [0.9, 0.05, 0.05], labels,
            lift_prior=prior, n_simulations=200_000,
        )
    for e, g in zip(expected, got):
        assert g["prob_best_overall"] == pytest.approx(e.prob_being_best, abs=0.01)
        assert g["prob_best_overall"] != 0.05


def test_monetary_without_prob_best_overall_does_not_warn(recwarn):
    BayesianEngine(seed=1).run_monetary_projection(
        visitors=[1000, 1000], conversions=[100, 110],
        biz_case=BusinessCaseInput(aovs={"A": 50.0, "B": 50.0}, runtime_days=14),
        variant_labels=["A", "B"],
    )
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_monetary_requires_variant_labels():
    with pytest.raises(ValueError, match="variant_labels is required"):
        BayesianEngine(seed=1).run_monetary_projection(
            visitors=[1000, 1000], conversions=[100, 110],
            biz_case=BusinessCaseInput(aovs={"A": 50.0, "B": 50.0}, runtime_days=14),
        )


# --- Monetary projection: uncertainty in each arm's mean AOV ---

def test_aov_uncertainty_widens_risk_at_equal_aov():
    """
    Each arm's mean AOV is uncertain (sd = aov_cv * AOV / sqrt(orders)), so
    with equal AOVs a larger aov_cv must raise the expected risk and pull
    prob_revenue_better toward 0.5, while prob_beat_control (rate-based)
    does not move. A shared or no AOV noise leaves all three unchanged.
    """
    visitors, conversions = [9000, 9100], [270, 300]
    aovs = {"Control": 50.0, "Challenger": 50.0}
    exact = _monetary(BayesianEngine(seed=31), visitors, conversions, aovs,
                      aov_cv=0.0, n_simulations=400_000)[0]
    noisy = _monetary(BayesianEngine(seed=31), visitors, conversions, aovs,
                      aov_cv=1.0, n_simulations=400_000)[0]
    assert noisy["expected_risk"] > 2.0 * exact["expected_risk"]
    assert abs(noisy["prob_revenue_better"] - \
               0.5) < abs(exact["prob_revenue_better"] - 0.5) - 0.05
    assert noisy["prob_beat_control"] == exact["prob_beat_control"]


def test_aov_uncertainty_shrinks_with_more_orders():
    """
    The same AOV gap with 100x the orders (and visitors) is far more certain:
    sd of each mean AOV falls by 10x, so prob_revenue_better moves toward 1.
    With 400 orders, rate uncertainty alone gives ~0.73; the AOV uncertainty
    (cv 1.5 / sqrt(400) = 7.5% per arm) brings it to ~0.63.
    """
    aovs = {"Control": 50.0, "Challenger": 52.0}
    few = _monetary(BayesianEngine(seed=32), [2000, 2000], [400, 400], aovs,
                    aov_cv=1.5, n_simulations=200_000)[0]
    many = _monetary(BayesianEngine(seed=32), [200_000, 200_000], [40_000, 40_000], aovs,
                     aov_cv=1.5, n_simulations=200_000)[0]
    assert 0.58 < few["prob_revenue_better"] < 0.68
    assert many["prob_revenue_better"] > 0.95
