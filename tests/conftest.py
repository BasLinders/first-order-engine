"""
tests/conftest.py: shared pytest configuration for the FOE test suite.

Fixtures defined here are available to every test file without importing.
Local fixture definitions in individual test files shadow conftest fixtures
of the same name, so adding this file does not break existing tests.
"""

import pytest

from foe.bayesian.operations import BayesianEngine, get_beta_prior, get_lift_prior
from foe.frequentist.operations import FrequentistEngine
from foe.core.models import ExperimentInput


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------
# Named distinctly (frequentist_engine / bayesian_engine) so they can coexist
# with the existing `engine` fixtures in each test file without shadowing them.

@pytest.fixture
def frequentist_engine() -> FrequentistEngine:
    """
    A stateless FrequentistEngine instance.
    Function-scoped (default). Each test receives a fresh instance.
    """
    return FrequentistEngine()


@pytest.fixture
def bayesian_engine() -> BayesianEngine:
    """
    A BayesianEngine with a fixed seed for deterministic Monte Carlo results.
    Function-scoped (default) so each test starts from the same RNG state,
    making probabilistic threshold assertions stable regardless of test order.
    """
    return BayesianEngine(seed=42)


# ---------------------------------------------------------------------------
# Prior fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uninformative_priors():
    """
    A flat Beta(1,1) prior paired with an uninformative lift prior.
    Applies virtually no pressure on the posterior. Data dominates.
    """
    return get_beta_prior(), get_lift_prior(0.0, "uninformative")


@pytest.fixture
def skeptical_priors():
    """
    A flat Beta(1,1) prior paired with a skeptical lift prior (std = 0.10).
    Penalises large observed lifts, pulling posteriors toward zero effect.
    """
    return get_beta_prior(), get_lift_prior(0.0, "skeptical")


# ---------------------------------------------------------------------------
# ExperimentInput factory
# ---------------------------------------------------------------------------
# Returns a callable so tests can construct inputs with minimal boilerplate.
# Usage (after adding `make_experiment` to the test function signature):
#
#   def test_foo(make_experiment):
#       data = make_experiment([1000, 1000], [100, 120])

@pytest.fixture
def make_experiment():
    """
    Factory fixture for ExperimentInput.

    Auto-generates variant labels (A, B, C, …) when none are supplied.
    Raises ValidationError on invalid data. Callers should use
    pytest.raises(ValidationError) when testing bad inputs directly.
    """
    def _make(
        visitors: list,
        conversions: list,
        labels: list = None,
    ) -> ExperimentInput:
        if labels is None:
            labels = [chr(65 + i) for i in range(len(visitors))]
        return ExperimentInput(
            visitors=visitors,
            conversions=conversions,
            labels=labels,
        )

    return _make
