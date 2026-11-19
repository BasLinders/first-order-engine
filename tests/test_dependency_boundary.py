"""
Enforces FOE's dependency boundary (docs/dependency-reduction.md).

Only foe/stats/ may import third-party statistical code, and only numerical
primitives. Every other module imports its statistics from foe.stats.

ALLOWED_ELSEWHERE lists the imports outside foe/stats/ that remain until their
replacement PR lands. It must only shrink: the test fails on any import not
listed here, and on any listed entry that is no longer imported (so it gets
removed when the code moves).
"""

import ast
from pathlib import Path

FOE_ROOT = Path(__file__).resolve().parent.parent / "foe"
STATS_PKG = FOE_ROOT / "stats"

RESTRICTED = ("scipy", "statsmodels", "patsy", "pingouin")
STATS_PKG_ALLOWED = {"scipy.special", "scipy.optimize"}

ALLOWED_ELSEWHERE = {
    "behavioral/operations.py": {"scipy.stats", "statsmodels.formula.api"},  # PR 2, PR 6
    "continuous/operations.py": {
        "scipy.stats",  # PR 6, PR 8
        "scipy.optimize",  # PR 4
        "scipy.special",  # PR 4
        "statsmodels.api",  # PR 2
        "statsmodels.formula.api",  # PR 2
        "statsmodels.discrete.discrete_model",  # PR 4
    },
    "frequentist/boschloo.py": {"scipy.stats", "statsmodels.stats.proportion"},  # PR 7, PR 2
    "frequentist/confidence.py": {"statsmodels.stats.proportion"},  # PR 2
    "frequentist/operations.py": {"statsmodels.formula.api"},  # PR 3
    "interaction/operations.py": {
        "statsmodels.api",  # PR 5
        "statsmodels.genmod.generalized_linear_model",  # PR 5
        "patsy",  # PR 5
    },
    "srm/operations.py": {"scipy.stats"},  # PR 6
}


def _restricted_imports(path: Path) -> set:
    """Restricted modules imported anywhere in the file, including inside functions."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            # "from scipy import stats" -> scipy.stats; "from scipy.stats import norm" -> scipy.stats
            if "." in node.module:
                names = [node.module]
            else:
                names = [f"{node.module}.{a.name}" for a in node.names]
        else:
            continue
        found.update(n for n in names if n.split(".")[0] in RESTRICTED)
    return found


def _foe_files():
    return sorted(p for p in FOE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_stats_package_imports_only_primitives():
    for path in _foe_files():
        if STATS_PKG in path.parents:
            bad = _restricted_imports(path) - STATS_PKG_ALLOWED
            assert not bad, f"{path.relative_to(FOE_ROOT)} imports {sorted(bad)}"


def test_engines_import_statistics_from_foe_stats():
    violations, stale = [], []
    seen = set()
    for path in _foe_files():
        if STATS_PKG in path.parents:
            continue
        rel = path.relative_to(FOE_ROOT).as_posix()
        found = _restricted_imports(path)
        allowed = ALLOWED_ELSEWHERE.get(rel, set())
        violations += [f"{rel}: {m}" for m in sorted(found - allowed)]
        stale += [f"{rel}: {m}" for m in sorted(allowed - found)]
        seen.add(rel)
    stale += [f"{rel}: (file not found)" for rel in ALLOWED_ELSEWHERE if rel not in seen]

    assert not violations, "Import these from foe.stats instead:\n" + "\n".join(violations)
    assert not stale, "Remove from ALLOWED_ELSEWHERE, no longer imported:\n" + "\n".join(stale)
