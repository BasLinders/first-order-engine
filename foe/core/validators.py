from typing import List

import pandas as pd


def validate_experiment_data(visitors: List[int], conversions: List[int]) -> None:
    """
    Validates that input arrays are mathematically and logically sound.

    Raises ValueError with a descriptive message if data is invalid.

    Checks performed (in order):
        1. At least two variants are present (control + one challenger).
        2. visitors and conversions have equal length.
        3. All visitor counts are greater than zero.
        4. All conversion counts are non-negative.
        5. No variant has more conversions than visitors.

    Args:
        visitors:    List of visitor counts, one per variant.
        conversions: List of conversion counts, one per variant.

    Raises:
        ValueError: On the first failing check, with the variant index
                    and offending values included in the message.
    """
    if len(visitors) < 2:
        raise ValueError(
            f"An experiment requires at least 2 variants (control + one challenger); "
            f"got {len(visitors)}."
        )
    if len(visitors) != len(conversions):
        raise ValueError(
            f"The number of visitor counts ({len(visitors)}) must match "
            f"the number of conversion counts ({len(conversions)})."
        )
    for i, v in enumerate(visitors):
        if v <= 0:
            raise ValueError(
                f"Variant at index {i}: visitor count must be greater than zero "
                f"(got {v})."
            )
    for i, c in enumerate(conversions):
        if c < 0:
            raise ValueError(
                f"Variant at index {i}: conversion count cannot be negative "
                f"(got {c})."
            )
    for i, (v, c) in enumerate(zip(visitors, conversions)):
        if c > v:
            raise ValueError(
                f"Variant at index {i}: conversions ({c}) exceed visitors ({v})."
            )


def validate_continuous_data(
    df: pd.DataFrame,
    kpi: str,
    group_col: str = "experience_variant_label",
    *,
    approach: str = "heuristic",
    unit: str = "per_transaction",
) -> None:
    """
    Validates a long-format continuous-metric DataFrame for the analysis engine.

    Raises ValueError with a descriptive message if the data cannot support the
    requested analysis. Structural problems that make a test impossible are
    errors; conditions that merely change interpretation (e.g. a per-visitor
    request on a file with no zeros) are NOT raised here — the engine surfaces
    those as non-fatal warnings in its result.

    Checks performed (in order):
        1. The grouping column and the KPI column are present.
        2. At least two variants are present (control + one challenger).
        3. The KPI column has at least one non-null value.
        4. For the Gamma family, the KPI has no negative values (the Gamma /
           two-part model is defined on x >= 0).
        5. Each variant retains enough usable rows for the requested unit:
           - per_transaction: at least 2 strictly positive rows per variant.
           - per_visitor:     at least 2 rows per variant (zeros allowed) and,
                              for the Gamma family, at least 2 positive rows in
                              the whole dataset so a spend Gamma can be fit.

    Args:
        df:        Long-format data, one row per analysis unit.
        kpi:       Name of the numeric metric column.
        group_col: Name of the categorical variant column.
        approach:  'heuristic' or 'gamma' (only the family matters here).
        unit:      'per_visitor' or 'per_transaction'.

    Raises:
        ValueError: On the first failing check.
    """
    is_gamma = str(approach).lower().startswith("gamma")
    per_visitor = unit == "per_visitor"

    if group_col not in df.columns:
        raise ValueError(f"Grouping column '{group_col}' is missing from the data.")
    if kpi not in df.columns:
        raise ValueError(f"KPI column '{kpi}' is missing from the data.")

    series = pd.to_numeric(df[kpi], errors="coerce")
    valid = df.loc[series.notna(), [group_col]].assign(_kpi=series[series.notna()])

    if valid.empty:
        raise ValueError(f"KPI column '{kpi}' has no numeric (non-null) values.")

    n_variants = valid[group_col].nunique()
    if n_variants < 2:
        raise ValueError(
            f"An experiment requires at least 2 variants to compare; "
            f"got {n_variants} in column '{group_col}'."
        )

    if is_gamma and (valid["_kpi"] < 0).any():
        n_neg = int((valid["_kpi"] < 0).sum())
        raise ValueError(
            f"The Gamma family requires non-negative values, but '{kpi}' contains "
            f"{n_neg} negative value(s). Use the heuristic approach for this metric."
        )

    # Per-variant usable-row checks.
    for label, grp in valid.groupby(group_col, observed=True):
        if per_visitor:
            if len(grp) < 2:
                raise ValueError(
                    f"Variant '{label}': per-visitor analysis needs at least 2 rows "
                    f"(got {len(grp)})."
                )
        else:  # per_transaction
            n_pos = int((grp["_kpi"] > 0).sum())
            if n_pos < 2:
                raise ValueError(
                    f"Variant '{label}': per-transaction analysis needs at least 2 "
                    f"positive-value rows (got {n_pos})."
                )

    if is_gamma and per_visitor:
        total_pos = int((valid["_kpi"] > 0).sum())
        if total_pos < 2:
            raise ValueError(
                "The per-visitor Gamma (two-part) model needs at least 2 positive "
                f"values across the dataset to fit the spend component; got {total_pos}."
            )
