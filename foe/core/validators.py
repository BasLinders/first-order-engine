from typing import List


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
