from typing import List

def validate_experiment_data(visitors: List[int], conversions: List[int]):
    """
    Validates that input data is statistically and logically sound.
    Raises ValueError if data is invalid.
    """
    if len(visitors) != len(conversions):
        raise ValueError("The number of visitor counts must match the number of conversion counts.")
    
    if any(v <= 0 for v in visitors):
        raise ValueError("Visitor counts must be greater than zero.")
        
    if any(c < 0 for c in conversions):
        raise ValueError("Conversion counts cannot be negative.")
        
    for v, c in zip(visitors, conversions):
        if c > v:
            raise ValueError(f"Impossible data detected: Conversions ({c}) exceed visitors ({v}).")
