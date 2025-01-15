def fib(high, low):
    """Calculates Fibonacci retracement levels.

    Args:
        high: The highest price in the dataset.
        low: The lowest price in the dataset.

    Returns:
        A dictionary containing the calculated Fibonacci levels.  Returns an empty dictionary if input is invalid.
    """
    if (
        not isinstance(high, (int, float))
        or not isinstance(low, (int, float))
        or high <= low
    ):
        return {}  # Handle invalid input

    diff = high - low
    levels = {
        "100%": high,
        "78.6%": high - diff * 0.236,
        "61.8%": high - diff * 0.382,
        "50%": high - diff * 0.5,
        "38.2%": high - diff * 0.618,
        "23.6%": high - diff * 0.764,
        "0%": low,
    }
    return levels
