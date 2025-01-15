import numpy as np
import pandas as pd


def weighted_moving_average(series, period):
    """Calculates Weighted Moving Average (WMA)."""
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def calculate_hma(prices, period):
    """Calculates the Hull Moving Average (HMA)."""
    # Step 1: WMA for n/2 periods
    wma_half_period = weighted_moving_average(prices, period // 2)

    # Step 2: WMA for n periods
    wma_full_period = weighted_moving_average(prices, period)

    # Step 3: Difference and multiply by 2
    diff = 2 * wma_half_period - wma_full_period

    # Step 4: WMA for sqrt(n) periods
    hma = weighted_moving_average(diff, int(np.sqrt(period)))

    return hma
    # Add HMA to DataFrame
    # df['hma'] = calculate_hma(df['close'], period=14)

    # Use HMA for trend determination
    # if df['hma'].iloc[-1] > df['hma'].iloc[-2]:
    #   trend = "upward"
    # elif df['hma'].iloc[-1] < df['hma'].iloc[-2]:
    #   trend = "downward"
    # else:
    trend = "neutral"
