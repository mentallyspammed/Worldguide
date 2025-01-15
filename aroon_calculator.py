import pandas as pd
import numpy as np

def calculate_aroon_up_manual(data: pd.DataFrame, window: int = 25) -> pd.Series:
    """Calculates Aroon Up manually."""
    aroon_up = []
    for i in range(len(data)):
        if i < window:
            aroon_up.append(np.nan)
        else:
            period_high = data["high"][i - window: i + 1]
            days_since_high = period_high.reset_index(drop=True).idxmax()
            up = ((window - days_since_high) / window) * 100
            aroon_up.append(up)
    return pd.Series(aroon_up, index=data.index, name="Aroon Up")

def calculate_aroon_down_manual(data: pd.DataFrame, window: int = 25) -> pd.Series:
    """Calculates Aroon Down manually."""
    aroon_down = []
    for i in range(len(data)):
        if i < window:
            aroon_down.append(np.nan)
        else:
            period_low = data["low"][i - window: i + 1]
            days_since_low = period_low.reset_index(drop=True).idxmin()
            down = ((window - days_since_low) / window) * 100
            aroon_down.append(down)
    return pd.Series(aroon_down, index=data.index, name="Aroon Down")
