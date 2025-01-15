import yfinance as yf
import pandas as pd
import numpy as np

def fibonacci_pivot_points(crypto_symbol, timeframe='1d'):
    """
    Calculates Fibonacci pivot points for a given cryptocurrency and timeframe.

    Args:
        crypto_symbol: The ticker symbol of the cryptocurrency (e.g., 'BTC-USD').
        timeframe: The timeframe for the data (e.g., '1d', '1h').

    Returns:
        A Pandas DataFrame with the calculated Fibonacci pivot points and the current price.
        Returns None if data cannot be retrieved or if the timeframe is invalid.
    """

    try:
        # Fetch historical data using yfinance (install yfinance first: pip install yfinance)
        data = yf.download(crypto_symbol, period=timeframe)

        if data.empty:
            print("Error: Could not retrieve data for the given symbol and timeframe.")
            return None
        
        # Extract relevant data
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]


        # Calculate Pivot Point (PP)
        pp = (high + low + close) / 3

        # Calculate Fibonacci-based levels using the most common ratios
        fib_ratios = [0.382, 0.618]
        resistance_levels = []
        support_levels = []


        for ratio in fib_ratios:
            resistance_levels.append(pp + (high - low) * ratio)
            support_levels.append(pp - (high - low) * ratio)

        # Store the results in a DataFrame
        results = pd.DataFrame({
            'Current Price': [close],
            'Pivot Point': [pp],
            'Resistance 1': resistance_levels[0],
            'Resistance 2': resistance_levels[1],
            'Support 1': support_levels[0],
            'Support 2': support_levels[1],
        })

        return results

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Example usage (replace with your desired crypto symbol and timeframe)
crypto_symbol = input("Enter the cryptocurrency ticker symbol (e.g., BTC-USD): ")
timeframe = input("Enter the timeframe (e.g., 1d, 1h): ")

results = fibonacci_pivot_points(crypto_symbol, timeframe)


if results is not None:
    print(results)