import os
import json
import logging
import requests
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime
from colorama import init, Fore, Style
from dotenv import load_dotenv
import hmac
import hashlib

# Initialize colorama for colored terminal output
init(autoreset=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
BASE_URL = "https://api.bybit.com"
TESTNET_URL = "https://api-testnet.bybit.com"
USE_TESTNET = False

if not API_KEY or not API_SECRET:
    logging.error(
        f"{Fore.RED}API keys not set. Please ensure BYBIT_API_KEY and BYBIT_API_SECRET are defined in a .env file.{Style.RESET_ALL}"
    )
    exit(1)


# --- Utility Functions ---
def get_base_url() -> str:
    """Return the correct API base URL based on Testnet usage."""
    return TESTNET_URL if USE_TESTNET else BASE_URL


def bybit_request(method: str, endpoint: str, params: dict = None) -> dict:
    """Send a signed request to the Bybit API."""
    timestamp = str(int(datetime.utcnow().timestamp() * 1000))
    params = params or {}
    param_str = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
    signature_payload = f"{timestamp}{API_KEY}{param_str}".encode()
    signature = hmac.new(
        API_SECRET.encode(), signature_payload, hashlib.sha256
    ).hexdigest()

    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json",
    }

    url = f"{get_base_url()}{endpoint}"
    try:
        response = requests.request(method, url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"API request failed: {e}")
        return {}


# --- Market Data Fetching ---
def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch historical kline (candlestick) data from Bybit API."""
    endpoint = "/v5/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = bybit_request("GET", endpoint, params)
    if response.get("retCode") == 0 and response.get("result"):
        df = pd.DataFrame(
            response["result"]["list"],
            columns=["start_time", "open", "high", "low", "close", "volume"],
        )
        df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
        df = df.astype(
            {
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
            }
        )
        return df
    else:
        logging.error(f"Failed to fetch klines for {symbol}. Response: {response}")
        return pd.DataFrame()


def fetch_orderbook(symbol: str, limit: int = 10) -> dict:
    """Fetch order book data from Bybit API."""
    endpoint = "/v5/market/orderbook"
    params = {"symbol": symbol, "limit": limit}
    response = bybit_request("GET", endpoint, params)
    if response.get("retCode") == 0 and response.get("result"):
        return response["result"]
    else:
        logging.error(f"Failed to fetch order book for {symbol}. Response: {response}")
        return {}


# --- TradingAnalyzer Class ---
class TradingAnalyzer:
    """
    A class for performing technical analysis on trading data, including:
    - Moving averages
    - Momentum calculation
    - Trend detection
    - Fibonacci retracement
    - Order book metrics
    - Merge sort for price analysis
    """

    def __init__(
        self,
        close_prices: pd.Series,
        high_price: float,
        low_price: float,
        orderbook_data: Dict,
    ):
        if not isinstance(close_prices, pd.Series):
            raise ValueError("close_prices must be a pandas Series.")
        if not isinstance(high_price, (int, float)):
            raise ValueError("high_price must be a number.")
        if not isinstance(low_price, (int, float)):
            raise ValueError("low_price must be a number.")
        if not isinstance(orderbook_data, dict):
            raise ValueError("orderbook_data must be a dictionary.")

        self.close_prices = close_prices
        self.high_price = high_price
        self.low_price = low_price
        self.orderbook_data = orderbook_data

    def calculate_sma(self, window: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        return self.close_prices.rolling(window=window).mean()

    def determine_trend(self, short_ma: pd.Series, long_ma: pd.Series) -> str:
        """Determine market trend based on moving averages."""
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return "Bullish"
        elif short_ma.iloc[-1] < long_ma.iloc[-1]:
            return "Bearish"
        else:
            return "Neutral"

    def calculate_fibonacci_levels(self) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        diff = self.high_price - self.low_price
        levels = {
            "23.6%": self.high_price - (diff * 0.236),
            "38.2%": self.high_price - (diff * 0.382),
            "50.0%": self.high_price - (diff * 0.5),
            "61.8%": self.high_price - (diff * 0.618),
            "78.6%": self.high_price - (diff * 0.786),
        }
        return levels

    def analyze(self) -> None:
        """Perform analysis and print the results."""
        short_ma = self.calculate_sma(window=5)
        long_ma = self.calculate_sma(window=10)
        trend = self.determine_trend(short_ma, long_ma)
        fib_levels = self.calculate_fibonacci_levels()

        print(f"\nTrend: {trend}")
        print("Fibonacci Levels:")
        for name, value in fib_levels.items():
            print(f"  {name}: {value:.2f}")


# --- Main Execution ---
def main():
    symbol = input(
        f"{Fore.CYAN}Enter trading symbol (e.g., BTCUSDT): {Style.RESET_ALL}"
    ).upper()
    interval = input(f"{Fore.CYAN}Enter timeframe (e.g., 1h, 15m): {Style.RESET_ALL}")

    df = fetch_klines(symbol, interval)
    if df.empty:
        print(f"{Fore.RED}Failed to fetch data for {symbol}.{Style.RESET_ALL}")
        return

    close_prices = df["close"]
    high_price = df["high"].max()
    low_price = df["low"].min()
    orderbook_data = fetch_orderbook(symbol)

    analyzer = TradingAnalyzer(close_prices, high_price, low_price, orderbook_data)
    analyzer.analyze()


if __name__ == "__main__":
    main()
