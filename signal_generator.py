import os
import logging
import requests
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from datetime import datetime
from dotenv import load_dotenv
from colorama import init, Fore, Style
import time
from typing import Dict, List, Tuple

# Initialize Colorama for colored outputs
init(autoreset=True)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")

# Logging setup
LOG_DIR = "botlogs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/trading_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("trading_bot")

# Constants
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W"]
MAX_RETRIES = 3
RETRY_DELAY = 5
SUPPORT_RESISTANCE_WINDOW = 14
CLUSTER_SENSITIVITY = 0.05

# --- API Utilities ---
def generate_signature(params: dict) -> str:
    """Generate HMAC SHA256 signature."""
    param_str = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
    return hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()


def fetch_data(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch historical market data."""
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "category": "linear",
    }
    for _ in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") == 0:
                df = pd.DataFrame(data["result"]["list"], columns=["start_time", "open", "high", "low", "close", "volume"])
                df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
                df = df.set_index("start_time").sort_index()
                df = df.astype({col: "float64" for col in ["open", "high", "low", "close", "volume"]})
                return df
        except requests.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            time.sleep(RETRY_DELAY)
    return pd.DataFrame()


def fetch_current_price(symbol: str) -> float:
    """Fetch the current price for a given symbol."""
    url = f"{BASE_URL}/v5/market/tickers"
    params = {"symbol": symbol, "category": "linear"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("retCode") == 0:
            return float(data["result"]["list"][0]["lastPrice"])
    except (requests.RequestException, ValueError) as e:
        logger.error(f"Error fetching current price: {e}")
    return None

# --- Indicator Calculations ---
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the DataFrame."""
    df["RSI"] = RSIIndicator(df["close"]).rsi()
    df["ATR"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["SMA_Fast"] = SMAIndicator(df["close"], window=5).sma_indicator()
    df["SMA_Slow"] = SMAIndicator(df["close"], window=20).sma_indicator()
    return df


def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels."""
    diff = high - low
    return {
        "23.6%": high - (diff * 0.236),
        "38.2%": high - (diff * 0.382),
        "50.0%": high - (diff * 0.5),
        "61.8%": high - (diff * 0.618),
        "78.6%": high - (diff * 0.786),
    }


def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculate pivot points and support/resistance levels."""
    pivot = (high + low + close) / 3
    return {
        "Pivot": pivot,
        "Resistance 1": (2 * pivot) - low,
        "Support 1": (2 * pivot) - high,
        "Resistance 2": pivot + (high - low),
        "Support 2": pivot - (high - low),
    }

# --- Signal Generation ---
def generate_signals(df: pd.DataFrame, levels: Dict[str, float]) -> List[Tuple[str, str, float, float, float]]:
    """Generate trading signals."""
    signals = []
    close_price = df["close"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    for level_name, level in levels.items():
        if close_price < level:
            tp = level + atr * 1.5
            sl = level - atr * 2
            signals.append(("Long", level_name, level, tp, sl))
        elif close_price > level:
            tp = level - atr * 1.5
            sl = level + atr * 2
            signals.append(("Short", level_name, level, tp, sl))
    return signals

# --- Display ---
def display_output(levels: Dict[str, float], signals: List[Tuple[str, str, float, float, float]]):
    """Display levels and signals in neon colors."""
    print(Fore.CYAN + "\nSupport and Resistance Levels:")
    for level_name, level in levels.items():
        print(Fore.MAGENTA + f"{level_name}: {level:.2f}")

    print(Fore.YELLOW + "\nGenerated Signals:")
    for signal_type, level_name, level, tp, sl in signals:
        color = Fore.GREEN if signal_type == "Long" else Fore.RED
        print(f"{color}{signal_type} at {level_name} ({level:.2f}) | TP: {tp:.2f} | SL: {sl:.2f}")

# --- Main ---
def main():
    symbol = input(Fore.CYAN + "Enter trading symbol (e.g., BTCUSDT): ").strip().upper()
    interval = input(Fore.CYAN + f"Enter interval ({', '.join(VALID_INTERVALS)}): ").strip()
    if interval not in VALID_INTERVALS:
        logger.error(Fore.RED + "Invalid interval.")
        return

    while True:
        df = fetch_data(symbol, interval)
        if not df.empty:
            df = calculate_indicators(df)
            current_price = fetch_current_price(symbol)
            if current_price is None:
                time.sleep(30)
                continue

            high, low, close = df["high"].max(), df["low"].min(), df["close"].iloc[-1]
            fib_levels = calculate_fibonacci_levels(high, low)
            pivot_levels = calculate_pivot_points(high, low, close)
            levels = {**fib_levels, **pivot_levels}
            signals = generate_signals(df, levels)
            display_output(levels, signals)
        time.sleep(60)


if __name__ == "__main__":
    main()
