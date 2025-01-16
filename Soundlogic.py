# merged_trading_script.py
# Unified trading bot script combining tabb.py and wgsta.py with enhanced precision and fixed errors.

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional
from zoneinfo import ZoneInfo
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from colorama import init, Fore, Style
import time

# Initialize colorama for colored terminal output
init(autoreset=True)

# --- Constants ---
LOG_DIR = "botlogs"
ST_LOUIS_TZ = ZoneInfo("America/Chicago")
MAX_RETRIES = 3
RETRY_DELAY = 5
PRECISION = 8  # Set precision for floating-point numbers

# Logging setup
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(
    LOG_DIR, f"trading_bot_{datetime.now(ST_LOUIS_TZ).strftime('%y%m%d')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("TradingBot")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = os.getenv("BYBIT_TESTNET", "False").lower() in ("true", "1", "t")


# --- Helper Functions ---
def retry_request(func, *args, **kwargs):
    """Retry wrapper for API requests."""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Retry {retries + 1}/{MAX_RETRIES} after error: {e}")
            retries += 1
            time.sleep(RETRY_DELAY * (2**retries))
    logger.error(f"Max retries exceeded for function {func.__name__}.")
    return None


# --- Bybit API Integration ---
class BybitAPI:
    def __init__(self):
        self.session = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET)
        self.logger = logger

    def fetch_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Fetch historical kline data."""
        try:
            response = self.session.get_kline(
                category="linear", symbol=symbol, interval=interval, limit=limit
            )
            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return pd.DataFrame()
            df = pd.DataFrame(
                response["result"]["list"],
                columns=[
                    "start_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover",
                ],
            )
            df["start_time"] = pd.to_datetime(
                pd.to_numeric(df["start_time"]), unit="ms", utc=True
            )
            return df.astype(
                {
                    col: float
                    for col in ["open", "high", "low", "close", "volume", "turnover"]
                }
            )
        except Exception as e:
            self.logger.error(f"Error fetching klines: {e}")
            return pd.DataFrame()

    def fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch the current price of a symbol."""
        try:
            response = self.session.get_tickers(category="linear", symbol=symbol)
            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return None
            ticker_info = response.get("result", {}).get("list", [])[0]
            return (
                float(ticker_info["lastPrice"]) if "lastPrice" in ticker_info else None
            )
        except Exception as e:
            self.logger.error(f"Error fetching current price: {e}")
            return None


# --- Technical Analysis ---
class TechnicalAnalyzer:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.api = BybitAPI()
        self.df = self.api.fetch_klines(symbol, interval, limit=200)
        if self.df.empty:
            raise ValueError(f"No data available for {symbol} at interval {interval}.")

    def add_indicators(self):
        """Add technical indicators to the DataFrame."""
        self.df["RSI"] = RSIIndicator(self.df["close"], window=14).rsi()
        self.df["ATR"] = AverageTrueRange(
            self.df["high"], self.df["low"], self.df["close"], window=14
        ).average_true_range()
        self.df["EMA_fast"] = EMAIndicator(self.df["close"], window=9).ema_indicator()
        self.df["EMA_slow"] = EMAIndicator(self.df["close"], window=21).ema_indicator()

    def calculate_fibonacci(self) -> Dict[str, float]:
        """Calculate Fibonacci levels."""
        high, low = self.df["high"].max(), self.df["low"].min()
        diff = high - low
        return {
            "Fib 161.8%": round(high + diff * 1.618, PRECISION),
            "Fib 100.0%": round(high, PRECISION),
            "Fib 78.6%": round(high - diff * 0.786, PRECISION),
            "Fib 61.8%": round(high - diff * 0.618, PRECISION),
            "Fib 50.0%": round(high - diff * 0.5, PRECISION),
            "Fib 38.2%": round(high - diff * 0.382, PRECISION),
            "Fib 23.6%": round(high - diff * 0.236, PRECISION),
            "Fib 0.0%": round(low, PRECISION),
        }


# --- Example Usage ---
if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "5"
    try:
        analyzer = TechnicalAnalyzer(symbol, interval)
        analyzer.add_indicators()
        fib_levels = analyzer.calculate_fibonacci()
        logger.info(f"Fibonacci Levels for {symbol}: {fib_levels}")
    except ValueError as e:
        logger.error(f"Error during analysis: {e}")
