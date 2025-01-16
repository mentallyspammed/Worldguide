import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from dotenv import load_dotenv
from colorama import init, Fore, Style
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

# Initialize colorama for neon terminal colors
init(autoreset=True)

# --- Constants ---
LOG_DIR = "botlogs"
ST_LOUIS_TZ = ZoneInfo("America/Chicago")
RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ADX_THRESHOLD = 25
FAST_MA_WINDOW = 12
SLOW_MA_WINDOW = 26

# Load environment variables
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = os.getenv("BYBIT_TESTNET", "False").lower() in ("true", "1", "t")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "5")

# Logging setup
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(
    LOG_DIR, f"trading_bot_{datetime.now(ST_LOUIS_TZ).strftime('%Y%m%d')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("TradingAnalyzer")


# --- Helper Functions ---
def calculate_fibonacci_pivots(
    high: float, low: float, close: float
) -> Dict[str, float]:
    """Calculates Fibonacci pivot points."""
    pivot = (high + low + close) / 3
    range_val = high - low
    return {
        "R3": pivot + range_val * 1.000,
        "R2": pivot + range_val * 0.618,
        "R1": pivot + range_val * 0.382,
        "Pivot": pivot,
        "S1": pivot - range_val * 0.382,
        "S2": pivot - range_val * 0.618,
        "S3": pivot - range_val * 1.000,
    }


class BybitAPI:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.testnet = TESTNET
        try:
            self.session = HTTP(
                api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Bybit API session: {e}")
            raise

    def fetch_current_price(self, symbol: str) -> Optional[float]:
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

    def fetch_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
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
            df.sort_values("start_time", inplace=True)
            return df.astype(
                {
                    col: float
                    for col in ["open", "high", "low", "close", "volume", "turnover"]
                }
            )
        except Exception as e:
            self.logger.error(f"Error fetching klines: {e}")
            return pd.DataFrame()


class TradingAnalyzer:
    def __init__(self, symbol: str, interval: str, logger: logging.Logger):
        self.logger = logger
        self.symbol = symbol
        self.interval = interval
        self.bybit = BybitAPI(logger)
        self.df = self.fetch_and_prepare_data()

        if self.df.empty:
            raise ValueError(
                "Failed to fetch data. Check your API key, symbol, and interval."
            )

        self._add_technical_indicators()

    def fetch_and_prepare_data(self) -> pd.DataFrame:
        df = self.bybit.fetch_klines(self.symbol, self.interval, limit=200)
        if not df.empty:
            return df.sort_values("start_time").reset_index(drop=True)
        return pd.DataFrame()

    def _add_technical_indicators(self):
        try:
            self.df["RSI"] = RSIIndicator(self.df["close"], window=RSI_WINDOW).rsi()
            self.df["ADX"] = ADXIndicator(
                self.df["high"], self.df["low"], self.df["close"]
            ).adx()
            self.df["fast_ma"] = EMAIndicator(
                self.df["close"], window=FAST_MA_WINDOW
            ).ema_indicator()
            self.df["slow_ma"] = EMAIndicator(
                self.df["close"], window=SLOW_MA_WINDOW
            ).ema_indicator()
        except Exception as e:
            self.logger.error(f"Error adding indicators: {e}")

    def determine_trend(self) -> str:
        adx = self.df["ADX"].iloc[-1]
        if adx < ADX_THRESHOLD:
            return "neutral"
        if self.df["fast_ma"].iloc[-1] > self.df["slow_ma"].iloc[-1]:
            return "bullish"
        return "bearish"

    def get_signal(self) -> Dict[str, Any]:
        trend = self.determine_trend()
        current_price = self.bybit.fetch_current_price(self.symbol)
        if not current_price:
            return {"position": "NONE"}

        high = self.df["high"].max()
        low = self.df["low"].min()
        close = self.df["close"].iloc[-1]
        fib_pivots = calculate_fibonacci_pivots(high, low, close)

        # Sort Fibonacci levels by proximity to the current price
        sorted_fibs = sorted(
            fib_pivots.items(), key=lambda x: abs(current_price - x[1])
        )
        nearest_fibs = sorted_fibs[:5]  # Nearest 5 Fibonacci levels
        closest_fib = nearest_fibs[0]  # Highlight the closest

        signal = {
            "trend": trend,
            "current_price": current_price,
            "nearest_fibs": nearest_fibs,
            "closest_fib": closest_fib,
            "position": "NONE",
            "entry_price": None,
            "take_profit": None,
        }

        if trend == "bullish" and self.df["RSI"].iloc[-1] < RSI_OVERSOLD:
            signal["position"] = "LONG"
            signal["entry_price"] = current_price
            signal["take_profit"] = (
                closest_fib[1] * 1.01
            )  # Example take profit: 1% above closest fib
        elif trend == "bearish" and self.df["RSI"].iloc[-1] > RSI_OVERBOUGHT:
            signal["position"] = "SHORT"
            signal["entry_price"] = current_price
            signal["take_profit"] = (
                closest_fib[1] * 0.99
            )  # Example take profit: 1% below closest fib

        return signal

    def display_signal(self, signal: Dict[str, Any]):
        neon_cyan = Fore.CYAN
        neon_yellow = Fore.YELLOW
        neon_green = Fore.GREEN
        neon_red = Fore.RED
        neon_magenta = Fore.MAGENTA

        print(f"{neon_cyan}=" * 60)
        print(f"{neon_magenta}Trading Signal Analysis{Style.RESET_ALL}")
        print(
            f"{neon_yellow}Current Price: {signal['current_price']:.2f}{Style.RESET_ALL}"
        )
        print(
            f"{neon_cyan}Trend Direction: {signal['trend'].capitalize()}{Style.RESET_ALL}"
        )
        print(f"\n{neon_magenta}Nearest Fibonacci Levels:{Style.RESET_ALL}")
        for level, value in signal["nearest_fibs"]:
            highlight = Fore.GREEN if (level, value) == signal["closest_fib"] else ""
            print(f"{highlight}{level}: {value:.2f}{Style.RESET_ALL}")

        if signal["position"] != "NONE":
            print(f"\n{neon_green}Position: {signal['position']}{Style.RESET_ALL}")
            print(
                f"{neon_cyan}Entry Price: {signal['entry_price']:.2f}{Style.RESET_ALL}"
            )
            print(
                f"{neon_cyan}Take Profit: {signal['take_profit']:.2f}{Style.RESET_ALL}"
            )
        else:
            print(f"{neon_red}No trade signal at this time.{Style.RESET_ALL}")
        print(f"{neon_cyan}=" * 60)


def main():
    analyzer = TradingAnalyzer(symbol=SYMBOL, interval=INTERVAL, logger=logger)
    logger.info(f"Starting bot for {SYMBOL} on {INTERVAL}-minute interval")

    while True:
        signal = analyzer.get_signal()
        analyzer.display_signal(signal)
        time.sleep(int(INTERVAL) * 60)


if __name__ == "__main__":
    main()
