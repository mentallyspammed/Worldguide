from dotenv import load_dotenv
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from ta.trend import EMAIndicator
from ta.momentum import rsi
from colorama import init, Fore, Back, Style
import time
import pytz
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

# Initialize colorama for neon-colored outputs
init(autoreset=True)


# --- Configuration ---
class Config:
    def __init__(self, symbol: str, interval: str):
        load_dotenv()
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.testnet = True
        self.symbol = symbol
        self.interval = interval
        self.log_level = "INFO"

        # Moving average periods
        self.ma_periods_short = 7
        self.ma_periods_long = 25
        self.fma_period = 13  # Fibonacci Moving Average period

        # RSI
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30

        # Timezone Configuration (St. Louis, Missouri)
        self.timezone = pytz.timezone("America/Chicago")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "API keys not set. Ensure BYBIT_API_KEY and BYBIT_API_SECRET are in .env"
            )

        # Base URL for Unified Trading Account (UTA)
        self.base_url = (
            "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"
        )  # For testnet


# --- Logging Setup ---
def setup_logger(name: str, level: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# --- Bybit API Client ---
class BybitAPI:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = HTTP(
            testnet=config.testnet,
            api_key=config.api_key,
            api_secret=config.api_secret,
        )

    def fetch_current_price(self) -> float | None:
        """Fetches the current real-time price of the symbol using pybit."""
        try:
            response = self.session.get_tickers(
                category="linear", symbol=self.config.symbol
            )

            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return None

            ticker_info = response.get("result", {}).get("list", [])[0]
            if not ticker_info or "lastPrice" not in ticker_info:
                self.logger.error("Current price data unavailable.")
                return None

            return float(ticker_info["lastPrice"])
        except InvalidRequestError as e:
            self.logger.error(f"Invalid request error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching current price: {e}")
            return None

    def fetch_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetches kline data from Bybit API for Unified Trading Account (UTA)
        (specifically for USDT Perpetual contracts) using pybit, ensures it's sorted by time,
        and handles timezone conversion.
        """
        try:
            response = self.session.get_kline(
                category="linear", symbol=symbol, interval=interval, limit=limit
            )

            # Check for API error messages
            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return pd.DataFrame()

            if "result" not in response or "list" not in response["result"]:
                self.logger.warning(f"No kline data found for symbol {symbol}.")
                return pd.DataFrame()

            # Adjust the columns based on the API response
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

            # Convert start_time to datetime and set timezone to UTC
            df["start_time"] = pd.to_datetime(df["start_time"], unit="ms", utc=True)

            # Sort by time to ensure the latest data is at the end
            df.sort_values("start_time", inplace=True)

            # Convert other columns to numeric types
            df = df.astype(
                {
                    col: float
                    for col in ["open", "high", "low", "close", "volume", "turnover"]
                }
            )

            return df

        except Exception as e:
            self.logger.error(f"Error fetching klines: {e}")
            return pd.DataFrame()


# --- Trading Analysis ---
class TradingAnalyzer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.api = BybitAPI(config, logger)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates moving averages, RSI, and volume metrics."""
        if df.empty:
            return df

        # Moving Averages
        df["EMA_Short"] = EMAIndicator(
            df["close"], window=self.config.ma_periods_short
        ).ema_indicator()
        df["EMA_Long"] = EMAIndicator(
            df["close"], window=self.config.ma_periods_long
        ).ema_indicator()
        df["FMA"] = EMAIndicator(
            df["close"], window=self.config.fma_period
        ).ema_indicator()

        # RSI
        df["RSI"] = rsi(df["close"], window=self.config.rsi_period)

        # Volume Analysis
        df["Avg_Volume"] = df["volume"].rolling(window=20).mean()
        return df

    def determine_trend(self, df: pd.DataFrame) -> str:
        """Determines the trend based on EMA and FMA crossovers."""
        if df.empty or "EMA_Short" not in df or "EMA_Long" not in df or "FMA" not in df:
            return "neutral"

        ema_short = df["EMA_Short"].iloc[-1]
        ema_long = df["EMA_Long"].iloc[-1]
        fma = df["FMA"].iloc[-1]

        if ema_short > ema_long > fma:
            return "bullish"
        elif ema_short < ema_long < fma:
            return "bearish"
        return "neutral"

    def analyze_momentum(self, df: pd.DataFrame) -> str:
        """Analyzes RSI to determine momentum."""
        if df.empty or "RSI" not in df:
            return "neutral"

        rsi_value = df["RSI"].iloc[-1]
        if rsi_value > self.config.rsi_overbought:
            return "overbought"
        elif rsi_value < self.config.rsi_oversold:
            return "oversold"
        return "neutral"

    def analyze_volume(self, df: pd.DataFrame) -> str:
        """Analyzes volume to determine if the current volume is above or below average."""
        if df.empty or "volume" not in df or "Avg_Volume" not in df:
            return "neutral"

        current_volume = df["volume"].iloc[-1]
        avg_volume = df["Avg_Volume"].iloc[-1]

        if current_volume > avg_volume:
            return "high"
        return "low"

    def calculate_fibonacci_support_resistance(
        self, df: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """Calculates support and resistance levels using Fibonacci retracement."""
        if df.empty:
            return {"support": [], "resistance": []}

        high = df["high"].max()
        low = df["low"].min()
        price_range = high - low

        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        supports = [low + (price_range * ratio) for ratio in fib_ratios]
        resistances = [high - (price_range * ratio) for ratio in fib_ratios]

        supports.sort()
        resistances.sort()

        current_price = df["close"].iloc[-1]
        nearest_support = max([s for s in supports if s < current_price], default=None)
        nearest_resistance = min(
            [r for r in resistances if r > current_price], default=None
        )

        return {
            "support": supports,
            "resistance": resistances,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
        }

    def generate_trade_signal(
        self, df: pd.DataFrame, current_price: float
    ) -> Optional[Dict[str, float]]:
        """
        Generates a trade signal based on RSI, moving averages, and Fibonacci levels.
        """
        if df.empty:
            return None

        trend = self.determine_trend(df)
        momentum = self.analyze_momentum(df)
        fib_levels = self.calculate_fibonacci_support_resistance(df)
        nearest_support = fib_levels["nearest_support"]
        nearest_resistance = fib_levels["nearest_resistance"]
        rsi_value = df["RSI"].iloc[-1]

        # Long Signal
        if (
            trend == "bullish"
            and momentum == "oversold"
            and abs(current_price - nearest_support) / current_price < 0.01
        ):
            return {
                "signal": "long",
                "entry_price": nearest_support,
                "rsi": rsi_value,
                "stop_loss": nearest_support * 0.99,  # Example: 1% below entry
                "take_profit": nearest_resistance,  # Example: Nearest resistance
            }

        # Short Signal
        if (
            trend == "bearish"
            and momentum == "overbought"
            and abs(current_price - nearest_resistance) / current_price < 0.01
        ):
            return {
                "signal": "short",
                "entry_price": nearest_resistance,
                "rsi": rsi_value,
                "stop_loss": nearest_resistance * 1.01,  # Example: 1% above entry
                "take_profit": nearest_support,  # Example: Nearest support
            }

        return None


# --- Main Function ---
def main():
    # Prompt user for symbol and timeframe
    valid_intervals = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W"]
    symbol = input("Enter the trading symbol (e.g., BTCUSDT): ").strip().upper()
    interval = input(f"Enter the timeframe {valid_intervals}: ").strip()

    # Validate inputs
    if not symbol.isalnum() or len(symbol) < 4:
        print(Fore.RED + "Invalid symbol. Example of valid symbols: BTCUSDT, ETHUSDT.")
        return
    if interval not in valid_intervals:
        print(
            Fore.RED + f"Invalid timeframe. Choose from: {', '.join(valid_intervals)}."
        )
        return

    # Initialize configuration and logging
    config = Config(symbol, interval)
    logger = setup_logger("trading_bot", config.log_level)
    analyzer = TradingAnalyzer(config, logger)

    # Continuous loop every 30 seconds
    while True:
        # Fetch the current price
        current_price = analyzer.api.fetch_current_price()
        if current_price is None:
            logger.error("Failed to fetch current price. Retrying in 30 seconds.")
            time.sleep(30)
            continue

        logger.info(
            f"{Fore.CYAN}Fetching kline data for {symbol} with a {interval}-minute timeframe."
        )
        df = analyzer.api.fetch_klines(config.symbol, config.interval)

        if df.empty:
            logger.error("No data available for analysis.")
            print(Fore.RED + "No data available for analysis.")
        else:
            # Get the very latest kline (which is now correctly the last row after sorting)
            latest_kline = df.iloc[-1]

            # Convert the latest kline's start_time to the user's timezone
            latest_kline_time_local = latest_kline["start_time"].tz_convert(
                config.timezone
            )

            # Calculate Indicators
            df = analyzer.calculate_indicators(df)

            # Trend Analysis
            trend = analyzer.determine_trend(df)

            # Momentum Analysis
            momentum = analyzer.analyze_momentum(df)

            # Volume Analysis
            volume = analyzer.analyze_volume(df)

            # Fibonacci Support/Resistance
            fib_levels = analyzer.calculate_fibonacci_support_resistance(df)

            # Trade Signal
            signal = analyzer.generate_trade_signal(df, current_price)

            # Format Analysis Results (including the localized timestamp and correct price)
            print(Fore.CYAN + "\n--- Analysis Results ---")
            print(f"Symbol: {Fore.YELLOW}{symbol}")
            print(f"Timeframe: {Fore.YELLOW}{interval}")
            print(
                f"Last Close Time ({config.timezone}): {Fore.YELLOW}{latest_kline_time_local.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(f"Current Price: {Fore.GREEN}{current_price}")
            print(
                f"Trend: {Fore.GREEN if trend == 'bullish' else Fore.RED if trend == 'bearish' else Fore.YELLOW}{trend}"
            )
            print(
                f"Momentum: {Fore.RED if momentum == 'overbought' else Fore.GREEN if momentum == 'oversold' else Fore.YELLOW}{momentum}"
            )
            print(f"Volume: {Fore.GREEN if volume == 'high' else Fore.RED}{volume}")
            if not df.empty:
                print(f"RSI: {Fore.YELLOW}{df['RSI'].iloc[-1]:.2f}")

            if fib_levels:
                print(Fore.CYAN + "\n--- Fibonacci Levels ---")
                for i, level in enumerate(fib_levels["support"]):
                    print(f"S{i+1}: {Fore.GREEN}{level:.2f}")
                for i, level in enumerate(fib_levels["resistance"]):
                    print(f"R{i+1}: {Fore.RED}{level:.2f}")
                print(
                    f"Nearest Support: {Fore.GREEN}{fib_levels['nearest_support']:.2f}"
                )
                print(
                    f"Nearest Resistance: {Fore.RED}{fib_levels['nearest_resistance']:.2f}"
                )

            if signal:
                print(Fore.CYAN + "\n--- Trade Signal ---")
                print(f"Signal: {Fore.MAGENTA}{signal['signal'].upper()}")
                print(f"Entry Price: {Fore.YELLOW}{signal['entry_price']:.2f}")
                print(f"RSI at Signal: {Fore.YELLOW}{signal['rsi']:.2f}")
                print(f"Stop Loss: {Fore.YELLOW}{signal['stop_loss']:.2f}")
                print(f"Take Profit: {Fore.YELLOW}{signal['take_profit']:.2f}")
            else:
                print(Fore.CYAN + "\n--- No Trade Signal Generated ---")

            print(Fore.CYAN + "\n-------------------------\n")

        # Wait for 30 seconds before next iteration
        logger.info(f"{Fore.CYAN}Waiting for 30 seconds before the next analysis...\n")
        time.sleep(30)


if __name__ == "__main__":
    main()
