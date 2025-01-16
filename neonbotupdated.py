"""
Trading bot for Bybit using technical analysis indicators
Last updated: 2024-01-25
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator, HullMovingAverage
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from dotenv import load_dotenv
from colorama import init, Fore, Style
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError
from retry import retry
# import pandas_ta as ta # Removed pandas_ta

# Initialize colorama
init(autoreset=True)

# --- Constants ---
LOG_DIR = "botlogs"
ST_LOUIS_TZ = ZoneInfo("America/Chicago")
MAX_RETRIES = 3
RETRY_DELAY = 5
VALID_INTERVALS = [1, 3, 5, 15, 30, 60, 240]
CLUSTER_SENSITIVITY = 0.05
SUPPORT_RESISTANCE_WINDOW = 14
VOLUME_LOOKBACK = 5
FAST_HMA_WINDOW = 12
SLOW_WMA_WINDOW = 26
RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ADX_THRESHOLD = 25
STOCH_WINDOW = 14
STOCH_SMOOTH_WINDOW = 3
VOLUME_SPIKE_THRESHOLD = 1.5

# Load environment variables
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = bool(os.getenv("BYBIT_TESTNET", "False").lower() in ("true", "1", "t"))
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "5")

# Initialize logging
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


class BybitAPI:
    """A class to interact with the Bybit API using pybit."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize the Bybit API client.

        Args:
            logger (logging.Logger): Logger instance for recording events
        """
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

    @retry(
        exceptions=(InvalidRequestError,),
        tries=MAX_RETRIES,
        delay=RETRY_DELAY,
        backoff=2,
    )
    def fetch_current_price(self) -> Optional[float]:
        """
        Fetches the current real-time price of the symbol.

        Returns:
            Optional[float]: Current price if available, None otherwise
        """
        try:
            response = self.session.get_tickers(category="linear", symbol=SYMBOL)

            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return None

            ticker_info = response.get("result", {}).get("list", [])[0]
            if not ticker_info or "lastPrice" not in ticker_info:
                self.logger.error("Current price data unavailable")
                return None

            return float(ticker_info["lastPrice"])
        except InvalidRequestError as e:
            self.logger.error(f"Invalid request error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error fetching current price: {e}")
            return None

    @retry(
        exceptions=(InvalidRequestError,),
        tries=MAX_RETRIES,
        delay=RETRY_DELAY,
        backoff=2,
    )
    def fetch_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetches kline/candlestick data from Bybit.

        Args:
            symbol (str): Trading pair symbol
            interval (str): Time interval for klines
            limit (int): Number of klines to fetch

        Returns:
            pd.DataFrame: DataFrame containing kline data
        """
        try:
            response = self.session.get_kline(
                category="linear", symbol=symbol, interval=interval, limit=limit
            )

            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return pd.DataFrame()

            if "result" not in response or "list" not in response["result"]:
                self.logger.warning(f"No kline data found for symbol {symbol}")
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

            # Convert and sort timestamps
            df["start_time"] = pd.to_datetime(
                pd.to_numeric(df["start_time"]), unit="ms", utc=True
            )
            df.sort_values("start_time", inplace=True)

            # Convert numeric columns to float
            numeric_columns = ["open", "high", "low", "close", "volume", "turnover"]
            df = df.astype({col: float for col in numeric_columns})

            return df

        except Exception as e:
            self.logger.error(f"Error fetching klines: {e}")
            return pd.DataFrame()


class TradingAnalyzer:
    """A class for performing trading analysis."""

    def __init__(self, symbol: str, interval: str, logger: logging.Logger):
        """
        Initialize the TradingAnalyzer.

        Args:
            symbol (str): Trading pair symbol
            interval (str): Time interval for analysis
            logger (logging.Logger): Logger instance

        Raises:
            ValueError: If interval is invalid or data fetch fails
        """
        self.logger = logger

        try:
            # Validate interval
            interval_value = int(interval)
            if interval_value not in VALID_INTERVALS:
                raise ValueError(
                    f"Invalid interval: {interval}. "
                    f"Valid intervals are: {VALID_INTERVALS}"
                )
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError(f"Interval must be a number. Got: {interval}") from e
            raise

        self.bybit = BybitAPI(logger)
        self.symbol = symbol.upper()
        self.interval = interval
        self.df = self.fetch_and_prepare_data()

        if self.df.empty:
            raise ValueError(
                "Failed to fetch data. Check your API key, symbol, and interval."
            )

        self._add_technical_indicators()
        self.current_price = self.bybit.fetch_current_price()
        self.fib_levels = self.calculate_fibonacci_levels()

    def fetch_and_prepare_data(self) -> pd.DataFrame:
        """
        Fetches and prepares the data for analysis.

        Returns:
            pd.DataFrame: Prepared data
        """
        df = self.bybit.fetch_klines(self.symbol, self.interval, limit=200)
        if not df.empty:
            df = df.sort_values("start_time")
            df = df.reset_index(drop=True)
            return df
        return pd.DataFrame()

    def _add_technical_indicators(self):
        """
        Adds technical indicators to the dataframe and ensures data integrity.

        Raises:
            ValueError: If critical indicators contain invalid data
        """
        try:
            # Calculate indicators using ta
            self.df["SMA_9"] = EMAIndicator(self.df["close"], window=9).ema_indicator()
            self.df["MACD"] = MACD(self.df["close"]).macd()
            self.df["RSI"] = RSIIndicator(self.df["close"], window=RSI_WINDOW).rsi()
            self.df["Stoch_RSI"] = StochasticOscillator(
                self.df["high"],
                self.df["low"],
                self.df["close"],
                window=STOCH_WINDOW,
                smooth_window=STOCH_SMOOTH_WINDOW,
            ).stoch()
            self.df["HMA_fast"] = HullMovingAverage(
                self.df["close"], window=FAST_HMA_WINDOW
            ).hma_indicator()
            self.df["WMA_slow"] = EMAIndicator(
                self.df["close"], window=SLOW_WMA_WINDOW
            ).ema_indicator()
            self.df["ADX"] = ADXIndicator(
                self.df["high"], self.df["low"], self.df["close"]
            ).adx()
            self.df["momentum"] = self.df["close"].diff()
            self.df["momentum_wma_10"] = (
                self.df["momentum"].rolling(window=10, win_type="triang").mean()
            )
            self.df["volume_ema_10"] = EMAIndicator(
                self.df["volume"], window=10
            ).ema_indicator()
            self.df["volume_spike"] = self.df["volume"] > (
                self.df["volume_ema_10"] * VOLUME_SPIKE_THRESHOLD
            )
            self.df = self.calculate_supertrend(self.df)
            self.df["VWAP"] = VolumeWeightedAveragePrice(
                high=self.df["high"],
                low=self.df["low"],
                close=self.df["close"],
                volume=self.df["volume"],
            ).volume_weighted_average_price()
            self.df["ATR"] = AverageTrueRange(
                high=self.df["high"],
                low=self.df["low"],
                close=self.df["close"],
                window=14,
            ).average_true_range()

            # Check for NaN values in critical indicators
            critical_indicators = [
                "SMA_9",
                "MACD",
                "RSI",
                "Stoch_RSI",
                "HMA_fast",
                "WMA_slow",
                "ADX",
                "momentum",
                "supertrend",
            ]

            nan_columns = [
                col for col in critical_indicators if self.df[col].isna().any()
            ]

            if nan_columns:
                self.logger.warning(
                    f"NaN values found in indicators: {nan_columns}. "
                    "Applying forward fill strategy."
                )
                # Forward fill NaN values
                self.df[nan_columns] = self.df[nan_columns].fillna(method="ffill")

                # Backward fill any remaining NaNs at the beginning
                self.df[nan_columns] = self.df[nan_columns].fillna(method="bfill")

            # Verify data integrity after filling
            if self.df[critical_indicators].isna().any().any():
                raise ValueError(
                    "Critical indicators contain invalid data after attempted repair"
                )

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            raise

    def calculate_supertrend(self, df, atr_period=10, multiplier=3):
        """
        Calculates the Supertrend indicator.

        Args:
            df (pd.DataFrame): Dataframe with OHLC data
            atr_period (int): ATR calculation period
            multiplier (int): Multiplier for ATR

        Returns:
            pd.DataFrame: Dataframe with Supertrend values added
        """

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate ATR
        price_diffs = [high - low, high - close.shift(), close.shift() - low]
        true_range = pd.concat(price_diffs, axis=1)
        true_range = true_range.abs().max(axis=1)
        atr = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()

        # HL2
        hl2 = (high + low) / 2

        # Upperband and lowerband calculation
        df["upperband"] = hl2 + (multiplier * atr)
        df["lowerband"] = hl2 - (multiplier * atr)

        # Calculate Supertrend
        df["in_uptrend"] = True
        for i in range(1, len(df)):
            curr, prev = i, i - 1

            if close[curr] > df["upperband"][prev]:
                df["in_uptrend"][curr] = True
            elif close[curr] < df["lowerband"][prev]:
                df["in_uptrend"][curr] = False
            else:
                df["in_uptrend"][curr] = df["in_uptrend"][prev]

                if (
                    df["in_uptrend"][curr]
                    and df["lowerband"][curr] < df["lowerband"][prev]
                ):
                    df["lowerband"][curr] = df["lowerband"][prev]

                if (
                    not df["in_uptrend"][curr]
                    and df["upperband"][curr] > df["upperband"][prev]
                ):
                    df["upperband"][curr] = df["upperband"][prev]

        df["supertrend"] = np.where(
            (df["in_uptrend"] == True), df["lowerband"], df["upperband"]
        )
        return df

    def calculate_fibonacci_levels(self) -> Dict[str, float]:
        """Calculate Fibonacci retracement and extension levels."""
        high = self.df["high"].max()
        low = self.df["low"].min()
        close = self.df["close"].iloc[-1]  # Use the most recent closing price

        pivot = (high + low + close) / 3
        range_val = high - low

        levels = {
            "R3": high + range_val * 0.618,  # Fibonacci Extension
            "R2": high + range_val * 0.382,  # Fibonacci Extension
            "R1": pivot + range_val * 0.618,
            "Pivot": pivot,
            "S1": pivot - range_val * 0.618,
            "S2": low - range_val * 0.382,  # Fibonacci Extension
            "S3": low - range_val * 0.618,  # Fibonacci Extension
        }
        return levels

    def determine_trend(self) -> str:
        """Determines the current trend direction."""
        if (
            self.df["close"].iloc[-1] > self.df["SMA_9"].iloc[-1]
            and self.df["ADX"].iloc[-1] > ADX_THRESHOLD
        ):
            if self.df["HMA_fast"].iloc[-1] > self.df["WMA_slow"].iloc[-1]:
                return "Strong Uptrend"
            return "Uptrend"
        elif (
            self.df["close"].iloc[-1] < self.df["SMA_9"].iloc[-1]
            and self.df["ADX"].iloc[-1] > ADX_THRESHOLD
        ):
            if self.df["HMA_fast"].iloc[-1] < self.df["WMA_slow"].iloc[-1]:
                return "Strong Downtrend"
            return "Downtrend"
        else:
            return "Sideways"

    def get_signal(self) -> Dict[str, Any]:
        """
        Generates trading signals based on technical indicators and Fibonacci levels.
        """
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        # Initialize signal dictionary
        signal = {
            "timestamp": datetime.now(ST_LOUIS_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": self.symbol,
            "current_price": self.current_price,
            "position": "NONE",
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "strength": 0,  # Signal strength (0-100)
            "fibonacci_levels": self.fib_levels,
            "trend": None,
        }

        # Calculate trend strength
        trend_strength = 0
        if latest["ADX"] > ADX_THRESHOLD:
            trend_strength = min((latest["ADX"] - ADX_THRESHOLD) * 2, 100)

        # Long signal conditions
        long_conditions = [
            latest["RSI"] < RSI_OVERSOLD,
            latest["MACD"] > 0 and prev["MACD"] < 0,  # MACD crossover
            latest["supertrend"] < self.current_price,
            latest["close"] > latest["WMA_slow"],
            latest["volume_spike"],
        ]

        # Short signal conditions
        short_conditions = [
            latest["RSI"] > RSI_OVERBOUGHT,
            latest["MACD"] < 0 and prev["MACD"] > 0,  # MACD crossover
            latest["supertrend"] > self.current_price,
            latest["close"] < latest["WMA_slow"],
            latest["volume_spike"],
        ]

        trend = self.determine_trend()
        signal["trend"] = trend

        # Entry, stop-loss, and take-profit logic based on trend and indicators
        if trend == "Strong Uptrend":
            long_strength = sum(long_conditions) * 20
            if long_strength > 60:
                signal["position"] = "LONG"
                signal["strength"] = long_strength
                signal["entry"] = self.current_price
                signal["stop_loss"] = self.fib_levels["S1"]
                signal["take_profit"] = self.fib_levels["R1"]

        elif trend == "Strong Downtrend":
            short_strength = sum(short_conditions) * 20
            if short_strength > 60:
                signal["position"] = "SHORT"
                signal["strength"] = short_strength
                signal["entry"] = self.current_price
                signal["stop_loss"] = self.fib_levels["R1"]
                signal["take_profit"] = self.fib_levels["S1"]

        # Add technical indicators
        signal["indicators"] = {
            "RSI": latest["RSI"],
            "MACD": latest["MACD"],
            "ADX": latest["ADX"],
            "Supertrend": latest["supertrend"],
            "Volume_EMA": latest["volume_ema_10"],
        }

        # Format signal output
        self._format_signal_output(signal)
        return signal

    def _format_signal_output(self, signal: Dict[str, Any]) -> None:
        """Format and print signal output with colors."""
        position_color = {"LONG": Fore.GREEN, "SHORT": Fore.RED, "NONE": Fore.YELLOW}

        print("\n" + "=" * 50)
        print(f"{Fore.CYAN}TRADING SIGNAL - {signal['timestamp']}{Style.RESET_ALL}")
        print(f"Symbol: {Fore.BLUE}{signal['symbol']}{Style.RESET_ALL}")
        print(
            f"Current Price: {Fore.YELLOW}{signal['current_price']:.2f}{Style.RESET_ALL}"
        )
        print(
            f"Position: {position_color[signal['position']]}{signal['position']}{Style.RESET_ALL}"
        )
        print(f"Trend: {Fore.MAGENTA}{signal['trend']}{Style.RESET_ALL}")

        if signal["position"] != "NONE":
            print(f"\nEntry Point: {Fore.CYAN}{signal['entry']:.2f}{Style.RESET_ALL}")
            print(f"Stop Loss: {Fore.RED}{signal['stop_loss']:.2f}{Style.RESET_ALL}")
            print(
                f"Take Profit: {Fore.GREEN}{signal['take_profit']:.2f}{Style.RESET_ALL}"
            )
            print(
                f"Signal Strength: {Fore.YELLOW}{signal['strength']}%{Style.RESET_ALL}"
            )

        print("\nFibonacci Levels:")
        for level, value in signal["fibonacci_levels"].items():
            print(f"{Fore.MAGENTA}{level}: {value:.2f}{Style.RESET_ALL}")

        print("\nTechnical Indicators:")
        for indicator, value in signal["indicators"].items():
            print(f"{Fore.CYAN}{indicator}: {value:.2f}{Style.RESET_ALL}")
        print("=" * 50 + "\n")


def main():
    try:
        analyzer = TradingAnalyzer(symbol=SYMBOL, interval=INTERVAL, logger=logger)

        logger.info(f"Starting trading bot for {SYMBOL} on {INTERVAL} minute interval")

        while True:
            signal = analyzer.get_signal()

            if signal.get("position") != "NONE":
                logger.info(
                    f"Signal generated: {signal['position']} at {signal['entry']} "
                    f"(Stop: {signal['stop_loss']}, Target: {signal['take_profit']})"
                )

            # Wait for the next interval
            time.sleep(int(INTERVAL) * 60)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {e}")


if __name__ == "__main__":
    main()
