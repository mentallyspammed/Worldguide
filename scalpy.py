import os
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from ta.pattern import candlestick_patterns
from dotenv import load_dotenv
from colorama import init, Fore, Style
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError
from retry import retry

from fib import fib

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
VOLUME_SPIKE_THRESHOLD = 1.5  # Define a threshold for volume spikes

# Load environment variables
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = bool(os.getenv("BYBIT_TESTNET", False))
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "5")

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TradingAnalyzer")

class BybitAPI:
    """A class to interact with the Bybit API using pybit."""

    def __init__(self, logger: logging.Logger):
        """Initializes the Bybit API client with configuration and logger."""
        self.logger = logger
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.testnet = TESTNET

        self.session = HTTP(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet
        )

    @retry(exceptions=(InvalidRequestError), tries=3, delay=2, backoff=2)
    def fetch_current_price(self) -> Optional[float]:
        """Fetches the current real-time price of the symbol using pybit."""
        try:
            response = self.session.get_tickers(
                category="linear", symbol=SYMBOL
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
            raise
        except Exception as e:
            self.logger.error(f"Error fetching current price: {e}")
            return None

    def fetch_orderbook(self, limit: int = 10) -> dict:
        """Fetch order book data from Bybit Unified Trading API."""
        try:
            response = self.session.get_orderbook(
                category="linear",
                symbol=SYMBOL,
                limit=limit,
            )

            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return {}

            return response.get("result", {})

        except Exception as e:
            self.logger.error(f"Error fetching order book: {e}")
            return {}

    def fetch_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """
        Fetches kline data from Bybit API for Unified Trading Account (UTA)
        (specifically for USDT Perpetual contracts) using pybit, ensures it's sorted by time,
        and handles timezone conversion.
        """
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit,
            )

            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return pd.DataFrame()

            if "result" not in response or "list" not in response["result"]:
                self.logger.warning(f"No kline data found for symbol {symbol}.")
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

            df["start_time"] = pd.to_datetime(pd.to_numeric(df["start_time"]), unit="ms", utc=True)
            df.sort_values("start_time", inplace=True)
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

def hull_moving_average(series, window):
    """Calculate the Hull Moving Average (HMA)"""
    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))
    
    wma_half_length = series.rolling(window=half_length).mean()
    wma_full_length = series.rolling(window=window).mean()
    
    raw_hma = (2 * wma_half_length) - wma_full_length
    hma = raw_hma.rolling(window=sqrt_length).mean()
    
    return hma

class TradingAnalyzer:
    """A class for performing trading analysis."""

    def __init__(self, symbol: str, interval: str, logger: logging.Logger):
        """Initializes the TradingAnalyzer with symbol, interval, and logger."""
        self.logger = logger
        self.bybit = BybitAPI(logger)
        self.symbol = symbol.upper()
        self.interval = interval
        self.df = self.fetch_and_prepare_data()
        if self.df.empty:
            raise ValueError("Failed to fetch data. Check your API key, symbol, and interval.")
        self._add_technical_indicators()

    def fetch_and_prepare_data(self):
        """Fetches and prepares the data for analysis."""
        df = self.bybit.fetch_klines(self.symbol, self.interval, limit=200)
        if not df.empty:
            df = df.sort_values("start_time")
            df = df.reset_index(drop=True)
            return df
        return pd.DataFrame()

    def _add_technical_indicators(self):
        """Adds technical indicators to the dataframe."""
        self.df["SMA_9"] = EMAIndicator(self.df["close"], window=9).ema_indicator()
        self.df["MACD"] = MACD(self.df["close"]).macd()
        self.df["RSI"] = RSIIndicator(self.df["close"], window=RSI_WINDOW).rsi()
        self.df["Stoch_RSI"] = StochasticOscillator(self.df["close"], self.df["low"], self.df["close"], window=STOCH_WINDOW, smooth_window=STOCH_SMOOTH_WINDOW).stoch()
        self.df["HMA_fast"] = hull_moving_average(self.df["close"], window=FAST_HMA_WINDOW)
        self.df["WMA_slow"] = EMAIndicator(self.df["close"], window=SLOW_WMA_WINDOW).ema_indicator()
        self.df["ADX"] = ADXIndicator(self.df["high"], self.df["low"], self.df["close"]).adx()
        self.df["momentum"] = self.df["close"].diff()
        self.df["momentum_wma_10"] = self.df["momentum"].rolling(window=10, win_type="triang").mean()
        self.df["volume_ma_10"] = EMAIndicator(self.df["volume"], window=10).ema_indicator()
        self.df["volume_spike"] = self.df["volume"] > (self.df["volume_ma_10"] * VOLUME_SPIKE_THRESHOLD)
        self.df = self.calculate_supertrend(self.df)
        self.df["VWAP"] = VolumeWeightedAveragePrice(high=self.df["high"], low=self.df["low"], close=self.df["close"], volume=self.df["volume"]).volume_weighted_average_price()
        self.df["ATR"] = AverageTrueRange(high=self.df["high"], low=self.df["low"], close=self.df["close"], window=14).average_true_range()

    def calculate_supertrend(self, df, atr_period=10, multiplier=3):
        """Calculates the Supertrend indicator."""
        hl2 = (df['high'] + df['low']) / 2
        atr = df['high'].rolling(window=atr_period).max() - df['low'].rolling(window=atr_period).min()
        atr = atr.rolling(window=atr_period).mean()
        df['upperband'] = hl2 + (multiplier * atr)
        df['lowerband'] = hl2 - (multiplier * atr)
        df['supertrend'] = np.nan
        df.loc[0, 'supertrend'] = df.loc[0, 'upperband']

        for i in range(1, len(df)):
            if df['close'][i] > df.loc[i-1, 'upperband']:
                df.loc[i, 'supertrend'] = df.loc[i, 'upperband']
            elif df['close'][i] < df.loc[i-1, 'lowerband']:
                df.loc[i, 'supertrend'] = df.loc[i, 'lowerband']
            else:
                df.loc[i, 'supertrend'] = df.loc[i-1, 'supertrend']

        return df

    def identify_support_resistance(self) -> Tuple[List[float], List[float]]:
        """Identifies support and resistance levels."""
        data = self.df["close"].values
        maxima, minima = [], []
        for i in range(SUPPORT_RESISTANCE_WINDOW, len(data) - SUPPORT_RESISTANCE_WINDOW):
            if data[i] == max(data[i - SUPPORT_RESISTANCE_WINDOW:i + SUPPORT_RESISTANCE_WINDOW]):
                maxima.append(data[i])
            if data[i] == min(data[i - SUPPORT_RESISTANCE_WINDOW:i + SUPPORT_RESISTANCE_WINDOW]):
                minima.append(data[i])
        return sorted(maxima, reverse=True), sorted(minima)

    def determine_trend(self) -> str:
        """Determines the current trend."""
        fast_hma = self.df["HMA_fast"].iloc[-1]
        slow_wma = self.df["WMA_slow"].iloc[-1]
        macd_value = self.df["MACD"].iloc[-1]

        if fast_hma > slow_wma and macd_value > 0:
            return "bullish"
        elif fast_hma < slow_wma and macd_value < 0:
            return "bearish"
        else:
            return "neutral"

    def is_near_level(self, price, level, threshold=0.01):
        """Checks if the price is near a specific level."""
        return abs(price - level) / price <= threshold

    def calculate_fibonacci_pivots(self, high, low, close):
        """Calculates Fibonacci pivot points."""
        pivot_point = (high + low + close) / 3
        s1 = pivot_point - 0.382 * (high - low)
        s2 = pivot_point - 0.618 * (high - low)
        s3 = pivot_point - 1.000 * (high - low)
        r1 = pivot_point + 0.382 * (high - low)
        r2 = pivot_point + 0.618 * (high - low)
        r3 = pivot_point + 1.000 * (high - low)
        return {
            "S3": s3,
            "S2": s2,
            "S1": s1,
            "R1": r1,
            "R2": r2,
            "R3": r3,
        }

    def determine_entry_signal(self, current_price: float, supports, resistances) -> Optional[Tuple[str, float, float]]:
        """Determines the entry signal based on current price and levels."""
        trend = self.determine_trend()
        if trend == "neutral":
            return None

        fast_hma = self.df["HMA_fast"].iloc[-1]
        slow_wma = self.df["WMA_slow"].iloc[-1]
        stoch_rsi = self.df["Stoch_RSI"].iloc[-1]
        volume_spike = self.df["volume_spike"].iloc[-1]
        supertrend = self.df["supertrend"].iloc[-1]
        atr = self.df["ATR"].iloc[-1]  # ATR value for volatility adjustments

        fib_pivots = self.calculate_fibonacci_pivots(
            self.df["high"].max(), self.df["low"].min(), self.df["close"].iloc[-1]
        )
        near_fib = any(self.is_near_level(current_price, level) for level in fib_pivots.values())
        near_support = any(self.is_near_level(current_price, s) for s in supports)
        near_resistance = any(self.is_near_level(current_price, r) for r in resistances)

        # Dynamic RSI thresholds based on volatility
        rsi_overbought = 80 if atr > 1 else 70
        rsi_oversold = 20 if atr > 1 else 30

        if trend == "bullish" and stoch_rsi < rsi_oversold and volume_spike:
            target = next((r for r in resistances if r > current_price), current_price * 1.02)
            return "long", current_price, target
        elif trend == "bearish" and stoch_rsi > rsi_overbought and volume_spike:
            target = next((s for s in supports if s < current_price), current_price * 0.98)
            return "short", current_price, target

        return None

    def extended_trend_indicator(self) -> str:
        """Provides an extended trend indicator analysis."""
        current_adx = self.df["ADX"].iloc[-1]
        fast_hma = self.df["HMA_fast"].iloc[-1]
        slow_wma = self.df["WMA_slow"].iloc[-1]
        macd_value = self.df["MACD"].iloc[-1]
        momentum = self.df["momentum"].iloc[-1]
        volume = self.df["volume"].iloc[-1]
        volume_ma_10 = self.df["volume_ma_10"].iloc[-1]
        stoch_rsi = self.df["Stoch_RSI"].iloc[-1]
        supertrend = self.df["supertrend"].iloc[-1]

        trend_strength = "weak" if current_adx < ADX_THRESHOLD else "strong"
        trend_direction = "neutral"

        if fast_hma > slow_wma and macd_value > 0:
            trend_direction = "bullish"
        elif fast_hma < slow_wma and macd_value < 0:
            trend_direction = "bearish"

        neon_yellow = Fore.LIGHTYELLOW_EX
        neon_cyan = Fore.LIGHTCYAN_EX

        self.logger.info(f"{neon_yellow}Trend Analysis:{Style.RESET_ALL}")
        self.logger.info(f"  ADX Value: {current_adx:.2f} ({trend_strength})")
        self.logger.info(f"  Fast HMA ({FAST_HMA_WINDOW}): {fast_hma:.2f}")
        self.logger.info(f"  Slow WMA ({SLOW_WMA_WINDOW}): {slow_wma:.2f}")
        self.logger.info(f"  MACD: {macd_value:.2f}")
        self.logger.info(f"  Momentum: {momentum:.2f}")
        self.logger.info(f"  Volume: {volume:.2f}")
        self.logger.info(f"  Volume MA (10): {volume_ma_10:.2f}")
        self.logger.info(f"  Stoch RSI: {stoch_rsi:.2f}")
        self.logger.info(f"  Supertrend: {supertrend:.2f}")
        self.logger.info(
            f"{neon_cyan}  Overall Trend: {trend_strength.capitalize()} {trend_direction.capitalize()}{Style.RESET_ALL}"
        )

        return f"{trend_strength} {trend_direction}"

    def find_hidden_bullish_divergence(self, lookback_period=20):
        """Finds hidden bullish divergence."""
        divergences = []
        macd_hist = self.df['MACD'].diff()  # Calculate MACD Histogram

        for i in range(lookback_period, len(self.df)):
            # 1. Lower lows in price:
            current_price_low = self.df['low'].iloc[i]
            previous_price_low = self.df['low'].iloc[i - lookback_period:i].min()
            if current_price_low < previous_price_low:
                # 2. Higher lows in MACD Histogram:
                current_macd_hist_low = macd_hist.iloc[i]
                previous_macd_hist_low = macd_hist.iloc[i - lookback_period:i].min()

                if current_macd_hist_low > previous_macd_hist_low:
                    # Potential Hidden Bullish Divergence
                    divergences.append(self.df.index[i])
        return divergences

    def confirm_with_candlestick(self, divergences):
        """Confirm hidden bullish divergence with a bullish candlestick pattern."""
        signals = []
        for date in divergences:
            if candlestick_patterns.bullish_engulfing(self.df.loc[date]):
                signals.append(date)
        return signals

    def place_trade(self, signal_dates):
        """Mock placing trades based on signals."""
        for date in signal_dates:
            self.logger.info(f"Placing trade on {date}")

    def analyze(self, current_price: float):
        """Performs analysis on the current price."""
        supports, resistances = self.identify_support_resistance()
        trend = self.determine_trend()
        entry_signal = self.determine_entry_signal(current_price, supports, resistances)

        high = self.df["high"].max()
        low = self.df["low"].min()
        fib_levels = fib(high, low)
        sorted_fib_levels = sorted(fib_levels.values())
        five_nearest_fib = sorted(sorted_fib_levels, key=lambda x: abs(x - current_price))[:5]

        fib_pivots = self.calculate_fibonacci_pivots(
            self.df["high"].max(), self.df["low"].min(), self.df["close"].iloc[-1]
        )

        self.extended_trend_indicator()
        neon_yellow = Fore.LIGHTYELLOW_EX
        neon_cyan = Fore.LIGHTCYAN_EX
        neon_green = Fore.LIGHTGREEN_EX
        neon_magenta = Fore.LIGHTMAGENTA_EX

        self.logger.info(f"{neon_yellow}Current Price: {current_price:.8f}{Style.RESET_ALL}")
        self.logger.info(f"{neon_yellow}Trend: {trend}{Style.RESET_ALL}")
        self.logger.info(f"{neon_yellow}5 Nearest Fibonacci Levels: {', '.join([f'{level:.8f}' for level in five_nearest_fib])}{Style.RESET_ALL}")

        self.logger.info(f"{neon_magenta}Fibonacci Pivots:{Style.RESET_ALL}")
        for level, value in fib_pivots.items():
            self.logger.info(f"{level}: {neon_cyan}{value:.8f}{Style.RESET_ALL}")

        if entry_signal:
            direction, entry_price, target_price = entry_signal
            self.logger.info(f"{neon_green}Trade Signal: {direction}{Style.RESET_ALL}")
            self.logger.info(f"Entry Price: {entry_price:.8f}")
            self.logger.info(f"Target Price: {target_price:.8f}")
        else:
            self.logger.info("No trade signal.")

        # Identify hidden bullish divergence
        divergences = self.find_hidden_bullish_divergence()
        signals = self.confirm_with_candlestick(divergences)

        # Place trades based on signals (mock)
        self.place_trade(signals)

def main():
    # Initialize Bybit API and Trading Analyzer
    trading_analyzer = TradingAnalyzer(SYMBOL, INTERVAL, logger)

    try:
        # Fetch current price
        current_price = trading_analyzer.bybit.fetch_current_price()
        if current_price is None:
            logger.error("Failed to fetch the current price.")
            return

        # Perform analysis
        trading_analyzer.analyze(current_price)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    init(autoreset=True)  # Initialize colorama for colored output
    main()