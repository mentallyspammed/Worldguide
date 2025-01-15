from dotenv import load_dotenv
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from ta.trend import EMAIndicator, MACD, ADXIndicator, AroonIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import rsi
from ta.volume import on_balance_volume, volume_price_trend
from colorama import init, Fore, Style
import time
import pytz
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError
from retry import retry

# Initialize colorama for neon-colored outputs
init(autoreset=True)

# --- Configuration ---

class TradingAnalyzer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def analyze(self, current_price):
         self.logger.info("Analysis function was called")
         return {} # just returning a dummy result
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
        self.ma_periods_short = 9
        self.ma_periods_long = 25
        self.fma_period = 20  # Fibonacci Moving Average period

        # RSI
        self.rsi_period = 14
        self.rsi_overbought = 66
        self.rsi_oversold = 33

        # Timezone Configuration (St. Louis, Missouri)
        self.timezone = pytz.timezone("America/Chicago")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "API keys not set. Ensure BYBIT_API_KEY and BYBIT_API_SECRET are in .env")

        # Base URL for Unified Trading Account (UTA)
        self.base_url = (
            "https://api-testnet.bybit.com"
            if self.testnet
            else "https://api.bybit.com"
        )  # For testnet

        # --- Constants ---
        self.LOG_DIR = "logs"
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
        self.RETRY_DELAY = int(os.getenv("RETRY_DELAY", 5))
        self.VALID_INTERVALS = ["1", "3", "5", "15",
                                "30", "60", "120", "240", "D", "W"]
        self.RETRY_ERROR_CODES = [429, 500, 502, 503, 504]
        self.SUPPORT_RESISTANCE_WINDOW = 20
        self.CLUSTER_SENSITIVITY = 0.05
        self.HIGHER_TIMEFRAMES = ["60", "240", "D"]


def setup_logger(name: str, level: str, symbol: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f" {symbol}_ {datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
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

    @retry(exceptions=(InvalidRequestError), tries=3, delay=2, backoff=2)
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
            raise
        except Exception as e:
            self.logger.error(f"Error fetching current price: {e}")
            return None

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

            # Check for API error messages
            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return pd.DataFrame()

            if "result" not in response or "list" not in response["result"]:
                self.logger.warning(
                    f"No kline data found for symbol {symbol}.")
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
            df["start_time"] = pd.to_datetime(
                df["start_time"], unit="ms", utc=True)

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

    def fetch_orderbook(self, limit: int = 10) -> dict:
        """Fetch order book data from Bybit Unified Trading API."""
        try:
            response = self.session.get_orderbook(
                category="linear",
                symbol=self.config.symbol,
                limit=limit,
            )

            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return {}

            return response.get("result", {})

        except Exception as e:
            self.logger.error(f"Error fetching order book: {e}")
            return {}

# --- Trading Analysis ---


class TradingAnalyzer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.api = BybitAPI(config, logger)
        self.levels = {}
        self.df = self.api.fetch_klines(
            config.symbol, config.interval, limit=200
        )


def wma(series: pd.Series, window: int) -> pd.Series:
    """Calculates the Weighted Moving Average (WMA)."""
    weights = np.arange(1, window + 1)
    return series.rolling(window=window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)


def hma(series: pd.Series, window: int) -> pd.Series:
    """Calculates the Hull Moving Average (HMA).

    Args:
        series: Pandas Series of prices.
        window: The time period for the HMA.

    Returns:
        Pandas Series representing the HMA.
    """
    m = np.int64(window)
    half_length = np.int64(m / 2)
    sqrt_length = np.int64(np.sqrt(m))

    wma1 = wma(series, half_length)
    wma2 = wma(series.shift(half_length), half_length)
    wma_diff = 2 * wma1 - wma2
    return wma(wma_diff, sqrt_length)

    def calculate_indicators(self) -> pd.DataFrame:
        """Calculates moving averages, RSI, and volume metrics."""
        if self.df.empty:
            return self.df

        # Moving Averages
        self.df["EMA_Short"] = EMAIndicator(
            self.df["close"], window=self.config.ma_periods_short
        ).ema_indicator()
        self.df["EMA_Long"] = EMAIndicator(
            self.df["close"], window=self.config.ma_periods_long
        ).ema_indicator()
        self.df["FMA"] = EMAIndicator(
            self.df["close"], window=self.config.fma_period
        ).ema_indicator()
        self.df["WMA"] = wma(
            self.df["close"], window=self.config.ma_periods_short
        )
        self.df["HMA"] = hma(
            self.df["close"], window=self.config.ma_periods_short
        )

        # RSI
        self.df["RSI"] = rsi(self.df["close"], window=self.config.rsi_period)

        # Volume Analysis
        self.df["Avg_Volume"] = self.df["volume"].rolling(window=20).mean()

        # MACD Calculation
        macd_data = self.calculate_macd()
        self.df["MACD"] = macd_data["macd"]
        self.df["Signal"] = macd_data["signal"]
        self.df["Histogram"] = macd_data["histogram"]

        # ADX Calculation
        adx_data = self.calculate_adx()
        self.df["ADX"] = adx_data["ADX"]
        self.df["+DI"] = adx_data["+DI"]
        self.df["-DI"] = adx_data["-DI"]

        # Aroon Calculation
        aroon_data = self.calculate_aroon()
        self.df["Aroon Up"] = aroon_data["Aroon Up"]
        self.df["Aroon Down"] = aroon_data["Aroon Down"]

        # Average True Range (ATR)
        self.df["ATR"] = self.calculate_atr()

        # Calculate OBV and VPT
        self.df["OBV"] = self.calculate_obv()
        self.df["VPT"] = self.calculate_vpt()

        return self.df

    def calculate_wma(self, window: int) -> pd.Series:
        """Calculates the Weighted Moving Average (WMA)."""
        return wma(self.df["close"], window=window)

    def calculate_hma(self, window: int) -> pd.Series:
        """Calculates the Hull Moving Average (HMA)."""
        return hma(self.df["close"], window=window)

    def calculate_momentum(self, period: int = 7) -> pd.Series:
        """Calculate momentum using Rate of Change (ROC)."""
        try:
            return (
                (self.df["close"] - self.df["close"].shift(period))
                / self.df["close"].shift(period)
            ) * 100
        except KeyError as e:
            self.logger.error(f"Error calculating momentum: {e}")
            return pd.Series(dtype="float64")
        except ZeroDivisionError:
            self.logger.error("Error calculating momentum: Division by zero")
            return pd.Series(dtype="float64")

    def calculate_macd(self) -> pd.DataFrame:
        """Calculate MACD and signal line"""
        try:
            close_prices = self.df["close"]
            ma_short = close_prices.ewm(span=12, adjust=False).mean()
            ma_long = close_prices.ewm(span=26, adjust=False).mean()
            macd = ma_short - ma_long
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            return pd.DataFrame({"macd": macd, "signal": signal, "histogram": histogram})
        except KeyError:
            self.logger.error("Missing the close column to calculate MACD.")
            return pd.DataFrame()

    def calculate_adx(self, window: int = 14) -> pd.DataFrame:
        """Calculates the Average Directional Index (ADX) with +DI and -DI."""
        adx_indicator = ADXIndicator(
            self.df["high"], self.df["low"], self.df["close"], window=window
        )
        return pd.DataFrame(
            {
                "ADX": adx_indicator.adx(),
                "+DI": adx_indicator.adx_pos(),
                "-DI": adx_indicator.adx_neg(),
            }
        )

    def calculate_aroon(self, window: int = 25) -> pd.DataFrame:
        """Calculates the Aroon Indicator."""
        aroon_indicator = AroonIndicator(self.df["close"], window=window)
        return pd.DataFrame(
            {
                "Aroon Up": aroon_indicator.aroon_up(),
                "Aroon Down": aroon_indicator.aroon_down(),
            }
        )

    def calculate_atr(self, window: int = 14) -> pd.Series:
        """Calculates the Average True Range (ATR)."""
        try:
            high_low = self.df["high"] - self.df["low"]
            high_close = (self.df["high"] - self.df["close"].shift()).abs()
            low_close = (self.df["low"] - self.df["close"].shift()).abs()

            tr = pd.concat([high_low, high_close, low_close],
                           axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            return atr
        except KeyError as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return pd.Series(dtype="float64")

    def calculate_obv(self) -> pd.Series:
        """Calculates On-Balance Volume (OBV)."""
        return on_balance_volume(self.df["close"], self.df["volume"])

    def calculate_vpt(self) -> pd.Series:
        """Calculates Volume Price Trend (VPT)."""
        return volume_price_trend(self.df["close"], self.df["volume"])

    def detect_macd_divergence(self) -> str:
        """Detects MACD divergence for trend confirmation. Returns "bullish", "bearish", or None string"""
        if self.df.empty or len(self.df) < 30:
            return None

        macd_df = self.calculate_macd()
        if macd_df.empty:
            return None

        prices = self.df["close"]
        macd_histogram = macd_df["histogram"]

        # Bullish divergence
        if (
            prices.iloc[-2] > prices.iloc[-1]
            and macd_histogram.iloc[-2] < macd_histogram.iloc[-1]
        ):
            return "bullish"  # Higher low on price, lower low on histogram

        # Bearish divergence
        elif (
            prices.iloc[-2] < prices.iloc[-1]
            and macd_histogram.iloc[-2] > macd_histogram.iloc[-1]
        ):
            return "bearish"  # Lower high on price, higher high on histogram

        return None

    def adapt_parameters(self, atr: float) -> Dict:
        """Adapt the parameters based on recent ATR"""

        base_params = {
            "momentum_period": 10,
            "momentum_ma_short": 12,
            "momentum_ma_long": 26,
            "volume_ma_period": 20,
            "atr_period": 14,
            "user_defined_weights": {
                "momentum_cross": 0.30,
                "positive_momentum": 0.40,
                "significant_volume": 0.30,
                "divergence": 0.20,
            },
        }

        if atr > 25:  # High volitility
            base_params["momentum_period"] = 15
            base_params["momentum_ma_short"] = 15
            base_params["momentum_ma_long"] = 30
            base_params["user_defined_weights"] = {
                "momentum_cross": 0.15,
                "positive_momentum": 0.40,
                "significant_volume": 0.45,
                "divergence": 0.25,
            }
        elif atr > 10:
            base_params["user_defined_weights"] = {
                "momentum_cross": 0.25,
                "positive_momentum": 0.35,
                "significant_volume": 0.40,
                "divergence": 0.30,
            }
        return base_params

    def determine_trend(self, df: pd.DataFrame) -> str:
        """Determines the trend based on EMA and FMA crossovers."""
        if (
            df.empty
            or "EMA_Short" not in df
            or "EMA_Long" not in df
            or "FMA" not in df
        ):
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
        nearest_support = max(
            [s for s in supports if s < current_price], default=None
        )
        nearest_resistance = min(
            [r for r in resistances if r > current_price], default=None
        )

        self.levels.update(
            {
                "support": supports,
                "resistance": resistances,
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
            }
        )

        return {
            "support": supports,
            "resistance": resistances,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
        }

    def calculate_pivot_points(
        self, high: float, low: float, close: float
    ) -> Dict[str, float]:
        """Calculates pivot points."""
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

        pivot_levels = {
            "Pivot": pivot,
            "R1": r1,
            "S1": s1,
            "R2": r2,
            "S2": s2,
            "R3": r3,
            "S3": s3,
        }

        self.levels.update(pivot_levels)
        return pivot_levels

    def analyze_macd(self) -> str:
        """Provides a basic analysis of the MACD indicator."""
        macd_line = self.df["MACD"].iloc[-1]
        signal_line = self.df["Signal"].iloc[-1]
        histogram = self.df["Histogram"].iloc[-1]

        if macd_line > signal_line and histogram > 0:
            return "bullish"  # MACD above Signal and Histogram positive
        elif macd_line < signal_line and histogram < 0:
            return "bearish"  # MACD below Signal and Histogram negative
        elif macd_line > signal_line and histogram < 0:
            return "potential bearish reversal"  # MACD above Signal but Histogram negative
        elif macd_line < signal_line and histogram > 0:
            return "potential bullish reversal"  # MACD below Signal but Histogram positive
        else:
            return "neutral"

    def _find_extrema(self, data: np.ndarray, window: int) -> Optional[np.ndarray]:
        """Finds local extrema (peaks and valleys) in the data."""
        extrema = []
        for i in range(window, len(data) - window):
            is_peak = data[i] > max(
                data[i-window:i]) and data[i] > max(data[i+1:i+1+window])
            is_valley = data[i] < min(
                data[i-window:i]) and data[i] < min(data[i+1:i+1+window])

            if is_peak or is_valley:
                extrema.append(data[i])

        if extrema:
            return np.array(extrema)
        else:
            self.logger.warning("No extrema found.")
            return None

    def _cluster_levels(self, data: np.ndarray, sensitivity: float) -> set:
        """Clusters levels based on sensitivity using a simple clustering algorithm."""
        if not data.any():  # Check if the array is empty.
            return set()

        clusters = []
        cluster = [data[0]]

        for i in range(1, len(data)):
            # Adjust sensitivity for larger numbers
            if np.abs(data[i] - np.mean(cluster)) < sensitivity*data[i]:
                cluster.append(data[i])
            else:
                clusters.append(cluster)
                cluster = [data[i]]
        clusters.append(cluster)  # Append the last cluster

        # Calculate mean for the clusters
        levels = set(np.mean(cluster) for cluster in clusters if cluster)

        return levels

    def identify_support_resistance(
        self, window: int = 14, sensitivity: float = 0.05
    ) -> Dict[str, List[float]]:
        """Identifies support and resistance levels using clustering."""
        if self.df.empty:
            return {"support": [], "resistance": []}

        data = self.df["close"].values
        levels = []

        extrema = self._find_extrema(data, window)

        if extrema is not None and len(extrema) > 0:
            clustered_levels = self._cluster_levels(extrema, sensitivity)
            levels = sorted(list(clustered_levels))

        if not levels:
            # Handle the case with no levels
            return {"support": [], "resistance": []}

        current_price = data[-1]
        support = [level for level in levels if level < current_price]
        resistance = [level for level in levels if level > current_price]

        self.levels.update({"support_clusters": support,
                           "resistance_clusters": resistance})
        return {"support": support, "resistance": resistance}

    def find_nearest_levels(
        self, current_price: float
    ) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
        """Finds the nearest support and resistance levels."""
        supports = []
        resistances = []

        # Check for cluster-based levels
        if "support_clusters" in self.levels:
            supports.extend([("Support (Cluster)", level, 0)
                            for level in self.levels["support_clusters"] if level < current_price])
        if "resistance_clusters" in self.levels:
            resistances.extend([("Resistance (Cluster)", level, 0)
                               for level in self.levels["resistance_clusters"] if level > current_price])

        # Check for Fibonacci levels
        if "nearest_support" in self.levels and self.levels["nearest_support"] is not None:
            supports.append(
                ("Support (Fibonacci)", self.levels["nearest_support"], 0))
        if "nearest_resistance" in self.levels and self.levels["nearest_resistance"] is not None:
            resistances.append(
                ("Resistance (Fibonacci)", self.levels["nearest_resistance"], 0))

        # Sort by distance
        nearest_supports = sorted(
            supports, key=lambda x: current_price - x[1])[:3]
        nearest_resistances = sorted(
            resistances, key=lambda x: x[1] - current_price)[:3]

        return nearest_supports, nearest_resistances

    def analyze_higher_timeframes(
        self, higher_timeframes: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Analyzes support/resistance on higher timeframes."""
        higher_tf_levels = {}
        for tf in higher_timeframes:
            df_higher = self.api.fetch_klines(
                self.config.symbol, tf, limit=200
            )
            if not df_higher.empty:
                analyzer_higher = TradingAnalyzer(self.config, self.logger)
                analyzer_higher.df = df_higher
                analyzer_higher.identify_support_resistance()
                higher_tf_levels[tf] = analyzer_higher.levels
            else:
                self.logger.warning(f"Could not fetch data for timeframe {tf}")

        return higher_tf_levels

    def determine_trend_momentum_adx(self) -> str:
        """Determines the trend direction based on momentum, MA, and ADX."""

        # Calculate momentum and moving averages
        momentum = self.calculate_momentum()
        sma_short = EMAIndicator(self.df["close"], window=5).ema_indicator()
        sma_long = EMAIndicator(self.df["close"], window=20).ema_indicator()

        # Ensure ADX data is available
        if "ADX" not in self.df.columns:
            adx_data = self.calculate_adx()
            self.df["ADX"] = adx_data["ADX"]
            self.df["+DI"] = adx_data["+DI"]
            self.df["-DI"] = adx_data["-DI"]

        # Get the most recent values
        adx = self.df["ADX"].iloc[-1]
        plus_di = self.df["+DI"].iloc[-1]
        minus_di = self.df["-DI"].iloc[-1]

        if adx > 25:  # Strong trend
            if (
                momentum.iloc[-1] > 0
                and sma_short.iloc[-1] > sma_long.iloc[-1]
                and plus_di > minus_di
            ):
                return "upward"
            elif (
                momentum.iloc[-1] < 0
                and sma_short.iloc[-1] < sma_long.iloc[-1]
                and minus_di > plus_di
            ):
                return "downward"
            else:
                return (
                    "neutral"
                )  # Strong trend but direction unclear based on other indicators
        else:
            return "neutral"  # Weak or no trend

    def suggest_trades(
        self,
        current_price: float,
        trend: str,
        nearest_supports: List[Tuple[str, float, float]],
        nearest_resistances: List[Tuple[str, float, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Suggests entry and target prices for long and short trades based on trend and S/R levels."""
        suggestions = {"long": {}, "short": {}}

        if trend == "upward" and nearest_supports:
            # Long entry near closest support with target at closest resistance
            entry_support = nearest_supports[0]
            target_resistance = (
                nearest_resistances[0] if nearest_resistances else None
            )

            suggestions["long"]["entry"] = entry_support[1]
            suggestions["long"]["target"] = (
                target_resistance[1]
                if target_resistance
                else current_price * 1.05
            )  # Default 5% above current price

        elif trend == "downward" and nearest_resistances:
            # Short entry near closest resistance with target at closest support
            entry_resistance = nearest_resistances[0]
            target_support = nearest_supports[0] if nearest_supports else None

            suggestions["short"]["entry"] = entry_resistance[1]
            suggestions["short"]["target"] = (
                target_support[1]
                if target_support
                else current_price * 0.95
            )  # Default 5% below current price

        return suggestions

    def _find_levels(self):
        # """Finds support and resistance levels using fractals."""
        levels = []
        for i in range(2, len(self.df) - 2):
            if self.is_support(self.df, i):
                l = self.df["low"][i]
                if isinstance(l, pd.Series):
                    l = l.iloc[0]
                if not any(level == l for level, _ in levels):  # Fixed this line
                    levels.append((l, "support"))

            elif self.is_resistance(self.df, i):
                l = self.df["high"][i]
                if isinstance(l, pd.Series):
                    l = l.iloc[0]
                if not any(level == l for level, _ in levels):  # Fixed this line
                    levels.append((l, "resistance"))
        self.levels = levels

    def analyze_rsi(self, rsi_value: float) -> str:
        """Provides a textual analysis of the RSI value."""
        if rsi_value > self.config.rsi_overbought:
            return "Overbought"
        elif rsi_value < self.config.rsi_oversold:
            return "Oversold"
        else:
            return "Neutral"

    def analyze(self, current_price: float):
        """Analyzes the market data and provides insights."""
        if self.df.empty:
            self.logger.error("DataFrame is empty. Cannot perform analysis.")
            return

        # Calculate various indicators
        self.df = self.calculate_indicators()

        # Trend Analysis
        trend = self.determine_trend_momentum_adx()

        # Momentum Analysis
        momentum = self.analyze_momentum(self.df)

        # Volume Analysis
        volume = self.analyze_volume(self.df)

        # MACD Analysis
        macd_analysis = self.analyze_macd()

        # RSI Analysis
        rsi_value = self.df["RSI"].iloc[-1]
        rsi_analysis = self.analyze_rsi(rsi_value)

        # Get high, low, close for Fibonacci and Pivot calculations
        high = self.df["high"].max()
        low = self.df["low"].min()
        close = self.df["close"].iloc[-1]

        # Fibonacci Support/Resistance
        fib_levels = self.calculate_fibonacci_support_resistance(self.df)

        # Pivot Point Levels
        self.calculate_pivot_points(high, low, close)

        # Dynamic S/R Levels
        self.identify_support_resistance()

        # Find nearest levels
        nearest_supports, nearest_resistances = self.find_nearest_levels(
            current_price
        )

        # Analyze higher timeframes
        higher_tf_levels = self.analyze_higher_timeframes(
            self.config.HIGHER_TIMEFRAMES
        )

        # Trade suggestions
        trade_suggestions = self.suggest_trades(
            current_price, trend, nearest_supports, nearest_resistances
        )

        # Format Analysis Results (including the localized timestamp and correct price)
        self.logger.info(
            f" {Fore.YELLOW}--- Analysis Results for {self.config.symbol} ({self.config.interval}) ---"
        )
        self.logger.info(
            f" {Fore.YELLOW}Symbol: {Fore.MAGENTA} {self.config.symbol}"
        )
        self.logger.info(
            f" {Fore.YELLOW}Timeframe: {Fore.MAGENTA} {self.config.interval}"
        )
        self.logger.info(
            f" {Fore.YELLOW}Last Close Time ({self.config.timezone}): {Fore.MAGENTA} {self.df['start_time'].iloc[-1].tz_convert(self.config.timezone).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.logger.info(
            f" {Fore.YELLOW}Current Price: {Fore.GREEN} {current_price}")
        self.logger.info(
            f" {Fore.YELLOW}Trend: {Fore.GREEN if trend == 'upward' else Fore.RED if trend == 'downward' else Fore.YELLOW} {trend}"
        )
        self.logger.info(
            f" {Fore.YELLOW}Momentum: {Fore.RED if momentum == 'overbought' else Fore.GREEN if momentum == 'oversold' else Fore.YELLOW} {momentum}"
        )
        self.logger.info(
            f" {Fore.YELLOW}Volume: {Fore.GREEN if volume == 'high' else Fore.RED} {volume}"
        )
        self.logger.info(
            f" {Fore.YELLOW}RSI: {Fore.MAGENTA} {rsi_value:.2f} ({rsi_analysis})"
        )
        self.logger.info(
            f" {Fore.YELLOW}MACD Analysis: {Fore.MAGENTA} {macd_analysis}")

        self.logger.info(f" {Fore.YELLOW}--- Fibonacci Levels ---")
        for name, level in fib_levels.items():
            if isinstance(level, list):
                self.logger.info(
                    f" {Fore.CYAN}{name}: {Fore.WHITE}{', '.join(map(str, level))}")
            else:
                self.logger.info(f" {Fore.CYAN}{name}: {Fore.WHITE}{level}")

        self.logger.info(f" {Fore.YELLOW}--- Pivot Points ---")
        for name, level in self.levels.items():
            if name in ["Pivot", "R1", "S1", "R2", "S2", "R3", "S3"]:
                self.logger.info(f" {Fore.CYAN}{name}: {Fore.WHITE}{level}")

        self.logger.info(
            f" {Fore.YELLOW}--- Dynamic Support/Resistance Clusters ---")
        if "support_clusters" in self.levels:
            self.logger.info(
                f" {Fore.CYAN}Support Clusters: {Fore.WHITE}{', '.join(map(str, self.levels['support_clusters']))}")
        if "resistance_clusters" in self.levels:
            self.logger.info(
                f" {Fore.CYAN}Resistance Clusters: {Fore.WHITE}{', '.join(map(str, self.levels['resistance_clusters']))}")

        self.logger.info(f" {Fore.YELLOW}--- Nearest Support/Resistance ---")
        self.logger.info(
            f" {Fore.CYAN}Nearest Supports: {Fore.WHITE}{', '.join([f'{label} at {value:.2f}' for label, value, vol in nearest_supports])}")
        self.logger.info(
            f" {Fore.CYAN}Nearest Resistances: {Fore.WHITE}{', '.join([f'{label} at {value:.2f}' for label, value, vol in nearest_resistances])}")

        self.logger.info(f" {Fore.YELLOW}--- Higher Timeframe Analysis ---")
        for tf, levels in higher_tf_levels.items():
            self.logger.info(f" {Fore.BLUE}Timeframe: {tf}")
            for level_type, level_values in levels.items():
                if level_type in ["support_clusters", "resistance_clusters"]:
                    self.logger.info(
                        f"  {Fore.CYAN}{level_type}: {Fore.WHITE}{', '.join(map(str, level_values))}")
                elif level_type in ["nearest_support", "nearest_resistance"] and level_values is not None:
                    self.logger.info(
                        f"  {Fore.CYAN}{level_type}: {Fore.WHITE}{level_values}")

        self.logger.info(f" {Fore.YELLOW}--- Trade Suggestions ---")
        if trade_suggestions["long"]:
            self.logger.info(
                f" {Fore.GREEN}Long Entry: {trade_suggestions['long'].get('entry', 'N/A')}")
            self.logger.info(
                f" {Fore.GREEN}Long Target: {trade_suggestions['long'].get('target', 'N/A')}")
        else:
            self.logger.info(f" {Fore.YELLOW}No Long Trade Suggestions")

        if trade_suggestions["short"]:
            self.logger.info(
                f" {Fore.RED}Short Entry: {trade_suggestions['short'].get('entry', 'N/A')}")
            self.logger.info(
                f" {Fore.RED}Short Target: {trade_suggestions['short'].get('target', 'N/A')}")
        else:
            self.logger.info(f" {Fore.YELLOW}No Short Trade Suggestions")
        self.logger.info(f" {Fore.YELLOW}--- End Analysis ---")

# --- Main Execution ---


def main():
    symbol = "BTCUSDT"
    interval = "15"
    config = Config(symbol, interval)
    logger = setup_logger("trading_bot", config.log_level, symbol)
    analyzer = TradingAnalyzer(config, logger)

    while True:
        try:
            current_price = analyzer.api.fetch_current_price()
            if current_price is not None:
                analyzer.analyze(current_price)
            else:
                logger.error("Failed to fetch current price. Retrying...")
        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}")
        time.sleep(60)  # Adjust as needed


if __name__ == "__main__":
    main()
