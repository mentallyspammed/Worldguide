import os
import logging
import requests
import pandas as pd
import numpy as np
from ta.trend import (
    SMAIndicator,
    MACD,
    EMAIndicator,
    ADXIndicator,
    AroonIndicator,
    WMAIndicator,
)
from ta.volatility import AverageTrueRange
from ta.momentum import rsi
from ta.volume import on_balance_volume, volume_price_trend, chaikin_money_flow
from datetime import datetime
from dotenv import load_dotenv
import hmac
import hashlib
import time
from typing import Dict, Tuple, List
from colorama import init, Fore, Style
from zoneinfo import ZoneInfo

# --- Constants ---
LOG_DIR = "botlogs"
ST_LOUIS_TZ = ZoneInfo("America/Chicago")
MAX_RETRIES = 3
RETRY_DELAY = 5
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W"]
CLUSTER_SENSITIVITY = 0.05
SUPPORT_RESISTANCE_WINDOW = 14
VOLUME_LOOKBACK = 5  # How many bars to look back for volume confirmation
FAST_MA_WINDOW = 12
SLOW_MA_WINDOW = 26
RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30


# --- Custom Exception ---
class APIError(Exception):
    """Custom exception for Bybit API errors."""

    def __init__(self, message, response=None, status_code=None, ret_code=None):
        self.message = message
        self.response = response
        self.status_code = status_code
        self.ret_code = ret_code
        super().__init__(self.message)


# --- Functions integrated from hma.py ---
def calculate_hma(series: pd.Series, window: int) -> pd.Series:
    half_window = int(window / 2)
    sqrt_window = int(np.sqrt(window))
    wma_half = calculate_wma(series, half_window)
    wma_full = calculate_wma(series, window)
    wma_subtracted = 2 * wma_half - wma_full
    return calculate_wma(wma_subtracted, sqrt_window)


# --- Functions integrated from wma.py ---
def calculate_wma(data: pd.Series, period: int) -> pd.Series:
    if len(data) < period:
        return pd.Series([np.nan] * len(data), index=data.index)
    weights = np.arange(1, period + 1)
    return data.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


# --- Configuration ---
class Config:
    """Handles loading of the configuration settings."""

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "API keys not set. Set BYBIT_API_KEY and BYBIT_API_SECRET in .env"
            )

    def __repr__(self):
        """Returns a string representation of the config, excluding API secret"""
        return (
            f"Config(api_key='{self.api_key[:6]}...'"
            f", base_url='{self.base_url}', log_level='{self.log_level}')"
        )


# --- Bybit API Client ---
class Bybit:
    """Handles all communication with the Bybit API."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = requests.Session()

    def _generate_signature(self, params: dict) -> str:
        """Generates the signature for Bybit API requests."""
        param_str = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
        return hmac.new(
            self.config.api_secret.encode(), param_str.encode(), hashlib.sha256
        ).hexdigest()

    def fetch_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Fetches kline data from Bybit API."""
        endpoint = "/v5/market/kline"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "category": "linear",
        }
        response = self._request("GET", endpoint, params)
        if response.get("retCode") == 0 and response.get("result"):
            klines = response["result"]["list"]
            df = pd.DataFrame(
                klines,
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
                pd.to_numeric(df["start_time"]), unit="ms"
            )  # Corrected
            df = df.astype(
                {
                    col: float
                    for col in ["open", "high", "low", "close", "volume", "turnover"]
                }
            )
            return df
        return pd.DataFrame()

    def fetch_current_price(self, symbol: str) -> float | None:
        """Fetches the latest price for a given symbol."""
        endpoint = "/v5/market/tickers"
        params = {"symbol": symbol, "category": "linear"}
        response = self._request("GET", endpoint, params)
        if response and response.get("retCode") == 0 and response.get("result"):
            try:
                return float(response["result"]["list"][0]["lastPrice"])
            except (KeyError, IndexError, ValueError):
                self.logger.error(
                    f"Could not extract last price from response: {response}"
                )
                return None
        return None

    def _request(self, method: str, endpoint: str, params: dict = None) -> dict:
        """Makes a request to the Bybit API and handles retries and errors."""
        retries = 0
        while retries < MAX_RETRIES:
            try:
                params = params or {}
                params["api_key"] = self.config.api_key
                params["timestamp"] = str(
                    int(datetime.now(ST_LOUIS_TZ).timestamp() * 1000)
                )
                params["sign"] = self._generate_signature(params)
                url = f"{self.config.base_url}{endpoint}"
                response = self.session.request(method, url, params=params)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

                try:
                    data = response.json()
                except requests.exceptions.JSONDecodeError:
                    self.logger.error(f"Failed to parse JSON response: {response.text}")
                    raise APIError(
                        "Invalid JSON response",
                        response=response,
                        status_code=response.status_code,
                    )

                if data.get("retCode") != 0:
                    self.logger.error(
                        f"Bybit API Error - Code: {data.get('retCode')}, Msg: {data.get('retMsg')}"
                    )
                    raise APIError(
                        f"Bybit API Error: {data.get('retMsg')}",
                        response=response,
                        status_code=response.status_code,
                        ret_code=data.get("retCode"),
                    )
                return data
            except requests.RequestException as e:
                self.logger.error(
                    f"Request Exception: {e}, Retrying {retries+1} of {MAX_RETRIES} after {RETRY_DELAY} seconds"
                )
                retries += 1
                time.sleep(RETRY_DELAY)
            except APIError as e:
                self.logger.error(
                    f"API Error: {e}. Retrying {retries+1} of {MAX_RETRIES} after {RETRY_DELAY} seconds"
                )
                retries += 1
                time.sleep(RETRY_DELAY)

        self.logger.error(
            f"Max retries exceeded for: {method} {endpoint} with params {params}"
        )
        raise APIError(f"Max retries exceeded for: {method} {endpoint}")

    def validate_interval(self, interval: str):
        """Validates given timeframe interval to be within accepted values."""
        if interval not in VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval: {interval}. Must be one of: {', '.join(VALID_INTERVALS)}"
            )


# --- Sorting Algorithms ---
def hybrid_merge_sort(arr, left, right):
    """Hybrid Merge Sort: Combines merge sort and insertion sort."""
    if right - left <= 10:  # Threshold for switching to Insertion Sort
        insertion_sort(arr, left, right)
    else:
        mid = (left + right) // 2
        hybrid_merge_sort(arr, left, mid)
        hybrid_merge_sort(arr, mid + 1, right)
        merge(arr, left, mid, right)


def insertion_sort(arr, left, right):
    """Insertion Sort for small subarrays."""
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def merge(arr, left, mid, right):
    """Merge function for Hybrid Merge Sort."""
    i, j, k = left, mid + 1, 0
    temp = [0] * (right - left + 1)

    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            j += 1
        k += 1

    while i <= mid:
        temp[k] = arr[i]
        i += 1
        k += 1

    while j <= right:
        temp[k] = arr[j]
        j += 1
        k += 1

    for i in range(len(temp)):
        arr[left + i] = temp[i]


# --- TradingAnalyzer ---
class TradingAnalyzer:
    """Handles trading analysis and decision logic."""

    def __init__(self, symbol: str, interval: str, logger: logging.Logger):
        self.logger = logger
        self.config = Config()
        self.bybit = Bybit(self.config, logger)
        self.symbol = self._validate_symbol(symbol)
        self.interval = self._validate_interval(interval)
        self.df = self.bybit.fetch_klines(self.symbol, self.interval, limit=200)
        self._add_technical_indicators(self.df)

    def _validate_symbol(self, symbol: str) -> str:
        """Validates the symbol, ensuring it is not empty and is in correct format."""
        if not symbol:
            raise ValueError("Symbol must be specified")
        if not symbol.isalpha() or len(symbol) < 4:
            raise ValueError(
                f"Invalid symbol: {symbol}. Symbol must contain letters and be longer than 3 character e.g. 'BTCUSDT'"
            )
        return symbol.upper()

    def _validate_interval(self, interval: str) -> str:
        """Validates the interval to ensure it is within acceptable intervals."""
        self.bybit.validate_interval(interval)
        return interval

    def _add_technical_indicators(self, df: pd.DataFrame):
        """Adds technical indicators to the DataFrame."""
        df["SMA_50"] = SMAIndicator(df["close"], window=50).sma_indicator()
        df["MACD"] = MACD(df["close"]).macd()
        df["RSI"] = rsi(df["close"])
        df["EMA_200"] = EMAIndicator(df["close"], window=200).ema_indicator()
        df["fast_ma"] = EMAIndicator(df["close"], window=FAST_MA_WINDOW).ema_indicator()
        df["slow_ma"] = EMAIndicator(df["close"], window=SLOW_MA_WINDOW).ema_indicator()
        df["ADX"] = ADXIndicator(df["high"], df["low"], df["close"]).adx()
        df["ATR"] = AverageTrueRange(
            df["high"], df["low"], df["close"]
        ).average_true_range()
        df["OBV"] = on_balance_volume(df["close"], df["volume"])
        df["VPT"] = volume_price_trend(df["close"], df["volume"])
        aroon = AroonIndicator(high=df["high"], low=df["low"])  # Corrected
        df["Aroon_Up"] = aroon.aroon_up()
        df["Aroon_Down"] = aroon.aroon_down()
        df["CMF"] = chaikin_money_flow(
            df["high"], df["low"], df["close"], df["volume"]
        )  # Corrected

    def identify_support_resistance(
        self,
        sensitivity: float = CLUSTER_SENSITIVITY,
        window: int = SUPPORT_RESISTANCE_WINDOW,
        volume_lookback: int = VOLUME_LOOKBACK,
    ) -> Tuple[List[float], List[float]]:
        """Identifies support and resistance levels with volume confirmation and sorts them."""
        data = self.df["close"].values
        volume = self.df["volume"].values
        maxima = []
        minima = []

        for i in range(window, len(data) - window):
            # Identify local maximum
            if data[i] > np.max(data[i - window : i]) and data[i] > np.max(
                data[i + 1 : i + window]
            ):
                peak_volume = np.mean(volume[i - volume_lookback : i + volume_lookback])
                if peak_volume > np.mean(volume):
                    maxima.append((data[i], peak_volume))
            # Identify local minimum
            elif data[i] < np.min(data[i - window : i]) and data[i] < np.min(
                data[i + 1 : i + window]
            ):
                trough_volume = np.mean(
                    volume[i - volume_lookback : i + volume_lookback]
                )
                if trough_volume > np.mean(volume):
                    minima.append((data[i], trough_volume))

        # --- Sort the levels using hybrid_merge_sort ---
        hybrid_merge_sort(maxima, 0, len(maxima) - 1)
        hybrid_merge_sort(minima, 0, len(minima) - 1)
        # -----------------------------------------------

        return [level for level, _ in maxima[:3]], [level for level, _ in minima[:3]]

    def determine_trend(self) -> str:
        """Determines the current trend based on moving average crossover."""
        if self.df.empty or len(self.df) < max(FAST_MA_WINDOW, SLOW_MA_WINDOW):
            return "neutral"

        current_fast_ma = self.df["fast_ma"].iloc[-1]
        current_slow_ma = self.df["slow_ma"].iloc[-1]
        previous_fast_ma = self.df["fast_ma"].iloc[-2]
        previous_slow_ma = self.df["slow_ma"].iloc[-2]

        # Using the most recent values
        if current_fast_ma > current_slow_ma and previous_fast_ma <= previous_slow_ma:
            return "bullish"
        elif current_fast_ma < current_slow_ma and previous_fast_ma >= previous_slow_ma:
            return "bearish"
        else:
            return "neutral"

    def determine_entry_signal(self, current_price: float) -> str | None:
        """Determines an entry signal based on trend, RSI, and support/resistance."""

        trend = self.determine_trend()
        if self.df.empty or self.df["RSI"].isnull().any() or not current_price:
            return None
        current_rsi = self.df["RSI"].iloc[-1]
        supports, resistances = self.identify_support_resistance()

        if (
            trend == "bullish"
            and current_rsi < RSI_OVERSOLD
            and any(
                abs(current_price - support) < 0.02 * current_price
                for support in supports
            )
        ):
            return "long"
        elif (
            trend == "bearish"
            and current_rsi > RSI_OVERBOUGHT
            and any(
                abs(current_price - resistance) < 0.02 * current_price
                for resistance in resistances
            )
        ):
            return "short"
        else:
            return "neutral"

    def analyze(self, current_price: float):
        """Analyzes the market using technical indicators and generates trading signals."""
        supports, resistances = self.identify_support_resistance()
        trend = self.determine_trend()
        entry_signal = self.determine_entry_signal(current_price)
        take_profit = round(
            current_price
            * (
                1.02
                if entry_signal == "long"
                else 0.98
                if entry_signal == "short"
                else 1.00
            ),
            2,
        )

        self.logger.info(f"{Fore.YELLOW}Current Price:{Fore.GREEN} {current_price:.2f}")
        self.logger.info(
            f"{Fore.YELLOW}Trend Direction:{Fore.CYAN} {trend.capitalize()}"
        )
        self.logger.info(f"{Fore.YELLOW}Support Levels:{Fore.BLUE} {supports}")
        self.logger.info(f"{Fore.YELLOW}Resistance Levels:{Fore.RED} {resistances}")
        self.logger.info(
            f"{Fore.YELLOW}Predicted Take Profit Level:{Fore.MAGENTA} {take_profit:.2f}"
        )

        if entry_signal and entry_signal != "neutral":
            self.logger.info(
                f"{Fore.GREEN}Entry Signal:{Fore.MAGENTA} {entry_signal.capitalize()}{Style.RESET_ALL}"
            )
        else:
            self.logger.info(
                f"{Fore.YELLOW}--- No Entry Signal at this time --- {Style.RESET_ALL}"
            )


# --- Logging Setup ---
def setup_logger(config: Config, name: str) -> logging.Logger:
    """Sets up a logger to write to file and stream to console"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"{name}_{timestamp}.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)

    try:
        logger.setLevel(config.log_level)
    except ValueError:
        logger.setLevel("INFO")  # Default
        logger.error(
            f"Invalid LOG_LEVEL '{config.log_level}' specified. Defaulting to 'INFO'"
        )

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            f"{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(stream_handler)
    return logger


# --- Main ---
def main():
    init(autoreset=True)
    config = Config()
    logger = setup_logger(config, "trading_bot")
    logger.info(f"{Fore.MAGENTA}Starting bot with {config}{Style.RESET_ALL}")
    while True:
        try:
            symbol = input(f"{Fore.CYAN}Enter trading symbol (e.g., BTCUSDT): ").upper()
            interval = input(
                f"{Fore.CYAN}Enter timeframe ({', '.join(VALID_INTERVALS)}): "
            )
            analyzer = TradingAnalyzer(symbol, interval, logger)
            break
        except ValueError as e:
            logger.error(f"Initialization error: {e}")
            continue

    while True:
        try:
            current_price = analyzer.bybit.fetch_current_price(analyzer.symbol)
            if current_price:
                analyzer.analyze(current_price)
            time.sleep(60)
        except APIError as e:
            logger.error(f"API Error during runtime: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error during runtime: {e}")
            continue
        except KeyboardInterrupt:
            logger.info("Bot terminated manually.")
            break


if __name__ == "__main__":
    main()
