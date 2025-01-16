import os
import logging
import requests
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.volatility import AverageTrueRange
from datetime import datetime
from dotenv import load_dotenv
import hmac
import hashlib
import time
from typing import Dict, Tuple, List
from colorama import init, Fore, Style
from zoneinfo import ZoneInfo
from functools import wraps

# --- Initialize Colorama ---
init(autoreset=True)

# --- Constants ---
LOG_DIR = "botlogs"
ST_LOUIS_TZ = ZoneInfo("America/Chicago")
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W"]
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]
MAX_RETRIES = 3
RETRY_DELAY = 5
SUPPORT_RESISTANCE_WINDOW = 14
CLUSTER_SENSITIVITY = 0.05
HIGHER_TIMEFRAMES = ["60", "240", "D"]

# --- Retry Decorator ---


def retry_request(
    max_retries: int = MAX_RETRIES,
     retry_delay: int = RETRY_DELAY):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    response = func(*args, **kwargs)
                    if isinstance(response, dict) and response.get(
                        'retCode') in RETRY_ERROR_CODES:
                        raise BybitAPIException(
    response.get(
        "retMsg",
         "Bybit API Error requiring retry."))
                    return response
                except (requests.exceptions.RequestException, BybitAPIException) as e:
                    if retries < max_retries:
                        retries += 1
                        time.sleep(retry_delay)
                        print(
    Fore.YELLOW +
    f"Retrying after error {e}. Attempt {retries} of {max_retries}",
     Style.RESET_ALL)
                    else:
                        print(
    Fore.RED +
    f"Max retries ({max_retries}) exceeded. Request failed.",
     Style.RESET_ALL)
                        raise
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    raise
        return wrapper
    return decorator

# --- Custom Exception for Bybit API ---


class BybitAPIException(Exception):
    pass

# --- Configuration ---


class Config:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
        if not self.api_key or not self.api_secret:
            raise ValueError(
    f"{
        Fore.RED}API keys not set. Set BYBIT_API_KEY and BYBIT_API_SECRET in .env file or as environment variables.{
            Style.RESET_ALL}")
# --- Bybit API Client ---


class Bybit:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = requests.Session()

    def _generate_signature(self, params: dict) -> str:
        param_str = "&".join(
    f"{key}={value}" for key,
    value in sorted(
        params.items()))
        return hmac.new(
    self.config.api_secret.encode(),
    param_str.encode(),
     hashlib.sha256).hexdigest()

    @retry_request()
    def _request(
    self,
    method: str,
    endpoint: str,
     params: dict = None) -> dict:
        params = params or {}
        params["api_key"] = self.config.api_key
        params["timestamp"] = str(
            int(datetime.now(ST_LOUIS_TZ).timestamp() * 1000))
        params["sign"] = self._generate_signature(params)
        url = f"{self.config.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, params=params)
            response.raise_for_status()
            json_response = response.json()
            if not json_response:
                raise BybitAPIException("Invalid JSON response from Bybit.")
            if json_response.get("retCode") in RETRY_ERROR_CODES:
                raise BybitAPIException(
    json_response.get(
        "retMsg",
         "Bybit API Error requiring retry."))
            if json_response.get("retCode") != 0:
                error_msg = json_response.get("retMsg", "An error occurred")
                raise BybitAPIException(error_msg)
            return json_response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request Error: {e}")
            raise
        except BybitAPIException as e:
            self.logger.error(f"Bybit API Error: {e}")
            raise
        except json.JSONDecodeError as e:
            raise BybitAPIException(f"Invalid JSON response: {e}") from e
        except Exception as e:
            print(
    Fore.RED +
    f"An unexpected error occurred in _request: {
        type(e).__name__}: {e}",
         Style.RESET_ALL)
            raise

    def fetch_klines(self, symbol: str, interval: str,
                     limit: int = 200) -> pd.DataFrame:
        if interval not in VALID_INTERVALS:
            raise ValueError(
    f"Invalid interval. Choose from: {VALID_INTERVALS}")
        endpoint = "/v5/market/kline"
        params = {
    "symbol": symbol,
    "interval": interval,
    "limit": limit,
     "category": "linear"}
        try:
            response = self._request("GET", endpoint, params)
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
     "turnover"],
            )
            df["start_time"] = pd.to_datetime(
    df["start_time"], unit="ms").dt.tz_localize(ST_LOUIS_TZ)
            df = df.astype(
                {col: float for col in ["open", "high", "low", "close", "volume", "turnover"]})
            return df
        except BybitAPIException as e:
            self.logger.error(f"Failed to fetch klines: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.exception(
    f"An unexpected error occurred in fetch_klines: {e}")
            return pd.DataFrame()

    def fetch_current_price(self, symbol: str) -> float | None:
        endpoint = "/v5/market/tickers"
        params = {"symbol": symbol, "category": "linear"}
        try:
            response = self._request("GET", endpoint, params=params)
            if response and response["result"]:
                return float(response["result"]["list"][0]["lastPrice"])
            else:
                raise BybitAPIException(
                    "Invalid price data received from Bybit.")
        except (KeyError, IndexError, TypeError) as e:
            self.logger.error(
    f"Error parsing price data: {e}. Response: {response}")
        except BybitAPIException as e:
            self.logger.error(f"Bybit API error fetching price: {e}")
        except Exception as e:
            self.logger.exception(
    f"An unexpected error occurred in fetch_current_price: {e}")
        return None
# --- Technical Analysis Module (Enhanced) ---


class TradingAnalyzer:
    def __init__(self, symbol: str, interval: str, logger: logging.Logger):
        self.symbol = symbol
        self.interval = interval
        self.logger = logger
        self.config = Config()
        self.bybit = Bybit(self.config, logger)
        self.df = self.bybit.fetch_klines(symbol, interval, limit=200)
        self.levels = {}
        self.pivot_points = {}

    def calculate_sma(self, window: int) -> pd.Series:
        return SMAIndicator(self.df["close"], window=window).sma_indicator()

    def calculate_ema(self, window: int) -> pd.Series:
        return EMAIndicator(self.df["close"], window=window).ema_indicator()

    def calculate_momentum(self, window: int = 14) -> pd.Series:
        return self.df["close"].diff(window)

    def calculate_fibonacci_retracement(
        self, high: float, low: float) -> Dict[str, float]:
        diff = high - low
        if diff == 0:
            self.logger.warning(
                "Cannot calculate Fibonacci: high and low are the same.")
            return {}
        fib_levels = {
            "Fib 161.8%": high + diff * 1.618,
            "Fib 100.0%": high,
            "Fib 78.6%": high - diff * 0.786,
            "Fib 61.8%": high - diff * 0.618,
            "Fib 50.0%": high - diff * 0.5,
            "Fib 38.2%": high - diff * 0.382,
            "Fib 23.6%": high - diff * 0.236,
            "Fib 0.0%": low,
        }
        return fib_levels

    def calculate_pivot_points(
        self, high: float, low: float, close: float) -> Dict[str, float]:
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
        self.pivot_points.update(pivot_levels)
        return pivot_levels

    def calculate_atr(self, window: int = 14) -> pd.Series:
        return AverageTrueRange(
    self.df["high"],
    self.df["low"],
    self.df["close"],
     window=window).average_true_range()

    def calculate_macd(
    self,
    fast_period: int = 12,
    slow_period: int = 26,
     signal_period: int = 9) -> pd.DataFrame:
        macd_indicator = MACD(
    self.df["close"],
    window_slow=slow_period,
    window_fast=fast_period,
     window_sign=signal_period)
        return pd.DataFrame({"MACD": macd_indicator.macd(),
    "Signal": macd_indicator.macd_signal(),
     "Histogram": macd_indicator.macd_diff()})


def identify_support_resistance(self,
    window: int = SUPPORT_RESISTANCE_WINDOW,
    sensitivity: float = CLUSTER_SENSITIVITY) -> Dict[str,
    Tuple[float,
     float]]:
        data = self.df["close"].values
        volumes = self.df["volume"].values
        levels = {}
        maxima_indices = [i for i in range(window, len(data) -
    window) if all(data[i] >= data[i -
    j] for j in range(1, window +
    1)) and all(data[i] >= data[i +
    j] for j in range(1, window +
     1))]
        minima_indices = [i for i in range(window, len(data) -
    window) if all(data[i] <= data[i -
    j] for j in range(1, window +
    1)) and all(data[i] <= data[i +
    j] for j in range(1, window +
     1))]
        maxima = data[maxima_indices]
        minima = data[minima_indices]
        all_points = np.concatenate((maxima, minima))
        if len(all_points) < 2:
            self.logger.warning(
                "Not enough data points to identify support/resistance levels.")
            return levels
        cluster_centers = []
        cluster_volumes = []
        for point in all_points:
            if not cluster_centers:
                cluster_centers.append(point)
                cluster_volumes.append(volumes[list(all_points).index(point)])
            else:
                added_to_cluster = False
                for i, center in enumerate(cluster_centers):
                    if abs(point - center) / center <= sensitivity:
                        cluster_centers[i] = (
                            cluster_centers[i] * len(cluster_volumes) + point) / (len(cluster_volumes) + 1)
                        cluster_volumes[i] += volumes[list(
                            all_points).index(point)]
                        added_to_cluster = True
                        break
                if not added_to_cluster:
                    cluster_centers.append(point)
                    cluster_volumes.append(
                        volumes[list(all_points).index(point)])
        current_price = self.df["close"].iloc[-1]
        for center, volume in zip(cluster_centers, cluster_volumes):
            price_diff_ratio = abs(current_price - center) / current_price
            if price_diff_ratio <= sensitivity:
                level_type = "Support" if center < current_price else "Resistance"
                levels[f"{level_type} (Cluster)"] = (
                    center, volume / len(cluster_volumes))
        self.levels.update(levels)
        return levels

    def find_nearest_levels(self, current_price: float) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]
        supports = [(label, value, vol) for label, (value, vol) in self.levels.items() if value < current_price and isinstance(value, (float, np.float64))]
        resistances = [(label, value, vol) for label, (value, vol) in self.levels.items() if value > current_price and isinstance(value, (float, np.float64))]
        nearest_supports = sorted(supports, key=lambda x: x[1], reverse=True)[:3]
        nearest_resistances = sorted(resistances, key=lambda x: x[1])[:3]
        return nearest_supports, nearest_resistances

    def determine_trend_momentum(self) -> str:
        momentum = self.calculate_momentum()
        sma_short = self.calculate_sma(5)
        sma_long = self.calculate_sma(20)
        if momentum.iloc[-1] > 0 and sma_short.iloc[-1] > sma_long.iloc[-1]:
            return "upward"
        elif momentum.iloc[-1] < 0 and sma_short.iloc[-1] < sma_long.iloc[-1]:
            return "downward"
        else:
            return "neutral"
        
    def analyze_higher_timeframes(self, higher_timeframes: List[str]) -> Dict[str, Dict[str, float]]:
        higher_tf_levels = {}
        for tf in higher_timeframes:
            df_higher = self.bybit.fetch_klines(self.symbol, tf, limit=200)
            if not df_higher.empty:
                analyzer_higher = TradingAnalyzer(self.symbol, tf, self.logger)
                analyzer_higher.identify_support_resistance()
                higher_tf_levels[tf] = analyzer_higher.levels
            else:
                self.logger.warning(f"Could not fetch data for timeframe {tf}")
        return higher_tf_levels

    def suggest_trades(self, current_price: float, trend: str, nearest_supports: List[Tuple[str, float, float]], nearest_resistances: List[Tuple[str, float, float]]) -> Dict[str, Dict[str, float]]:
        suggestions = {"long": {}, "short": {}}
        if trend == "upward" and nearest_supports:
            entry_support = nearest_supports[0]
            target_resistance = nearest_resistances[0] if nearest_resistances else None
            suggestions["long"]["entry"] = entry_support[1]
            suggestions["long"]["target"] = target_resistance[1] if target_resistance else current_price * 1.05
        elif trend == "downward" and nearest_resistances:
            entry_resistance = nearest_resistances[0]
            target_support = nearest_supports[0] if nearest_supports else None
            suggestions["short"]["entry"] = entry_resistance[1]
            suggestions["short"]["target"] = target_support[1] if target_support else current_price * 0.95
        return suggestions
def analyze(self, current_price: float):
        if self.df.empty:
            self.logger.error("DataFrame is empty. Cannot perform analysis.")
            return
        self.df["fast_ma"] = self.calculate_ema(5)
        self.df["slow_ma"] = self.calculate_ema(20)
        self.df["volume_ma"] = self.calculate_sma(20)
        self.df["ma_diff"] = self.df["fast_ma"] - self.df["slow_ma"]
        self.df["momentum_like"] = self.df["ma_diff"].diff()
        self.df["atr"] = self.calculate_atr()
        high = self.df["high"].max()
        low = self.df["low"].min()
        close = self.df["close"].iloc[-1]
        fib_levels = self.calculate_fibonacci_retracement(high, low)
        self.calculate_pivot_points(high, low, close)
        self.identify_support_resistance()
        nearest_supports, nearest_resistances = self.find_nearest_levels(current_price)
        higher_tf_levels = self.analyze_higher_timeframes(HIGHER_TIMEFRAMES)
        trend = self.determine_trend_momentum()
        trade_suggestions = self.suggest_trades(current_price, trend, nearest_supports, nearest_resistances)
        self.logger.info(f"{Fore.YELLOW}Current Price ({self.interval}):{Fore.GREEN} {current_price:.2f}")
        self.logger.info(f"{Fore.YELLOW}Trend:{Fore.CYAN} {trend}")
        if self.df["atr"] is not None:
            self.logger.info(f"{Fore.YELLOW}ATR:{Fore.MAGENTA} {self.df['atr'].iloc[-1]:.2f}")
        else:
            self.logger.info(f"{Fore.YELLOW}ATR:{Fore.MAGENTA} None")
        self.logger.info(f"{Fore.YELLOW}Support/Resistance Levels ({self.interval}):")
        for level, (value, volume) in self.levels.items():
            if isinstance(value, (float, np.float64)):
                label_color = Fore.BLUE if "Support" in level else Fore.RED
                self.logger.info(f"{label_color} {level}: {Fore.CYAN} {value:.2f} {Fore.MAGENTA}(Vol: {volume:.2f}){Style.RESET_ALL}")
            else:
                self.logger.warning(f"Skipping invalid level value: {value}")
        self.logger.info(f"{Fore.YELLOW}Pivot Point Levels ({self.interval}):")
        for level, value in self.pivot_points.items():
            label_color = Fore.GREEN if level == "Pivot" else (Fore.BLUE if level.startswith("S") else Fore.RED)
            self.logger.info(f"{label_color} {level}: {Fore.CYAN} {value:.2f}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.YELLOW}Fibonacci Levels ({self.interval}):")
        for level, value in fib_levels.items():
            self.logger.info(f"{Fore.CYAN} {level}: {Fore.GREEN} {value:.2f}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.YELLOW}Higher Timeframe Levels:")
        for tf, levels in higher_tf_levels.items():
            self.logger.info(f"{Fore.CYAN}  Timeframe: {tf}")
            for level, (value, volume) in levels.items():
                if isinstance(value, (float, np.float64)):
                    label_color = Fore.BLUE if "Support" in level else Fore.RED
                    self.logger.info(f"    {label_color} {level}: {Fore.CYAN} {value:.2f} {Fore.MAGENTA}(Vol: {volume:.2f}){Style.RESET_ALL}")
                else:
                    self.logger.warning(f"Skipping invalid level value: {value}")
        self.logger.info(f"{Fore.YELLOW}Nearest Support Levels:")
        for level, value, volume in nearest_supports:
            self.logger.info(f"{Fore.BLUE} {level}: {Fore.GREEN} {value:.2f} {Fore.MAGENTA}(Vol: {volume:.2f})")
        self.logger.info(f"{Fore.YELLOW}Nearest Resistance Levels:")
        for level, value, volume in nearest_resistances:
            self.logger.info(f"{Fore.RED} {level}: {Fore.BLUE} {value:.2f} {Fore.MAGENTA}(Vol: {volume:.2f})")
        self.logger.info(f"{Fore.YELLOW}Trade Suggestions:")
        if trade_suggestions["long"]:
            self.logger.info(f"{Fore.GREEN}  Long Entry:{Fore.CYAN} {trade_suggestions['long']['entry']:.2f}")
            self.logger.info(f"{Fore.GREEN}  Long Target:{Fore.CYAN} {trade_suggestions['long']['target']:.2f}")
        else:
            self.logger.info(f"{Fore.GREEN}  No long entry suggested.")
        if trade_suggestions["short"]:
            self.logger.info(f"{Fore.RED}  Short Entry:{Fore.CYAN} {trade_suggestions['short']['entry']:.2f}")
            self.logger.info(f"{Fore.RED}  Short Target:{Fore.CYAN} {trade_suggestions['short']['target']:.2f}")
        else:
            self.logger.info(f"{Fore.RED}  No short entry suggested.")

# --- Logging Setup ---
def setup_logger(name: str) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"{name}_{timestamp}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(f"{Fore.BLUE}%(asctime)s{
