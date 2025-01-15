import os
import logging
import requests
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD, EMAIndicator, ADXIndicator, AroonIndicator
import hma
import wma
from ta.volatility import AverageTrueRange
from ta.momentum import rsi
from ta.volume import on_balance_volume, volume_price_trend
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
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 5))
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W"]
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]
SUPPORT_RESISTANCE_WINDOW = 14
CLUSTER_SENSITIVITY = 0.05
HIGHER_TIMEFRAMES = ["60", "240", "D"]


# --- Configuration ---
class Config:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "API keys not set. Set BYBIT_API_KEY and BYBIT_API_SECRET in .env"
            )


# --- Bybit API Client ---
class Bybit:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = requests.Session()

    def _generate_signature(self, params: dict) -> str:
        param_str = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
        return hmac.new(
            self.config.api_secret.encode(), param_str.encode(), hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, endpoint: str, params: dict = None) -> dict:
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

                if response.status_code in RETRY_ERROR_CODES:
                    self.logger.warning(
                        f"Rate limited or server error. Retrying {retries + 1}/{MAX_RETRIES} after {RETRY_DELAY} seconds."
                    )
                    time.sleep(RETRY_DELAY * (2**retries))
                    retries += 1
                    continue

                if response.status_code != 200:
                    self.logger.error(
                        f"Bybit API Error {response.status_code}: {response.text}"
                    )
                    return {"retCode": -1, "retMsg": f"HTTP {response.status_code}"}

                json_response = response.json()
                if not json_response:
                    return {"retCode": -1, "retMsg": "Invalid JSON"}

                if json_response.get("retCode") != 0:
                    self.logger.error(f"Bybit API returned non-zero: {json_response}")
                    return json_response

                return json_response

            except requests.RequestException as e:
                self.logger.error(f"API request failed: {e}")
                retries += 1
                time.sleep(RETRY_DELAY * (2**retries))
            except Exception as e:
                self.logger.exception(f"An unexpected error occurred: {e}")
                return {"retCode": -1, "retMsg": "Unexpected Error"}

        return {"retCode": -1, "retMsg": "Max retries exceeded"}

    def fetch_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
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
            df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
            df = df.astype(
                {
                    col: float
                    for col in ["open", "high", "low", "close", "volume", "turnover"]
                }
            )
            return df
        else:
            self.logger.error(
                f"Failed to fetch klines for {symbol} ({interval}): {response.get('retMsg', 'Unknown error')}"
            )
            return pd.DataFrame()

    def fetch_current_price(self, symbol: str) -> float | None:
        endpoint = "/v5/market/tickers"
        params = {"symbol": symbol, "category": "linear"}
        response = self._request("GET", endpoint, params)

        if response and response.get("retCode") == 0 and response.get("result"):
            try:
                price = float(response["result"]["list"][0]["lastPrice"])
                if price <= 0:
                    raise ValueError("Price must be positive.")
                return price
            except (KeyError, IndexError, ValueError) as e:
                self.logger.error(f"Invalid price data from Bybit: {e}")
                return None
        else:
            self.logger.error(
                f"Failed to fetch price for {symbol}: {response.get('retMsg', 'Unknown error')}"
            )
            return None


# --- Technical Analysis Module (Enhanced) ---
class TradingAnalyzer:
    def __init__(
        self,
        symbol: str,
        interval: str,
        logger: logging.Logger,
    ):
        self.symbol = symbol
        self.interval = interval
        self.logger = logger
        self.config = Config()
        self.bybit = Bybit(self.config, logger)
        self.df = self.bybit.fetch_klines(symbol, interval, limit=200)
        self.levels = {}
        self.pivot_points = {}  # To store pivot point levels

    def calculate_sma(self, window: int) -> pd.Series:
        return SMAIndicator(self.df["close"], window=window).sma_indicator()

    def calculate_ema(self, window: int) -> pd.Series:
        return EMAIndicator(self.df["close"], window=window).ema_indicator()

    def calculate_wma(self, window: int) -> pd.Series:
        """Calculates the Weighted Moving Average (WMA)."""
        return wma(self.df["close"], window=window)

    def calculate_hma(self, window: int) -> pd.Series:
        """Calculates the Hull Moving Average (HMA)."""
        return hma(self.df["close"], window=window)

    def calculate_momentum(self, window: int = 14) -> pd.Series:
        return self.df["close"].diff(window)

    def calculate_fibonacci_retracement(
        self, high: float, low: float
    ) -> Dict[str, float]:
        diff = high - low
        if diff == 0:
            self.logger.warning(
                "Cannot calculate Fibonacci: high and low are the same."
            )
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

        self.pivot_points.update(pivot_levels)
        return pivot_levels

    def calculate_atr(self, window: int = 14) -> pd.Series:
        return AverageTrueRange(
            self.df["high"], self.df["low"], self.df["close"], window=window
        ).average_true_range()

    def calculate_macd(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> pd.DataFrame:
        macd_indicator = MACD(
            self.df["close"],
            window_slow=slow_period,
            window_fast=fast_period,
            window_sign=signal_period,
        )
        return pd.DataFrame(
            {
                "MACD": macd_indicator.macd(),
                "Signal": macd_indicator.macd_signal(),
                "Histogram": macd_indicator.macd_diff(),
            }
        )

    def calculate_rsi(self, window: int = 14) -> pd.Series:
        """Calculates the Relative Strength Index (RSI)."""
        return rsi(self.df["close"], window=window)

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

    def calculate_obv(self) -> pd.Series:
        """Calculates On-Balance Volume (OBV)."""
        return on_balance_volume(self.df["close"], self.df["volume"])

    def calculate_vpt(self) -> pd.Series:
        """Calculates Volume Price Trend (VPT)."""
        return volume_price_trend(self.df["close"], self.df["volume"])

    def identify_support_resistance(
        self,
        window: int = SUPPORT_RESISTANCE_WINDOW,
        sensitivity: float = CLUSTER_SENSITIVITY,
    ) -> Dict[str, Tuple[float, float]]:
        data = self.df["close"].values
        volumes = self.df["volume"].values
        levels = {}

        # Find local maxima and minima
        maxima_indices = [
            i
            for i in range(window, len(data) - window)
            if all(data[i] >= data[i - j] for j in range(1, window + 1))
            and all(data[i] >= data[i + j] for j in range(1, window + 1))
        ]
        minima_indices = [
            i
            for i in range(window, len(data) - window)
            if all(data[i] <= data[i - j] for j in range(1, window + 1))
            and all(data[i] <= data[i + j] for j in range(1, window + 1))
        ]
        maxima = data[maxima_indices]
        minima = data[minima_indices]

        # Combine maxima and minima
        all_points = np.concatenate((maxima, minima))

        if len(all_points) < 2:
            self.logger.warning(
                "Not enough data points to identify support/resistance levels."
            )
            return levels

        # Manually cluster points into groups
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
                        # Update the cluster center as a weighted average
                        cluster_centers[i] = (
                            cluster_centers[i] * len(cluster_volumes) + point
                        ) / (len(cluster_volumes) + 1)
                        # Update volume
                        cluster_volumes[i] += volumes[list(all_points).index(point)]
                        added_to_cluster = True
                        break
                if not added_to_cluster:
                    cluster_centers.append(point)
                    cluster_volumes.append(volumes[list(all_points).index(point)])

        # Filter levels and associate with volume
        current_price = self.df["close"].iloc[-1]

        for center, volume in zip(cluster_centers, cluster_volumes):
            price_diff_ratio = abs(current_price - center) / current_price
            if price_diff_ratio <= sensitivity:
                level_type = "Support" if center < current_price else "Resistance"
                levels[f"{level_type} (Cluster)"] = (
                    center,
                    volume / len(cluster_volumes),
                )  # Average volume

        self.levels.update(levels)
        return levels

    def find_nearest_levels(
        self, current_price: float
    ) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
        supports = [
            (label, value, vol)
            for label, (value, vol) in self.levels.items()
            if value < current_price and isinstance(value, (float, np.float64))
        ]
        resistances = [
            (label, value, vol)
            for label, (value, vol) in self.levels.items()
            if value > current_price and isinstance(value, (float, np.float64))
        ]

        nearest_supports = sorted(supports, key=lambda x: x[1], reverse=True)[:3]
        nearest_resistances = sorted(resistances, key=lambda x: x[1])[:3]
        return nearest_supports, nearest_resistances

    def analyze_rsi(self, rsi_value: float) -> str:
        """Analyzes the RSI value for overbought/oversold conditions."""
        if rsi_value > 70:
            return "overbought"
        elif rsi_value < 30:
            return "oversold"
        else:
            return "neutral"

    def analyze_macd(self) -> str:
        """Provides a basic analysis of the MACD indicator."""
        macd_data = self.calculate_macd()
        macd_line = macd_data["MACD"].iloc[-1]
        signal_line = macd_data["Signal"].iloc[-1]
        histogram = macd_data["Histogram"].iloc[-1]

        if macd_line > signal_line and histogram > 0:
            return "bullish"  # MACD above Signal and Histogram positive
        elif macd_line < signal_line and histogram < 0:
            return "bearish"  # MACD below Signal and Histogram negative
        elif macd_line > signal_line and histogram < 0:
            return (
                "potential bearish reversal"  # MACD above Signal but Histogram negative
            )
        elif macd_line < signal_line and histogram > 0:
            return (
                "potential bullish reversal"  # MACD below Signal but Histogram positive
            )
        else:
            return "neutral"

    def determine_trend_momentum_adx(self) -> str:
        """Determines the trend direction based on momentum, MA, and ADX."""
        momentum = self.calculate_momentum()
        sma_short = self.calculate_sma(5)
        sma_long = self.calculate_sma(20)
        adx_data = self.calculate_adx()

        adx = adx_data["ADX"].iloc[-1]
        plus_di = adx_data["+DI"].iloc[-1]
        minus_di = adx_data["-DI"].iloc[-1]

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
                return "neutral"  # Strong trend but direction unclear based on other indicators
        else:
            return "neutral"  # Weak or no trend

    def analyze_higher_timeframes(
        self, higher_timeframes: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Analyzes support/resistance on higher timeframes."""
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
            target_resistance = nearest_resistances[0] if nearest_resistances else None

            suggestions["long"]["entry"] = entry_support[1]
            suggestions["long"]["target"] = (
                target_resistance[1] if target_resistance else current_price * 1.05
            )  # Default 5% above current price

        elif trend == "downward" and nearest_resistances:
            # Short entry near closest resistance with target at closest support
            entry_resistance = nearest_resistances[0]
            target_support = nearest_supports[0] if nearest_supports else None

            suggestions["short"]["entry"] = entry_resistance[1]
            suggestions["short"]["target"] = (
                target_support[1] if target_support else current_price * 0.95
            )  # Default 5% below current price

        return suggestions

    def analyze(self, current_price: float):
        """Analyzes the market data and provides insights."""
        if self.df.empty:
            self.logger.error("DataFrame is empty. Cannot perform analysis.")
            return

        # Calculate various indicators
        self.df["fast_ma"] = self.calculate_ema(5)
        self.df["slow_ma"] = self.calculate_ema(20)
        self.df["wma"] = self.calculate_wma(14)
        self.df["hma"] = self.calculate_hma(14)
        self.df["volume_ma"] = self.calculate_sma(20)
        self.df["ma_diff"] = self.df["fast_ma"] - self.df["slow_ma"]
        self.df["momentum_like"] = self.df["ma_diff"].diff()
        self.df["atr"] = self.calculate_atr()
        self.df["obv"] = self.calculate_obv()
        self.df["vpt"] = self.calculate_vpt()

        adx_data = self.calculate_adx()
        self.df["ADX"] = adx_data["ADX"]
        self.df["+DI"] = adx_data["+DI"]
        self.df["-DI"] = adx_data["-DI"]

        aroon_data = self.calculate_aroon()
        self.df["Aroon Up"] = aroon_data["Aroon Up"]
        self.df["Aroon Down"] = aroon_data["Aroon Down"]

        rsi_value = self.calculate_rsi()
        rsi_analysis = self.analyze_rsi(rsi_value.iloc[-1])
        self.df["RSI"] = rsi_value

        high = self.df["high"].max()
        low = self.df["low"].min()
        close = self.df["close"].iloc[-1]

        # Calculate support/resistance levels
        fib_levels = self.calculate_fibonacci_retracement(high, low)
        self.calculate_pivot_points(high, low, close)
        self.identify_support_resistance()  # Dynamic S/R
        nearest_supports, nearest_resistances = self.find_nearest_levels(current_price)

        # Analyze higher timeframes
        higher_tf_levels = self.analyze_higher_timeframes(HIGHER_TIMEFRAMES)

        # Determine trend and momentum
        trend = self.determine_trend_momentum_adx()

        # MACD Analysis
        macd_analysis = self.analyze_macd()

        # Suggest trades
        trade_suggestions = self.suggest_trades(
            current_price, trend, nearest_supports, nearest_resistances
        )

        # Log the analysis results
        self.logger.info(
            f"{Fore.YELLOW}Current Price ({self.interval}):{Fore.GREEN} {current_price:.2f}"
        )
        self.logger.info(f"{Fore.YELLOW}Trend:{Fore.CYAN} {trend}")

        # Log ATR
        if self.df["atr"] is not None:
            self.logger.info(
                f"{Fore.YELLOW}ATR:{Fore.MAGENTA} {self.df['atr'].iloc[-1]:.2f}"
            )
        else:
            self.logger.info(f"{Fore.YELLOW}ATR:{Fore.MAGENTA} None")

        # Logging calculated levels for the current timeframe
        self.logger.info(f"{Fore.YELLOW}Support/Resistance Levels ({self.interval}):")
        for level, (value, volume) in self.levels.items():
            if isinstance(
                value, (float, np.float64)
            ):  # Check if 'value' is a float or numpy.float64
                label_color = Fore.BLUE if "Support" in level else Fore.RED
                self.logger.info(
                    f"{label_color} {level}: {Fore.CYAN} {value:.2f} {Fore.MAGENTA}(Vol: {volume:.2f}){Style.RESET_ALL}"
                )
            else:
                self.logger.warning(f"Skipping invalid level value: {value}")

        # Log pivot point levels
        self.logger.info(f"{Fore.YELLOW}Pivot Point Levels ({self.interval}):")
        for level, value in self.pivot_points.items():
            label_color = (
                Fore.GREEN
                if level == "Pivot"
                else (Fore.BLUE if level.startswith("S") else Fore.RED)
            )
            self.logger.info(
                f"{label_color} {level}: {Fore.CYAN} {value:.2f}{Style.RESET_ALL}"
            )

        # Log Fibonacci levels
        self.logger.info(f"{Fore.YELLOW}Fibonacci Levels ({self.interval}):")
        for level, value in fib_levels.items():
            self.logger.info(
                f"{Fore.CYAN} {level}: {Fore.GREEN} {value:.2f}{Style.RESET_ALL}"
            )

        # Log higher timeframe levels
        self.logger.info(f"{Fore.YELLOW}Higher Timeframe Levels:")
        for tf, levels in higher_tf_levels.items():
            self.logger.info(f"{Fore.CYAN}  Timeframe: {tf}")
            for level, (value, volume) in levels.items():
                if isinstance(value, (float, np.float64)):
                    label_color = Fore.BLUE if "Support" in level else Fore.RED
                    self.logger.info(
                        f"    {label_color} {level}: {Fore.CYAN} {value:.2f} {Fore.MAGENTA}(Vol: {volume:.2f}){Style.RESET_ALL}"
                    )
                else:
                    self.logger.warning(f"Skipping invalid level value: {value}")

        self.logger.info(f"{Fore.YELLOW}Nearest Support Levels:")
        for level, value, volume in nearest_supports:
            self.logger.info(
                f"{Fore.BLUE} {level}: {Fore.GREEN} {value:.2f} {Fore.MAGENTA}(Vol: {volume:.2f})"
            )

        self.logger.info(f"{Fore.YELLOW}Nearest Resistance Levels:")
        for level, value, volume in nearest_resistances:
            self.logger.info(
                f"{Fore.RED} {level}: {Fore.BLUE} {value:.2f} {Fore.MAGENTA}(Vol: {volume:.2f})"
            )

        # Log trade suggestions
        self.logger.info(f"{Fore.YELLOW}Trade Suggestions:")
        if trade_suggestions["long"]:
            self.logger.info(
                f"{Fore.GREEN}  Long Entry:{Fore.CYAN} {trade_suggestions['long']['entry']:.2f}"
            )
            self.logger.info(
                f"{Fore.GREEN}  Long Target:{Fore.CYAN} {trade_suggestions['long']['target']:.2f}"
            )
        else:
            self.logger.info(f"{Fore.GREEN}  No long entry suggested.")

        if trade_suggestions["short"]:
            self.logger.info(
                f"{Fore.RED}  Short Entry:{Fore.CYAN} {trade_suggestions['short']['entry']:.2f}"
            )
            self.logger.info(
                f"{Fore.RED}  Short Target:{Fore.CYAN} {trade_suggestions['short']['target']:.2f}"
            )
        else:
            self.logger.info(f"{Fore.RED}  No short entry suggested.")

        # Log WMA, HMA, MACD, ADX, Aroon, OBV, VPT, and RSI
        self.logger.info(
            f"{Fore.YELLOW}WMA (14):{Fore.CYAN} {self.df['wma'].iloc[-1]:.2f}"
        )
        self.logger.info(
            f"{Fore.YELLOW}HMA (14):{Fore.CYAN} {self.df['hma'].iloc[-1]:.2f}"
        )
        self.logger.info(f"{Fore.YELLOW}MACD Analysis: {Fore.MAGENTA}{macd_analysis}")
        self.logger.info(
            f"{Fore.YELLOW}ADX:{Fore.CYAN} {self.df['ADX'].iloc[-1]:.2f} {Fore.YELLOW}(+DI:{Fore.GREEN} {self.df['+DI'].iloc[-1]:.2f}{Fore.YELLOW}, -DI:{Fore.RED} {self.df['-DI'].iloc[-1]:.2f}{Fore.YELLOW})"
        )
        self.logger.info(
            f"{Fore.YELLOW}Aroon Up: {Fore.GREEN}{self.df['Aroon Up'].iloc[-1]:.2f}, Aroon Down: {Fore.RED}{self.df['Aroon Down'].iloc[-1]:.2f}"
        )
        self.logger.info(f"{Fore.YELLOW}OBV:{Fore.CYAN} {self.df['obv'].iloc[-1]:.2f}")
        self.logger.info(f"{Fore.YELLOW}VPT:{Fore.CYAN} {self.df['vpt'].iloc[-1]:.2f}")
        self.logger.info(
            f"{Fore.YELLOW}RSI:{Fore.CYAN} {rsi_value.iloc[-1]:.2f} ({rsi_analysis})"
        )


# --- Logging Setup ---
def setup_logger(name: str) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            f"{Fore.BLUE}%(asctime)s{Fore.RESET} - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(stream_handler)

    return logger


# --- Main Function ---
def main():
    init(autoreset=True)
    logger = setup_logger("trading_bot")

    try:
        symbol = input(f"{Fore.CYAN}Enter trading symbol (e.g., BTCUSDT): ").upper()
        interval = input(f"{Fore.CYAN}Enter timeframe ({', '.join(VALID_INTERVALS)}): ")

        if interval not in VALID_INTERVALS:
            logger.error(f"{Fore.RED}Invalid interval: {interval}")
            return

        analyzer = TradingAnalyzer(symbol, interval, logger)

        while True:
            current_price = analyzer.bybit.fetch_current_price(symbol)
            if current_price is None:
                time.sleep(30)
                continue

            analyzer.df = analyzer.bybit.fetch_klines(symbol, interval, limit=200)
            if analyzer.df.empty:
                logger.error(
                    f"{Fore.RED}Failed to fetch klines for {symbol} ({interval})."
                )
                time.sleep(30)
                continue

            analyzer.analyze(current_price)

            time.sleep(60)

    except KeyboardInterrupt:
        logger.info(f"{Fore.YELLOW}Exiting script.")
    except Exception as e:
        logger.exception(f"{Fore.RED}An error occurred: {e}")


if __name__ == "__main__":
    main()
