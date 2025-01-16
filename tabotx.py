import os
import logging
import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, AwesomeOscillatorIndicator, StochRSIIndicator
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

# --- Constants ---
LOG_DIR = "botlogs"  # Directory to store log files
ST_LOUIS_TZ = ZoneInfo("America/Chicago")  # Time zone for timestamps
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))  # Maximum API request retries
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 5))  # Delay between retries (seconds)
VALID_INTERVALS = [
    "1",
    "3",
    "5",
    "15",
    "30",
    "60",
    "120",
    "240",
    "D",
    "W",
]  # Valid chart intervals
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]  # HTTP error codes that trigger retries
SUPPORT_RESISTANCE_WINDOW = 14  # Lookback window for identifying S/R levels
CLUSTER_SENSITIVITY = 0.05  # Sensitivity for filtering S/R levels by distance to price
# Higher timeframes for multi-timeframe analysis (you can add more)
HIGHER_TIMEFRAMES = ["60", "240", "D"]


# --- Configuration ---
class Config:
    """Loads configuration from environment variables."""

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "API keys not set. Please set BYBIT_API_KEY and BYBIT_API_SECRET in your .env file."
            )


# --- Bybit API Client ---
class Bybit:
    """Handles interactions with the Bybit V5 API."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = requests.Session()

    def _generate_signature(self, params: dict) -> str:
        """Generates a signature for the request."""
        param_str = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
        return hmac.new(
            self.config.api_secret.encode(), param_str.encode(), hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, endpoint: str, params: dict = None) -> dict:
        """Sends a request to the Bybit API with retries and error handling."""
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

                # Handle rate limits and server errors with retries
                if response.status_code in RETRY_ERROR_CODES:
                    self.logger.warning(
                        f"{Fore.YELLOW}Rate limited or server error. "
                        f"Retrying {retries + 1}/{MAX_RETRIES} after {RETRY_DELAY} seconds."
                    )
                    time.sleep(RETRY_DELAY * (2**retries))  # Exponential backoff
                    retries += 1
                    continue

                # Handle non-200 responses
                if response.status_code != 200:
                    self.logger.error(
                        f"{Fore.RED}Bybit API Error {response.status_code}: {response.text}"
                    )
                    return {"retCode": -1, "retMsg": f"HTTP {response.status_code}"}

                json_response = response.json()
                if not json_response:
                    return {"retCode": -1, "retMsg": "Invalid JSON"}

                # Handle Bybit-specific error codes
                if json_response.get("retCode") != 0:
                    self.logger.error(
                        f"{Fore.RED}Bybit API returned non-zero: {json_response}"
                    )
                    # Here you can add more specific error handling based on retCode
                    # For example:
                    # if json_response["retCode"] == 10001:  # Order not found
                    #   return handle_order_not_found(json_response)
                    return json_response

                return json_response

            except requests.RequestException as e:
                self.logger.error(f"{Fore.RED}API request failed: {e}")
                retries += 1
                time.sleep(RETRY_DELAY * (2**retries))  # Exponential backoff
            except Exception as e:
                self.logger.exception(f"{Fore.RED}An unexpected error occurred: {e}")
                return {"retCode": -1, "retMsg": "Unexpected Error"}

        return {"retCode": -1, "retMsg": "Max retries exceeded"}

    def fetch_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Fetches klines (candlestick data) for a given symbol and interval."""
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
                f"{Fore.RED}Failed to fetch klines for {symbol} ({interval}): "
                f"{response.get('retMsg', 'Unknown error')}"
            )
            return pd.DataFrame()

    def fetch_current_price(self, symbol: str) -> float | None:
        """Fetches the current price for a given symbol."""
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
                self.logger.error(f"{Fore.RED}Invalid price data from Bybit: {e}")
                return None
        else:
            self.logger.error(
                f"{Fore.RED}Failed to fetch price for {symbol}: "
                f"{response.get('retMsg', 'Unknown error')}"
            )
            return None


# --- Technical Analysis Module ---
class TechnicalAnalyzer:
    """Performs technical analysis on market data."""

    def __init__(self, symbol: str, interval: str, logger: logging.Logger):
        self.symbol = symbol
        self.interval = interval
        self.logger = logger
        self.bybit = Bybit(Config(), logger)  # Initialize Bybit client
        self.df = self.bybit.fetch_klines(symbol, interval, limit=200)
        self.levels = {}  # Store calculated levels (Fibonacci, Pivot Points, S/R)

    def calculate_sma(self, window: int) -> pd.Series:
        """Calculates Simple Moving Average."""
        return SMAIndicator(self.df["close"], window=window).sma_indicator()

    def calculate_ema(self, window: int) -> pd.Series:
        """Calculates Exponential Moving Average."""
        return EMAIndicator(self.df["close"], window=window).ema_indicator()

    def calculate_momentum(self, window: int = 14) -> pd.Series:
        """Calculates Momentum."""
        return self.df["close"].diff(window)

    def calculate_fibonacci_retracement(
        self, high: float, low: float
    ) -> Dict[str, float]:
        """Calculates Fibonacci retracement levels."""
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

        # Store Fibonacci levels, dynamically assigning labels based on current price
        current_price = self.df["close"].iloc[-1]
        self.levels.update(
            {
                f"Support ({label})"
                if value <= current_price
                else f"Resistance ({label})": value
                for label, value in fib_levels.items()
            }
        )

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

        self.levels.update(pivot_levels)
        return pivot_levels

    def kmeans_clustering(
        self, data: np.ndarray, n_clusters: int, max_iterations: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs K-Means clustering without using scikit-learn."""
        # Initialize centroids randomly
        centroids = data[np.random.choice(data.shape[0], n_clusters, replace=False)]

        for _ in range(max_iterations):
            # Assign points to the nearest centroid
            distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Update centroids
            new_centroids = np.array(
                [data[labels == i].mean(axis=0) for i in range(n_clusters)]
            )

            # Check for convergence
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return centroids, labels

    def identify_support_resistance(
        self,
        window: int = SUPPORT_RESISTANCE_WINDOW,
        sensitivity: float = CLUSTER_SENSITIVITY,
    ) -> Dict[str, Tuple[float, float]]:
        """Identifies support and resistance levels based on price action, clustering, and volume."""
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

        # Combine and reshape for clustering
        all_points = np.concatenate((maxima, minima)).reshape(-1, 1)

        if len(all_points) < 2:
            self.logger.warning(
                "Not enough data points to identify support/resistance levels."
            )
            return levels

        # Use K-Means clustering
        n_clusters = min(int(len(all_points) * 0.5), 8)  # Cap the number of clusters
        cluster_centers, cluster_labels = self.kmeans_clustering(all_points, n_clusters)

        current_price = self.df["close"].iloc[-1]

        # Filter levels based on sensitivity to price and associate with volume
        for i, center in enumerate(cluster_centers):
            price_diff_ratio = abs(current_price - center) / current_price
            if price_diff_ratio <= sensitivity:
                # Calculate average volume at this level
                level_indices = [
                    idx
                    for j, idx in enumerate(maxima_indices + minima_indices)
                    if cluster_labels[j] == i
                ]
                avg_volume = np.mean(volumes[level_indices]) if level_indices else 0

                level_type = "Support" if center < current_price else "Resistance"
                levels[f"{level_type} (Cluster)"] = (center[0], avg_volume)

        # Store levels
        self.levels.update(levels)

        return levels

    def find_nearest_levels(
        self, current_price: float
    ) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
        """Finds nearest support and resistance levels, including volume information."""
        supports = []
        resistances = []

        for label, level_data in self.levels.items():
            if isinstance(level_data, tuple) and len(level_data) == 2:
                value, vol = level_data
                if value < current_price:
                    supports.append((label, value, vol))
                elif value > current_price:
                    resistances.append((label, value, vol))
            elif isinstance(level_data, float):  # if key has no volume data
                if level_data < current_price:
                    supports.append((label, level_data, 0.0))  # assign volume as 0.0
                elif level_data > current_price:
                    resistances.append((label, level_data, 0.0))  # assign volume as 0.0

        nearest_supports = sorted(supports, key=lambda x: x[1], reverse=True)[:3]
        nearest_resistances = sorted(resistances, key=lambda x: x[1])[:3]
        return nearest_supports, nearest_resistances

    def calculate_atr(self, window: int = 14) -> pd.Series:
        """Calculates Average True Range."""
        return AverageTrueRange(
            self.df["high"], self.df["low"], self.df["close"], window=window
        ).average_true_range()

    def calculate_macd(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> pd.DataFrame:
        """Calculates MACD indicator."""
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

    def detect_macd_divergence(self) -> str | None:
        """Detects MACD divergence."""
        macd_df = self.calculate_macd()
        if macd_df is None or len(macd_df) < 3:
            self.logger.warning("Insufficient data for MACD divergence detection.")
            return None

        close_prices = self.df["close"].tail(3).values
        macd_values = macd_df["MACD"].tail(3).values

        if (
            close_prices[0] > close_prices[1] < close_prices[2]
            and macd_values[0] < macd_values[1] > macd_values[2]
        ):
            self.logger.info("Bullish MACD divergence detected.")
            return "bullish"
        elif (
            close_prices[0] < close_prices[1] > close_prices[2]
            and macd_values[0] > macd_values[1] < macd_values[2]
        ):
            self.logger.info("Bearish MACD divergence detected.")
            return "bearish"
        else:
            return None

    def determine_trend_momentum(self) -> str:
        """Determines the current trend direction based on momentum and moving averages."""
        momentum = self.calculate_momentum()
        sma_short = self.calculate_sma(5)
        sma_long = self.calculate_sma(20)

        if momentum.iloc[-1] > 0 and sma_short.iloc[-1] > sma_long.iloc[-1]:
            return "upward"
        elif momentum.iloc[-1] < 0 and sma_short.iloc[-1] < sma_long.iloc[-1]:
            return "downward"
        else:
            return "neutral"

    def predict_next_level(
        self,
        current_price: float,
        nearest_supports: List[Tuple[str, float]],
        nearest_resistances: List[Tuple[str, float]],
    ) -> str:
        """Predicts the next likely price level based on support and resistance."""
        if not nearest_supports and not nearest_resistances:
            return "neutral"

        closest_resistance = min(
            nearest_resistances, key=lambda x: abs(x[1] - current_price), default=None
        )
        closest_support = min(
            nearest_supports, key=lambda x: abs(x[1] - current_price), default=None
        )

        if closest_resistance is None:
            return "downward"
        if closest_support is None:
            return "upward"

        if abs(closest_resistance[1] - current_price) < abs(
            closest_support[1] - current_price
        ):
            return "upward"
        else:
            return "downward"

    def analyze_higher_timeframes(
        self, higher_timeframes: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Analyzes support/resistance on higher timeframes."""
        higher_tf_levels = {}
        for tf in higher_timeframes:
            df_higher = self.bybit.fetch_klines(self.symbol, tf, limit=200)
            if not df_higher.empty:
                analyzer_higher = TechnicalAnalyzer(self.symbol, tf, self.logger)
                # Assuming you have a method to identify S/R levels, similar to what was done for the primary timeframe
                analyzer_higher.identify_support_resistance()
                higher_tf_levels[tf] = analyzer_higher.levels
            else:
                self.logger.warning(f"Could not fetch data for timeframe {tf}")

        return higher_tf_levels

    def analyze(self, current_price: float):
        """Analyzes the market data and provides insights."""
        high = self.df["high"].max()
        low = self.df["low"].min()
        close = self.df["close"].iloc[-1]

        # Calculate support/resistance levels
        self.calculate_fibonacci_retracement(high, low)
        self.calculate_pivot_points(high, low, close)
        self.identify_support_resistance()  # New method for dynamic S/R
        nearest_supports, nearest_resistances = self.find_nearest_levels(current_price)

        # Analyze higher timeframes
        higher_tf_levels = self.analyze_higher_timeframes(HIGHER_TIMEFRAMES)

        # Determine trend and momentum
        trend = self.determine_trend_momentum()

        # Calculate ATR for volatility
        atr = self.calculate_atr()

        # Predict next likely level
        next_level_prediction = self.predict_next_level(
            current_price, nearest_supports, nearest_resistances
        )

        # Log the analysis results
        self.logger.info(
            f"{Fore.YELLOW}Current Price ({self.interval}):{Fore.GREEN} {current_price:.2f}"
        )
        self.logger.info(f"{Fore.YELLOW}Trend:{Fore.CYAN} {trend}")

        if atr is not None:
            self.logger.info(f"{Fore.YELLOW}ATR:{Fore.MAGENTA} {atr.iloc[-1]:.2f}")
        else:
            self.logger.info(f"{Fore.YELLOW}ATR:{Fore.MAGENTA} None")

        # Logging calculated levels for the current timeframe
        self.logger.info(f"{Fore.YELLOW}Support/Resistance Levels ({self.interval}):")
        for level, level_data in self.levels.items():
            if isinstance(level_data, tuple) and len(level_data) == 2:
                value, volume = level_data
                label_color = Fore.BLUE if "Support" in level else Fore.RED
                self.logger.info(
                    f"{label_color} {level}: {Fore.CYAN} {value:.2f} {Fore.MAGENTA}(Vol: {volume:.2f}){Style.RESET_ALL}"
                )
            elif isinstance(level_data, float):  # for levels without volume
                label_color = Fore.BLUE if "Support" in level else Fore.RED
                self.logger.info(
                    f"{label_color} {level}: {Fore.CYAN} {level_data:.2f} {Fore.MAGENTA}(Vol: 0.00){Style.RESET_ALL}"
                )

        # Log higher timeframe levels
        self.logger.info(f"{Fore.YELLOW}Higher Timeframe Levels:")
        for tf, levels in higher_tf_levels.items():
            self.logger.info(f"{Fore.CYAN}  Timeframe: {tf}")
            for level, level_data in levels.items():
                if isinstance(level_data, tuple) and len(level_data) == 2:
                    value, volume = level_data
                    label_color = Fore.BLUE if "Support" in level else Fore.RED
                    self.logger.info(
                        f"    {label_color} {level}: {Fore.CYAN} {value:.2f} {Fore.MAGENTA}(Vol: {volume:.2f}){Style.RESET_ALL}"
                    )
                elif isinstance(level_data, float):  # Handle levels without volume data
                    label_color = Fore.BLUE if "Support" in level else Fore.RED
                    self.logger.info(
                        f"    {label_color} {level}: {Fore.CYAN} {level_data:.2f} {Fore.MAGENTA}(Vol: 0.00){Style.RESET_ALL}"
                    )
                else:
                    self.logger.warning(f"Skipping invalid level value: {level_data}")

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

        self.logger.info(
            f"{Fore.YELLOW}Prediction:{Fore.MAGENTA} {next_level_prediction}"
        )


# --- Logging Setup ---
def setup_logger(name: str) -> logging.Logger:
    """Sets up a logger with file and stream handlers."""
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
    """Main function to run the trading bot."""
    init(autoreset=True)  # Initialize colorama
    config = Config()
    logger = setup_logger("trading_bot")

    try:
        bybit = Bybit(config, logger)

        symbol = input(f"{Fore.CYAN}Enter trading symbol (e.g., BTCUSDT): ").upper()
        interval = input(f"{Fore.CYAN}Enter timeframe ({', '.join(VALID_INTERVALS)}): ")

        if interval not in VALID_INTERVALS:
            logger.error(f"{Fore.RED}Invalid interval: {interval}")
            return

        while True:
            current_price = bybit.fetch_current_price(symbol)
            if current_price is None:
                time.sleep(30)  # Wait before retrying
                continue

            analyzer = TechnicalAnalyzer(symbol, interval, logger)
            if analyzer.df.empty:
                logger.error(
                    f"{Fore.RED}Failed to fetch klines for {symbol} ({interval})."
                )
                time.sleep(30)
                continue
            analyzer.analyze(current_price)

            time.sleep(30)  # Adjust analysis frequency as needed

    except Exception as e:
        logger.exception(f"{Fore.RED}An error occurred: {e}")


if __name__ == "__main__":
    main()
