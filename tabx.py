import os
import logging
import requests
import pandas as pd
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
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]  # Valid chart intervals
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]  # HTTP error codes that trigger retries

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
                params["timestamp"] = str(int(datetime.now(ST_LOUIS_TZ).timestamp() * 1000))
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

    def fetch_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """Fetches klines (candlestick data) for a given symbol and interval."""
        endpoint = "/v5/market/kline"
        params = {"symbol": symbol, "interval": interval, "limit": limit, "category": "linear"}
        response = self._request("GET", endpoint, params)

        if response.get("retCode") == 0 and response.get("result"):
            klines = response["result"]["list"]
            df = pd.DataFrame(
                klines,
                columns=["start_time", "open", "high", "low", "close", "volume", "turnover"],
            )
            df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
            df = df.astype(
                {col: float for col in ["open", "high", "low", "close", "volume", "turnover"]}
            )
            return df
        else:
            self.logger.error(
                f"{Fore.RED}Failed to fetch klines for {symbol}: "
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

    def __init__(self, df: pd.DataFrame, logger: logging.Logger):
        self.df = df
        self.logger = logger
        self.levels = {}  # Store calculated levels (Fibonacci, Pivot Points)

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
        self, high: float, low: float, current_price: float
    ) -> Dict[str, float]:
        """Calculates Fibonacci retracement levels."""
        diff = high - low
        if diff == 0:
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

        for label, value in fib_levels.items():
            level_label = (
                f"Support ({label})" if value <= current_price else f"Resistance ({label})"
            )
            self.levels[level_label] = value

        return fib_levels

    def calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculates pivot points."""
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

        self.levels.update(
            {
                "Pivot": pivot,
                "R1": r1,
                "S1": s1,
                "R2": r2,
                "S2": s2,
                "R3": r3,
                "S3": s3,
            }
        )
        return self.levels

    def find_nearest_levels(
        self, current_price: float
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Finds nearest support and resistance levels."""
        supports = [
            (label, value) for label, value in self.levels.items() if value < current_price
        ]
        resistances = [
            (label, value) for label, value in self.levels.items() if value > current_price
        ]
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

        if abs(closest_resistance[1] - current_price) < abs(closest_support[1] - current_price):
            return "upward"
        else:
            return "downward"

    def analyze(self, current_price: float):
        """Analyzes the market data and provides insights."""
        high = self.df["high"].max()
        low = self.df["low"].min()
        close = self.df["close"].iloc[-1]

        # Calculate support/resistance levels
        self.calculate_fibonacci_retracement(high, low, current_price)
        self.calculate_pivot_points(high, low, close)
        nearest_supports, nearest_resistances = self.find_nearest_levels(current_price)

        # Determine trend and momentum
        trend = self.determine_trend_momentum()

        # Calculate ATR for volatility
        atr = self.calculate_atr()

        # Predict next likely level
        next_level_prediction = self.predict_next_level(
            current_price, nearest_supports, nearest_resistances
        )

        # Log the analysis results
        self.logger.info(f"{Fore.YELLOW}Current Price:{Fore.GREEN} {current_price:.2f}")
        self.logger.info(f"{Fore.YELLOW}Trend:{Fore.CYAN} {trend}")

        if atr is not None:
            self.logger.info(f"{Fore.YELLOW}ATR:{Fore.MAGENTA} {atr.iloc[-1]:.2f}")
        else:
            self.logger.info(f"{Fore.YELLOW}ATR:{Fore.MAGENTA} None")

        self.logger.info(f"{Fore.YELLOW}Fibonacci Levels (Support/Resistance):")
        for level, value in self.levels.items():
            label = (
                f"{Fore.BLUE}Support ({level})"
                if value < current_price
                else f"{Fore.RED}Resistance ({level})"
                if value > current_price
                else level
            )
            self.logger.info(f"{label}: {Fore.CYAN} {value:.2f}")

        self.logger.info(f"{Fore.YELLOW}Nearest Support Levels:")
        for level, value in nearest_supports:
            self.logger.info(f"{Fore.BLUE} {level}: {Fore.GREEN} {value:.2f}")

        self.logger.info(f"{Fore.YELLOW}Nearest Resistance Levels:")
        for level, value in nearest_resistances:
            self.logger.info(f"{Fore.RED} {level}: {Fore.BLUE} {value:.2f}")

        self.logger.info(f"{Fore.YELLOW}Prediction:{Fore.MAGENTA} {next_level_prediction}")

# --- Logging Setup ---
def setup_logger(name: str) -> logging.Logger:
    """Sets up a logger with file and stream handlers."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(f"{Fore.BLUE}%(asctime)s{Fore.RESET} - %(levelname)s - %(message)s")
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

            df = bybit.fetch_klines(symbol, interval, limit=200)  # Fetch 200 candles
            if df.empty:
                time.sleep(30)  # Wait before retrying
                continue

            analyzer = TechnicalAnalyzer(df, logger)
            analyzer.analyze(current_price)

            time.sleep(30)  # Adjust analysis frequency as needed

    except Exception as e:
        logger.exception(f"{Fore.RED}An error occurred: {e}")

if __name__ == "__main__":
    main()