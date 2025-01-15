import os
import logging
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import hmac
import hashlib
import time
from typing import Dict, Tuple, List
from colorama import init, Fore, Style
from zoneinfo import ZoneInfo

# Initialize colorama
init(autoreset=True)

# Load env vars
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

# Constants
LOG_DIR = "botlogs"
os.makedirs(LOG_DIR, exist_ok=True)
ST_LOUIS_TZ = ZoneInfo("America/Chicago")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 5))
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]


# --- Helper Functions ---
def generate_signature(params: dict) -> str:
    """Generate HMAC SHA256 signature."""
    param_str = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
    return hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()


def setup_logger(symbol: str) -> logging.Logger:
    """Set up a logger for the given symbol."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"{symbol}_{timestamp}.log")

    logger = logging.getLogger(symbol)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Stream handler (console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            Fore.BLUE + "%(asctime)s" + Fore.RESET + " - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(stream_handler)

    return logger


def safe_json_response(response: requests.Response, logger=None) -> dict | None:
    """Safely parse JSON response."""
    try:
        return response.json()
    except ValueError:
        if logger:
            logger.error(Fore.RED + f"Invalid JSON response: {response.text}")
        return None


def bybit_request(method: str, endpoint: str, params: dict = None, logger=None) -> dict:
    """Send a signed request to the Bybit Unified Trading API."""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            params = params or {}
            params["api_key"] = API_KEY
            params["timestamp"] = str(int(datetime.now(ST_LOUIS_TZ).timestamp() * 1000))
            params["sign"] = generate_signature(params)

            url = f"{BASE_URL}{endpoint}"
            response = requests.request(method, url, params=params)

            if response.status_code in RETRY_ERROR_CODES:
                if logger:
                    logger.warning(
                        Fore.YELLOW
                        + f"Rate limited or server error. Retrying after {RETRY_DELAY} seconds."
                    )
                time.sleep(RETRY_DELAY * (2**retries))  # Exponential backoff
                retries += 1
                continue

            if response.status_code != 200:
                if logger:
                    logger.error(
                        Fore.RED
                        + f"Non-200 response from Bybit: {response.status_code} - {response.text}"
                    )
                return {"retCode": -1, "retMsg": f"HTTP {response.status_code}"}

            json_response = safe_json_response(response, logger)
            if not json_response:
                return {"retCode": -1, "retMsg": "Invalid JSON"}

            return json_response

        except requests.RequestException as e:
            if logger:
                logger.error(Fore.RED + f"API request failed: {e}")
            retries += 1
            time.sleep(RETRY_DELAY * (2**retries))

        except Exception as e:
            if logger:
                logger.error(Fore.RED + f"An unexpected error occurred: {e}")
            return {"retCode": -1, "retMsg": "Unexpected Error"}

    return {"retCode": -1, "retMsg": "Max retries exceeded"}


def fetch_klines(
    symbol: str, interval: str, limit: int = 200, logger=None
) -> pd.DataFrame:
    """Fetch historical kline (candlestick) data from Bybit Unified Trading API."""
    endpoint = "/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "category": "linear",
    }
    response = bybit_request("GET", endpoint, params, logger)
    if response.get("retCode") == 0 and response.get("result"):
        data = response["result"]["list"]
        columns = ["start_time", "open", "high", "low", "close", "volume"]
        # Check if "turnover" column exists in the response
        if (
            response["result"]["list"]
            and len(response["result"]["list"][0]) > 6
            and response["result"]["list"][0][6]
        ):
            columns.append("turnover")
        df = pd.DataFrame(data, columns=columns)
        df["start_time"] = pd.to_datetime(
            pd.to_numeric(df["start_time"], errors="coerce"), unit="ms"
        )
        df = df.astype({col: float for col in columns if col != "start_time"})

        # Data Validation
        required_dtypes = {col: float for col in columns if col != "start_time"}
        for col, dtype in required_dtypes.items():
            if df[col].dtype != dtype:
                if logger:
                    logger.error(
                        Fore.RED
                        + f"Data validation failed column: {col} type {df[col].dtype} and should be {dtype}"
                    )
                return pd.DataFrame()  # Return empty dataframe to stop processing

        return df
    else:
        if logger:
            logger.error(
                Fore.RED + f"Failed to fetch klines for {symbol}. Response: {response}"
            )
        return pd.DataFrame()


def fetch_current_price(symbol: str, logger=None) -> float:
    """Fetch the current price for the given symbol from Bybit."""
    endpoint = "/v5/market/tickers"
    params = {"symbol": symbol, "category": "linear"}
    response = bybit_request("GET", endpoint, params, logger)
    if response.get("retCode") == 0 and response.get("result"):
        try:
            current_price = float(response["result"]["list"][0]["lastPrice"])
            if not isinstance(current_price, float) or current_price <= 0:
                if logger:
                    logger.error(
                        Fore.RED + f"Invalid current price received: {current_price}"
                    )
                return None
            return current_price
        except (KeyError, IndexError, ValueError) as e:
            if logger:
                logger.error(
                    Fore.RED
                    + f"Failed to extract current price from response {response}: {e}"
                )
            return None
    else:
        if logger:
            logger.error(
                Fore.RED
                + f"Failed to fetch current price for {symbol}. Response: {response}"
            )
        return None


# --- TradingAnalyzer Class ---
class TradingAnalyzer:
    def __init__(self, df: pd.DataFrame, logger: logging.Logger):
        self.df = df
        self.logger = logger
        self.levels = {}  # Unified levels dictionary
        self.fib_levels = {}

    def calculate_sma(self, window: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        try:
            return self.df["close"].rolling(window=window).mean()
        except KeyError:
            self.logger.error(
                Fore.RED + "Missing required 'close' column to calculate SMA"
            )
            return pd.Series(dtype="float64")

    def calculate_momentum(self, period: int = 7) -> pd.Series:
        """Calculate momentum using Rate of Change (ROC)."""
        try:
            return (
                (self.df["close"] - self.df["close"].shift(period))
                / self.df["close"].shift(period)
            ) * 100
        except KeyError as e:
            self.logger.error(Fore.RED + f"Error calculating momentum: {e}")
            return pd.Series(dtype="float64")
        except ZeroDivisionError:
            self.logger.error(Fore.RED + "Error calculating momentum: Division by zero")
            return pd.Series(dtype="float64")

    def calculate_fibonacci_retracement(
        self, high: float, low: float, current_price: float
    ) -> Dict[str, float]:
        """
        Calculates Fibonacci retracement levels and updates support/resistance in self.levels.
        Uses the current price in the calculation.
        """
        try:
            diff = high - low
            if diff == 0:
                return {}

            # Calculate Fibonacci levels based on high and low
            fib_levels = {
                "Fib 23.6%": high - diff * 0.236,
                "Fib 38.2%": high - diff * 0.382,
                "Fib 50.0%": high - diff * 0.5,
                "Fib 61.8%": high - diff * 0.618,
                "Fib 78.6%": high - diff * 0.786,
            }

            # Classify each level as support or resistance based on current price
            for label, value in fib_levels.items():
                if value < current_price:
                    # Add a prefix to the level label to indicate support
                    support_label = f"Support ({label})"
                    self.levels[support_label] = value
                elif value > current_price:
                    # Add a prefix to the level label to indicate resistance
                    resistance_label = f"Resistance ({label})"
                    self.levels[resistance_label] = value

            self.fib_levels = (
                fib_levels  # Update self.fib_levels with calculated levels
            )
            return self.fib_levels
        except Exception as e:
            self.logger.error(Fore.RED + f"Error calculating Fibonacci levels: {e}")
            return {}

    def calculate_pivot_points(self, high: float, low: float, close: float):
        """Calculate pivot points, supports, and resistances."""
        try:
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)

            self.levels.update(
                {
                    "pivot": pivot,
                    "r1": r1,
                    "s1": s1,
                    "r2": r2,
                    "s2": s2,
                    "r3": r3,
                    "s3": s3,
                }
            )
        except Exception as e:
            self.logger.error(Fore.RED + f"Error calculating pivot points: {e}")
            self.levels = {}

    def find_nearest_levels(
        self, current_price: float, num_levels: int = 5
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Find the nearest support and resistance levels to the current price."""
        try:
            support_levels = [
                (label, value)
                for label, value in self.levels.items()
                if value < current_price
            ]
            resistance_levels = [
                (label, value)
                for label, value in self.levels.items()
                if value > current_price
            ]

            nearest_supports = sorted(
                support_levels, key=lambda x: abs(x[1] - current_price), reverse=True
            )[-num_levels:]
            nearest_supports = sorted(nearest_supports, key=lambda x: x[1])

            nearest_resistances = sorted(
                resistance_levels, key=lambda x: abs(x[1] - current_price)
            )[:num_levels]
            nearest_resistances = sorted(nearest_resistances, key=lambda x: x[1])

            return nearest_supports, nearest_resistances
        except Exception as e:
            self.logger.error(Fore.RED + f"Error finding nearest levels: {e}")
            return [], []

    def calculate_atr(self, window: int = 14) -> pd.Series:
        """Calculates the Average True Range (ATR)."""
        try:
            high_low = self.df["high"] - self.df["low"]
            high_close = (self.df["high"] - self.df["close"].shift()).abs()
            low_close = (self.df["low"] - self.df["close"].shift()).abs()

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            return atr
        except KeyError as e:
            self.logger.error(Fore.RED + f"Error calculating ATR: {e}")
            return pd.Series(dtype="float64")

    def calculate_momentum_ma(self) -> None:
        """Calculate momentum and its moving averages."""
        self.df["momentum"] = self.df["close"].diff(10)
        self.df["momentum_ma_short"] = self.df["momentum"].rolling(window=12).mean()
        self.df["momentum_ma_long"] = self.df["momentum"].rolling(window=26).mean()
        self.df["volume_ma"] = self.df["volume"].rolling(window=20).mean()

    def calculate_macd(self) -> pd.DataFrame:
        """Calculate MACD and signal line"""
        try:
            close_prices = self.df["close"]
            ma_short = close_prices.ewm(span=12, adjust=False).mean()
            ma_long = close_prices.ewm(span=26, adjust=False).mean()
            macd = ma_short - ma_long
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            return pd.DataFrame(
                {"macd": macd, "signal": signal, "histogram": histogram}
            )
        except KeyError:
            self.logger.error(Fore.RED + "Missing the close column to calculate MACD.")
            return pd.DataFrame()

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
            "trend_strength_threshold": 0.6,  # Increased sensitivity
            "sideways_atr_multiplier": 1.5,  # Reduced for more sensitivity
            "user_defined_weights": {
                "ema_alignment": 0.4,  # Increased weight
                "momentum": 0.3,
                "volume_confirmation": 0.2,  # Increased weight
                "divergence": 0.1,
            },
        }

        if atr > 25:  # High volatility
            base_params.update(
                {
                    "momentum_period": 15,
                    "momentum_ma_short": 15,
                    "momentum_ma_long": 30,
                    "trend_strength_threshold": 0.7,
                    "user_defined_weights": {
                        "ema_alignment": 0.45,
                        "momentum": 0.25,
                        "volume_confirmation": 0.2,
                        "divergence": 0.1,
                    },
                }
            )
        elif atr > 10:  # Medium volatility
            base_params.update(
                {
                    "trend_strength_threshold": 0.65,
                    "user_defined_weights": {
                        "ema_alignment": 0.4,
                        "momentum": 0.3,
                        "volume_confirmation": 0.25,
                        "divergence": 0.05,
                    },
                }
            )

        return base_params

    def calculate_ema(self, window: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        try:
            return self.df["close"].ewm(span=window, adjust=False).mean()
        except KeyError:
            self.logger.error(
                Fore.RED + "Missing required 'close' column to calculate EMA"
            )
            return pd.Series(dtype="float64")

    def determine_trend_momentum(self) -> dict:
        """
        Determine the trend using a more sensitive approach with EMA alignment, momentum, and volume.
        """
        if self.df.empty or len(self.df) < 26:
            return {"trend": "Insufficient Data", "strength": 0}

        atr = self.calculate_atr()
        dynamic_params = self.adapt_parameters(atr.iloc[-1])

        # Calculate EMAs
        ema_short = self.calculate_ema(dynamic_params["momentum_ma_short"])
        ema_mid = self.calculate_ema(dynamic_params["momentum_period"])
        ema_long = self.calculate_ema(dynamic_params["momentum_ma_long"])

        # Calculate Momentum
        self.df["momentum"] = self.df["close"].diff(dynamic_params["momentum_period"])
        momentum = self.df["momentum"].rolling(window=3).mean().iloc[-1]

        # Volume Confirmation
        volume_ma = (
            self.df["volume"].rolling(window=dynamic_params["volume_ma_period"]).mean()
        )
        volume_confirmation = (
            self.df["volume"].iloc[-1] > volume_ma.iloc[-1] * 1.25
        )  # Increased sensitivity

        # MACD Divergence
        divergence = self.detect_macd_divergence()

        # Sideways Market Detection
        atr_range = atr.iloc[-1] * dynamic_params["sideways_atr_multiplier"]
        is_sideways = (
            abs(self.df["high"].iloc[-1] - self.df["low"].iloc[-1]) < atr_range
        )

        # Trend Scoring
        bullish_score = 0
        bearish_score = 0
        user_defined_weights = dynamic_params["user_defined_weights"]

        # EMA Alignment
        if ema_short.iloc[-1] > ema_mid.iloc[-1] > ema_long.iloc[-1]:
            bullish_score += user_defined_weights["ema_alignment"]
        elif ema_short.iloc[-1] < ema_mid.iloc[-1] < ema_long.iloc[-1]:
            bearish_score += user_defined_weights["ema_alignment"]

        # Momentum
        if momentum > 0:
            bullish_score += user_defined_weights["momentum"]
        elif momentum < 0:
            bearish_score += user_defined_weights["momentum"]

        # Volume Confirmation
        if volume_confirmation:
            if momentum > 0:
                bullish_score += user_defined_weights["volume_confirmation"]
            else:
                bearish_score += user_defined_weights["volume_confirmation"]

        # Divergence
        if divergence == "bullish":
            bullish_score += user_defined_weights["divergence"]
        elif divergence == "bearish":
            bearish_score += user_defined_weights["divergence"]

        # Determine Trend
        trend_strength_threshold = dynamic_params["trend_strength_threshold"]
        if is_sideways:
            return {"trend": "Sideways", "strength": 0}
        elif (
            bullish_score > bearish_score and bullish_score >= trend_strength_threshold
        ):
            return {"trend": "Uptrend", "strength": bullish_score}
        elif (
            bearish_score > bullish_score and bearish_score >= trend_strength_threshold
        ):
            return {"trend": "Downtrend", "strength": bearish_score}
        else:
            return {"trend": "Neutral", "strength": max(bullish_score, bearish_score)}

    def predict_next_level(
        self,
        current_price: float,
        nearest_supports: List[Tuple[str, float]],
        nearest_resistances: List[Tuple[str, float]],
    ) -> str:
        """Predict the most likely next support or resistance level the price is moving towards."""
        if not nearest_supports or not nearest_resistances:
            return "No clear prediction"

        closest_support = min(nearest_supports, key=lambda x: abs(x[1] - current_price))
        closest_resistance = min(
            nearest_resistances, key=lambda x: abs(x[1] - current_price)
        )

        if abs(closest_support[1] - current_price) < abs(
            closest_resistance[1] - current_price
        ):
            return f"Likely headed towards support at {closest_support[0]}: {closest_support[1]:.2f}"
        else:
            return f"Likely headed towards resistance at {closest_resistance[0]}: {closest_resistance[1]:.2f}"

    def analyze(self, current_price: float):
        """
        Perform comprehensive analysis and log the results.
        Fibonacci levels are now calculated using the current price.
        """
        high = self.df["high"].max()
        low = self.df["low"].min()
        close = self.df["close"].iloc[
            -1
        ]  # Use the last closing price for pivot calculation

        # Calculate Fibonacci levels using current price and update support/resistance levels
        self.calculate_fibonacci_retracement(high, low, current_price)
        self.calculate_pivot_points(high, low, close)
        nearest_supports, nearest_resistances = self.find_nearest_levels(current_price)
        trend_data = self.determine_trend_momentum()
        trend = trend_data.get(
            "trend", "Unknown"
        )  # Fallback to unknown if there is not a trend
        strength = trend_data.get("strength", 0)
        atr = self.calculate_atr()
        next_level_prediction = self.predict_next_level(
            current_price, nearest_supports, nearest_resistances
        )

        self.logger.info(
            f"{Fore.YELLOW}Current Price: {Fore.GREEN}{current_price:.2f}{Style.RESET_ALL}"
        )
        self.logger.info(
            f"{Fore.YELLOW}Trend: {Fore.CYAN}{trend}{Style.RESET_ALL} with strength: {Fore.CYAN}{strength}{Style.RESET_ALL}"
        )
        self.logger.info(
            f"{Fore.YELLOW}ATR: {Fore.MAGENTA}{atr.iloc[-1]:.2f}{Style.RESET_ALL}"
        )

        # Log Fibonacci levels with their support/resistance labels
        self.logger.info(
            f"{Fore.YELLOW}Fibonacci Levels (Support/Resistance):{Style.RESET_ALL}"
        )
        for level, value in self.fib_levels.items():
            # Determine if the level is support or resistance based on current price
            if value < current_price:
                label = f"{Fore.BLUE}Support ({level})"  # Blue for support
            elif value > current_price:
                label = f"{Fore.RED}Resistance ({level})"  # Red for resistance
            else:
                label = level  # No special color if equal to current price
            self.logger.info(f"{label}: {Fore.CYAN}{value:.2f}")

        self.logger.info(f"{Fore.YELLOW}Nearest Support Levels:{Style.RESET_ALL}")
        for level, value in nearest_supports:
            self.logger.info(f"{Fore.BLUE}   {level}: {Fore.GREEN}{value:.2f}")

        self.logger.info(f"{Fore.YELLOW}Nearest Resistance Levels:{Style.RESET_ALL}")
        for level, value in nearest_resistances:
            self.logger.info(f"{Fore.RED}   {level}: {Fore.BLUE}{value:.2f}")
        self.logger.info(
            f"{Fore.YELLOW}Prediction: {Fore.MAGENTA}{next_level_prediction}{Style.RESET_ALL}"
        )


# --- Main Function ---
def main():
    # Validate API credentials
    if not API_KEY or not API_SECRET:
        print(
            Fore.RED
            + "API keys not set. Please ensure BYBIT_API_KEY and BYBIT_API_SECRET are defined in a .env file."
            + Style.RESET_ALL
        )
        exit(1)

    symbol = input(
        Fore.CYAN + "Enter trading symbol (e.g., BTCUSDT): " + Style.RESET_ALL
    ).upper()
    interval = input(
        Fore.CYAN
        + f"Enter timeframe (e.g., {', '.join(VALID_INTERVALS)}): "
        + Style.RESET_ALL
    )
    if interval not in VALID_INTERVALS:
        print(Fore.RED + f"Invalid interval selected {interval}")
        exit(1)

    logger = setup_logger(symbol)

    while True:
        current_price = fetch_current_price(symbol, logger)  # Pass logger
        if current_price is None:
            time.sleep(30)  # Wait before retrying
            continue  # Skip to next iteration if price is none

        df = fetch_klines(symbol, interval, logger=logger)  # Pass logger
        if df.empty:
            logger.error(Fore.RED + "Failed to fetch data.")
            time.sleep(30)
            continue

        analyzer = TradingAnalyzer(df, logger)
        analyzer.analyze(current_price)
        time.sleep(30)


if __name__ == "__main__":
    main()
