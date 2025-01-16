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
import talib as ta

# Initialize colorama
init(autoreset = True)

# Load env vars
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

# Constants
LOG_DIR = "botlogs"
os.makedirs(LOG_DIR, exist_ok = True)
ST_LOUIS_TZ = ZoneInfo("America/Chicago")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 5))
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]

# --- Helper Functions ---
def generate_signature(params: dict) -> str:
 """Generate HMAC SHA256 signature."""
param_str = "&".join(f" {
    key
}= {
    value
}" for key, value in sorted(params.items()))
return hmac.new(
    API_SECRET.encode(), param_str.encode(), hashlib.sha256
).hexdigest()

def setup_logger(symbol: str) -> logging.Logger:
 """Set up a logger for the given symbol."""
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(LOG_DIR, f" {
    symbol
}_ {
    timestamp
}.log")

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

def safe_json_response(response: requests.Response, logger = None) -> dict | None:
 """Safely parse JSON response."""
try:
return response.json()
except ValueError:
if logger:
logger.error(
    Fore.RED + f"Invalid JSON response: {
        response.text
    }"
)
return None

def bybit_request(method: str, endpoint: str, params: dict = None, logger = None) -> dict:
 """Send a signed request to the Bybit Unified Trading API."""
retries = 0
while retries < MAX_RETRIES:
try:
params = params or {}
params["api_key"] = API_KEY
params["timestamp"] = str(int(datetime.now(ST_LOUIS_TZ).timestamp() * 1000))
params["sign"] = generate_signature(params)

url = f" {
    BASE_URL
} {
    endpoint
}"
response = requests.request(method, url, params = params)

if response.status_code in RETRY_ERROR_CODES:
if logger:
logger.warning(
    Fore.YELLOW
    + f"Rate limited or server error. Retrying after {
        RETRY_DELAY
    } seconds."
)
time.sleep(RETRY_DELAY * (2 ** retries)) # Exponential backoff
retries += 1
continue

if response.status_code != 200:
if logger:
logger.error(
    Fore.RED
    + f"Non-200 response from Bybit: {
        response.status_code
    } - {
        response.text
    }"
)
return {
    "retCode": -1, "retMsg": f"HTTP {
        response.status_code
    }"
}

json_response = safe_json_response(response, logger)
if not json_response:
return {
    "retCode": -1, "retMsg": "Invalid JSON"
}

return json_response

except requests.RequestException as e:
if logger:
logger.error(Fore.RED + f"API request failed: {
    e
}")
retries += 1
time.sleep(RETRY_DELAY * (2 ** retries))

except Exception as e:
if logger:
logger.error(Fore.RED + f"An unexpected error occurred: {
    e
}")
return {
    "retCode": -1, "retMsg": "Unexpected Error"
}

return {
    "retCode": -1, "retMsg": "Max retries exceeded"
}

def fetch_klines(symbol: str, interval: str, limit: int = 200, logger = None) -> pd.DataFrame:
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
    response["result"]["list"] and len(response["result"]["list"][0]) > 6 and response["result"]["list"][0][6]
):
columns.append("turnover")
df = pd.DataFrame(data, columns = columns)
df["start_time"] = pd.to_datetime(
    pd.to_numeric(df["start_time"], errors = "coerce"), unit = "ms"
)
df = df.astype({
    col: float for col in columns if col != "start_time"
})

# Data Validation
required_dtypes = {
    col: float for col in columns if col != "start_time"
}
for col, dtype in required_dtypes.items():
if df[col].dtype != dtype:
if logger:
logger.error(
    Fore.RED
    + f"Data validation failed column: {
        col
    } type {
        df[col].dtype
    } and should be {
        dtype
    }"
)
return pd.DataFrame() # Return empty dataframe to stop processing

return df
else :
if logger:
logger.error(
    Fore.RED + f"Failed to fetch klines for {
        symbol
    }. Response: {
        response
    }"
)
return pd.DataFrame()

def fetch_current_price(symbol: str, logger = None) -> float:
 """Fetch the current price for the given symbol from Bybit."""
endpoint = "/v5/market/tickers"
params = {
    "symbol": symbol, "category": "linear"
}
response = bybit_request("GET", endpoint, params, logger)
if response.get("retCode") == 0 and response.get("result"):
try:
current_price = float(response["result"]["list"][0]["lastPrice"])
if not isinstance(current_price, float) or current_price <= 0:
if logger:
logger.error(
    Fore.RED + f"Invalid current price received: {
        current_price
    }"
)
return None
return current_price
except (KeyError, IndexError, ValueError) as e:
if logger:
logger.error(
    Fore.RED
    + f"Failed to extract current price from response {
        response
    }: {
        e
    }"
)
return None
else :
if logger:
logger.error(
    Fore.RED
    + f"Failed to fetch current price for {
        symbol
    }. Response: {
        response
    }"
)
return None

# --- TradingAnalyzer Class ---
class TradingAnalyzer:
 """
    A class to perform technical analysis on financial data using pandas DataFrames and TA-Lib.
    """

def __init__(self, df: pd.DataFrame, logger: logging.Logger):
 """
        Initializes the TradingAnalyzer with a pandas DataFrame and a logger.
        Args:
          df (pd.DataFrame): DataFrame containing trading data with at least 'high', 'low', 'close', and 'volume' columns.
          logger (logging.Logger): Logger instance for logging.
        Raises:
          ValueError: If the DataFrame does not have essential columns ('high', 'low', 'close', 'volume').
          TypeError: If df is not a pandas DataFrame or if logger is not logging Logger
        """
if not isinstance(df, pd.DataFrame):
raise TypeError("df must be pandas DataFrame")
if not isinstance(logger, logging.Logger):
raise TypeError("logger must be a logging.Logger object")
required_cols = ['high', 'low', 'close', 'volume']
if not all(col in df.columns for col in required_cols):
raise ValueError(f"DataFrame must contain {
    required_cols
}")
self.df = df.copy() # Work with a copy to avoid modifying the original
self.logger = logger
self.logger.info("TradingAnalyzer initialized.")
self.levels = {} # Unified levels dictionary
self.fib_levels = {}

def _validate_window(self, window: int) -> None:
"""
        Validates window to be a positive int and not greater than the length of df.
        Args:
            window (int): The length of the window to evaluate over
        Raises:
            ValueError: If window is not a positive int or if greater than the data length.
            TypeError: If window is not an int
        """
if not isinstance(window, int):
raise TypeError("window must be an int")
if window <= 0:
raise ValueError("window must be positive")
if window > len(self.df):
raise ValueError(f"window ( {
    window
}) must be less than or equal to length of DataFrame ( {
    len(self.df)})")

def calculate_sma(self, window: int) -> pd.Series:
 """
        Calculates the Simple Moving Average.
        Args:
            window (int): The window period for the SMA calculation.
        Returns:
            pandas.Series: The SMA values as a pandas Series.
        Raises:
            ValueError: If invalid window size.
        """
try:
self._validate_window(window)
sma = ta.SMA(self.df['close'], timeperiod = window)
self.logger.debug(f"SMA calculated with window {
    window
}.")
return sma
except Exception as e:
self.logger.error(f"Error calculating SMA: {
    e
}")
raise # Re-raise the exception after logging

def calculate_ema(self, window: int) -> pd.Series:
 """
        Calculates the Exponential Moving Average.
        Args:
            window (int): The window period for EMA calculation.
        Returns:
             pandas.Series: The EMA values as pandas Series.
        Raises:
             ValueError: If invalid window size.
        """
try:
self._validate_window(window)
ema = ta.EMA(self.df['close'], timeperiod = window)
self.logger.debug(f"EMA calculated with window {
    window
}.")
return ema
except Exception as e:
self.logger.error(f"Error calculating EMA: {
    e
}")
raise # Re-raise the exception after logging

def calculate_momentum(self, period: int = 7) -> pd.Series:
 """
        Calculates price momentum.
        Args:
            period (int): The lookback period for momentum calculation.
        Returns:
            pandas.Series: The momentum values.
        Raises:
            ValueError: If period is not a positive integer, or if out of bounds
        """
try:
if not isinstance(period, int):
raise TypeError("Period must be an integer")
if period <= 0:
raise ValueError("Period must be positive")
if period >= len(self.df):
raise ValueError(f"Period ( {
    period
}) must be less than length of DataFrame ( {
    len(self.df)})")

momentum = ta.MOM(self.df['close'], timeperiod = period)
self.logger.debug(f"Momentum calculated with period: {
    period
}.")
return momentum
except Exception as e:
self.logger.error(f"Error calculating momentum: {
    e
}")
raise # Re-raise the exception after logging

def calculate_fibonacci_retracement(self, high: float, low: float, current_price: float) -> Dict[str, float]:
 """
        Calculates Fibonacci retracement levels using the Camarilla method for tighter levels
        and more significant retracements, enhancing dynamic support/resistance identification.
        """
try:
diff = high - low
if diff == 0:
return {}

# Extended Fibonacci levels for a more comprehensive analysis
fib_levels = {
    "Fib 161.8%": high + diff * 1.618,
    "Fib 100.0%": high, # Typically the high, acting as resistance
    "Fib 78.6%": high - diff * 0.786,
    "Fib 61.8%": high - diff * 0.618,
    "Fib 50.0%": high - diff * 0.5,
    "Fib 38.2%": high - diff * 0.382,
    "Fib 23.6%": high - diff * 0.236,
    "Fib 0.0%": low, # Typically the low, acting as support
}

# Classify each level as support or resistance based on current price
for label, value in fib_levels.items():
if value <= current_price:
support_label = f"Support ( {
    label
})"
self.levels[support_label] = value
elif value > current_price:
resistance_label = f"Resistance ( {
    label
})"
self.levels[resistance_label] = value

self.fib_levels = fib_levels
return self.fib_levels
except Exception as e:
self.logger.error(Fore.RED + f"Error calculating Fibonacci levels: {
    e
}")
return {}

def calculate_pivot_points(self, high: float, low: float, close: float) -> None:
 """
        Calculate pivot points using the Camarilla method for tighter levels, enhancing
        support and resistance identification.
        """
try:
pivot_range = high - low

# Camarilla pivot point calculations for tighter levels
pivot = (high + low + close) / 3
r1 = close + pivot_range * 1.1 / 12
s1 = close - pivot_range * 1.1 / 12
r2 = close + pivot_range * 1.1 / 6
s2 = close - pivot_range * 1.1 / 6
r3 = close + pivot_range * 1.1 / 4
s3 = close - pivot_range * 1.1 / 4
r4 = close + pivot_range * 1.1 / 2
s4 = close - pivot_range * 1.1 / 2

# Update levels with Camarilla pivot points
self.levels.update({
    "Pivot": pivot,
    "R1 (Camarilla)": r1,
    "S1 (Camarilla)": s1,
    "R2 (Camarilla)": r2,
    "S2 (Camarilla)": s2,
    "R3 (Camarilla)": r3,
    "S3 (Camarilla)": s3,
    "R4 (Camarilla)": r4,
    "S4 (Camarilla)": s4,
})
except Exception as e:
self.logger.error(Fore.RED + f"Error calculating Camarilla pivot points: {
    e
}")
self.levels.clear() # Clear levels on error to avoid partial updates

def find_nearest_levels(self, current_price: float, num_levels: int = 5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
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
    support_levels, key = lambda x: abs(x[1] - current_price), reverse = True
)[-num_levels:]
nearest_supports = sorted(nearest_supports, key = lambda x: x[1])

nearest_resistances = sorted(
    resistance_levels, key = lambda x: abs(x[1] - current_price)
)[:num_levels]
nearest_resistances = sorted(nearest_resistances, key = lambda x: x[1])

return nearest_supports, nearest_resistances
except Exception as e:
def adapt_parameters(self, atr: float) -> Dict:
 """
        Adapts trading parameters based on Average True Range (ATR) for volatility adjustments.
        Args:
           atr (float): The average true range.
        Returns:
           Dict: Dictionary of adaptive trading parameters.
        Raises:
            ValueError: if atr not a float or a positive number
        """
try:
if not isinstance(atr, float):
raise TypeError("atr must be a float")
if atr <= 0:
raise ValueError("atr must be positive")
stop_loss_factor = 2 # default
take_profit_factor = 3 # default

if atr < 0.5:
stop_loss_factor = 1.5
take_profit_factor = 2.0
elif atr > 1.0:
stop_loss_factor = 2.5
take_profit_factor = 3.5
adapted_params = {
    'stop_loss_factor': stop_loss_factor,
    'take_profit_factor': take_profit_factor
}
self.logger.debug(f"Parameters adapted with atr: {
    atr
} with return params of: {
    adapted_params
}")
return adapted_params
except (TypeError, ValueError) as e:
self.logger.error(f"Error adapting parameters: {
    e
}")
return {} # Return an empty dictionary or some default value on error
except Exception as e:
self.logger.error(f"An unexpected error occurred: {
    e
}")
return {}
self.logger.error(Fore.RED + f"Error finding nearest levels: {
    e
}")
return [], []

def calculate_atr(self, window: int = 14) -> pd.Series:
 """
        Calculates the Average True Range (ATR).
        Args:
           window (int): The window size for ATR calculation.
        Returns:
           pandas.Series: The ATR values
        Raises:
           ValueError: If invalid window size
        """
try:
self._validate_window(window)
atr = ta.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod = window)
self.logger.debug(f"ATR calculated with window {
    window
}")
return atr
except Exception as e:
self.logger.error(f"Error calculating ATR: {
    e
}")
raise # Re-raise exception

def calculate_macd(self, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
 """
        Calculates the Moving Average Convergence Divergence (MACD).
        Args:
            short_window (int): The short window for MACD calculation.
            long_window (int): The long window for MACD calculation.
            signal_window (int): The signal window for MACD calculation.
        Returns:
           pandas.DataFrame: DataFrame containing MACD, signal, and histogram.
        Raises:
           ValueError: If short_window, long_window, or signal_window are not positive integers or if they are not reasonable sizes relative to the dataframe
        """
try:
if not all(isinstance(win, int) for win in [short_window, long_window, signal_window]):
raise TypeError("short_window, long_window and signal_window must be integers")

if not all(win > 0 for win in [short_window, long_window, signal_window]):
raise ValueError("short_window, long_window, and signal_window must be positive integers")
if not short_window < long_window:
raise ValueError("short_window must be less than long_window")
if short_window > len(self.df) or long_window > len(self.df):
raise ValueError("short_window and long_window can not exceed length of the DataFrame.")

macd, signal, hist = ta.MACD(self.df['close'], fastperiod = short_window, slowperiod = long_window, signalperiod = signal_window)
macd_df = pd.DataFrame({
    'MACD': macd,
    'Signal': signal,
    'Histogram': hist
})
self.logger.debug(f"MACD calculated with short_window= {
    short_window
}, long_window= {
    long_window
}, signal_window= {
    signal_window
}.")
return macd_df
except Exception as e:
self.logger.error(f"Error calculating MACD: {
    e
}")
raise # Re-raise the exception after logging

def detect_macd_divergence(self) -> str:
 """
        Detects MACD divergence, identifying potential trend reversals based on the last 3 macd points.
        Returns:
           str: String of bullish or bearish if detected otherwise null.
        """
try:
macd_df = self.calculate_macd()
if len(macd_df) < 3:
self.logger.warning(f"Not enough data to detect macd divergence")
return None
close_prices = self.df['close'].tail(3).tolist()
macd_values = macd_df['MACD'].tail(3).tolist()
if not macd_values:
self.logger.warning(f"Not enough data for MACD in dataframe")
return None
# Check for bullish divergence: price makes a lower low, MACD makes a higher low
if close_prices[0] > close_prices[1] < close_prices[2] and macd_values[0] < macd_values[1] > macd_values[2]:
self.logger.info("MACD bullish divergence detected.")
return "bullish"
# Check for bearish divergence: price makes a higher high, MACD makes a lower high
elif close_prices[0] < close_prices[1] > close_prices[2] and macd_values[0] > macd_values[1] < macd_values[2]:
self.logger.info("MACD bearish divergence detected.")
return "bearish"
else :
self.logger.debug("No MACD divergence detected")
return None


def determine_trend_momentum(self) -> dict:
 """
        Determine trend direction based on momentum and moving average.
        Returns:
           dict: dictionary of direction based on momentum and moving averages.
        """
try:
momentum = self.calculate_momentum()
sma_short = self.calculate_sma(5) # short_term
sma_long = self.calculate_sma(20) # long-term

if momentum is None or sma_short is None or sma_long is None:
self.logger.warning("Unable to determine trend as not enough data for momentum or sma")
return None
if momentum.iloc[-1] > 0 and sma_short.iloc[-1] > sma_long.iloc[-1]:
self.logger.debug("Overall trend is upward")
return {
    "direction": "upward"
}
elif momentum.iloc[-1] < 0 and sma_short.iloc[-1] < sma_long.iloc[-1]:
self.logger.debug("Overall trend is downward")
return {
    "direction": "downward"
} else :
self.logger.debug("Overall trend is neutral")
return {
    "direction": "neutral"
}
except Exception as e:
self.logger.error(f"Error determining trend: {
    e
}")
return None

def predict_next_level(self, current_price: float, nearest_supports: List[Tuple[str, float]], nearest_resistances: List[Tuple[str, float]]) -> str:
 """
        Predicts the next probable level to be met based on distances to supports and resistances.
        Args:
            current_price (float): The current price.
            nearest_supports (List[Tuple[str,float]]): the nearest support levels to the current price.
            nearest_resistances (List[Tuple[str,float]]): The nearest resistance levels to the current price.
        Returns:
           str: Direction towards the next level to be hit.
        Raises:
           ValueError: If the inputs are incorrect or non-standard
        """
try:
if not isinstance(current_price, float):
raise TypeError("Current price must be a float")
if not isinstance(nearest_supports, list) or not isinstance(nearest_resistances, list):
raise TypeError("nearest_supports and nearest_resistances must be a list of tuples.")
if not all(isinstance(item, tuple) and len(item) == 2 for item in nearest_supports):
raise ValueError("Nearest Support must be a list of tuples.")
if not all(isinstance(item, tuple) and len(item) == 2 for item in nearest_resistances):
raise ValueError("Nearest resistance must be a list of tuples")

if not nearest_supports and not nearest_resistances:
self.logger.debug("Neither support nor resistance to predict toward")
return "neutral"

if not nearest_supports: # all resistances so predict upwards
self.logger.debug("No nearest supports so predicting up.")
return "upward"
if not nearest_resistances: # all supports so predict down
self.logger.debug("No nearest resistances so predicting down")
return "downward"
closest_resistance = min(nearest_resistances, key = lambda x: abs(x[1] - current_price))
closest_support = min(nearest_supports, key = lambda x: abs(x[1] - current_price))

if abs(closest_resistance[1] - current_price) < abs(closest_support[1] - current_price):
self.logger.debug(f"Predicting upward with target being : {
    closest_resistance[0]}: {
    closest_resistance[1]}")
return "upward"
else :
self.logger.debug(f"Predicting downward with target being : {
    closest_support[0]}: {
    closest_support[1]}")
return "downward"
except Exception as e:
self.logger.error(f"Error predicting next level: {
    e
}")
raise

def analyze(self, current_price: float):
 """
        Perform comprehensive analysis and log the results.
        Fibonacci levels are now calculated using the current price.
        """
high = self.df["high"].max()
low = self.df["low"].min()
close = self.df['close'].iloc[-1] # Use the last closing price for pivot calculation

# Calculate Fibonacci levels using current price and update support/resistance levels
self.calculate_fibonacci_retracement(high, low, current_price)
self.calculate_pivot_points(high, low, close)
nearest_supports, nearest_resistances = self.find_nearest_levels(current_price)
trend_data = self.determine_trend_momentum()
trend = trend_data.get('direction', 'Unknown') # Fallback to unknown if there is not a trend
atr = self.calculate_atr()
next_level_prediction = self.predict_next_level(current_price, nearest_supports, nearest_resistances)

self.logger.info(
f" {
    Fore.YELLOW
}Current Price: {
    Fore.GREEN
} {
    current_price:.2f
} {
    Style.RESET_ALL
}"
)
self.logger.info(f" {
    Fore.YELLOW
}Trend: {
    Fore.CYAN
} {
    trend
} {
    Style.RESET_ALL
}")
self.logger.info(f" {
    Fore.YELLOW
}ATR: {
    Fore.MAGENTA
} {
    atr.iloc[-1]:.2f
} {
    Style.RESET_ALL
}")

# Log Fibonacci levels with their support/resistance labels
self.logger.info(f" {
    Fore.YELLOW
}Fibonacci Levels (Support/Resistance): {
    Style.RESET_ALL
}")
for level, value in self.fib_levels.items():
# Determine if the level is support or resistance based on current price
if value < current_price:
label = f" {
    Fore.BLUE
}Support ( {
    level
})" # Blue for support
elif value > current_price:
label = f" {
    Fore.RED
}Resistance ( {
    level
})" # Red for resistance
else :
label = level # No special color if equal to current price
self.logger.info(f" {
    label
}: {
    Fore.CYAN
} {
    value:.2f
}")

self.logger.info(f" {
    Fore.YELLOW
}Nearest Support Levels: {
    Style.RESET_ALL
}")
for level, value in nearest_supports:
self.logger.info(f" {
    Fore.BLUE
} {
    level
}: {
    Fore.GREEN
} {
    value:.2f
}")

self.logger.info(f" {
    Fore.YELLOW
}Nearest Resistance Levels: {
    Style.RESET_ALL
}")
for level, value in nearest_resistances:
self.logger.info(f" {
    Fore.RED
} {
    level
}: {
    Fore.BLUE
} {
    value:.2f
}")
self.logger.info(
f" {
    Fore.YELLOW
}Prediction: {
    Fore.MAGENTA
} {
    next_level_prediction
} {
    Style.RESET_ALL
}"
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
    Fore.CYAN + f"Enter timeframe (e.g., {
        ', '.join(VALID_INTERVALS)}): " + Style.RESET_ALL
)
if interval not in VALID_INTERVALS:
print(Fore.RED + f"Invalid interval selected {
    interval
}")
exit(1)

logger = setup_logger(symbol)

while True:
current_price = fetch_current_price(symbol, logger) # Pass logger
if current_price is None:
time.sleep(30) # Wait before retrying
continue #Skip to next iteration if price is none

df = fetch_klines(symbol, interval, logger = logger) # Pass logger
if df.empty:
logger.error(Fore.RED + "Failed to fetch data.")
time.sleep(30)
continue

analyzer = TradingAnalyzer(df, logger)
analyzer.analyze(current_price)
time.sleep(30)

if __name__ == "__main__":
main()