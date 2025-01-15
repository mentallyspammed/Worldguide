import os
import logging
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import hmac
import hashlib
from typing import Dict
from colorama import init, Fore, Style
from zoneinfo import ZoneInfo
import time

# Initialize colorama for colored terminal output
init(autoreset=True)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
BASE_URL = "https://api.bybit.com"

# Create the `botlogs` directory if it doesn't exist
LOG_DIR = "botlogs"
os.makedirs(LOG_DIR, exist_ok=True)

# Timezone for St. Louis, Missouri
ST_LOUIS_TZ = ZoneInfo("America/Chicago")

def setup_logger(symbol: str) -> logging.Logger:
    """Set up a logger for the given symbol."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"{symbol}_{timestamp}.log")

    logger = logging.getLogger(symbol)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Stream handler (console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)

    return logger

def safe_json_response(response):
    """Safely parse JSON response."""
    try:
        return response.json()
    except ValueError:
        return None

def generate_signature(params: dict) -> str:
    """Generate HMAC SHA256 signature."""
    param_str = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
    return hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()

def bybit_request(method: str, endpoint: str, params: dict = None) -> dict:
    """Send a signed request to the Bybit Unified Trading API."""
    try:
        params = params or {}
        params["api_key"] = API_KEY
        params["timestamp"] = str(int(datetime.now(ST_LOUIS_TZ).timestamp() * 1000))
        params["sign"] = generate_signature(params)

        url = f"{BASE_URL}{endpoint}"
        response = requests.request(method, url, params=params)

        if response.status_code != 200:
            logging.error(f"Non-200 response from Bybit: {response.status_code} - {response.text}")
            return {"retCode": -1, "retMsg": f"HTTP {response.status_code}"}
        
        json_response = safe_json_response(response)
        if not json_response:
            logging.error("Invalid JSON response received.")
            return {"retCode": -1, "retMsg": "Invalid JSON"}
        
        return json_response
    except requests.RequestException as e:
        logging.error(f"API request failed: {e}")
        return {"retCode": -1, "retMsg": str(e)}

def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch historical kline (candlestick) data from Bybit Unified Trading API."""
    endpoint = "/v5/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit, "category": "linear"}
    response = bybit_request("GET", endpoint, params)
    if response.get("retCode") == 0 and response.get("result"):
        data = response["result"]["list"]
        columns = ["start_time", "open", "high", "low", "close", "volume"]
        if len(data[0]) == 7:  # If there's an additional column like "turnover"
            columns.append("turnover")
        df = pd.DataFrame(data, columns=columns)
        df["start_time"] = pd.to_datetime(pd.to_numeric(df["start_time"]), unit="ms")
        df = df.astype({col: float for col in columns if col != "start_time"})
        return df
    else:
        logging.error(f"Failed to fetch klines for {symbol}. Response: {response}")
        return pd.DataFrame()

def fetch_current_price(symbol: str) -> float:
    """Fetch the current price for the given symbol from Bybit."""
    endpoint = "/v5/market/tickers"
    params = {"symbol": symbol, "category": "linear"}
    response = bybit_request("GET", endpoint, params)
    if response.get("retCode") == 0 and response.get("result"):
        try:
            current_price = float(response["result"]["list"][0]["lastPrice"])
            return current_price
        except (KeyError, IndexError, ValueError):
            logging.error(f"Failed to extract current price from response: {response}")
            return None
    else:
        logging.error(f"Failed to fetch current price for {symbol}. Response: {response}")
        return None

class TradingAnalyzer:
    """Comprehensive class for performing technical analysis."""

    def __init__(self, close_prices: pd.Series, high_prices: pd.Series, low_prices: pd.Series, logger: logging.Logger):
        self.close_prices = close_prices
        self.high_prices = high_prices
        self.low_prices = low_prices
        self.logger = logger
        self.pivot_points = {}

    def calculate_pivot_points(self):
        """Calculate pivot points, supports, and resistances."""
        previous_close = self.close_prices.iloc[-1]
        previous_high = self.high_prices.iloc[-1]
        previous_low = self.low_prices.iloc[-1]

        pivot = (previous_high + previous_low + previous_close) / 3
        self.pivot_points = {
            "pivot": pivot,
            "r1": (2 * pivot) - previous_low,
            "s1": (2 * pivot) - previous_high,
            "r2": pivot + (previous_high - previous_low),
            "s2": pivot - (previous_high - previous_low),
        }

    def find_nearest_levels(self, current_price: float) -> str:
        """Identify the next significant support or resistance."""
        self.calculate_pivot_points()
        supports = [v for k, v in self.pivot_points.items() if k.startswith('s')]
        resistances = [v for k, v in self.pivot_points.items() if k.startswith('r')]

        next_support = min((s for s in supports if s <= current_price), default=None)
        next_resistance = min((r for r in resistances if r >= current_price), default=None)

        if next_resistance is None or (next_support is not None and abs(current_price - next_support) < abs(current_price - next_resistance)):
            return f"Approaching Support at {next_support:.2f}"
        else:
            return f"Approaching Resistance at {next_resistance:.2f}"

# Main execution with auto-refresh
if __name__ == "__main__":
    symbol = input(f"{Fore.CYAN}Enter trading symbol (e.g., BTCUSDT): {Style.RESET_ALL}").upper()
    interval = input(f"{Fore.CYAN}Enter timeframe (e.g., 1, 5, 15, 60, 240, D): {Style.RESET_ALL}")

    logger = setup_logger(symbol)

    while True:
        try:
            df = fetch_klines(symbol, interval)
            if df.empty:
                logger.error(f"{Fore.RED}Failed to fetch data for {symbol}.{Style.RESET_ALL}")
                continue

            current_price = fetch_current_price(symbol)
            if current_price is None:
                logger.error(f"{Fore.RED}Could not fetch the current price.{Style.RESET_ALL}")
                continue

            analyzer = TradingAnalyzer(
                close_prices=df["close"],
                high_prices=df["high"],
                low_prices=df["low"],
                logger=logger
            )
            next_level = analyzer.find_nearest_levels(current_price)
            logger.info(f"Current Price: {current_price:.2f} | {next_level}")

            time.sleep(60)  # Refresh every 60 seconds
        except KeyboardInterrupt:
            logger.info(f"{Fore.GREEN}Exiting script.{Style.RESET_ALL}")
            break
