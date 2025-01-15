import os
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import hmac
import hashlib
from typing import Dict
from colorama import init, Fore, Style

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
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(stream_handler)

    return logger


def safe_json_response(response):
    """Safely parse JSON response."""
    try:
        return response.json()
    except ValueError:
        return None


def bybit_request(method: str, endpoint: str, params: dict = None) -> dict:
    """Send a signed request to the Bybit Unified Trading API."""
    try:
        params = params or {}
        params["api_key"] = API_KEY
        params["timestamp"] = str(int(datetime.now(timezone.utc).timestamp() * 1000))
        params["sign"] = generate_signature(params)

        url = f"{BASE_URL}{endpoint}"
        response = requests.request(method, url, params=params)

        if response.status_code != 200:
            logging.error(
                f"Non-200 response from Bybit: {response.status_code} - {response.text}"
            )
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
    params = {"symbol": symbol, "interval": interval, "limit": limit}
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


def fetch_orderbook(symbol: str, limit: int = 10) -> dict:
    """Fetch order book data from Bybit Unified Trading API."""
    endpoint = "/v5/market/orderbook"
    params = {"symbol": symbol, "limit": limit}
    response = bybit_request("GET", endpoint, params)
    if response.get("retCode") == 0 and response.get("result"):
        return response["result"]
    else:
        logging.error(f"Failed to fetch order book for {symbol}. Response: {response}")
        return {}


def generate_signature(params: dict) -> str:
    """Generate HMAC SHA256 signature."""
    param_str = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
    return hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()


# Main Execution
if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        print(
            f"{Fore.RED}API keys not set. Please ensure BYBIT_API_KEY and BYBIT_API_SECRET are defined in a .env file.{Style.RESET_ALL}"
        )
        exit(1)

    symbol = input(
        f"{Fore.CYAN}Enter trading symbol (e.g., BTCUSDT): {Style.RESET_ALL}"
    ).upper()
    interval = input(f"{Fore.CYAN}Enter timeframe (e.g., 1h, 15m): {Style.RESET_ALL}")

    logger = setup_logger(symbol)

    df = fetch_klines(symbol, interval)
    if df.empty:
        logger.error(f"Failed to fetch data for {symbol}.")
        exit(1)

    orderbook_data = fetch_orderbook(symbol)
    if not orderbook_data:
        logger.warning(
            f"Failed to fetch order book data for {symbol}. Using empty data."
        )
