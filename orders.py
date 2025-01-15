import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from termcolor import colored
from typing import List, Tuple

# Load environment variables
load_dotenv()
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")
BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")

# Configure logging
LOG_DIR = "botlogs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{LOG_DIR}/trading_bot.log"),
    ],
)
logger = logging.getLogger("trading_bot")

# Constants
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W"]
SUPPORT_RESISTANCE_WINDOW = 14
MAX_RETRIES = 3
RETRY_DELAY = 5


def generate_signature(params: dict) -> str:
    """Generate a HMAC SHA256 signature."""
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hmac.new(BYBIT_SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()


def fetch_data(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch historical market data."""
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "category": "linear",
    }
    params["api_key"] = BYBIT_API_KEY
    params["timestamp"] = str(int(time.time() * 1000))
    params["sign"] = generate_signature(params)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") == 0:
                df = pd.DataFrame(data["result"]["list"], columns=["Open Time", "Open", "High", "Low", "Close", "Volume", "Turnover"])
                df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
                df = df.set_index("Open Time").sort_index()
                df = df.astype({col: "float64" for col in ["Open", "High", "Low", "Close", "Volume"]})
                return df
            else:
                logger.error(f"API error: {data.get('retMsg', 'Unknown error')}")
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
    return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and append technical indicators to the DataFrame."""
    if df.empty:
        logger.warning("Empty DataFrame passed to indicator calculations.")
        return df

    macd = MACD(df["Close"])
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["ADX"] = ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    return df


def calculate_fibonacci_levels(df: pd.DataFrame) -> List[float]:
    """Calculate Fibonacci support and resistance levels."""
    high = df["High"].max()
    low = df["Low"].min()
    diff = high - low
    levels = [
        high,
        high - diff * 0.236,
        high - diff * 0.382,
        high - diff * 0.5,
        high - diff * 0.618,
        low,
    ]
    return levels


def generate_signal(df: pd.DataFrame, levels: List[float]) -> List[Tuple[str, float, float, float]]:
    """Generate signals based on strong and weak support/resistance levels."""
    signals = []
    close_price = df["Close"].iloc[-1]
    atr = df["ATR"].iloc[-1]

    for i, level in enumerate(levels):
        if close_price < level:  # Strong support
            tp = level + atr * 1.5
            sl = level - atr * 2
            signals.append(("Long", level, tp, sl))
        elif close_price > level:  # Weak resistance
            tp = level - atr * 1.5
            sl = level + atr * 2
            signals.append(("Short", level, tp, sl))

    return signals


def display_output(levels: List[float], signals: List[Tuple[str, float, float, float]]):
    """Display support/resistance levels and signals with neon colors."""
    print(colored("\nSupport and Resistance Levels (Fibonacci):", "cyan", attrs=["bold"]))
    for i, level in enumerate(levels):
        color = "green" if i % 2 == 0 else "magenta"
        print(colored(f"Level {i + 1}: {level:.2f}", color))

    print(colored("\nGenerated Signals:", "yellow", attrs=["bold"]))
    for signal_type, level, tp, sl in signals:
        color = "blue" if signal_type == "Long" else "red"
        print(
            colored(
                f"{signal_type} at {level:.2f} | TP: {tp:.2f} | SL: {sl:.2f}",
                color,
                attrs=["bold"],
            )
        )


def main():
    symbol = input("Enter trading symbol (e.g., BTCUSDT): ").strip().upper()
    interval = input(f"Enter interval ({', '.join(VALID_INTERVALS)}): ").strip()
    if interval not in VALID_INTERVALS:
        logger.error("Invalid interval.")
        return

    while True:
        df = fetch_data(symbol, interval)
        if not df.empty:
            df = calculate_indicators(df)
            levels = calculate_fibonacci_levels(df)
            signals = generate_signal(df, levels)
            display_output(levels, signals)
        time.sleep(60)


if __name__ == "__main__":
    main()


import os
import logging
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from termcolor import colored
import time
from typing import List, Tuple

# Load environment variables
load_dotenv()
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")
BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")

# Configure logging
LOG_DIR = "botlogs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{LOG_DIR}/trading_bot.log"),
    ],
)
logger = logging.getLogger("trading_bot")

# Constants
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W"]
MAX_RETRIES = 3
RETRY_DELAY = 5


def generate_signature(params: dict) -> str:
    """Generate a HMAC SHA256 signature."""
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hmac.new(BYBIT_SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()


def fetch_data(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch historical market data."""
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "category": "linear",
    }
    params["api_key"] = BYBIT_API_KEY
    params["timestamp"] = str(int(time.time() * 1000))
    params["sign"] = generate_signature(params)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") == 0:
                df = pd.DataFrame(data["result"]["list"], columns=["Open Time", "Open", "High", "Low", "Close", "Volume", "Turnover"])
                df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
                df = df.set_index("Open Time").sort_index()
                df = df.astype({col: "float64" for col in ["Open", "High", "Low", "Close", "Volume"]})
                return df
            else:
                logger.error(f"API error: {data.get('retMsg', 'Unknown error')}")
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
    return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and append technical indicators to the DataFrame."""
    if df.empty:
        logger.warning("Empty DataFrame passed to indicator calculations.")
        return df

    macd = MACD(df["Close"])
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["ADX"] = ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    return df


def fibonacci(n: int) -> List[int]:
    """Generate Fibonacci numbers up to the nth number."""
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[2:]  # Exclude the first two numbers (0 and 1)


def fibonacci_moving_average(data: List[float], n: int) -> List[float]:
    """Calculate Fibonacci moving averages for the given data."""
    fib_numbers = fibonacci(n)
    moving_averages = []

    for fib in fib_numbers:
        if fib > len(data):
            break
        avg = sum(data[:fib]) / fib
        moving_averages.append(avg)

    return moving_averages


def calculate_fibonacci_levels(df: pd.DataFrame) -> List[float]:
    """Calculate Fibonacci support and resistance levels."""
    high = df["High"].max()
    low = df["Low"].min()
    diff = high - low
    levels = [
        high,
        high - diff * 0.236,
        high - diff * 0.382,
        high - diff * 0.5,
        high - diff * 0.618,
        low,
    ]
    return levels


def generate_signal(df: pd.DataFrame, levels: List[float]) -> List[Tuple[str, float, float, float]]:
    """Generate signals based on Fibonacci moving averages and levels."""
    signals = []
    close_price = df["Close"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    moving_averages = fibonacci_moving_average(df["Close"].tolist(), 5)

    for i, level in enumerate(levels):
        if close_price < level:  # Strong support
            tp = level + atr * 1.5
            sl = level - atr * 2
            signals.append(("Long", level, tp, sl))
        elif close_price > level:  # Weak resistance
            tp = level - atr * 1.5
            sl = level + atr * 2
            signals.append(("Short", level, tp, sl))

    return signals, moving_averages


def display_output(levels: List[float], signals: List[Tuple[str, float, float, float]], moving_averages: List[float]):
    """Display support/resistance levels, signals, and Fibonacci moving averages."""
    print(colored("\nSupport and Resistance Levels (Fibonacci):", "cyan", attrs=["bold"]))
    for i, level in enumerate(levels):
        color = "green" if i % 2 == 0 else "magenta"
        print(colored(f"Level {i + 1}: {level:.2f}", color))

    print(colored("\nGenerated Signals:", "yellow", attrs=["bold"]))
    for signal_type, level, tp, sl in signals:
        color = "blue" if signal_type == "Long" else "red"
        print(
            colored(
                f"{signal_type} at {level:.2f} | TP: {tp:.2f} | SL: {sl:.2f}",
                color,
                attrs=["bold"],
            )
        )

    print(colored("\nFibonacci Moving Averages:", "cyan", attrs=["bold"]))
    for i, avg in enumerate(moving_averages, 1):
        print(colored(f"Fibonacci MA {i}: {avg:.2f}", "magenta"))


def main():
    symbol = input("Enter trading symbol (e.g., BTCUSDT): ").strip().upper()
    interval = input(f"Enter interval ({', '.join(VALID_INTERVALS)}): ").strip()
    if interval not in VALID_INTERVALS:
        logger.error("Invalid interval.")
        return

    while True:
        df = fetch_data(symbol, interval)
        if not df.empty:
            df = calculate_indicators(df)
            levels = calculate_fibonacci_levels(df)
            signals, moving_averages = generate_signal(df, levels)
            display_output(levels, signals, moving_averages)
        time.sleep(60)


if __name__ == "__main__":
    main()


import os
import logging
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from termcolor import colored
import time
from typing import List, Tuple

# Load environment variables
load_dotenv()
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")
BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")

# Configure logging
LOG_DIR = "botlogs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{LOG_DIR}/trading_bot.log"),
    ],
)
logger = logging.getLogger("trading_bot")

# Constants
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W"]
MAX_RETRIES = 3
RETRY_DELAY = 5


def generate_signature(params: dict) -> str:
    """Generate a HMAC SHA256 signature."""
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hmac.new(BYBIT_SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()


def fetch_data(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch historical market data."""
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "category": "linear",
    }
    params["api_key"] = BYBIT_API_KEY
    params["timestamp"] = str(int(time.time() * 1000))
    params["sign"] = generate_signature(params)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") == 0:
                df = pd.DataFrame(data["result"]["list"], columns=["Open Time", "Open", "High", "Low", "Close", "Volume", "Turnover"])
                df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
                df = df.set_index("Open Time").sort_index()
                df = df.astype({col: "float64" for col in ["Open", "High", "Low", "Close", "Volume"]})
                return df
            else:
                logger.error(f"API error: {data.get('retMsg', 'Unknown error')}")
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
    return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and append technical indicators to the DataFrame."""
    if df.empty:
        logger.warning("Empty DataFrame passed to indicator calculations.")
        return df

    macd = MACD(df["Close"])
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["ADX"] = ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    return df


def fibonacci(n: int) -> List[int]:
    """Generate Fibonacci numbers up to the nth number."""
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[2:]  # Exclude the first two numbers (0 and 1)


def fibonacci_moving_average(data: List[float], n: int) -> List[float]:
    """Calculate Fibonacci moving averages for the given data."""
    fib_numbers = fibonacci(n)
    moving_averages = []

    for fib in fib_numbers:
        if fib > len(data):
            break
        avg = sum(data[:fib]) / fib
        moving_averages.append(avg)

    return moving_averages


def calculate_fibonacci_levels(df: pd.DataFrame) -> List[float]:
    """Calculate Fibonacci support and resistance levels."""
    high = df["High"].max()
    low = df["Low"].min()
    diff = high - low
    levels = [
        high,
        high - diff * 0.236,
        high - diff * 0.382,
        high - diff * 0.5,
        high - diff * 0.618,
        low,
    ]
    return levels


def generate_signal(df: pd.DataFrame, levels: List[float]) -> List[Tuple[str, float, float, float]]:
    """Generate signals based on Fibonacci moving averages and levels."""
    signals = []
    close_price = df["Close"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    moving_averages = fibonacci_moving_average(df["Close"].tolist(), 5)

    for i, level in enumerate(levels):
        if close_price < level:  # Strong support
            tp = level + atr * 1.5
            sl = level - atr * 2
            signals.append(("Long", level, tp, sl))
        elif close_price > level:  # Weak resistance
            tp = level - atr * 1.5
            sl = level + atr * 2
            signals.append(("Short", level, tp, sl))

    return signals, moving_averages


def display_output(levels: List[float], signals: List[Tuple[str, float, float, float]], moving_averages: List[float]):
    """Display support/resistance levels, signals, and Fibonacci moving averages."""
    print(colored("\nSupport and Resistance Levels (Fibonacci):", "cyan", attrs=["bold"]))
    for i, level in enumerate(levels):
        color = "green" if i % 2 == 0 else "magenta"
        print(colored(f"Level {i + 1}: {level:.2f}", color))

    print(colored("\nGenerated Signals:", "yellow", attrs=["bold"]))
    for signal_type, level, tp, sl in signals:
        color = "blue" if signal_type == "Long" else "red"
        print(
            colored(
                f"{signal_type} at {level:.2f} | TP: {tp:.2f} | SL: {sl:.2f}",
                color,
                attrs=["bold"],
            )
        )

    print(colored("\nFibonacci Moving Averages:", "cyan", attrs=["bold"]))
    for i, avg in enumerate(moving_averages, 1):
        print(colored(f"Fibonacci MA {i}: {avg:.2f}", "magenta"))


def main():
    symbol = input("Enter trading symbol (e.g., BTCUSDT): ").strip().upper()
    interval = input(f"Enter interval ({', '.join(VALID_INTERVALS)}): ").strip()
    if interval not in VALID_INTERVALS:
        logger.error("Invalid interval.")
        return

    while True:
        df = fetch_data(symbol, interval)
        if not df.empty:
            df = calculate_indicators(df)
            levels = calculate_fibonacci_levels(df)
            signals, moving_averages = generate_signal(df, levels)
            display_output(levels, signals, moving_averages)
        time.sleep(60)


if __name__ == "__main__":
    main()
import os
import logging
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from termcolor import colored
import time
from typing import List, Tuple

# Load environment variables
load_dotenv()
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")
BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")

# Configure logging
LOG_DIR = "botlogs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{LOG_DIR}/trading_bot.log"),
    ],
)
logger = logging.getLogger("trading_bot")

# Constants
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W"]
MAX_RETRIES = 3
RETRY_DELAY = 5


def generate_signature(params: dict) -> str:
    """Generate a HMAC SHA256 signature."""
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hmac.new(BYBIT_SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()


def fetch_data(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch historical market data."""
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "category": "linear",
    }
    params["api_key"] = BYBIT_API_KEY
    params["timestamp"] = str(int(time.time() * 1000))
    params["sign"] = generate_signature(params)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") == 0:
                df = pd.DataFrame(data["result"]["list"], columns=["Open Time", "Open", "High", "Low", "Close", "Volume", "Turnover"])
                df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
                df = df.set_index("Open Time").sort_index()
                df = df.astype({col: "float64" for col in ["Open", "High", "Low", "Close", "Volume"]})
                return df
            else:
                logger.error(f"API error: {data.get('retMsg', 'Unknown error')}")
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
    return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and append technical indicators to the DataFrame."""
    if df.empty:
        logger.warning("Empty DataFrame passed to indicator calculations.")
        return df

    macd = MACD(df["Close"])
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["ADX"] = ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    return df


def calculate_fibonacci_levels(current_price: float, pivot_point: float) -> dict:
    """
    Calculate Fibonacci levels based on the current price and pivot point.

    Parameters:
    current_price (float): The current price of the asset.
    pivot_point (float): The pivot point calculated from volume.

    Returns:
    dict: A dictionary containing Fibonacci levels.
    """
    fibonacci_ratios = [0.236, 0.382, 0.618, 0.786]
    levels = {}

    for ratio in fibonacci_ratios:
        level = pivot_point + (current_price - pivot_point) * ratio
        levels[f"Fibonacci Level {ratio}"] = level

    return levels


def generate_signal(df: pd.DataFrame, fibonacci_levels: dict) -> List[Tuple[str, float, float, float]]:
    """Generate signals based on Fibonacci levels."""
    signals = []
    close_price = df["Close"].iloc[-1]
    atr = df["ATR"].iloc[-1]

    for level_name, level in fibonacci_levels.items():
        if close_price < level:  # Strong support
            tp = level + atr * 1.5
            sl = level - atr * 2
            signals.append(("Long", level_name, level, tp, sl))
        elif close_price > level:  # Weak resistance
            tp = level - atr * 1.5
            sl = level + atr * 2
            signals.append(("Short", level_name, level, tp, sl))

    return signals


def display_output(fibonacci_levels: dict, signals: List[Tuple[str, float, float, float]]):
    """Display Fibonacci levels and generated signals."""
    print(colored("\nFibonacci Levels (Pivot-Based):", "cyan", attrs=["bold"]))
    for level_name, level in fibonacci_levels.items():
        print(colored(f"{level_name}: {level:.2f}", "magenta"))

    print(colored("\nGenerated Signals:", "yellow", attrs=["bold"]))
    for signal_type, level_name, level, tp, sl in signals:
        color = "blue" if signal_type == "Long" else "red"
        print(
            colored(
                f"{signal_type} at {level_name} ({level:.2f}) | TP: {tp:.2f} | SL: {sl:.2f}",
                color,
                attrs=["bold"],
            )
        )


def main():
    symbol = input("Enter trading symbol (e.g., BTCUSDT): ").strip().upper()
    interval = input(f"Enter interval ({', '.join(VALID_INTERVALS)}): ").strip()
    if interval not in VALID_INTERVAL

