import os
import logging
import requests
import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator, MACD, EMAIndicator, ADXIndicator, AroonIndicator, WMAIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import rsi
from ta.volume import on_balance_volume, volume_price_trend
from datetime import datetime
from dotenv import load_dotenv
import hmac
import hashlib
import time
from typing import Dict, Tuple, List, Optional
from termcolor import colored
from zoneinfo import ZoneInfo
import json

# --- Configuration ---
class Config:
    """Handles configuration settings."""
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")

        if not self.api_key or not self.api_secret:
            raise ValueError(colored("API keys not set. Set BYBIT_API_KEY and BYBIT_API_SECRET in .env", "red"))

        # Log the loaded API URL
        logger = logging.getLogger("trading_bot")
        logger.info(f"Bybit API Base URL: {self.base_url}")


# --- Constants ---
class Constants:
    """Holds constant values."""
    LOG_DIR = "botlogs"
    UTC_TZ = ZoneInfo("UTC")
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", 5))
    VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W"]
    RETRY_ERROR_CODES = [429, 500, 502, 503, 504]
    SUPPORT_RESISTANCE_WINDOW = 14
    CLUSTER_SENSITIVITY = 0.05
    HIGHER_TIMEFRAMES = ["60", "240", "D"]


# --- Logging Setup ---
def setup_logger(name, log_file=None, level=logging.INFO):
    """Sets up a logger with the given name, log file, and level."""
    if not os.path.exists(Constants.LOG_DIR):
        os.makedirs(Constants.LOG_DIR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logging

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Create a separate logger for API requests/responses
    api_logger = logging.getLogger(f"{name}_api")
    api_logger.setLevel(logging.DEBUG)
    api_logger.propagate = False
    api_console_handler = logging.StreamHandler()
    api_console_handler.setFormatter(formatter)
    api_logger.addHandler(api_console_handler)

    if log_file:
        api_file_handler = logging.FileHandler(log_file.replace(".log", "_api.log"))
        api_file_handler.setFormatter(formatter)
        api_logger.addHandler(api_file_handler)

    return logger

# --- Bybit API Client ---
class Bybit:
    """Handles communication with the Bybit API."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.api_logger = logging.getLogger(f"{logger.name}_api")
        self.session = requests.Session()

    def _generate_signature(self, params: dict) -> str:
        """Generates a signature for the Bybit API request."""
        param_str = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
        return hmac.new(
            self.config.api_secret.encode(), param_str.encode(), hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, endpoint: str, params: dict = None) -> dict:
        """Makes a request to the Bybit API with retries."""
        retries = 0
        while retries < Constants.MAX_RETRIES:
            try:
                params = params or {}
                params["api_key"] = self.config.api_key
                params["timestamp"] = str(int(datetime.now(Constants.UTC_TZ).timestamp() * 1000))
                params["sign"] = self._generate_signature(params)

                url = f"{self.config.base_url}{endpoint}"
                self.api_logger.debug(f"API Request - Method: {method}, URL: {url}, Params: {params}")
                response = self.session.request(method, url, params=params)

                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                json_response = response.json()
                self.api_logger.debug(f"API Response: {json_response}")

                if json_response.get("retCode", -1) != 0:
                    self.logger.error(f"Bybit API returned non-zero retCode: {json_response}")
                    return json_response  # Return the error response

                return json_response

            except requests.exceptions.HTTPError as e:
                if response.status_code in Constants.RETRY_ERROR_CODES:
                    self.logger.warning(
                        f"HTTP error {response.status_code}. Retrying {retries + 1}/{Constants.MAX_RETRIES} after {Constants.RETRY_DELAY} seconds."
                    )
                    time.sleep(Constants.RETRY_DELAY * (2 ** retries))
                    retries += 1
                    continue
                else:
                    self.logger.error(f"Bybit API HTTP error {response.status_code}: {e}")
                    return {"retCode": -1, "retMsg": f"HTTP {response.status_code}"}
            except requests.RequestException as e:
                self.logger.error(f"API request failed: {e}")
                retries += 1
                time.sleep(Constants.RETRY_DELAY * (2 ** retries))
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON response: {response.text}, Error: {e}")
                return {"retCode": -1, "retMsg": "Invalid JSON"}
            except Exception as e:
                self.logger.exception(f"An unexpected error occurred: {e}")
                return {"retCode": -1, "retMsg": "Unexpected Error"}

        return {"retCode": -1, "retMsg": "Max retries exceeded"}

    def fetch_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """Fetches kline data from Bybit."""
        endpoint = "/v5/market/kline"
        params = {
            "symbol": symbol, "interval": interval, "limit": limit, "category": "linear"
        }
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
                f"Failed to fetch klines for {symbol} ({interval}): {response.get('retMsg', 'Unknown error')}"
            )
            return pd.DataFrame()

    def fetch_current_price(self, symbol: str) -> Optional[float]:
       """Fetches the current price of a symbol from Bybit."""
       endpoint = "/v5/market/tickers"
       params = {
           "symbol": symbol, "category": "linear"
       }
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

# --- Technical Indicators ---
class TechnicalIndicators:
    """Handles the calculation of technical indicators."""

    def __init__(self, df: pd.DataFrame, logger: logging.Logger):
        """Initialise the technical indicator class

              Args:
                    df(pd.DataFrame): Data frame of price data
                    logger(logging.Logger): Logger
        """
        self.df = df
        self.logger = logger
        self._ensure_numeric_df()

    def _ensure_numeric_df(self):
          """Ensure columns used for indicators are numeric"""
          if not self.df.empty:
                numeric_cols = ['high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in self.df.columns:
                        try:
                           self.df[col] = pd.to_numeric(self.df[col], errors='raise')
                        except Exception as e:
                             self.logger.error(f"Column {col} not numeric, error: {e}")
                             raise

    def calculate_atr(self, window: int = 14) -> pd.Series:
        """Calculates the Average True Range (ATR)."""
        if self.df.empty:
            self.logger.error("DataFrame empty - can't calculate ATR.")
            return pd.Series(dtype=float)
        return AverageTrueRange(self.df["high"], self.df["low"], self.df["close"], window=window).average_true_range()

    def calculate_sma(self, window: int = 20) -> pd.Series:
        """Calculates the Simple Moving Average (SMA)."""
        if self.df.empty:
            self.logger.error("DataFrame empty - can't calculate SMA")
            return pd.Series(dtype=float)
        return SMAIndicator(self.df["close"], window=window).sma_indicator()

    def calculate_ema(self, window: int = 20) -> pd.Series:
         """Calculates the Exponential Moving Average (EMA)."""
         if self.df.empty:
              self.logger.error("DataFrame is empty - cannot calculate EMA")
              return pd.Series(dtype=float)
         return EMAIndicator(self.df["close"], window=window).ema_indicator()

    def calculate_macd(self) -> pd.Series:
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        if self.df.empty:
            self.logger.error("DataFrame empty - can't calculate MACD")
            return pd.Series(dtype=float)
        return MACD(self.df["close"]).macd()

    def calculate_rsi(self, window: int = 14) -> pd.Series:
        """Calculates the Relative Strength Index (RSI)."""
        if self.df.empty:
            self.logger.error("DataFrame empty - can't calculate RSI.")
            return pd.Series(dtype=float)
        return rsi(self.df["close"], window=window)

    def calculate_obv(self) -> pd.Series:
        """Calculates On-Balance Volume (OBV)."""
        if self.df.empty:
             self.logger.error("DataFrame empty - can't calculate OBV")
             return pd.Series(dtype=float)
        return on_balance_volume(self.df["close"], self.df["volume"])

    def calculate_adx(self, window: int = 14) -> pd.Series:
        """Calculates the Average Directional Index (ADX)."""
        if self.df.empty:
            self.logger.error("DataFrame empty - can't calculate ADX.")
            return pd.Series(dtype=float)
        return ADXIndicator(self.df["high"], self.df["low"], self.df["close"], window=window).adx()

    def calculate_aroon(self, window: int = 25) -> Tuple[pd.Series, pd.Series]:
        """Calculates the Aroon indicator."""
        if self.df.empty:
            self.logger.error("Dataframe is empty - can't calculate Aroon.")
            return pd.Series(dtype=float), pd.Series(dtype=float)
        aroon_ind = AroonIndicator(self.df["high"], self.df["low"], self.df["close"], window)
        return aroon_ind.aroon_up(), aroon_ind.aroon_down()

    def calculate_wma(self, window=20) -> pd.Series:
        """Calculates the Weighted Moving Average (WMA)."""
        if self.df.empty:
           self.logger.error("Dataframe is empty - can't calculate WMA")
           return pd.Series(dtype=float)
        return WMAIndicator(self.df["close"], window=window).wma()

    def calculate_vpt(self) -> pd.Series:
         """Calculates Volume Price Trend (VPT)."""
         if self.df.empty:
            self.logger.error("Dataframe is empty - can't calculate VPT.")
            return pd.Series(dtype=float)
         return volume_price_trend(self.df["close"], self.df["volume"])

# --- Trading Analyzer ---
class TradingAnalyzer:
   """Base class for analyzing trading data."""
   def __init__(self, symbol: str, interval: str, logger: logging.Logger, bybit: Bybit):
       self.symbol = symbol
       self.interval = interval
       self.logger = logger
       self.bybit = bybit
       self.df = pd.DataFrame()  # Placeholder for DataFrame
       self.levels = {}
       self.pivot_points = {}
       self.indicators = TechnicalIndicators(self.df, self.logger)

   def analyze(self, current_price: float):
       """Placeholder for analysis logic."""
       pass

   def calculate_custom_pivot_points(self, high, low, close, weights=(0.3, 0.3, 0.4)):
       """Customizable Pivot Points calculation."""
       pivot = sum([high * weights[0], low * weights[1], close * weights[2]])
       r1 = 2 * pivot - low
       s1 = 2 * pivot - high
       r2 = pivot + (high - low)
       s2 = pivot - (high - low)
       r3 = high + 2 * (pivot - low)
       s3 = low - 2 * (high - pivot)

       pivot_levels = {
         "Pivot (Custom)": pivot,
         "R1 (Custom)": r1,
         "S1 (Custom)": s1,
         "R2 (Custom)": r2,
         "S2 (Custom)": s2,
         "R3 (Custom)": r3,
         "S3 (Custom)": s3,
       }
       self.pivot_points.update(pivot_levels)
       return pivot_levels

   def determine_trend_ma(self, short_window=50, long_window=200) -> str:
       """Determines the trend based on moving averages."""
       sma_short = self.indicators.calculate_sma(window=short_window)
       sma_long = self.indicators.calculate_sma(window=long_window)
       if sma_short.empty or sma_long.empty:
            self.logger.warning("SMA data is empty cannot calculate MA trend")
            return "neutral"

       if sma_short.iloc[-1] > sma_long.iloc[-1]:
           return "upward"
       elif sma_short.iloc[-1] < sma_long.iloc[-1]:
           return "downward"
       else:
           return "neutral"

   def determine_trend(self, indicator: pd.Series, threshold: float = 0.0) -> str:
        """ Determines trend from indicator data"""
        if indicator.empty:
           self.logger.warning("Indicator series empty cannot determine trend")
           return "neutral"

        if indicator.iloc[-1] > threshold:
            return "upward"
        elif indicator.iloc[-1] < threshold:
            return "downward"
        else:
            return "neutral"

   def determine_trend_momentum(self, indicator: pd.Series, overbought=70, oversold=30) -> str:
      """Determines the trend based on momentum and volume indicators."""
      if indicator.empty:
         self.logger.warning("Indicator series empty cannot determine momentum trend")
         return "neutral"

      if indicator.iloc[-1] > overbought:
            return "downward"  # Potential reversal from overbought
      elif indicator.iloc[-1] < oversold:
            return "upward"  # Potential reversal from oversold
      else:
            return "neutral"

   def determine_trend_volume(self, indicator: pd.Series) -> str:
        """Determines the trend based on volume indicators."""
        if indicator.empty:
              self.logger.warning("Indicator series empty - cannot determine volume trend.")
              return "neutral"

        if len(indicator) < 2:
           self.logger.warning("Not enough datapoints to check volume trend.")
           return "neutral"

        if indicator.iloc[-1] > indicator.iloc[-2]:
            return "upward"  # Increasing Volume suggests upward trend
        elif indicator.iloc[-1] < indicator.iloc[-2]:
           return "downward"  # Decreasing Volume suggests downward trend
        else:
           return "neutral"

# --- Enhanced Trading Analyzer ---
class TradingAnalyzerEnhanced(TradingAnalyzer):
   """Enhanced trading analyzer with advanced features."""

   def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_sensitivity = None  # To store dynamic sensitivity

   def initialize_data(self):
      """Fetches initial kline data and sets up the DataFrame."""
      self.df = self.bybit.fetch_klines(self.symbol, self.interval, limit=200)
      if self.df.empty:
            self.logger.error(f"Failed to fetch initial klines for {self.symbol} ({self.interval}).")
            return False
      self.indicators.df = self.df  # Pass DataFrame for technical indicator calculations
      return True

   def update_dynamic_sensitivity(self):
         """Dynamically adjust clustering sensitivity based on ATR."""
         if len(self.df) < 15:
           self.logger.warning("Not enough data to calculate ATR yet.")
           return
         atr = self.indicators.calculate_atr().iloc[-1]
         if atr > 0:
            self.dynamic_sensitivity = Constants.CLUSTER_SENSITIVITY * atr
         else:
            self.logger.warning("ATR is zero or negative. Cannot update sensitivity.")


   def analyze(self, current_price: float):
        """Performs enhanced trading analysis."""
        if self.df.empty:
            self.logger.error("DataFrame is empty. Cannot perform analysis.")
            return

        #Recalculate indicators after updating the dataframe
        self.indicators = TechnicalIndicators(self.df, self.logger)
        self.update_dynamic_sensitivity()


        #Example analysis using multiple indicators
        macd_trend = self.determine_trend(self.indicators.calculate_macd())
        rsi_trend = self.determine_trend_momentum(self.indicators.calculate_rsi())
        obv_trend = self.determine_trend_volume(self.indicators.calculate_obv())
        atr_value = self.indicators.calculate_atr().iloc[-1]
        sma_trend = self.determine_trend_ma()
        aroon_up, aroon_down = self.indicators.calculate_aroon()

        self.logger.info(f"Analysis for {self.symbol} ({self.interval}):")
        self.logger.info(f"  Current Price: {current_price:.2f}")
        self.logger.info(f"  MACD Trend: {macd_trend}")
        self.logger.info(f"  RSI Trend: {rsi_trend}")
        self.logger.info(f"  OBV Trend: {obv_trend}")
        self.logger.info(f"  ATR: {atr_value:.2f}")
        self.logger.info(f"  SMA Trend: {sma_trend}")
        self.logger.info(f"  Aroon Up: {aroon_up.iloc[-1]:.2f}, Aroon Down: {aroon_down.iloc[-1]:.2f}")


        # Add more sophisticated analysis based on indicator combinations, pivot points, etc.


#Example usage
config = Config()
logger = setup_logger("trading_bot", "bot.log")
bybit = Bybit(config, logger)
analyzer = TradingAnalyzerEnhanced("BTCUSDT", "5", logger, bybit)

if analyzer.initialize_data():
    current_price = bybit.fetch_current_price("BTCUSDT")
    if current_price:
        analyzer.analyze(current_price)