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

                if response.status_code in Constants.RETRY_ERROR_CODES:
                    self.logger.warning(
                        f"Rate limited or server error. Retrying {retries + 1}/{Constants.MAX_RETRIES} after {Constants.RETRY_DELAY} seconds."
                    )
                    time.sleep(Constants.RETRY_DELAY * (2 ** retries))
                    retries += 1
                    continue

                if response.status_code != 200:
                    self.logger.error(f"Bybit API Error {response.status_code}: {response.text}")
                    return {
                        "retCode": -1, "retMsg": f"HTTP {response.status_code}"
                    }
                try:
                   json_response = response.json()
                   self.api_logger.debug(f"API Response: {json_response}")
                except json.JSONDecodeError:
                     self.logger.error(f"Invalid JSON Response Received: {response.text}")
                     return {
                       "retCode": -1, "retMsg": "Invalid JSON"
                     }

                if not json_response:
                      return {
                            "retCode": -1, "retMsg": "Invalid JSON"
                        }


                if json_response.get("retCode", -1) != 0:
                   self.logger.error(f"Bybit API returned non-zero: {json_response}")
                   return json_response

                return json_response

            except requests.RequestException as e:
                self.logger.error(f"API request failed: {e}")
                retries += 1
                time.sleep(Constants.RETRY_DELAY * (2 ** retries))
            except Exception as e:
                self.logger.exception(f"An unexpected error occurred: {e}")
                return {
                    "retCode": -1, "retMsg": "Unexpected Error"
                }

        return {
            "retCode": -1, "retMsg": "Max retries exceeded"
        }

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
        """Calculates the Average True Range (ATR).

              Args:
                     window(int): Length for the indicator window default 14

              Returns:
                  pd.Series: The ATR series for this window
        """
        if self.df.empty:
            self.logger.error("DataFrame empty - can't calculate ATR.")
            return pd.Series(dtype=float)
        try:
            return AverageTrueRange(self.df["high"], self.df["low"], self.df["close"], window=window).average_true_range()
        except Exception as e:
              self.logger.error(f"Failed to calculate ATR: {e}")
              return pd.Series(dtype=float)

    def calculate_sma(self, window: int = 20) -> pd.Series:
        """Calculates the Simple Moving Average (SMA).

              Args:
                    window(int): Length for the indicator window default 20

              Returns:
                  pd.Series: The SMA series for this window
        """
        if self.df.empty:
            self.logger.error("DataFrame empty - can't calculate SMA")
            return pd.Series(dtype=float)
        try:
           return SMAIndicator(self.df["close"], window=window).sma_indicator()
        except Exception as e:
           self.logger.error(f"Failed to calculate SMA: {e}")
           return pd.Series(dtype=float)


    def calculate_ema(self, window: int = 20) -> pd.Series:
         """Calculates the Exponential Moving Average (EMA).

               Args:
                    window(int): Length for the indicator window default 20

              Returns:
                   pd.Series: The EMA series for this window
        """
         if self.df.empty:
              self.logger.error("DataFrame is empty - cannot calculate EMA")
              return pd.Series(dtype=float)
         try:
            return EMAIndicator(self.df["close"], window=window).ema_indicator()
         except Exception as e:
            self.logger.error(f"Failed to calculate EMA: {e}")
            return pd.Series(dtype=float)

    def calculate_macd(self) -> pd.Series:
        """Calculates the Moving Average Convergence Divergence (MACD).

                Returns:
                    pd.Series: MACD series.
        """
        if self.df.empty:
            self.logger.error("DataFrame empty - can't calculate MACD")
            return pd.Series(dtype=float)
        try:
           return MACD(self.df["close"]).macd()
        except Exception as e:
           self.logger.error(f"Failed to calculate MACD: {e}")
           return pd.Series(dtype=float)

    def calculate_rsi(self, window: int = 14) -> pd.Series:
        """Calculates the Relative Strength Index (RSI).

              Args:
                   window(int): Length for the indicator window default 14

               Returns:
                   pd.Series: RSI series.
        """
        if self.df.empty:
            self.logger.error("DataFrame empty - can't calculate RSI.")
            return pd.Series(dtype=float)
        try:
            return rsi(self.df["close"], window=window)
        except Exception as e:
            self.logger.error(f"Failed to calculate RSI: {e}")
            return pd.Series(dtype=float)


    def calculate_obv(self) -> pd.Series:
        """Calculates On-Balance Volume (OBV).

               Returns:
                   pd.Series: OBV series.
       """
        if self.df.empty:
             self.logger.error("DataFrame empty - can't calculate OBV")
             return pd.Series(dtype=float)
        try:
           return on_balance_volume(self.df["close"], self.df["volume"])
        except Exception as e:
            self.logger.error(f"Failed to calculate OBV: {e}")
            return pd.Series(dtype=float)


    def calculate_adx(self, window: int = 14) -> pd.Series:
        """Calculates the Average Directional Index (ADX).

              Args:
                  window(int): Length for the indicator window default 14

                Returns:
                  pd.Series: ADX series
         """
        if self.df.empty:
            self.logger.error("DataFrame empty - can't calculate ADX.")
            return pd.Series(dtype=float)
        try:
           return ADXIndicator(self.df["high"], self.df["low"], self.df["close"], window=window).adx()
        except Exception as e:
           self.logger.error(f"Failed to calculate ADX {e}")
           return pd.Series(dtype=float)

    def calculate_aroon(self, window: int = 25) -> pd.Series:
        """Calculates the Aroon indicator.

                Args:
                    window(int): Length for the indicator window default 25

                Returns:
                   pd.Series: Aroon indicator series
          """
        if self.df.empty:
            self.logger.error("Dataframe is empty - can't calculate Aroon.")
            return pd.Series(dtype=float)

        # Ensure numeric types for 'high', 'low', 'close'
        self._ensure_numeric_df() # moved to constructor

        try:
           aroon_ind = AroonIndicator(self.df["high"], self.df["low"], self.df["close"], window)
           self.df['aroon_up'] = aroon_ind.aroon_up()
           self.df['aroon_down'] = aroon_ind.aroon_down()
           return aroon_ind.aroon_indicator()
        except Exception as e:
           self.logger.error(f"Failed to calculate Aroon {e}")
           return pd.Series(dtype=float)


    def calculate_wma(self, window=20) -> pd.Series:
        """Calculates the Weighted Moving Average (WMA).

             Args:
                    window(int): Length for the indicator window default 20

            Returns:
                 pd.Series: WMA indicator series.
        """
        if self.df.empty:
           self.logger.error("Dataframe is empty - can't calculate WMA")
           return pd.Series(dtype=float)
        try:
          return WMAIndicator(self.df["close"], window=window).wma()
        except Exception as e:
           self.logger.error(f"Failed to calculate WMA: {e}")
           return pd.Series(dtype=float)

    def calculate_vpt(self) -> pd.Series:
         """Calculates Volume Price Trend (VPT).

             Returns:
                pd.Series: VPT indicator series.
         """
         if self.df.empty:
            self.logger.error("Dataframe is empty - can't calculate VPT.")
            return pd.Series(dtype=float)
         try:
           return volume_price_trend(self.df["close"], self.df["volume"])
         except Exception as e:
            self.logger.error(f"Failed to calculate VPT: {e}")
            return pd.Series(dtype=float)

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
       """Determines the trend based on moving averages.

             Args:
                  short_window (int): Short moving average
                  long_window (int): Long moving average

            Returns:
                 str: String indicating trend 'upward', 'downward', or 'neutral'
       """
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
        """ Determines trend from indicator data
          Args:
              indicator(pd.Series): The technical indicator data
              threshold(float): Threshold of indicator that it must pass to show upward/downward trend

          Returns:
               str: String of the trend
       """
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
         if len(self.df) > 0:
           self.dynamic_sensitivity = Constants.CLUSTER_SENSITIVITY * atr / self.df["close"].iloc[-1]
         else:
           self.logger.warning("DataFrame is empty. Cannot update dynamic sensitivity.")

   def identify_support_resistance_dynamic(self, window=Constants.SUPPORT_RESISTANCE_WINDOW):
         """Enhanced S/R identification with dynamic sensitivity and volume weighting."""
         self.update_dynamic_sensitivity()
         data = self.df["close"].values
         volumes = self.df["volume"].values
         levels = {}

         maxima_indices = [
            i for i in range(window, len(data) - window)
            if all(data[i] >= data[i - j] for j in range(1, window + 1)) and
            all(data[i] >= data[i + j] for j in range(1, window + 1))
         ]
         minima_indices = [
           i for i in range(window, len(data) - window)
           if all(data[i] <= data[i - j] for j in range(1, window + 1)) and
            all(data[i] <= data[i + j] for j in range(1, window + 1))
          ]

         maxima = data[maxima_indices]
         minima = data[minima_indices]
         volumes_maxima = volumes[maxima_indices]
         volumes_minima = volumes[minima_indices]

         all_points = np.concatenate((maxima, minima))
         all_volumes = np.concatenate((volumes_maxima, volumes_minima))

         cluster_centers, cluster_volumes = [], []
         for point, volume in zip(all_points, all_volumes):
           if not cluster_centers:
                cluster_centers.append(point)
                cluster_volumes.append(volume)
           else:
                added_to_cluster = False
                for i, center in enumerate(cluster_centers):
                    if abs(point - center) / center <= self.dynamic_sensitivity:
                        cluster_centers[i] = (
                          cluster_centers[i] * cluster_volumes[i] + point * volume
                        ) / (cluster_volumes[i] + volume)
                        cluster_volumes[i] += volume
                        added_to_cluster = True
                        break
                if not added_to_cluster:
                     cluster_centers.append(point)
                     cluster_volumes.append(volume)

         current_price = self.df["close"].iloc[-1]
         for center, volume in zip(cluster_centers, cluster_volumes):
            level_type = "Support" if center < current_price else "Resistance"
            levels[f"{level_type} (Dynamic)"] = (center, volume / sum(cluster_volumes))

         self.levels.update(levels)
         return levels

   def analyze(self, current_price: float):
      """Performs analysis using all available methods and information."""
      if not self.initialize_data():
          self.logger.error("Could not initialize data - Analysis Cancelled")
          return
      self.identify_support_resistance_dynamic()
      close = self.df["close"].iloc[-1]
      high = self.df["high"].iloc[-1]
      low = self.df["low"].iloc[-1]
      self.calculate_custom_pivot_points(high, low, close)

      # Calculate Indicators
      rsi_ind = self.indicators.calculate_rsi()
      macd_ind = self.indicators.calculate_macd()
      ema_ind = self.indicators.calculate_ema()
      adx_ind = self.indicators.calculate_adx()
      obv_ind = self.indicators.calculate_obv()
      aroon_ind = self.indicators.calculate_aroon()
      wma_ind = self.indicators.calculate_wma()
      vpt_ind = self.indicators.calculate_vpt()

      trend_ma = self.determine_trend_ma()
      trend_rsi = self.determine_trend_momentum(rsi_ind)
      trend_macd = self.determine_trend(macd_ind,0.0)
      trend_adx = self.determine_trend(adx_ind, 25)
      trend_obv = self.determine_trend_volume(obv_ind)
      trend_aroon = self.determine_trend(aroon_ind,0.0)
      trend_wma = self.determine_trend_ma(short_window=wma_ind.iloc[-1], long_window=ema_ind.iloc[-1])
      trend_vpt = self.determine_trend_volume(vpt_ind)


      analysis = {
           "symbol": self.symbol,
           "interval": self.interval,
           "price": current_price,
            "timestamp":  datetime.now(Constants.UTC_TZ).isoformat(),
           "supports_resistances": self.levels,
           "pivot_points": self.pivot_points,
           "trend_ma": trend_ma,
           "trend_rsi": trend_rsi,
           "trend_macd": trend_macd,
            "trend_adx": trend_adx,
           "trend_obv": trend_obv,
          "trend_aroon": trend_aroon,
         "trend_wma": trend_wma,
        "trend_vpt": trend_vpt,
      }
      self.logger.info(json.dumps(analysis, indent=4, default=str)) # Make sure to convert non standard types to string

def main():
    logger = setup_logger("trading_bot", log_file="trading_bot.log")
    config = Config()
    bybit_client = Bybit(config, logger)

    symbol = "BTCUSDT"
    interval = "5"

    analyzer = TradingAnalyzerEnhanced(symbol, interval, logger, bybit_client)

    try:
        current_price = bybit_client.fetch_current_price(symbol)

        if current_price is not None:
             analyzer.analyze(current_price)
        else:
             logger.error("Could not get current price")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()