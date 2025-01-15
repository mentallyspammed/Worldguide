from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from config import Config
from bybit_api import BybitAPI


class TradingAnalyzer:
    """Analyzes market data for trading insights."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initializes the TradingAnalyzer with configuration and logger."""
        self.config = config
        self.logger = logger
        self.api = BybitAPI(config, logger)
        self.levels = {}  # Store calculated support/resistance levels
        self.df = pd.DataFrame()
        self._load_historical_data()

    def _load_historical_data(self):
        """Loads historical data from Bybit API."""
        self.logger.info("Loading historical kline data...")
        self.df = self.api.fetch_klines(
            symbol=self.config.symbol,
            interval=self.config.interval,
            limit=self.config.data_limit,
        )
        if self.df.empty:
             self.logger.error(
                 f"Failed to load historical klines. Check the API connectivity and config")
        else:
            self.logger.info("Historical kline data loaded successfully.")

    def analyze(self, current_price: float) -> Optional[Dict]:
        """Analyzes the market data and returns insights."""

        if self.df.empty:
            self.logger.error(
                "DataFrame is empty. Cannot perform analysis. Please check API connectivity and configuration"
            )
            return None  # Return None if no data
        else:
            self.logger.info(
                "DataFrame is not empty. Continuing with analysis.")

        # Recalculate indicators on new data
        self.calculate_indicators()

        if self.df.empty:
            self.logger.error(
                "DataFrame is empty after calculating indicators. Cannot perform analysis"
            )
            return None
        else:
            self.logger.info(
                "DataFrame is not empty after calculating indicators. Continuing with analysis.")

        # ... (Rest of your analysis logic as before, using self.df) ...
        # But with improved functions calls
        trend = self._determine_trend_adx()  # improved by combining momentum and adx
        momentum = self.analyze_momentum(self.df)
        volume = self.analyze_volume(self.df)
        macd_analysis = self._analyze_macd()
        rsi_value = self.df["RSI"].iloc[-1]
        rsi_analysis = self._analyze_rsi(rsi_value)
        aroon_analysis = self.analyze_aroon()

        # Calculate levels (using updated _find_levels)
        self._find_levels()  # update levels attribute
        nearest_supports = self.find_nearest_level(current_price, "support")
        nearest_resistances = self.find_nearest_level(
            current_price, "resistance")

        atr = self.df["ATR"].iloc[-1]

        # Adapt Parameters
        adapted_params = self.adapt_parameters(atr)

        # Calculate Fibonacci pivot levels
        fib_levels = self.calculate_fibonacci_pivot()

        # Generate trade suggestions with weights (updated function call)
        trade_suggestions, long_score, short_score = self._generate_trade_signals(
            current_price,
            trend,
            nearest_supports,
            nearest_resistances,
            adapted_params,
            momentum,
            volume,
            rsi_analysis,
            macd_analysis,
            aroon_analysis,
        )

        # Compile analysis results
        results = {
            "current_price": current_price,
            "trend": trend,
            "momentum": momentum,
            # ... rest of your results ...
            "volume": volume,
            "rsi_value": rsi_value,
            "rsi_analysis": rsi_analysis,
            "macd_analysis": macd_analysis,
            "aroon_analysis": aroon_analysis,
            "nearest_supports": nearest_supports,
            "nearest_resistances": nearest_resistances,
            "fibonacci_levels": fib_levels.to_dict(),  # add the fib levels to the result
            "trade_suggestions": trade_suggestions,
            "long_score": long_score,
            "short_score": short_score,
        }
        return results

    def calculate_indicators(self) -> None:
        """Calculates technical indicators."""
        if self.df.empty:
            return

        # Calculate moving averages
        self.df["WMA_Short"] = self.wma(
            self.df["close"], window=self.config.ma_periods_short
        )
        self.df["WMA_Long"] = self.wma(
            self.df["close"], window=self.config.ma_periods_long
        )
        self.df["FMA"] = self.ema(self.df["close"], window=self.config.fma_period)
        self.df["HMA"] = self.hma(
            self.df["close"], window=self.config.ma_periods_short
        )
        # Calculate RSI
        self.df["RSI"] = self.rsi(self.df["close"], window=self.config.rsi_period)

        # Calculate MACD
        macd_data = self.calculate_macd()
        if not macd_data.empty:
            self.df["MACD"] = macd_data["macd"]
            self.df["Signal"] = macd_data["signal"]
            self.df["Histogram"] = macd_data["histogram"]

        # Calculate ADX
        adx_data = self.calculate_adx()
        if not adx_data.empty:
            self.df["ADX"] = adx_data["ADX"]
            self.df["+DI"] = adx_data["+DI"]
            self.df["-DI"] = adx_data["-DI"]

        # Calculate Aroon
        aroon_data = self.calculate_aroon()
        if not aroon_data.empty:
            self.df["Aroon Up"] = aroon_data["Aroon Up"]
            self.df["Aroon Down"] = aroon_data["Aroon Down"]

        # Calculate ATR, OBV and VPT
        self.df["ATR"] = self.calculate_atr()
        self.df["OBV"] = self.calculate_obv()
        self.df["VPT"] = self.calculate_vpt()

    def wma(self, series: pd.Series, window: int) -> pd.Series:
        """Calculates the Weighted Moving Average (WMA)."""
        weights = np.arange(1, window + 1)
        return series.rolling(window=window).apply(
            lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
        )

    def ema(self, series: pd.Series, window: int) -> pd.Series:
        """Calculates the Exponential Moving Average (EMA)."""
        return series.ewm(span=window, min_periods=window).mean()

    def hma(self, series: pd.Series, window: int) -> pd.Series:
        """Calculates the Hull Moving Average (HMA)."""
        half_length = int(window / 2)
        sqrt_length = int(np.sqrt(window))
        wma1 = self.wma(series, half_length)
        wma2 = self.wma(series.shift(half_length), half_length)
        wma_diff = 2 * wma1 - wma2
        return self.wma(wma_diff, sqrt_length)

    def rsi(self, series: pd.Series, window: int) -> pd.Series:
        """Calculates the Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self) -> pd.DataFrame:
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        ema_12 = self.ema(self.df["close"], window=12)
        ema_26 = self.ema(self.df["close"], window=26)
        macd = ema_12 - ema_26
        signal = self.ema(macd, window=9)
        histogram = macd - signal
        return pd.DataFrame({"macd": macd, "signal": signal, "histogram": histogram})

    def calculate_adx(self) -> pd.DataFrame:
        """Calculates the Average Directional Index (ADX)."""
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]
        window = 14

        plus_dm = high.diff()
        plus_dm = plus_dm.where((plus_dm > low.diff().abs()) & (plus_dm > 0), 0)

        minus_dm = low.diff().abs()
        minus_dm = minus_dm.where(
            (minus_dm > high.diff()) & (minus_dm > 0), 0
        )

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window).mean()
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        adx = dx.rolling(window).mean()

        return pd.DataFrame({
            "ADX": adx,
            "+DI": plus_di,
            "-DI": minus_di,
        })

    def calculate_aroon(self) -> pd.DataFrame:
        """Calculates the Aroon Indicator."""
        window = 25
        close = self.df["close"]
        aroon_up = (
            100
            * (window - close.rolling(window).apply(lambda x: x.argmin()))
            / window
        )
        aroon_down = (
            100
            * (window - close.rolling(window).apply(lambda x: x.argmax()))
            / window
        )

        return pd.DataFrame({"Aroon Up": aroon_up, "Aroon Down": aroon_down})

    def calculate_atr(self) -> pd.Series:
        """Calculates the Average True Range (ATR)."""
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]
        window = 14

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window).mean()
        return atr

    def calculate_obv(self) -> pd.Series:
        """Calculates the On Balance Volume (OBV)."""
        close = self.df["close"]
        volume = self.df["volume"]
        obv = (volume * np.sign(close.diff())).fillna(0).cumsum()
        return obv

    def calculate_vpt(self) -> pd.Series:
        """Calculates the Volume Price Trend (VPT)."""
        close = self.df["close"]
        volume = self.df["volume"]
        vpt = (volume * (close.diff() / close.shift())).fillna(0).cumsum()
        return vpt

    def calculate_fibonacci_pivot(self, previous_day_high: float = None, previous_day_low: float = None, previous_day_close: float = None) -> pd.DataFrame:
        """Calculates Fibonacci pivot points and support/resistance levels.

        Args:
            previous_day_high (float): High of the previous day
            previous_day_low (float): Low of the previous day
            previous_day_close (float): Close of the previous day

        Returns:
            pd.DataFrame: DataFrame with Fibonacci levels.
        """
        if previous_day_high is None or previous_day_low is None or previous_day_close is None:
            if len(self.df) == 0:
                raise ValueError("Dataframe can not be empty")
            # Use the last row of the dataframe to calculate the pivot
            pivot_high = self.df["high"].iloc[-1]
            pivot_low = self.df["low"].iloc[-1]
            pivot_close = self.df["close"].iloc[-1]
        else:
            pivot_high = previous_day_high
            pivot_low = previous_day_low
            pivot_close = previous_day_close

        pivot = (pivot_high + pivot_low + pivot_close) / 3

        levels = [
            0.0,
            0.236,
            0.382,
            0.5,
            0.618,
            0.786,
            1.0,
        ]  # Fibonacci retracement/extension levels

        fib_levels = pd.DataFrame()

        fib_levels["levels"] = levels
        fib_levels["support"] = pivot - (pivot_high - pivot_low) * fib_levels["levels"]
        fib_levels["resistance"] = pivot + (pivot_high - pivot_low) * fib_levels["levels"]

        # Adjust for readability
        fib_levels["levels"] = fib_levels["levels"]
        fib_levels = fib_levels.set_index('levels')
        return fib_levels

    # ... (Other improved methods would be implemented here) ...
    def _determine_trend_adx(self) -> str:
        """Determines the trend using ADX."""
        # Revised Trend identification logic using ADX
        adx = self.df["ADX"].iloc[-1]
        plus_di = self.df["+DI"].iloc[-1]
        minus_di = self.df["-DI"].iloc[-1]

        if adx > self.config.adx_threshold:  # Using adx_threshold from config
            if plus_di > minus_di:
                return "Uptrend"
            elif minus_di > plus_di:
                return "Downtrend"
        return "Sideways"

    def _find_levels(self):
        """Finds support and resistance levels using fractals."""
        levels = []
        for i in range(2, len(self.df) - 2):
            if self.is_support(self.df, i):
                l = self.df["low"][i] # get the actual value
                if isinstance(l, pd.Series):
                     l = l.iloc[0] #get the first element
                if l not in levels:
                    levels.append((l, "support"))  # add type of level

            elif self.is_resistance(self.df, i):
                 l = self.df["high"][i]
                 if isinstance(l, pd.Series):
                     l = l.iloc[0] #get the first element
                 if l not in levels:
                    levels.append((l, "resistance"))  # add type of level

        self.levels = levels  # store levels as attribute

    def is_support(self, df, i):
        """Checks if a point is a support level."""
        cond1 = df["low"][i] < df["low"][i - 1]
        cond2 = df["low"][i] < df["low"][i + 1]
        cond3 = df["low"][i + 1] < df["low"][i + 2]
        cond4 = df["low"][i - 1] < df["low"][i - 2]
        return all([cond1, cond2, cond3, cond4])

    def is_resistance(self, df, i):
        """Checks if a point is a resistance level."""
        cond1 = df["high"][i] > df["high"][i - 1]
        cond2 = df["high"][i] > df["high"][i + 1]
        cond3 = df["high"][i + 1] > df["high"][i + 2]
        cond4 = df["high"][i - 1] > df["high"][i - 2]
        return all([cond1, cond2, cond3, cond4])

    def find_nearest_level(self, current_price, level_type="support"):
        """Finds the nearest support or resistance level to the current price."""
        if not self.levels:  # Check if levels have been calculated
            return None

        nearest_level = None
        min_distance = float("inf")  # initialize for comparison

        for level, type in self.levels:
            if type == level_type:
                distance = abs(current_price - level)

                if distance < min_distance:
                    min_distance = distance
                    nearest_level = level

        return nearest_level

    # ... (the rest of your code with other improved functions)

    def analyze_momentum(self, df: pd.DataFrame) -> str:
        """Analyzes momentum using RSI."""
        rsi_value = df["RSI"].iloc[-1]
        if rsi_value > self.config.rsi_overbought:
            return "Overbought"
        elif rsi_value < self.config.rsi_oversold:
            return "Oversold"
        else:
            return "Neutral"

    def analyze_volume(self, df: pd.DataFrame) -> str:
        """Analyzes volume using OBV and VPT."""
        obv_trend = df["OBV"].diff().iloc[-1]
        vpt_trend = df["VPT"].diff().iloc[-1]

        if obv_trend > 0 and vpt_trend > 0:
            return "Strong Buying"
        elif obv_trend < 0 and vpt_trend < 0:
            return "Strong Selling"
        elif obv_trend > 0 > vpt_trend:
            return "Possible Divergence (Bullish)"  # OBV positive, VPT negative
        elif obv_trend < 0 < vpt_trend:
            