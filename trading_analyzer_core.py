import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import rsi
from ta.volume import on_balance_volume, volume_price_trend
from bybit_api import BybitAPI


class TradingAnalyzer:
    """Analyzes market data for trading insights."""

    def __init__(self, config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.api = BybitAPI(config, logger)
        self.df = self.api.fetch_klines(config.symbol, config.interval, limit=200)

    def analyze(self, current_price: float) -> Dict:
        """Analyzes the market data and returns insights."""
        if self.df.empty:
            self.logger.error("DataFrame is empty. Cannot perform analysis.")
            return {}

        # Calculate indicators
        self.calculate_indicators()

        # Perform analysis
        trend = self.determine_trend()
        rsi_value = self.df["RSI"].iloc[-1]
        rsi_analysis = self.analyze_rsi(rsi_value)

        # Calculate levels
        nearest_supports, nearest_resistances = self.find_nearest_levels(current_price)

        # Compile results
        return {
            "symbol": self.config.symbol,
            "interval": self.config.interval,
            "current_price": current_price,
            "trend": trend,
            "rsi_value": rsi_value,
            "rsi_analysis": rsi_analysis,
            "nearest_supports": nearest_supports,
            "nearest_resistances": nearest_resistances,
        }

    def calculate_indicators(self):
        """Calculates indicators like RSI and moving averages."""
        self.df["RSI"] = rsi(self.df["close"], window=self.config.rsi_period)

    def determine_trend(self) -> str:
        """Determines the trend based on moving averages."""
        short_ma = EMAIndicator(
            self.df["close"], self.config.ma_periods_short
        ).ema_indicator()
        long_ma = EMAIndicator(
            self.df["close"], self.config.ma_periods_long
        ).ema_indicator()
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return "Bullish"
        elif short_ma.iloc[-1] < long_ma.iloc[-1]:
            return "Bearish"
        return "Neutral"

    def analyze_rsi(self, rsi_value: float) -> str:
        """Analyzes RSI value."""
        if rsi_value > self.config.rsi_overbought:
            return "Overbought"
        elif rsi_value < self.config.rsi_oversold:
            return "Oversold"
        return "Neutral"

    def find_nearest_levels(self, current_price: float) -> Tuple[List, List]:
        """Finds nearest support and resistance levels."""
        supports = [("Support", current_price * 0.95, 0.05)]
        resistances = [("Resistance", current_price * 1.05, 0.05)]
        return supports, resistances
