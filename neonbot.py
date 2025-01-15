import os
import logging
import requests
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD, EMAIndicator, ADXIndicator
from ta.momentum import rsi
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hmac
import hashlib
import time
from typing import List, Tuple
from colorama import init, Fore, Style
from zoneinfo import ZoneInfo
from fib import fib

# --- Constants ---
LOG_DIR = "botlogs"
ST_LOUIS_TZ = ZoneInfo("America/Chicago")
MAX_RETRIES = 3
RETRY_DELAY = 5
VALID_INTERVALS = [1,_3,_5,_15,_30,_60,_240]
CLUSTER_SENSITIVITY = 0.05
SUPPORT_RESISTANCE_WINDOW = 14
VOLUME_LOOKBACK = 5
FAST_MA_WINDOW = 12
SLOW_MA_WINDOW = 26
RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ADX_THRESHOLD = 25

# ... (rest of the code remains the same until the TradingAnalyzer class)

class TradingAnalyzer:
    def __init__(self, symbol: str, interval: str, logger: logging.Logger):
        self.logger = logger
        self.config = Config()
        self.bybit = Bybit(self.config, logger)
        self.symbol = symbol.upper()
        self.interval = interval
        self.df = self.fetch_and_prepare_data()
        if self.df.empty:
            raise ValueError(
                "Failed to fetch data. Check your API key, symbol, and interval.")
        self._add_technical_indicators()

    def calculate_fibonacci_pivots(self, high, low, close):
        pivot_point = (high + low + close) / 3
        s1 = pivot_point - 0.382 * (high - low)
        s2 = pivot_point - 0.618 * (high - low)
        s3 = pivot_point - 1.000 * (high - low)
        r1 = pivot_point + 0.382 * (high - low)
        r2 = pivot_point + 0.618 * (high - low)
        r3 = pivot_point + 1.000 * (high - low)
        return {
            "S3": s3,
            "S2": s2,
            "S1": s1,
            "R1": r1,
            "R2": r2,
            "R3": r3,
        }

    def analyze(self, current_price: float):
        supports, resistances = self.identify_support_resistance()
        trend = self.determine_trend()
        entry_signal = self.determine_entry_signal(
            current_price, supports, resistances)

        # Fibonacci Levels
        high = self.df["high"].max()
        low = self.df["low"].min()
        fib_levels = fib(high, low)
        sorted_fib_levels = sorted(fib_levels.values())
        five_nearest_fib = sorted(
            sorted_fib_levels,
            key=lambda x: abs(x - current_price)
        )[:5]

        # Fibonacci Pivots
        fib_pivots = self.calculate_fibonacci_pivots(
            self.df["high"].max(), self.df["low"].min(), self.df["close"].iloc[-1]
        )

        self.extended_trend_indicator()
        self.logger.info(
            f"{Fore.YELLOW}Current Price: {current_price:.2f}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.YELLOW}Trend: {trend}{Style.RESET_ALL}")
        self.logger.info(
            f"{Fore.YELLOW}5 Nearest Fibonacci Levels: {five_nearest_fib}{Style.RESET_ALL}")

        # Print Fibonacci Pivots
        print(f"{Fore.MAGENTA}Fibonacci Pivots:{Style.RESET_ALL}")
        for level, value in fib_pivots.items():
            print(f"{level}: {value:.2f}")

        if entry_signal:
            self.logger.info(
                f"{Fore.GREEN}Trade Signal: {entry_signal}{Style.RESET_ALL}")
        else:
            self.logger.info("No trade signal.")

    def fetch_and_prepare_data(self):
        df = self.bybit.fetch_klines(self.symbol, self.interval, limit=200)
        if not df.empty:
            df = df.sort_values("start_time")
            df = df.reset_index(drop=True)
            return df
        else:
            return pd.DataFrame()

    def _add_technical_indicators(self):
        self.df["SMA_9"] = SMAIndicator(
            self.df["close"], window=9).sma_indicator()
        self.df["MACD"] = MACD(self.df["close"]).macd()
        self.df["RSI"] = rsi(self.df["close"], window=RSI_WINDOW)
        self.df["EMA_200"] = EMAIndicator(
            self.df["close"], window=200).ema_indicator()
        self.df["fast_ma"] = EMAIndicator(
            self.df["close"], window=FAST_MA_WINDOW).ema_indicator()
        self.df["slow_ma"] = EMAIndicator(
            self.df["close"], window=SLOW_MA_WINDOW).ema_indicator()
        self.df["ADX"] = ADXIndicator(
            self.df["high"],
            self.df["low"],
            self.df["close"]).adx()
        self.df["momentum"] = self.df["close"].diff()
        self.df["momentum_wma_10"] = self.df["momentum"].rolling(
            window=10, win_type="triang").mean()
        self.df["volume_ma_10"] = SMAIndicator(
            self.df["volume"], window=10).sma_indicator()

    def identify_support_resistance(self) -> Tuple[List[float], List[float]]:
        data = self.df["close"].values
        maxima, minima = [], []
        for i in range(
                SUPPORT_RESISTANCE_WINDOW,
                len(data) - SUPPORT_RESISTANCE_WINDOW):
            if data[i] == max(data[i_-_SUPPORT_RESISTANCE_WINDOW:i_+_SUPPORT_RESISTANCE_WINDOW]):
                maxima.append(data[i])
            if data[i] == min(data[i_-_SUPPORT_RESISTANCE_WINDOW:i_+_SUPPORT_RESISTANCE_WINDOW]):
                minima.append(data[i])
        return sorted(maxima, reverse=True), sorted(minima)

    def determine_trend(self) -> str:
        current_adx = self.df["ADX"].iloc[-1]
        if current_adx < ADX_THRESHOLD:
            return "neutral"
        fast_ma, slow_ma = self.df["fast_ma"].iloc[-1], self.df["slow_ma"].iloc[-1]
        if fast_ma > slow_ma:
            return "bullish"
        elif fast_ma < slow_ma:
            return "bearish"
        return "neutral"

    def is_near_level(self, price, level, threshold=0.01):
        return abs(price - level) / price <= threshold

    def determine_entry_signal(
            self,
            current_price: float,
            supports,
            resistances) -> str | None:
        trend = self.determine_trend()
        if trend == "neutral":
            return None

        fast_ma = self.df["fast_ma"].iloc[-1]
        slow_ma = self.df["slow_ma"].iloc[-1]
        open_price = self.df["open"].iloc[-1]
        hma_9 = self.df["SMA_9"].iloc[-1]  # Using SMA_9 as a substitute for HMA_9

        # Fibonacci Pivots
        fib_pivots = self.calculate_fibonacci_pivots(
            self.df["high"].max(), self.df["low"].min(), self.df["close"].iloc[-1]
        )
        near_fib = any(self.is_near_level(current_price, level)
                       for level in fib_pivots.values())
        near_support = any(self.is_near_level(current_price, s)
                           for s in supports)
        near_resistance = any(
            self.is_near_level(
                current_price,
                r) for r in resistances)

        if (
            fast_ma > slow_ma
            and open_price < hma_9
            and self.df["close"].iloc[-1] > hma_9
            and self.df["volume"].iloc[-1] > self.df["volume_ma_10"].iloc[-1]
            and self.df["momentum"].iloc[-1] > self.df["momentum_wma_10"].iloc[-1]
            and (near_fib or near_support or near_resistance)
        ):
            return "long"

        return None

    def extended_trend_indicator(self) -> str:
        current_adx = self.df["ADX"].iloc[-1]
        fast_ma = self.df["fast_ma"].iloc[-1]
        slow_ma = self.df["slow_ma"].iloc[-1]
        macd_value = self.df["MACD"].iloc[-1]

        trend_strength = "weak" if current_adx < ADX_THRESHOLD else "strong"
        trend_direction = "neutral"

        if fast_ma > slow_ma and macd_value > 0:
            trend_direction = "bullish"
        elif fast_ma < slow_ma and macd_value < 0:
            trend_direction = "bearish"

        self.logger.info(f"{Fore.YELLOW}Trend Analysis:{Style.RESET_ALL}")
        self.logger.info(f"  ADX Value: {current_adx:.2f} ({trend_strength})")
        self.logger.info(f"  Fast EMA ({FAST_MA_WINDOW}): {fast_ma:.2f}")
        self.logger.info(f"  Slow EMA ({SLOW_MA_WINDOW}): {slow_ma:.2f}")
        self.logger.info(f"  MACD: {macd_value:.2f}")
        self.logger.info(
            f"{Fore.CYAN}  Overall Trend: {trend_strength.capitalize()} {trend_direction.capitalize()}{Style.RESET_ALL}")

        return f"{trend_strength} {trend_direction}"

