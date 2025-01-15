import logging
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np
from bybit_api import BybitAPI
import threading


class TradingAnalyzer:
    def __init__(self, config: Dict, logger: logging.Logger, api: BybitAPI):
        self.config = config
        self.logger = logger
        self.api = api
        self.historical_data = deque(maxlen=self.config.data_limit)
        self.last_analysis = None
        self.analysis_lock = threading.Lock()

    def calculate_pivot_levels(
        self, high: float, low: float, close: float
    ) -> Dict[str, float]:
        """Calculates standard pivot points and Fibonacci extensions."""
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

        fib_382 = pivot + (r1 - s1) * 0.382
        fib_618 = pivot + (r1 - s1) * 0.618

        return {
            "pivot": pivot,
            "r1": r1,
            "s1": s1,
            "r2": r2,
            "s2": s2,
            "r3": r3,
            "s3": s3,
            "fib_382": fib_382,
            "fib_618": fib_618,
        }

    def calculate_vpt(self, closes: np.array, volumes: np.array) -> np.array:
        """Calculates Volume Price Trend (VPT) using numpy."""
        price_change = np.diff(closes)
        vpt = np.zeros_like(closes)
        vpt[1:] = np.cumsum(volumes[1:] * price_change / closes[:-1])
        return vpt

    def detect_support_resistance(
        self, highs: np.array, lows: np.array, closes: np.array, window_size: int = 20
    ) -> List[float]:
        """
        Detects support and resistance levels using fractals and Fibonacci levels.
        """
        support_levels = []
        resistance_levels = []

        # Fractal-based S/R
        for i in range(window_size, len(highs) - window_size):
            is_support = (
                lows[i] <= lows[i - window_size : i + window_size + 1].min()
                and closes[i] > lows[i]
            )
            is_resistance = (
                highs[i] >= highs[i - window_size : i + window_size + 1].max()
                and closes[i] < highs[i]
            )

            if is_support:
                support_levels.append(lows[i])
            if is_resistance:
                resistance_levels.append(highs[i])

        # Fibonacci-based S/R
        fib_levels = self.calculate_fibonacci_levels(
            highs, lows, closes
        )  # added closes
        support_levels.extend(fib_levels.get("supports", []))
        resistance_levels.extend(fib_levels.get("resistances", []))

        support_levels = self._cluster_levels(support_levels)
        resistance_levels = self._cluster_levels(resistance_levels)

        return support_levels + resistance_levels

    def calculate_fibonacci_levels(
        self, highs: np.array, lows: np.array, closes: np.array
    ) -> Dict[str, List[float]]:
        """
        Calculates Fibonacci retracement levels as support and resistance.
        """
        # Fibonacci retracement levels
        high = highs[-1]
        low = lows[-1]
        diff = high - low
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]  # Common Fibonacci levels

        supports = []
        resistances = []
        for level in levels:
            retracement = high - diff * level
            if retracement < closes[-1]:
                supports.append(retracement)
            else:
                resistances.append(retracement)

        return {"supports": supports, "resistances": resistances}

    def _cluster_levels(
        self, levels: List[float], tolerance: float = 0.01
    ) -> List[float]:
        """Clusters similar price levels together."""
        if not levels:
            return []

        clustered_levels = []
        levels.sort()
        current_cluster = [levels[0]]

        for i in range(1, len(levels)):
            if (levels[i] - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(levels[i])
            else:
                clustered_levels.append(np.mean(current_cluster))
                current_cluster = [levels[i]]

        clustered_levels.append(np.mean(current_cluster))
        return clustered_levels

    def calculate_rsi(self, closes: np.array, period: int = 14) -> np.array:
        """Calculates the Relative Strength Index (RSI)."""
        delta = np.diff(closes)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = np.zeros_like(closes)
        rsi[:period] = 100 - (100 / (1 + rs))

        for i in range(period, len(closes)):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(
        self,
        closes: np.array,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[np.array, np.array, np.array]:
        """Calculates MACD, signal line, and histogram."""
        ema_fast = self.calculate_ema(closes, fast_period)
        ema_slow = self.calculate_ema(closes, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal_period)
        macd_histogram = macd_line - signal_line

        return macd_line, signal_line, macd_histogram

    def calculate_ema(self, data: np.array, period: int) -> np.array:
        """Calculates Exponential Moving Average (EMA)."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[period - 1] = np.mean(data[:period])

        for i in range(period, len(data)):
            ema[i] = (data[i] * alpha) + (ema[i - 1] * (1 - alpha))

        return ema

    def calculate_sma(self, data: np.array, period: int) -> np.array:
        """Calculates Simple Moving Average (SMA)."""
        sma = np.zeros_like(data)
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1 : i + 1])
        return sma

    def calculate_bbands(
        self, closes: np.array, period: int = 20, std_dev: int = 2
    ) -> Tuple[np.array, np.array, np.array]:
        """Calculates Bollinger Bands."""
        sma = self.calculate_sma(closes, period)
        rolling_std = np.zeros_like(closes)

        for i in range(period - 1, len(closes)):
            rolling_std[i] = np.std(closes[i - period + 1 : i + 1])

        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)

        return upper_band, sma, lower_band

    def calculate_aroon(
        self, highs: np.array, lows: np.array, period: int = 14
    ) -> Tuple[np.array, np.array]:
        """Calculates Aroon Up and Aroon Down."""
        aroon_up = np.zeros_like(highs)
        aroon_down = np.zeros_like(lows)

        for i in range(period, len(highs)):
            high_period = highs[i - period : i + 1]
            low_period = lows[i - period : i + 1]

            aroon_up[i] = ((period - np.argmax(high_period)) / period) * 100
            aroon_down[i] = ((period - np.argmin(low_period)) / period) * 100

        return aroon_up, aroon_down

    def calculate_obv(self, closes: np.array, volumes: np.array) -> np.array:
        """Calculates On-Balance Volume (OBV)."""
        obv = np.zeros_like(closes)
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif closes[i] < closes[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]
        return obv

    def _calculate_adx(
        self, highs: np.array, lows: np.array, closes: np.array, period: int = 14
    ) -> np.array:
        """Calculates the Average Directional Index (ADX)."""

        if len(highs) < period or len(lows) < period or len(closes) < period:
            return np.zeros_like(closes)

        tr = np.maximum(
            np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1])),
            np.abs(lows[1:] - closes[:-1]),
        )
        tr = np.insert(tr, 0, 0)  # Ensure tr has the same length as the inputs

        plus_dm = np.maximum(highs[1:] - highs[:-1], 0)
        minus_dm = np.maximum(lows[:-1] - lows[1:], 0)

        plus_dm = np.insert(plus_dm, 0, 0)
        minus_dm = np.insert(minus_dm, 0, 0)

        smooth_tr = self.calculate_ema(tr, period)
        smooth_plus_dm = self.calculate_ema(plus_dm, period)
        smooth_minus_dm = self.calculate_ema(minus_dm, period)

        plus_di = 100 * (smooth_plus_dm / smooth_tr)
        minus_di = 100 * (smooth_minus_dm / smooth_tr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        adx = self.calculate_ema(dx, period)

        return adx

    def detect_macd_divergence(
        self, closes: np.array, macd: np.array, lookback: int
    ) -> str:
        """Detects MACD divergence."""
        if len(closes) < lookback or len(macd) < lookback:
            return "None"

        price_slice = closes[-lookback:]
        macd_slice = macd[-lookback:]

        if len(price_slice) < 2 or len(macd_slice) < 2:
            return "None"

        # Bearish Divergence
        if price_slice[0] < price_slice[-1] and macd_slice[0] > macd_slice[-1]:
            return "Bearish"
        # Bullish Divergence
        elif price_slice[0] > price_slice[-1] and macd_slice[0] < macd_slice[-1]:
            return "Bullish"
        else:
            return "None"

    def calculate_weighted_score(
        self,
        closes: np.array,
        volumes: np.array,
        rsi: float,
        macd_hist: float,
        macd_divergence: str,
        ema_short: np.array,
        ema_long: np.array,
    ) -> Tuple[float, float]:
        # Ensure required data
        if (
            len(closes) < self.config.ma_periods_long
            or len(closes) < self.config.ma_periods_short
        ):
            return 0, 0

        # Momentum Score
        momentum_score = 0
        if rsi > self.config.rsi_overbought:
            momentum_score = -1
        elif rsi < self.config.rsi_oversold:
            momentum_score = 1

        # Volume Score
        volume_score = 0
        volume_trend = self.analyze_volume_trend(volumes)
        if volume_trend == "increasing":
            volume_score = 1
        elif volume_trend == "decreasing":
            volume_score = -1

        # Divergence Score
        divergence_score = 0
        if macd_divergence == "Bullish":
            divergence_score = 1
        elif macd_divergence == "Bearish":
            divergence_score = -1

        # Crossover Score
        crossover_score = 0
        if ema_short[-1] > ema_long[-1] and ema_short[-2] <= ema_long[-2]:
            crossover_score = 1
        elif ema_short[-1] < ema_long[-1] and ema_short[-2] >= ema_long[-2]:
            crossover_score = -1

        long_score = (
            self.config.momentum_weight * max(0, momentum_score)
            + self.config.volume_weight * max(0, volume_score)
            + self.config.divergence_weight * max(0, divergence_score)
            + self.config.crossover_weight * max(0, crossover_score)
        )
        short_score = (
            self.config.momentum_weight * min(0, momentum_score)
            + self.config.volume_weight * min(0, volume_score)
            + self.config.divergence_weight * min(0, divergence_score)
            + self.config.crossover_weight * min(0, crossover_score)
        )

        return long_score, short_score

    def adapt_parameters(self, closes: np.array, atr: np.array):
        """Adapt parameters based on volatility using ATR."""

        if len(closes) < self.config.atr_period or len(atr) < self.config.atr_period:
            return

        current_atr = atr[-1]
        if current_atr > self.config.high_volatility_threshold:
            self.config.rsi_overbought = 80
            self.config.rsi_oversold = 20
            self.config.take_profit_multiplier = (
                self.config.take_profit_multiplier * 1.2
            )
        elif current_atr < self.config.low_volatility_threshold:
            self.config.rsi_overbought = 60
            self.config.rsi_oversold = 40
            self.config.take_profit_multiplier = (
                self.config.take_profit_multiplier * 0.8
            )
        else:
            self.config.rsi_overbought = 70
            self.config.rsi_oversold = 30
            self.config.take_profit_multiplier = (
                self.config.take_profit_multiplier
            )  # retains default values

    def calculate_atr(
        self, highs: np.array, lows: np.array, closes: np.array, period: int = 14
    ) -> np.array:
        """Calculates Average True Range (ATR)."""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return np.zeros_like(closes)

        tr = np.maximum(
            np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1])),
            np.abs(lows[1:] - closes[:-1]),
        )
        tr = np.insert(tr, 0, 0)  # Ensures TR has same length as closes
        atr = self.calculate_ema(tr, period)
        return atr

    def analyze_volume_trend(self, volumes: np.array) -> str:
        """
        Analyzes the trend of volume.

        Args:
            volumes: Array of volume data.

        Returns:
            "increasing", "decreasing", or "neutral".
        """
        if len(volumes) < self.config.volume_trend_period:
            return "neutral"

        # Calculate the moving average of volume
        volume_ma = self.calculate_sma(volumes, self.config.volume_trend_period)

        # Determine the trend based on the moving average
        if volume_ma[-1] > volume_ma[-2]:
            return "increasing"
        elif volume_ma[-1] < volume_ma[-2]:
            return "decreasing"
        else:
            return "neutral"

    def determine_trend(
        self, closes: np.array, macd: np.array, aroon_up: np.array, aroon_down: np.array
    ) -> str:
        """
        Determines the trend based on MACD and Aroon indicators.

        Args:
            closes: Closing prices.
            macd: MACD line.
            aroon_up: Aroon Up values.
            aroon_down: Aroon Down values.

        Returns:
            "upward", "downward", or "neutral"
        """

        # Use EMA for trend determination
        ema_short = self.calculate_ema(closes, self.config.ma_periods_short)
        ema_long = self.calculate_ema(closes, self.config.ma_periods_long)

        if (
            ema_short[-1] > ema_long[-1]
            and macd[-1] > 0
            and aroon_up[-1] > aroon_down[-1]
        ):
            return "upward"
        elif (
            ema_short[-1] < ema_long[-1]
            and macd[-1] < 0
            and aroon_down[-1] > aroon_up[-1]
        ):
            return "downward"
        else:
            return "neutral"

    def _get_nearest_level(
        self, current_price: float, levels: List[float], is_support: bool = True
    ) -> Optional[float]:
        """Gets the nearest support or resistance level"""
        if not levels:
            return None

        if is_support:
            valid_levels = [level for level in levels if level < current_price]
            if not valid_levels:
                return None
            return max(valid_levels, key=lambda x: current_price - x)

        else:  # resistance
            valid_levels = [level for level in levels if level > current_price]
            if not valid_levels:
                return None
            return min(valid_levels, key=lambda x: x - current_price)

    def generate_signals(
        self,
        current_price: float,
        rsi: float,
        macd: float,
        macd_signal: float,
        bb_middle: float,
        bb_upper: float,
        bb_lower: float,
        trend: str,
        support_resistance: List[float],
        adx: float,
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Generates long and short entry signals based on indicators, trend, S/R, ADX and weighted score.
        """
        long_signal = None
        short_signal = None

        if adx < self.config.adx_threshold:
            return None, None  # if trend is weak dont trade

        long_score, short_score = (
            self.last_analysis.get("long_score", 0),
            self.last_analysis.get("short_score", 0),
        )  # access scores from analysis output

        # Check for Long Entry
        if trend == "upward" and long_score > self.config.signal_threshold:
            entry_level = self._get_nearest_level(
                current_price, support_resistance, is_support=True
            )
            if entry_level:
                short_entry_level = self._get_nearest_level(
                    current_price, support_resistance, is_support=False
                )

                if short_entry_level:
                    long_signal = {
                        "entry": entry_level,
                        "target": short_entry_level,  # Opposite level for target
                    }
        # Check for Short Entry
        elif trend == "downward" and short_score < -self.config.signal_threshold:
            entry_level = self._get_nearest_level(
                current_price, support_resistance, is_support=False
            )
            if entry_level:
                long_entry_level = self._get_nearest_level(
                    current_price, support_resistance, is_support=True
                )
                if long_entry_level:
                    short_signal = {"entry": entry_level, "target": long_entry_level}

        return long_signal, short_signal

    def analyze(self, symbol: str) -> Optional[Dict]:
        """Analyzes trading data and generates trading insights."""

        try:
            kline_data = self.api.fetch_klines(
                symbol,
                self.config.interval,
                limit=self.config.data_limit,
            )
            if kline_data.empty:
                self.logger.error("Failed to fetch kline data")
                return None

            self.historical_data.extend(kline_data.to_dict("records"))

            closes = np.array([float(kline["close"]) for kline in self.historical_data])
            highs = np.array([float(kline["high"]) for kline in self.historical_data])
            lows = np.array([float(kline["low"]) for kline in self.historical_data])
            volumes = np.array(
                [float(kline["volume"]) for kline in self.historical_data]
            )

            rsi = self.calculate_rsi(closes, period=self.config.rsi_period)
            macd, macdsignal, macdhist = self.calculate_macd(
                closes,
                fast_period=self.config.macd_fast,
                slow_period=self.config.macd_slow,
                signal_period=self.config.macd_signal,
            )
            bb_upper, bb_middle, bb_lower = self.calculate_bbands(
                closes, period=self.config.bbands_period
            )
            aroon_up, aroon_down = self.calculate_aroon(
                highs, lows, period=self.config.aroon_period
            )
            atr = self.calculate_atr(highs, lows, closes, period=self.config.atr_period)

            # Volume Analysis
            avg_volume = np.mean(volumes)
            is_volume_spike = volumes[-1] > avg_volume * self.config.volume_threshold
            volume_trend = self.analyze_volume_trend(
                volumes
            )  # Added volume trend analysis

            obv = self.calculate_obv(closes, volumes)
            vpt = self.calculate_vpt(closes, volumes)
            support_resistance = self.detect_support_resistance(
                highs, lows, closes, window_size=self.config.support_resistance_window
            )
            pivot_levels = self.calculate_pivot_levels(highs[-1], lows[-1], closes[-1])
            adx = self._calculate_adx(
                highs, lows, closes, period=self.config.adx_period
            )  # adx value
            macd_divergence = self.detect_macd_divergence(
                closes, macd, self.config.macd_divergence_lookback
            )  # macd divergence

            # Trend Analysis (using multiple indicators)
            trend = self.determine_trend(closes, macd, aroon_up, aroon_down)

            # Generate the weighted score based on momentum, volume, divergence, and crossover
            ema_short = self.calculate_ema(closes, self.config.ma_periods_short)
            ema_long = self.calculate_ema(closes, self.config.ma_periods_long)
            long_score, short_score = self.calculate_weighted_score(
                closes,
                volumes,
                rsi[-1],
                macdhist[-1],
                macd_divergence,
                ema_short,
                ema_long,
            )

            # Adapt parameters based on volatility
            self.adapt_parameters(closes, atr)

            current_price = closes[-1]

            with self.analysis_lock:
                self.last_analysis = {
                    "current_price": current_price,
                    "rsi": rsi[-1],
                    "macd": macd[-1],
                    "macd_signal": macdsignal[-1],
                    "macd_hist": macdhist[-1],
                    "bb_upper": bb_upper[-1],
                    "bb_middle": bb_middle[-1],
                    "bb_lower": bb_lower[-1],
                    "aroon_up": aroon_up[-1],
                    "aroon_down": aroon_down[-1],
                    "volume": volumes[-1],
                    "avg_volume": avg_volume,
                    "is_volume_spike": is_volume_spike,
                    "volume_trend": volume_trend,  # Added volume trend
                    "obv": obv[-1],
                    "vpt": vpt[-1],
                    "support_resistance": support_resistance,
                    "pivot_levels": pivot_levels,
                    "trend": trend,
                    "adx": adx[-1],
                    "macd_divergence": macd_divergence,
                    "long_score": long_score,
                    "short_score": short_score,
                    "timestamp": time.time(),
                }

            long_signal, short_signal = self.generate_signals(
                current_price,
                rsi[-1],
                macd[-1],
                macdsignal[-1],
                bb_middle[-1],
                bb_upper[-1],
                bb_lower[-1],
                trend,
                support_resistance,  # pass in support and resistance
                adx[-1],
            )
            with self.analysis_lock:
                self.last_analysis["long_signal"] = (
                    long_signal  # add long signal to last analysis output
                )
                self.last_analysis["short_signal"] = (
                    short_signal  # add short signal to last analysis output
                )

            self.logger.debug(f"Analysis results: {self.last_analysis}")
            return self.last_analysis

        except Exception as e:
            self.logger.error(f"Error during analysis: {e}", exc_info=True)
            return None
