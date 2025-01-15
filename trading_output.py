import logging
from typing import Dict, List, Tuple, Optional
from colorama import Fore, Style
from config import Config


class TradingOutput:
    """Handles the output of trading analysis results."""

    def __init__(self, logger: logging.Logger, config: Config):
        """Initializes with the logger and configuration."""
        self.logger = logger
        self.config = config

    def display_analysis(self, results: Dict):
        """Displays the detailed analysis results with color."""
        current_price = results.get("current_price", "N/A")
        self.logger.info(
            f" {Fore.YELLOW}--- Analysis Results for {self.config.symbol} ({self.config.interval}) ---{Style.RESET_ALL}"
        )
        self.logger.info(
            f" {Fore.YELLOW}Symbol: {Fore.MAGENTA} {self.config.symbol}{Style.RESET_ALL}"
        )
        self.logger.info(
            f" {Fore.YELLOW}Timeframe: {Fore.MAGENTA} {self.config.interval}{Style.RESET_ALL}"
        )
        self.logger.info(
            f" {Fore.YELLOW}Current Price: {Fore.CYAN} {current_price:.4f}{Style.RESET_ALL}"
        )

        self._log_trend_analysis(results.get("trend", "N/A"))
        self._log_momentum_analysis(results)
        self._log_volume_analysis(results.get("volume", "N/A"))
        self._log_rsi_analysis(
            results.get("rsi_analysis", "N/A"), results.get("rsi_value", "N/A")
        )
        self._log_macd_analysis(results.get("macd_analysis", "N/A"))
        self._log_aroon_analysis(results.get("aroon_analysis", "N/A"))
        self._log_support_resistance(
            results.get("nearest_supports"),
            results.get("nearest_resistances"),
            current_price,
        )
        self._log_fibonacci_levels(results.get("fibonacci_levels", {}))
        self._log_trade_suggestions(results.get("trade_suggestions", []))
        self._log_scores(
            results.get("long_score", "N/A"), results.get("short_score", "N/A")
        )
        self.logger.info(Style.RESET_ALL)

    def _log_trend_analysis(self, trend: str):
        self.logger.info(
            f" {Fore.YELLOW}Trend:{Style.RESET_ALL} {self._colorize_trend(trend)}"
        )

    def _log_momentum_analysis(self, results: Dict):
        momentum = results.get("momentum", "N/A")
        self.logger.info(
            f" {Fore.YELLOW}Momentum:{Style.RESET_ALL} {self._colorize_momentum(momentum)}"
        )

    def _log_volume_analysis(self, volume_analysis: str):
        self.logger.info(
            f" {Fore.YELLOW}Volume:{Style.RESET_ALL} {self._colorize_volume_analysis(volume_analysis)}"
        )

    def _log_rsi_analysis(self, rsi_analysis: str, rsi_value: float):
        self.logger.info(
            f" {Fore.YELLOW}RSI:{Style.RESET_ALL} {self._colorize_rsi_analysis(rsi_analysis)} ({Fore.CYAN}{rsi_value:.2f}{Style.RESET_ALL})"
        )

    def _log_macd_analysis(self, macd_analysis: str):
        self.logger.info(
            f" {Fore.YELLOW}MACD:{Style.RESET_ALL} {self._colorize_macd_analysis(macd_analysis)}"
        )

    def _log_aroon_analysis(self, aroon_analysis: str):
        self.logger.info(
            f" {Fore.YELLOW}Aroon:{Style.RESET_ALL} {self._colorize_aroon_analysis(aroon_analysis)}"
        )

    def _log_support_resistance(
        self,
        supports: Optional[float],
        resistances: Optional[float],
        current_price: float,
    ):
        self.logger.info(f" {Fore.YELLOW}Support/Resistance:{Style.RESET_ALL}")
        if supports is not None:
            distance = current_price - supports
            self.logger.info(
                f"  {Fore.GREEN}Nearest Support:{Style.RESET_ALL} {Fore.CYAN}{supports:.4f}{Style.RESET_ALL} (Distance: {Fore.GREEN}{distance:.4f}{Style.RESET_ALL})"
            )
        else:
            self.logger.info(
                f"  {Fore.GREEN}Nearest Support:{Style.RESET_ALL} {Fore.YELLOW}None Detected{Style.RESET_ALL}"
            )

        if resistances is not None:
            distance = current_price - resistances
            self.logger.info(
                f"  {Fore.RED}Nearest Resistance:{Style.RESET_ALL} {Fore.CYAN}{resistances:.4f}{Style.RESET_ALL} (Distance: {Fore.RED}{distance:.4f}{Style.RESET_ALL})"
            )
        else:
            self.logger.info(
                f"  {Fore.RED}Nearest Resistance:{Style.RESET_ALL} {Fore.YELLOW}None Detected{Style.RESET_ALL}"
            )

    def _log_fibonacci_levels(self, fib_levels: Dict):
        """Logs Fibonacci pivot levels."""
        self.logger.info(f" {Fore.YELLOW}Fibonacci Pivot Levels:{Style.RESET_ALL}")
        if fib_levels:
            for level_name, levels in fib_levels.items():
                if isinstance(levels, dict):
                    self.logger.info(
                        f"  {Fore.MAGENTA}{level_name.capitalize()}:{Style.RESET_ALL}"
                    )
                    for level, value in levels.items():
                        self.logger.info(
                            f"    - {level}: {Fore.CYAN}{value:.4f}{Style.RESET_ALL}"
                        )
        else:
            self.logger.info(
                f"   {Fore.YELLOW}No Fibonacci pivot levels calculated.{Style.RESET_ALL}"
            )

    def _log_trade_suggestions(self, trade_suggestions: List[Dict]):
        self.logger.info(f" {Fore.YELLOW}Trade Suggestions:{Style.RESET_ALL}")
        if trade_suggestions:
            for suggestion in trade_suggestions:
                signal = suggestion.get("signal")
                confidence = suggestion.get("confidence")
                if signal == "long":
                    self.logger.info(
                        f"   {Fore.GREEN}Long Signal:{Style.RESET_ALL} Confidence: {Fore.CYAN}{confidence:.2f}{Style.RESET_ALL}"
                    )
                elif signal == "short":
                    self.logger.info(
                        f"   {Fore.RED}Short Signal:{Style.RESET_ALL} Confidence: {Fore.CYAN}{confidence:.2f}{Style.RESET_ALL}"
                    )
        else:
            self.logger.info(
                f"   {Fore.YELLOW}No trade suggestions at this time.{Style.RESET_ALL}"
            )

    def _log_scores(self, long_score: float, short_score: float):
        self.logger.info(
            f" {Fore.YELLOW}Scores:{Style.RESET_ALL} Long: {Fore.GREEN}{long_score:.2f}{Style.RESET_ALL}, Short: {Fore.RED}{short_score:.2f}{Style.RESET_ALL}"
        )

    # --- Colorizing Functions ---

    def _colorize_rsi_value(self, rsi_value: float) -> str:
        if rsi_value >= self.config.rsi_overbought:
            return f"{Fore.RED}Overbought{Style.RESET_ALL}"
        elif rsi_value <= self.config.rsi_oversold:
            return f"{Fore.GREEN}Oversold{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}Neutral{Style.RESET_ALL}"

    def _colorize_macd_analysis(self, macd_analysis: str) -> str:
        if macd_analysis == "Bullish":
            return f"{Fore.GREEN}Bullish{Style.RESET_ALL}"
        elif macd_analysis == "Bearish":
            return f"{Fore.RED}Bearish{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}Neutral{Style.RESET_ALL}"

    def _colorize_aroon_analysis(self, aroon_analysis: str) -> str:
        if aroon_analysis == "Strong Uptrend":
            return f"{Fore.GREEN}Strong Uptrend{Style.RESET_ALL}"
        elif aroon_analysis == "Strong Downtrend":
            return f"{Fore.RED}Strong Downtrend{Style.RESET_ALL}"
        elif aroon_analysis == "Weak Uptrend":
            return f"{Fore.GREEN}Weak Uptrend{Style.RESET_ALL}"
        elif aroon_analysis == "Weak Downtrend":
            return f"{Fore.RED}Weak Downtrend{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}Neutral{Style.RESET_ALL}"

    def _colorize_momentum(self, momentum: str) -> str:
        if momentum == "Overbought":
            return f"{Fore.RED}Overbought{Style.RESET_ALL}"
        elif momentum == "Oversold":
            return f"{Fore.GREEN}Oversold{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}Neutral{Style.RESET_ALL}"

    def _colorize_volume_analysis(self, volume_analysis: str) -> str:
        if volume_analysis == "Strong Buying":
            return f"{Fore.GREEN}Strong Buying{Style.RESET_ALL}"
        elif volume_analysis == "Strong Selling":
            return f"{Fore.RED}Strong Selling{Style.RESET_ALL}"
        elif volume_analysis == "Possible Divergence (Bullish)":
            return f"{Fore.GREEN}Possible Divergence (Bullish){Style.RESET_ALL}"
        elif volume_analysis == "Possible Divergence (Bearish)":
            return f"{Fore.RED}Possible Divergence (Bearish){Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}Neutral{Style.RESET_ALL}"

    def _colorize_trend(self, trend: str) -> str:
        if trend == "Uptrend":
            return f"{Fore.GREEN}Uptrend{Style.RESET_ALL}"
        elif trend == "Downtrend":
            return f"{Fore.RED}Downtrend{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}Sideways{Style.RESET_ALL}"
