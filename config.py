import os
import json


class Config:
    def __init__(self, config_file="config.py"):
        self.config = self._load_config(config_file)
        self.api_key = self.config.get("bybit_api_key")
        self.api_secret = self.config.get("bybit_api_secret")
        self.testnet = self.config.get("testnet", False)
        self.symbol = self.config.get("symbol", "BTCUSDT")
        self.interval = self.config.get("interval", "5")
        self.data_limit = self.config.get("data_limit", 200)
        self.ma_periods_short = self.config.get("ma_periods_short", 20)
        self.ma_periods_long = self.config.get("ma_periods_long", 50)
        self.fma_period = self.config.get("fma_period", 9)
        self.rsi_period = self.config.get("rsi_period", 14)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        self.adx_threshold = self.config.get("adx_threshold", 25)
        self.atr_multiplier = self.config.get("atr_multiplier", 1.0)
        self.signal_threshold = self.config.get("signal_threshold", 3)

    def _load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(
                f"Warning: Configuration file '{config_file}' not found. Using default values or environment variables."
            )
            return {}  # Return empty dict, attributes will use defaults or None


# Example config.json:
# {
#   "bybit_api_key": "YOUR_API_KEY",
#   "bybit_api_secret": "YOUR_API_SECRET",
#   "testnet": false,
#   "symbol": "BTCUSDT",
#   "interval": "15",
#   "data_limit": 300,
#   "ma_periods_short": 10,
#   "ma_periods_long": 30,
#   "fma_period": 7,
#   "rsi_period": 14,
#   "rsi_overbought": 75,
#   "rsi_oversold": 25,
#   "adx_threshold": 20,
#   "atr_multiplier": 1.5,
#   "signal_threshold": 4
# }
