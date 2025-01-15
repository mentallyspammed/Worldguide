import logging
import os
import pandas as pd
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError
from dotenv import load_dotenv
import time
from retry import retry
from typing import Optional
from config import Config

class BybitAPI:
    """A class to interact with the Bybit API using pybit."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initializes the Bybit API client with configuration and logger."""
        load_dotenv()  # Load variables from .env file
        self.logger = logger
        self.config = config
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.testnet = config.testnet

        # Initialize the HTTP session without the endpoint
        self.session = HTTP(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet
        )

    @retry(exceptions=(InvalidRequestError), tries=3, delay=2, backoff=2)
    def fetch_current_price(self) -> Optional[float]:
        """Fetches the current real-time price of the symbol using pybit."""
        try:
            response = self.session.get_tickers(
                category="linear", symbol=self.config.symbol
            )

            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return None

            ticker_info = response.get("result", {}).get("list", [])[0]
            if not ticker_info or "lastPrice" not in ticker_info:
                self.logger.error("Current price data unavailable.")
                return None

            return float(ticker_info["lastPrice"])
        except InvalidRequestError as e:
            self.logger.error(f"Invalid request error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error fetching current price: {e}")
            return None

    def fetch_orderbook(self, limit: int = 10) -> dict:
        """Fetch order book data from Bybit Unified Trading API."""
        try:
            response = self.session.get_orderbook(
                category="linear",
                symbol=self.config.symbol,
                limit=limit,
            )

            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return {}

            return response.get("result", {})

        except Exception as e:
            self.logger.error(f"Error fetching order book: {e}")
            return {}

    def fetch_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """
        Fetches kline data from Bybit API for Unified Trading Account (UTA)
        (specifically for USDT Perpetual contracts) using pybit, ensures it's sorted by time,
        and handles timezone conversion.
        """
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit,
            )

            # Check for API error messages
            if response["retCode"] != 0:
                self.logger.error(f"Bybit API Error: {response['retMsg']}")
                return pd.DataFrame()

            if "result" not in response or "list" not in response["result"]:
                self.logger.warning(f"No kline data found for symbol {symbol}.")
                return pd.DataFrame()

            # Adjust the columns based on the API response
            df = pd.DataFrame(
                response["result"]["list"],
                columns=[
                    "start_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover",
                ],
            )

            # Convert start_time to datetime and set timezone to UTC
            df["start_time"] = pd.to_datetime(pd.to_numeric(df["start_time"]), unit="ms", utc=True)

            # Sort by time to ensure the latest data is at the end
            df.sort_values("start_time", inplace=True)

            # Convert other columns to numeric types
            df = df.astype(
                {
                    col: float
                    for col in ["open", "high", "low", "close", "volume", "turnover"]
                }
            )

            return df

        except Exception as e:
            self.logger.error(f"Error fetching klines: {e}")
            return pd.DataFrame()