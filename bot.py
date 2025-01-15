import os
import logging
import json
from dotenv import load_dotenv
import asyncio
import random
from pybit.unified_trading import HTTP, WebSocket
import pandas as pd
import ta
from datetime import datetime
import time
from urllib.parse import urlencode
from typing import Dict, Any, List, Tuple

# --- Constants ---
CATEGORY_LINEAR = "linear"
ORDER_TYPE_MARKET = "Market"
ORDER_TYPE_LIMIT = "Limit"
TIME_IN_FORCE = "GoodTillCancel"
ORDER_SIDE_BUY = "Buy"
ORDER_SIDE_SELL = "Sell"
RETRY_COUNT = 3
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOOP_INTERVAL = 300
GRID_SPACING_PERCENTAGE = 0.5
NUM_GRID_LEVELS = 5
RISK_PERCENTAGE = 0.5

# --- Custom Exceptions ---
class BybitAPIError(Exception):
    def __init__(self, message, ret_code=None):
        super().__init__(message)
        self.ret_code = ret_code

class ConfigurationError(Exception):
    pass

class InvalidParameterError(Exception):
    pass

class InsufficientFundsError(Exception):
  pass

# --- Bybit API Class ---
class BybitAPI:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.bybit.com", ws_url: str = "wss://stream.bybit.com/v5/public/linear"):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.ws_url = ws_url
        self.retry_count = RETRY_COUNT
        self.client = HTTP(api_key=api_key, api_secret=api_secret, recv_window=5000) # Increased the receive windows
        self.logger.info(f"BybitAPI initialized. Base URL: {base_url}, WebSocket URL: {ws_url}")

    async def _request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        full_url = f"{self.base_url}{endpoint}"
        for attempt in range(self.retry_count):
            try:
                self.logger.debug(f"Making {method} request to: {full_url} - Attempt {attempt + 1}/{self.retry_count}, Params: {params}, Data: {data}")
                await asyncio.sleep(2 ** attempt + random.uniform(0, 1))  # Exponential backoff with jitter

                if method == "GET":
                     if params:
                        response = self.client.request(method=method, path=endpoint, max_retries=self.retry_count, recv_window=5000, params=params)
                     else:
                       response = self.client.request(method=method, path=endpoint, max_retries=self.retry_count, recv_window=5000)

                elif method == "POST":
                    response = self.client.request(method=method, path=endpoint, max_retries=self.retry_count, recv_window=5000, data=data)

                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if response.get("retCode") != 0:
                   self.logger.error(f"API Error: {response['retCode']} - {response.get('retMsg')} for {endpoint} with params {params}")
                   raise BybitAPIError(response['retMsg'], response['retCode'])

                return response

            except requests.exceptions.RequestException as e:
                self.logger.exception(f"Request error during {method} {full_url}: {e}")
                if attempt == self.retry_count - 1:
                    raise BybitAPIError(f"Max retries exceeded for {full_url}, request failed: {e}") from e
            except Exception as e:
                self.logger.exception(f"Unexpected error during request: {e}")
                if attempt == self.retry_count - 1:
                    raise BybitAPIError(f"Max retries exceeded, unexpected error: {e}") from e

    async def get_symbol_info(self, symbol: str) -> Dict:
        params = {"category": CATEGORY_LINEAR, "symbol": symbol}
        try:
            data = await self._request("GET", "/v5/market/instruments-info", params=params)
            return data.get('result',{}).get('list', [None])[0] if data.get('result',{}).get('list') else None
        except Exception as e:
            self.logger.exception(f"Error getting symbol info: {e}")
            return None
    
    async def get_current_price(self, symbol: str, category: str) -> float:
        params = {"symbol": symbol, "category": category}
        try:
            response = await self._request("GET", "/v5/market/tickers", params=params)
            if response and response["retCode"] == 0 and response.get("result") and len(response["result"]["list"]) > 0:
                return float(response["result"]["list"][0]["lastPrice"])
            else:
                self.logger.error(f"Error in get_current_price response: {response}")
                return None
        except Exception as e:
            self.logger.exception(f"get_current_price failed: {e}")
            return None

    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List:
        params = {"category": CATEGORY_LINEAR, "symbol": symbol, "interval": interval, "limit": limit}
        try:
            data = await self._request("GET", "/v5/market/kline", params=params)
            return data.get('result', {}).get('list')
        except Exception as e:
            self.logger.exception(f"Error getting klines: {e}")
            return None
        
    async def get_balance(self):
        try:
            response = await self._request("GET", "/v5/account/wallet-balance", params={"accountType": "UNIFIED"})
            if response and response['retCode'] == 0 and response["result"]["list"]:
                for item in response['result']['list']:
                    if item.get('coin', []):
                        for coin in item["coin"]:
                            if coin['coin'] == 'USDT':
                                return float(coin['walletBalance'])
            self.logger.error(f"Unexpected response format from get_balance: {response}")
            return None
        except Exception as e:
            self.logger.exception(f"get_balance request failed: {e}")
            return None
        
    async def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: float = None, timeInForce: str = TIME_IN_FORCE, reduce_only: bool = False, close_on_trigger: bool = False, category: str = CATEGORY_LINEAR, stop_loss: float = None, take_profit: float = None):
        """Places an order, including support for stop loss and take profit orders."""
        data = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": timeInForce,
            "reduceOnly": reduce_only,
            "closeOnTrigger": close_on_trigger,
        }

        # Include price if it's a limit order
        if order_type == ORDER_TYPE_LIMIT and price is not None:
            data["price"] = str(price)

        # Include stop loss and take profit if provided
        if stop_loss is not None:
            data["stopLoss"] = str(stop_loss)
        if take_profit is not None:
            data["takeProfit"] = str(take_profit)

        try:
            response = await self._request("POST", "/v5/order/create", data=data)
            if response and response.get("retCode") == 0:
                self.logger.info(f"Order placed successfully: {response}")
                return response
            else:
                self.logger.error(f"Failed to place order: {response}")
                return None
        except Exception as e:
            self.logger.exception(f"Error placing order: {e}")
            return None

    async def get_active_stop_loss_order(self, symbol: str, category: str):
        """Fetches the current active stop loss order for a given symbol."""
        params = {"category": category, "symbol": symbol, "orderFilter": ORDER_TYPE_MARKET}  # Adjusted to fetch stop-loss orders
        try:
            response = await self._request("GET", "/v5/order/realtime", params=params)
            if response and response["retCode"] == 0 and response["result"]["list"]:
                # Filter out orders that are not stop-loss orders
                stop_loss_orders = [
                    order
                    for order in response["result"]["list"]
                    if order.get("stopOrderType") == "StopLoss"
                ]
                if stop_loss_orders:
                    return stop_loss_orders[0]  # Return the first stop-loss order found
                else:
                    self.logger.info(f"No active stop-loss order found for {symbol}")
                    return None
            else:
                self.logger.error(f"Failed to fetch stop-loss orders: {response}")
                return None
        except Exception as e:
            self.logger.exception(f"Error fetching stop-loss orders: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: str, category: str):
        data = {
            "category": category,
            "symbol": symbol,
            "orderId": order_id
        }
        try:
            response = await self._request("POST", "/v5/order/cancel", data=data)
            if response and response.get('retCode') == 0:
                self.logger.info(f"Order cancelled successfully: {response}")
                return response
            else:
                self.logger.error(f"Failed to cancel order: {response}")
                return None
        except Exception as e:
            self.logger.exception(f"Error cancelling order: {e}")
            return None
        
    async def get_positions(self, symbol: str, category: str) -> Dict:
        """Fetches the current positions for a given symbol."""
        params = {"category": category, "symbol": symbol}
        try:
            response = await self._request("GET", "/v5/position/list", params=params)
            if response and response.get('retCode') == 0:
                self.logger.info(f"Successfully fetched position data for {symbol}")
                return response
            else:
                self.logger.error(f"Failed to fetch position data: {response}")
                return None
        except Exception as e:
            self.logger.exception(f"Error fetching position data: {e}")
            return None
    
# --- Trading Bot Class ---
class TradingBot:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self._load_config(config_file)
        self.logger = logging.getLogger(__name__)
        self.log_level = self.config.get('log_level', DEFAULT_LOG_LEVEL).upper()

        # Generate log filename using current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_filename = f"botlog_{timestamp}.log"

        self._configure_logging(self.log_level, self.log_filename)
        self.logger.info("TradingBot initializing...")

        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        if not self.api_key or not self.api_secret:
            self.logger.error("BYBIT_API_KEY and BYBIT_API_SECRET must be set in the .env file.")
            raise ValueError("Missing API keys in .env file.")
        self.api = BybitAPI(self.api_key, self.api_secret)
        
        self.symbols = self.config.get('symbols')
        self.loop_interval = self.config.get("loop_interval", DEFAULT_LOOP_INTERVAL)
        self.max_websocket_reconnect_attempts = self.config.get("max_websocket_reconnect_attempts", 3)
        self.initial_reconnect_delay = self.config.get("initial_reconnect_delay", 5)
        self.max_reconnect_delay = self.config.get("max_reconnect_delay", 60)
        self.ws_clients = []
        self.running = True
        
        self.logger.info("TradingBot initialized.")

    def _load_config(self, config_file: str) -> Dict:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_file}")
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in: {config_file}")
            raise ConfigurationError(f"Invalid JSON format in: {config_file}")

    def _configure_logging(self, log_level: str, log_filename: str):
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')
        self.logger.setLevel(numeric_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler for debugging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.info(f"Logging configured. Level: {log_level}, File: {log_filename}")
        
    async def run(self):
        self.logger.info("Starting TradingBot main loop.")
        await self._configure_trading_parameters(self.symbols)

        for symbol_config in self.symbols:
            symbol = symbol_config["symbol"]
            category = symbol_config.get("category", CATEGORY_LINEAR)
            task = asyncio.create_task(self.websocket_handler(symbol, category))

    async def _configure_trading_parameters(self, symbols: List[Dict]):
        for symbol_config in symbols:
            symbol = symbol_config["symbol"]
            try:
                symbol_info = await self.api.get_symbol_info(symbol)
                if not symbol_info:
                    self.logger.error(f"Could not fetch information for {symbol}. Skipping.")
                    continue

                min_trade_qty = float(symbol_info['lotSizeFilter']['minOrderQty'])
                price_scale = symbol_info['priceScale']
                qty_step = float(symbol_info['lotSizeFilter']['qtyStep'])

                symbol_config["min_trade_qty"] = min_trade_qty
                symbol_config["price_scale"] = price_scale
                symbol_config["qty_step"] = qty_step
                symbol_config.setdefault("rsi_oversold_level", 30)
                symbol_config.setdefault("rsi_overbought_level", 70)
                symbol_config.setdefault("rsi_lookback", 14)
                symbol_config.setdefault("trade_quantity", min_trade_qty)
                symbol_config.setdefault("bollinger_window", 20)
                symbol_config.setdefault("bollinger_std_dev", 2)

                if symbol_config["trade_quantity"] < min_trade_qty:
                    symbol_config["trade_quantity"] = min_trade_qty
                    self.logger.warning(
                        f"Trade quantity for {symbol} was below minimum. Adjusted to: {min_trade_qty}"
                    )

                self.logger.info(
                    f"Configured symbol {symbol}, "
                    f"trade qty: {symbol_config['trade_quantity']}, "
                    f"min trade qty: {min_trade_qty}, "
                    f"price precision: {price_scale}, "
                    f"qty precision: {qty_step}"
                )

            except Exception as e:
                self.logger.exception(f"Error configuring parameters for {symbol}: {e}")

    async def websocket_handler(self, symbol: str, category: str):
        retries = 0
        while retries < self.max_websocket_reconnect_attempts:
            try:
                ws_client = WebSocket(
                    testnet=False,  # Change to False for mainnet
                    channel_type="linear",
                )
                ws_client.trade_stream(symbol, self.process_trade_data)
                self.ws_clients.append(ws_client)
                self.logger.info(f"WebSocket connected for {symbol}.")

                while self.running:
                    await asyncio.sleep(self.loop_interval)
                    await self.trading_strategy(symbol, category)
                
                break

            except Exception as ws_exception:
                self.logger.exception(f"Websocket error for {symbol}: {ws_exception}")
                retries += 1
                delay = min(self.initial_reconnect_delay * (2 ** (retries - 1)), self.max_reconnect_delay)
                self.logger.warning(f"Attempting reconnection for {symbol} in {delay} seconds, attempt {retries}/{self.max_websocket_reconnect_attempts}")
                await asyncio.sleep(delay)

            finally:
                if self.ws_clients:
                    try:
                        for ws_client in self.ws_clients:
                           await ws_client.exit()
                        self.logger.info("WebSocket connection closed.")
                    except Exception as e:
                        self.logger.error(f"Error closing WebSocket: {e}")

                if retries == self.max_websocket_reconnect_attempts:
                    self.logger.error(f"Max reconnection attempts reached for {symbol}.")
                    break
                if not self.running:
                    self.logger.info(f"Websocket handler exiting for {symbol}.")
                    break

    async def process_trade_data(self, trade_data: Dict):
        self.logger.debug(f"Raw Websocket Data: {trade_data}")
        try:
            if 'topic' in trade_data and 'data' in trade_data:
                symbol = trade_data['topic'].split('.')[1]

                self.logger.debug(f"Trade data received, symbol: {symbol}")
                for symbol_config in self.symbols:
                    if symbol_config["symbol"] == symbol:
                        symbol_config["last_price"] = float(trade_data['data'][0]['p'])
                        break

        except Exception as e:
            self.logger.exception(f"Error in trade processing loop: {e}")

    async def trading_strategy(self, symbol: str, category: str):
        """
        Executes the trading strategy based on the provided symbol and category.
        """
        try:
            symbol_config = next((sc for sc in self.symbols if sc["symbol"] == symbol), None)
            if not symbol_config:
                self.logger.error(f"Symbol configuration not found for {symbol}")
                return

            if symbol_config.get("trading_enabled", True):
                # --- Get Klines ---
                klines = await self.api.get_klines(
                    symbol,
                    symbol_config.get("time_interval", "5"),
                    limit=max(
                        symbol_config.get("rsi_lookback", 14),
                        symbol_config.get("bollinger_window", 20),
                    ) + 10,  # Ensure enough data for calculations, and a little more
                )
                if not klines:
                    self.logger.error(f"No kline data received for {symbol}")
                    return

                # --- Convert klines to DataFrame and check data integrity ---
                df = pd.DataFrame(
                    klines,
                    columns=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "turnover",
                    ],
                )
                df["close"] = pd.to_numeric(df["close"])
                if df["close"].isnull().any():
                    self.logger.error(f"Missing closing prices in klines for {symbol}")
                    return

                # --- Indicators ---
                rsi_values = ta.momentum.RSIIndicator(
                    close=df["close"], window=symbol_config["rsi_lookback"]
                ).rsi()
                rsi = rsi_values.iloc[-1]

                bollinger = ta.volatility.BollingerBands(
                    close=df["close"],
                    window=symbol_config["bollinger_window"],
                    window_dev=symbol_config["bollinger_std_dev"],
                )
                upper_band = bollinger.bollinger_hband().iloc[-1]
                lower_band = bollinger.bollinger_lband().iloc[-1]
                ma_band = bollinger.bollinger_mavg().iloc[-1]

                current_price = symbol_config.get("last_price")
                if not current_price:
                    self.logger.warning(f"Current price not available for {symbol}")
                    return

                # --- Trading Logic ---
                # RSI and Bollinger Bands Strategy
                if rsi < symbol_config["rsi_oversold_level"] and current_price < lower_band:
                    self.logger.info(f"{symbol} - RSI: {rsi}, Price: {current_price} below Lower Band: {lower_band} - Potential Buy")
                    await self.execute_trade_logic(
                        symbol,
                        ORDER_SIDE_BUY,
                        symbol_config["trade_quantity"],
                        current_price,
                        ORDER_TYPE_MARKET,
                        category,
                    )
                elif rsi > symbol_config["rsi_overbought_level"] and current_price > upper_band:
                    self.logger.info(f"{symbol} - RSI: {rsi}, Price: {current_price} above Upper Band: {upper_band} - Potential Sell")
                    await self.execute_trade_logic(
                        symbol,
                        ORDER_SIDE_SELL,
                        symbol_config["trade_quantity"],
                        current_price,
                        ORDER_TYPE_MARKET,
                        category,
                    )

                # Grid Trading Logic
                buy_levels, sell_levels = await self.generate_grid_levels(symbol, current_price, category)
                if buy_levels:
                  await self.place_grid_orders(symbol, category, buy_levels, ORDER_SIDE_BUY, symbol_config["trade_quantity"])
                if sell_levels:
                  await self.place_grid_orders(symbol, category, sell_levels, ORDER_SIDE_SELL, symbol_config["trade_quantity"])

                # Trailing Stop Loss
                await self.manage_trailing_stop(
                    symbol,
                    symbol_config.get("trailing_stop_loss_callback"),
                    category
                )
            else:
                self.logger.info(f"Trading disabled for {symbol}.")

        except Exception as e:
            self.logger.exception(f"Error in trading strategy for {symbol}: {e}")
    
    async def manage_trailing_stop(self, symbol: str, trailing_stop_percentage: float, category: str):
        """Manages the trailing stop order for a given symbol."""
        position = await self.api.get_positions(symbol=symbol, category=category)
        if position and position['result']['list']:
            position_data = position['result']['list'][0]
            size = float(position_data.get('size', 0))
            if size > 0:  # Check if there's an open position
                current_price = await self.api.get_current_price(symbol=symbol, category=category)
                if current_price:
                    side = position_data['side']
                    mark_price = float(position_data['markPrice'])

                    if side == ORDER_SIDE_BUY:
                        new_stop_loss_price = mark_price * (1 - trailing_stop_percentage / 100)
                        stop_loss_side = ORDER_SIDE_SELL
                    elif side == ORDER_SIDE_SELL:
                        new_stop_loss_price = mark_price * (1 + trailing_stop_percentage / 100)
                        stop_loss_side = ORDER_SIDE_BUY
                    else:
                        self.logger.error(f"Invalid side: {side}")
                        return

                    active_sl_order = await self.api.get_active_stop_loss_order(symbol=symbol, category=category)

                    if not active_sl_order:
                        self.logger.info(f"No existing stop loss order, creating a new one at {new_stop_loss_price}")
                        await self.api.place_order(symbol=symbol, side=stop_loss_side, order_type=ORDER_TYPE_LIMIT, qty=size, price=new_stop_loss_price, timeInForce=TIME_IN_FORCE, reduce_only=True, category=category)
                    else:
                        existing_sl_price = float(active_sl_order.get('stopLoss'))
                        existing_order_id = active_sl_order.get('orderId')

                        # Adjust threshold as needed
                        threshold = 0.001  # Example threshold for price difference

                        if (side == ORDER_SIDE_BUY and new_stop_loss_price > existing_sl_price * (1 + threshold)) or \
                           (side == ORDER_SIDE_SELL and new_stop_loss_price < existing_sl_price * (1 - threshold)):
                            self.logger.info(f"Updating stop loss order. New price: {new_stop_loss_price}, Old price: {existing_sl_price}")
                            cancel_response = await self.api.cancel_order(symbol=symbol, order_id=existing_order_id, category=category)
                            if cancel_response and cancel_response.get('retCode') == 0:
                                await self.api.place_order(symbol=symbol, side=stop_loss_side, order_type=ORDER_TYPE_LIMIT, qty=size, price=new_stop_loss_price, timeInForce=TIME_IN_FORCE, reduce_only=True, category=category)
                            else:
                                self.logger.error(f"Failed to cancel old stop loss order {existing_order_id}: {cancel_response}")
                        else:
                            self.logger.info("Trailing stop is up to date.")
        elif position and not position['result']['list']:
            self.logger.info(f"No active position found for symbol {symbol} to manage trailing stop.")
        else:
            self.logger.error(f"Error fetching position data for {symbol} to manage trailing stop: {position}")

    async def generate_grid_levels(self, symbol: str, current_price: float, category: str) -> Tuple[List[float], List[float]]:
        """Generates grid levels for grid trading based on current price and configured parameters."""
        grid_spacing = GRID_SPACING_PERCENTAGE / 100
        num_levels = NUM_GRID_LEVELS

        buy_levels = []
        sell_levels = []

        for i in range(1, num_levels + 1):
            buy_price = current_price * (1 - i * grid_spacing)
            sell_price = current_price * (1 + i * grid_spacing)
            buy_levels.append(buy_price)
            sell_levels.append(sell_price)

        return buy_levels, sell_levels

    async def place_grid_orders(self, symbol: str, category: str, levels: List[float], side: str, order_qty: float):
        """Places grid orders at specified price levels."""
        for level in levels:
            try:
                # Place the order
                order_response = await self.api.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=ORDER_TYPE_LIMIT,
                    qty=order_qty,
                    price=level,
                    timeInForce=TIME_IN_FORCE,
                    category=category
                )

                if order_response and order_response.get('retCode') == 0:
                    order_id = order_response['result'].get('orderId')
                    self.logger.info(f"Placed {side} Limit Order at {level:.4f}, Order ID: {order_id}")
                else:
                    self.logger.error(f"Could not place {side} Limit Order at {level:.4f}. Response: {order_response}")

            except Exception as e:
                self.logger.exception(f"Error placing grid order for {symbol} at level {level}: {e}")

    async def execute_trade_logic(self, symbol: str, side: str, qty: float, price: float, order_type: str, category: str):
        """
        Executes the trade logic based on the given parameters.
        """
        try:
            self.logger.info(f"Executing trade logic: {side} {qty} {symbol} at {price} (Order Type: {order_type})")
            order_data = await self.api.place_order(
                symbol,
                side,
                order_type,
                qty,
                price,
                category=category
            )

            if order_data and order_data.get("retCode") == 0:
                self.logger.info(f"Order placed successfully: {order_data}")
            else:
                self.logger.error(f"Failed to place order: {order_data}")

        except Exception as e:
            self.logger.exception(f"Error executing trade logic: {e}")

# --- Main Execution ---
async def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_filepath = os.path.join(script_dir, 'config.json')

    bot = TradingBot(config_filepath)
    try:
        await bot.run()
    except KeyboardInterrupt:
        bot.logger.info("Bot stopped by user.")
        bot.running = False
        # Properly stop and join WebSocket threads
        if bot.ws_clients:
            for ws_client in bot.ws_clients:
               await ws_client.exit()
    except Exception as e:
        bot.logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
``````json
{
    "log_level": "INFO",
    "loop_interval": 30,
    "max_websocket_reconnect_attempts": 3,
    "initial_reconnect_delay": 5,
    "max_reconnect_delay": 60,
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "category": "linear",
            "trading_enabled": true,
            "rsi_oversold_level": 30,
            "rsi_overbought_level": 70,
            "rsi_lookback": 14,
            "trade_quantity": 0.001,
            "bollinger_window": 20,
            "bollinger_std_dev": 2,
            "trailing_stop_loss_callback": 0.3,
            "time_interval": "5"
        },
        {
            "symbol": "ETHUSDT",
            "category": "linear",
            "trading_enabled": true,
            "rsi_oversold_level": 30,
            "rsi_overbought_level": 70,
            "rsi_lookback": 14,
            "trade_quantity": 0.01,
            "bollinger_window": 20,
            "bollinger_std_dev": 2,
            "trailing_stop_loss_