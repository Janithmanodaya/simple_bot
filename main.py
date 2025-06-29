# Binance Trading Bot - Advance EMA Cross Strategy
# Author: Jules (AI Software Engineer)
# Date: [Current Date] - Will be filled by system or actual date
#
# This script implements the "Advance EMA Cross" trading strategy (ID: 8) for Binance Futures.
# It connects to the Binance API (testnet or mainnet), fetches market data, calculates EMAs,
# identifies trading signals based on EMA crossovers with price validation, manages position sizing
# based on user-defined risk, places orders (entry, stop-loss, take-profit), and dynamically
# adjusts SL/TP based on P&L movements.

import importlib.util
import sys
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import pandas as pd # For kline data handling and analysis
import time # For delays and timing loops
import math # For rounding, floor, ceil operations

# --- Configuration Defaults ---
DEFAULT_RISK_PERCENT = 1.0       # Default account risk percentage per trade (e.g., 1.0 for 1%)
DEFAULT_LEVERAGE = 20            # Default leverage (e.g., 20x)
DEFAULT_MAX_CONCURRENT_POSITIONS = 5 # Default maximum number of concurrent open positions
DEFAULT_MARGIN_TYPE = "ISOLATED" # Default margin type: "ISOLATED" or "CROSS"

# --- Global State Variables ---
# Stores details of active trades. Key: symbol (e.g., "BTCUSDT")
# Value: dict with trade info like order IDs, entry/SL/TP prices, quantity, side.
active_trades = {}

# --- Utility and Configuration Functions ---

def load_api_keys(env):
    """
    Loads API keys from keys.py based on the specified environment.
    Exits the script if keys.py is not found or keys are not configured.

    Args:
        env (str): The environment ("testnet" or "mainnet").

    Returns:
        tuple: (api_key, api_secret)
    """
    try:
        spec = importlib.util.spec_from_file_location("keys", "keys.py")
        if spec is None:
            print("Error: Could not load keys.py. Check if the file exists and is a valid Python module.")
            sys.exit(1)
        keys_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(keys_module) # type: ignore

        api_key, api_secret = None, None
        if env == "testnet":
            api_key = getattr(keys_module, "api_testnet", None)
            api_secret = getattr(keys_module, "secret_testnet", None)
        elif env == "mainnet":
            api_key = getattr(keys_module, "api_mainnet", None)
            api_secret = getattr(keys_module, "secret_mainnet", None)
        else:
            # This case should ideally not be reached if env is validated before calling
            raise ValueError("Invalid environment specified for loading API keys.")

        # Check if keys are placeholders or missing
        placeholders = ["<your-testnet-api-key>", "<your-testnet-secret>",
                        "<your-mainnet-api-key>", "<your-mainnet-secret>"]
        if not api_key or not api_secret or api_key in placeholders or api_secret in placeholders:
            print(f"Error: API key/secret for {env} not found or not configured in keys.py.")
            print("Please open keys.py and replace the placeholder values with your actual API credentials.")
            sys.exit(1)
        return api_key, api_secret
    except FileNotFoundError:
        print("Error: keys.py not found. Please create it in the same directory as main.py and add your API credentials.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading API keys: {e}")
        sys.exit(1)

def get_user_configurations():
    """
    Prompts the user for various trading configurations and returns them as a dictionary.
    Includes input validation for each configuration item.
    """
    print("\n--- Strategy Configuration ---")
    configs = {}

    # Select Environment
    while True:
        env_input = input("Select environment (testnet/mainnet): ").lower().strip()
        if env_input in ["testnet", "mainnet"]:
            configs["environment"] = env_input
            break
        print("Invalid environment. Please enter 'testnet' or 'mainnet'.")

    # Load API keys based on selected environment
    configs["api_key"], configs["api_secret"] = load_api_keys(configs["environment"])

    # Account Risk Percentage
    while True:
        try:
            risk_input = input(f"Enter account risk percentage per trade (e.g., 1 for 1%, default: {DEFAULT_RISK_PERCENT}%): ")
            risk_percent = float(risk_input or DEFAULT_RISK_PERCENT)
            if 0 < risk_percent <= 100: # Practical limit, can be adjusted
                configs["risk_percent"] = risk_percent / 100  # Convert to decimal for calculations
                break
            print("Risk percentage must be a positive value (e.g., 0.5, 1, 2, up to 100).")
        except ValueError:
            print("Invalid input. Please enter a number (e.g., 1.5 for 1.5%).")

    # Leverage
    while True:
        try:
            leverage_input = input(f"Enter leverage (e.g., 10 for 10x, default: {DEFAULT_LEVERAGE}x): ")
            leverage = int(leverage_input or DEFAULT_LEVERAGE)
            if 1 <= leverage <= 125:  # Binance max leverage can vary per symbol, 125 is a common cap.
                configs["leverage"] = leverage
                break
            print("Leverage must be an integer between 1 and 125.")
        except ValueError:
            print("Invalid input. Please enter an integer (e.g., 10).")

    # Maximum Concurrent Positions
    while True:
        try:
            max_pos_input = input(f"Enter maximum concurrent positions (default: {DEFAULT_MAX_CONCURRENT_POSITIONS}): ")
            max_concurrent_positions = int(max_pos_input or DEFAULT_MAX_CONCURRENT_POSITIONS)
            if max_concurrent_positions > 0:
                configs["max_concurrent_positions"] = max_concurrent_positions
                break
            print("Maximum concurrent positions must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer (e.g., 5).")

    # Margin Type
    while True:
        margin_input = input(f"Enter margin type (ISOLATED/CROSS, default: {DEFAULT_MARGIN_TYPE}): ").upper().strip()
        margin_type = margin_input or DEFAULT_MARGIN_TYPE
        if margin_type in ["ISOLATED", "CROSS"]:
            configs["margin_type"] = margin_type
            break
        print("Invalid margin type. Please enter 'ISOLATED' or 'CROSS'.")

    configs["strategy_id"] = 8
    configs["strategy_name"] = "Advance EMA Cross"
    print("--- Configuration Complete ---")
    return configs

# --- Binance API Interaction Functions ---

def initialize_binance_client(configs):
    """
    Initializes and returns the Binance API client using provided configurations.
    Tests connectivity by pinging the server.

    Args:
        configs (dict): Dictionary containing API key, secret, and environment.

    Returns:
        binance.client.Client or None: Initialized client object, or None on failure.
    """
    api_key = configs["api_key"]
    api_secret = configs["api_secret"]
    is_testnet = configs["environment"] == "testnet"

    try:
        client = Client(api_key, api_secret, testnet=is_testnet)
        client.ping()  # Test connectivity
        print(f"\nSuccessfully connected to Binance {configs['environment'].title()} API.")
        server_time_response = client.get_server_time()
        print(f"Binance Server Time: {pd.to_datetime(server_time_response['serverTime'], unit='ms')} UTC")
        return client
    except BinanceAPIException as e:
        print(f"Binance API Exception during client initialization or ping: {e}")
        return None
    except Exception as e: # Catch other potential errors like network issues
        print(f"Error initializing Binance client: {e}")
        return None

def get_historical_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=500):
    """
    Fetches historical klines (candlesticks) for a given symbol from Binance.
    Converts the data into a pandas DataFrame with a datetime index.

    Args:
        client (binance.client.Client): Initialized Binance client.
        symbol (str): Trading symbol (e.g., "BTCUSDT").
        interval (str): Kline interval (e.g., Client.KLINE_INTERVAL_15MINUTE).
        limit (int): Number of klines to fetch (min 500 for strategy, Binance max varies).

    Returns:
        pd.DataFrame: DataFrame with kline data, or an empty DataFrame on failure.
    """
    print(f"\nFetching historical klines for {symbol}, interval {interval}, limit {limit}...")
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            print(f"No kline data returned for {symbol}. This might be due to an invalid symbol or no recent trades.")
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Drop rows with NaN in critical columns that might have resulted from 'coerce'
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)


        print(f"Successfully fetched {len(df)} klines for {symbol}.")
        if len(df) < 200:  # Minimum for EMA200 calculation
             print(f"Warning: Fetched only {len(df)} klines for {symbol}. EMA200 might be inaccurate or unavailable.")
        return df

    except BinanceAPIException as e:
        print(f"Binance API Exception while fetching klines for {symbol}: {e}")
    except ValueError as ve: # Catch errors from pd.to_numeric or pd.to_datetime if data format is unexpected
        print(f"ValueError processing kline data for {symbol}: {ve}")
    except Exception as e: # Catch other potential errors
        print(f"An unexpected error occurred while fetching klines for {symbol}: {e}")
    return pd.DataFrame()


def get_account_balance(client, asset="USDT"):
    """
    Fetches the account balance for a specific asset (default USDT) from Binance Futures.

    Args:
        client (binance.client.Client): Initialized Binance client.
        asset (str): The asset symbol for which to fetch the balance (e.g., "USDT").

    Returns:
        float: Account balance for the specified asset, or 0.0 on failure.
    """
    print(f"\nFetching account balance for {asset}...")
    try:
        balances = client.futures_account_balance()
        for balance_info in balances:
            if balance_info['asset'] == asset:
                current_balance = float(balance_info['balance'])
                print(f"Account Balance ({asset}): {current_balance}")
                return current_balance
        print(f"Warning: {asset} not found in futures account balance.")
        return 0.0
    except BinanceAPIException as e:
        print(f"Binance API Exception while getting account balance: {e}")
    except ValueError: # If 'balance' is not a valid float
        print(f"Error: Could not convert balance to float for asset {asset}.")
    except Exception as e:
        print(f"An unexpected error occurred while getting account balance: {e}")
    return 0.0

def get_open_positions(client):
    """
    Fetches and displays all open positions from Binance Futures.

    Args:
        client (binance.client.Client): Initialized Binance client.

    Returns:
        list: A list of open position dictionaries, or an empty list on failure.
    """
    print("\nFetching open positions...")
    try:
        positions = client.futures_position_information()
        # Filter for positions with non-zero amount
        open_positions = [p for p in positions if p.get('positionAmt') and float(p['positionAmt']) != 0]

        if not open_positions:
            print("No open positions.")
            return []

        print("Current Open Positions:")
        for pos in open_positions:
            print(f"  Symbol: {pos['symbol']}, Amount: {pos['positionAmt']}, "
                  f"Entry Price: {pos['entryPrice']}, PnL: {pos['unRealizedProfit']}")
        return open_positions
    except BinanceAPIException as e:
        print(f"Binance API Exception while getting open positions: {e}")
    except ValueError: # If 'positionAmt' or other critical fields are not valid floats
        print("Error: Could not parse position data.")
    except Exception as e:
        print(f"An unexpected error occurred while getting open positions: {e}")
    return []

def get_open_orders(client, symbol=None):
    """
    Fetches and displays all open orders from Binance Futures, optionally for a specific symbol.

    Args:
        client (binance.client.Client): Initialized Binance client.
        symbol (str, optional): Trading symbol to filter orders. Defaults to None (all symbols).

    Returns:
        list: A list of open order dictionaries, or an empty list on failure.
    """
    action = f"Fetching open orders{(' for ' + symbol) if symbol else ''}..."
    print(f"\n{action}")
    try:
        if symbol:
            orders = client.futures_get_open_orders(symbol=symbol)
        else:
            orders = client.futures_get_open_orders()

        if not orders:
            print("No open orders found.")
            return []

        print("Current Open Orders:")
        for order in orders:
            print(f"  Symbol: {order['symbol']}, Order ID: {order['orderId']}, Type: {order['type']}, "
                  f"Side: {order['side']}, Price: {order['price']}, Stop Price: {order.get('stopPrice', 'N/A')}, "
                  f"Orig Qty: {order['origQty']}")
        return orders
    except BinanceAPIException as e:
        print(f"Binance API Exception while {action.lower()}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while {action.lower()}: {e}")
    return []

def set_leverage_on_symbol(client, symbol, leverage):
    """
    Sets the leverage for a specific symbol on Binance Futures.

    Args:
        client (binance.client.Client): Initialized Binance client.
        symbol (str): Trading symbol.
        leverage (int): Desired leverage.

    Returns:
        bool: True if successful, False otherwise.
    """
    print(f"\nSetting leverage for {symbol} to {leverage}x...")
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        print(f"Leverage for {symbol} set to {leverage}x successfully.")
        return True
    except BinanceAPIException as e:
        print(f"Binance API Exception setting leverage for {symbol} to {leverage}x: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while setting leverage for {symbol}: {e}")
    return False

def set_margin_type_on_symbol(client, symbol, margin_type):
    """
    Sets the margin type (ISOLATED/CROSS) for a specific symbol on Binance Futures.

    Args:
        client (binance.client.Client): Initialized Binance client.
        symbol (str): Trading symbol.
        margin_type (str): Desired margin type ("ISOLATED" or "CROSS").

    Returns:
        bool: True if successful or no change needed, False otherwise.
    """
    print(f"\nSetting margin type for {symbol} to {margin_type}...")
    try:
        client.futures_change_margin_type(symbol=symbol, marginType=margin_type.upper())
        print(f"Margin type for {symbol} set to {margin_type} successfully.")
        return True
    except BinanceAPIException as e:
        if e.code == -4046:  # Error code for "No need to change margin type"
            print(f"Margin type for {symbol} is already {margin_type}.")
            return True
        print(f"Binance API Exception setting margin type for {symbol} to {margin_type}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while setting margin type for {symbol}: {e}")
    return False

def place_new_order(client, symbol_info, side, order_type, quantity, price=None, stop_price=None, reduce_only=None):
    """
    Places a new order on Binance Futures.
    Uses symbol_info for formatting price and quantity according to precision rules.

    Args:
        client (binance.client.Client): Initialized Binance client.
        symbol_info (dict): Exchange information for the symbol, including precision rules.
        side (str): "BUY" or "SELL".
        order_type (str): e.g., "MARKET", "LIMIT", "STOP_MARKET", "TAKE_PROFIT_MARKET".
        quantity (float): Order quantity.
        price (float, optional): Order price for LIMIT type orders.
        stop_price (float, optional): Stop price for STOP_MARKET/TAKE_PROFIT_MARKET orders.
        reduce_only (bool, optional): If True, order will only reduce an existing position.

    Returns:
        dict or None: The order response dictionary if successful, None otherwise.
    """
    symbol = symbol_info['symbol']
    price_precision = int(symbol_info.get('pricePrecision', 8)) # Default to 8 if not found
    qty_precision = int(symbol_info.get('quantityPrecision', 8)) # Default to 8 if not found

    # Format quantity
    formatted_quantity = f"{quantity:.{qty_precision}f}"

    print(f"\nPlacing new order: {side} {formatted_quantity} {symbol} Type: {order_type}")
    params = {
        "symbol": symbol,
        "side": side.upper(),
        "type": order_type.upper(),
        "quantity": formatted_quantity,
    }

    if order_type.upper() in ["LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if price is None:
            print(f"Error: Price must be specified for {order_type} order on {symbol}.")
            return None
        params["price"] = f"{price:.{price_precision}f}"
        params["timeInForce"] = "GTC"  # Good 'Til Canceled, common for limit orders

    if order_type.upper() in ["STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if stop_price is None:
            print(f"Error: Stop price must be specified for {order_type} order on {symbol}.")
            return None
        params["stopPrice"] = f"{stop_price:.{price_precision}f}"

    if reduce_only is not None:
        params["reduceOnly"] = str(reduce_only).lower() # API expects 'true' or 'false'

    try:
        order = client.futures_create_order(**params)
        print("Order placed successfully:")
        # Display key order details
        print(f"  Symbol: {order.get('symbol')}, Order ID: {order.get('orderId')}, Type: {order.get('type')}, Side: {order.get('side')}")
        print(f"  Price: {order.get('price')}, Stop Price: {order.get('stopPrice', 'N/A')}, Avg Price: {order.get('avgPrice', 'N/A')}, Qty: {order.get('origQty')}, Status: {order.get('status')}")
        return order
    except BinanceAPIException as e:
        print(f"Binance API Exception placing {order_type} order for {symbol}: {e}")
    except BinanceOrderException as e: # More specific exception for order issues
        print(f"Binance Order Exception placing {order_type} order for {symbol}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while placing {order_type} order for {symbol}: {e}")
    return None

# --- Indicator Calculation Functions ---

def calculate_ema(df, period, column='close'):
    """
    Calculates the Exponential Moving Average (EMA) for a given period and DataFrame column.

    Args:
        df (pd.DataFrame): DataFrame containing the price data (must have `column`).
        period (int): The period for EMA calculation (e.g., 100, 200).
        column (str): The column name in `df` to use for EMA calculation (default 'close').

    Returns:
        pd.Series or None: A pandas Series containing the EMA values, or None if input is invalid.
    """
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in DataFrame for EMA calculation.")
        return None
    if len(df) < period:
        # Not an error, but EMA will be less accurate or all NaNs for initial part
        print(f"Warning: Data length ({len(df)}) is less than EMA period ({period}). EMA may be inaccurate or contain NaNs.")
    try:
        return df[column].ewm(span=period, adjust=False).mean()
    except Exception as e:
        print(f"Error calculating EMA for period {period} on column {column}: {e}")
        return None

# --- Strategy Logic Functions ---

def check_ema_crossover_conditions(df, short_ema_col='EMA100', long_ema_col='EMA200', validation_candles=20):
    """
    Checks for EMA crossover signals (EMA100/EMA200) and validates them based on price action
    in the preceding `validation_candles`.

    Args:
        df (pd.DataFrame): DataFrame with kline data including 'EMA100' and 'EMA200' columns.
        short_ema_col (str): Column name for the shorter period EMA.
        long_ema_col (str): Column name for the longer period EMA.
        validation_candles (int): Number of previous candles to check for price validation.

    Returns:
        str or None: "LONG" if a validated buy signal, "SHORT" for a sell signal, None otherwise.
    """
    if not all(col in df.columns for col in [short_ema_col, long_ema_col, 'low', 'high']):
        print(f"Error: Required columns ('{short_ema_col}', '{long_ema_col}', 'low', 'high') not all found in DataFrame for crossover check.")
        return None
    # Need at least 2 candles for crossover logic, plus `validation_candles` for the lookback period.
    if len(df) < validation_candles + 2:
        print(f"Warning: Not enough data ({len(df)} candles) for crossover check with {validation_candles} validation candles. Need at least {validation_candles + 2}.")
        return None

    # Ensure EMAs are not NaN for the candles being checked
    if df[[short_ema_col, long_ema_col]].iloc[- (validation_candles + 2):].isnull().values.any():
        print(f"Warning: NaN values found in EMA columns within the last {validation_candles + 2} candles. Cannot reliably check crossover.")
        return None

    # Crossover detection using the last two completed candles.
    # .iloc[-1] is the latest completed candle, .iloc[-2] is the one before it.
    prev_short_ema = df[short_ema_col].iloc[-2]
    curr_short_ema = df[short_ema_col].iloc[-1]
    prev_long_ema = df[long_ema_col].iloc[-2]
    curr_long_ema = df[long_ema_col].iloc[-1]

    # Validation period: `validation_candles` immediately preceding the crossover candle.
    # These are candles from index -(validation_candles + 1) up to index -2 (relative to end of DataFrame).
    validation_period_df = df.iloc[-(validation_candles + 1) : -1]
    if len(validation_period_df) < validation_candles:
        print(f"Warning: Insufficient candles in validation period. Expected {validation_candles}, got {len(validation_period_df)}.")
        return None


    # LONG Entry Condition: Short EMA crosses above Long EMA
    if prev_short_ema <= prev_long_ema and curr_short_ema > curr_long_ema:
        print(f"Potential LONG signal: {short_ema_col} ({curr_short_ema:.4f}) crossed above {long_ema_col} ({curr_long_ema:.4f}).")
        # Validation: In the previous `validation_candles`, price must not have touched/crossed EMA100 or EMA200.
        valid_signal = True
        for i in range(len(validation_period_df)):
            candle = validation_period_df.iloc[i]
            ema100_val_at_candle = validation_period_df[short_ema_col].iloc[i]
            ema200_val_at_candle = validation_period_df[long_ema_col].iloc[i]

            # Check if the candle's range [low, high] intersects with either EMA value at that point.
            touched_ema100 = candle['low'] <= ema100_val_at_candle <= candle['high']
            touched_ema200 = candle['low'] <= ema200_val_at_candle <= candle['high']

            if touched_ema100 or touched_ema200:
                reason = []
                if touched_ema100: reason.append(f"EMA100 ({ema100_val_at_candle:.4f}) touched (Low:{candle['low']:.4f}, High:{candle['high']:.4f})")
                if touched_ema200: reason.append(f"EMA200 ({ema200_val_at_candle:.4f}) touched (Low:{candle['low']:.4f}, High:{candle['high']:.4f})")
                print(f"Validation FAILED for LONG at candle {validation_period_df.index[i]}: {'; '.join(reason)}.")
                valid_signal = False
                break

        if valid_signal:
            print("LONG signal VALIDATED: Price stayed clear of EMAs in the validation period.")
            return "LONG"

    # SHORT Entry Condition: Short EMA crosses below Long EMA
    if prev_short_ema >= prev_long_ema and curr_short_ema < curr_long_ema:
        print(f"Potential SHORT signal: {short_ema_col} ({curr_short_ema:.4f}) crossed below {long_ema_col} ({curr_long_ema:.4f}).")
        valid_signal = True
        for i in range(len(validation_period_df)):
            candle = validation_period_df.iloc[i]
            ema100_val_at_candle = validation_period_df[short_ema_col].iloc[i]
            ema200_val_at_candle = validation_period_df[long_ema_col].iloc[i]

            touched_ema100 = candle['low'] <= ema100_val_at_candle <= candle['high']
            touched_ema200 = candle['low'] <= ema200_val_at_candle <= candle['high']

            if touched_ema100 or touched_ema200:
                reason = []
                if touched_ema100: reason.append(f"EMA100 ({ema100_val_at_candle:.4f}) touched (Low:{candle['low']:.4f}, High:{candle['high']:.4f})")
                if touched_ema200: reason.append(f"EMA200 ({ema200_val_at_candle:.4f}) touched (Low:{candle['low']:.4f}, High:{candle['high']:.4f})")
                print(f"Validation FAILED for SHORT at candle {validation_period_df.index[i]}: {'; '.join(reason)}.")
                valid_signal = False
                break

        if valid_signal:
            print("SHORT signal VALIDATED: Price stayed clear of EMAs in the validation period.")
            return "SHORT"

    return None # No validated signal

# --- Stop-Loss/Take-Profit and Order Management Functions ---

def calculate_swing_high_low(df_klines, window=20, current_candle_idx=-1):
    """
    Calculates the swing high and swing low from the `window` number of candles
    preceding the `current_candle_idx`.

    Args:
        df_klines (pd.DataFrame): DataFrame with kline data, must include 'high' and 'low' columns.
        window (int): Number of previous candles to consider for swing points.
        current_candle_idx (int): Index of the current candle (e.g., -1 for the latest completed candle).
                                 The window is taken from candles *before* this index.

    Returns:
        tuple (float, float) or (None, None): (swing_high, swing_low), or (None, None) if data is insufficient.
    """
    # Ensure enough data points before the current_candle_idx
    if len(df_klines) < window + abs(current_candle_idx) or current_candle_idx - window < -len(df_klines):
        print(f"Warning: Not enough data (length {len(df_klines)}) to calculate swing high/low with window {window} before index {current_candle_idx}.")
        return None, None

    # Define the slice for relevant candles: from (current_candle_idx - window) to (current_candle_idx - 1)
    # Example: if current_candle_idx is -1 (last candle), window is 20, slice is from -21 to -2.
    start_slice = current_candle_idx - window
    end_slice = current_candle_idx

    relevant_candles = df_klines.iloc[start_slice : end_slice]

    if relevant_candles.empty:
        print(f"Warning: No relevant candles found for swing high/low calculation (slice [{start_slice}:{end_slice}] on DF length {len(df_klines)}).")
        # Fallback to the single candle immediately preceding current_candle_idx if available
        if len(df_klines) >= abs(current_candle_idx) + 1:
             prev_candle = df_klines.iloc[current_candle_idx-1]
             return prev_candle['high'], prev_candle['low']
        return None, None

    swing_high = relevant_candles['high'].max()
    swing_low = relevant_candles['low'].min()
    return swing_high, swing_low

def calculate_sl_tp_values(entry_price, side, ema100_value, df_klines, current_candle_idx=-1):
    """
    Calculates initial Stop-Loss (SL) and Take-Profit (TP) prices based on strategy rules.

    Args:
        entry_price (float): The price at which the trade is entered.
        side (str): "LONG" or "SHORT".
        ema100_value (float): The value of EMA100 at the time of entry.
        df_klines (pd.DataFrame): DataFrame with kline data, for swing high/low calculation.
        current_candle_idx (int): Index of the entry candle in df_klines.

    Returns:
        tuple (float, float) or (None, None): (sl_price, tp_price), or (None, None) on failure.
    """
    tp_percentage = 0.01  # Primary TP: 1% from entry price
    max_sl_from_entry_percentage = 0.01  # Max SL distance: 1% from entry price

    # Calculate Take-Profit price
    if side == "LONG":
        take_profit_price = entry_price * (1 + tp_percentage)
    elif side == "SHORT":
        take_profit_price = entry_price * (1 - tp_percentage)
    else:
        print(f"Error: Invalid trade side '{side}' for SL/TP calculation.")
        return None, None

    # Calculate Stop-Loss price
    # Option 1: SL slightly beyond the 100-EMA line
    # "Slightly beyond" defined as a small percentage buffer of the EMA value itself.
    ema_buffer_percentage = 0.0005 # 0.05% of EMA value as buffer
    if side == "LONG":
        sl_based_on_ema100 = ema100_value * (1 - ema_buffer_percentage)
    else: # SHORT
        sl_based_on_ema100 = ema100_value * (1 + ema_buffer_percentage)

    # Check if SL based on EMA100 is within the 1% max distance from entry price
    sl_dist_ema_percentage = abs(entry_price - sl_based_on_ema100) / entry_price

    final_sl_price = None
    if sl_dist_ema_percentage <= max_sl_from_entry_percentage:
        final_sl_price = sl_based_on_ema100
        print(f"SL calculated based on EMA100: {final_sl_price:.4f} (Distance from entry: {sl_dist_ema_percentage*100:.2f}%)")
    else:
        # Option 2: EMA-based SL is >1% away, use last swing high/low
        print(f"SL based on EMA100 ({sl_based_on_ema100:.4f}, Distance: {sl_dist_ema_percentage*100:.2f}%) is > {max_sl_from_entry_percentage*100:.0f}% away. Using swing point.")
        swing_high, swing_low = calculate_swing_high_low(df_klines, window=20, current_candle_idx=current_candle_idx)

        swing_buffer_percentage = 0.0005 # 0.05% of swing point value as buffer
        if side == "LONG":
            if swing_low is None:
                print(f"Warning: Swing low not available for LONG SL. Falling back to max SL % from entry.")
                final_sl_price = entry_price * (1 - max_sl_from_entry_percentage)
            else:
                final_sl_price = swing_low * (1 - swing_buffer_percentage)
                print(f"SL calculated based on Swing Low ({swing_low:.4f}), buffered to: {final_sl_price:.4f}")
        else: # SHORT
            if swing_high is None:
                print(f"Warning: Swing high not available for SHORT SL. Falling back to max SL % from entry.")
                final_sl_price = entry_price * (1 + max_sl_from_entry_percentage)
            else:
                final_sl_price = swing_high * (1 + swing_buffer_percentage)
                print(f"SL calculated based on Swing High ({swing_high:.4f}), buffered to: {final_sl_price:.4f}")

        # Cap swing-based SL if it's excessively far (e.g., > 1.5 * max_sl_from_entry_percentage)
        # This prevents extreme SLs if swing points are very distant.
        max_swing_sl_dev_percentage = max_sl_from_entry_percentage * 1.5
        current_swing_sl_dev = abs(entry_price - final_sl_price) / entry_price
        if current_swing_sl_dev > max_swing_sl_dev_percentage:
            print(f"Warning: Swing-based SL ({final_sl_price:.4f}, dev: {current_swing_sl_dev*100:.2f}%) is too far. Capping at {max_swing_sl_dev_percentage*100:.2f}% dev.")
            if side == "LONG":
                final_sl_price = entry_price * (1 - max_swing_sl_dev_percentage)
            else: # SHORT
                final_sl_price = entry_price * (1 + max_swing_sl_dev_percentage)
            print(f"Capped SL to: {final_sl_price:.4f}")

    # Final validation: ensure SL is not on the wrong side of entry price or identical.
    if (side == "LONG" and final_sl_price >= entry_price) or \
       (side == "SHORT" and final_sl_price <= entry_price):
        print(f"Warning: Calculated SL {final_sl_price:.4f} for {side} is invalid relative to entry {entry_price:.4f}. Adjusting to max SL %.")
        if side == "LONG":
            final_sl_price = entry_price * (1 - max_sl_from_entry_percentage)
        else: # SHORT
            final_sl_price = entry_price * (1 + max_sl_from_entry_percentage)
        print(f"Adjusted SL to: {final_sl_price:.4f}")

    print(f"Final Initial SL: {final_sl_price:.4f}, TP: {take_profit_price:.4f} for {side} trade from entry {entry_price:.4f}")
    return final_sl_price, take_profit_price

def check_and_adjust_sl_tp_dynamic(current_price, entry_price, initial_sl, initial_tp, current_sl, current_tp, side):
    """
    Checks if Stop-Loss (SL) or Take-Profit (TP) needs adjustment based on dynamic breakeven logic.

    Args:
        current_price (float): The current market price of the asset.
        entry_price (float): The price at which the trade was entered.
        initial_sl (float): The initial stop-loss price set for the trade.
        initial_tp (float): The initial take-profit price set for the trade.
        current_sl (float): The current active stop-loss price.
        current_tp (float): The current active take-profit price.
        side (str): "LONG" or "SHORT".

    Returns:
        tuple (float, float) or (None, None): (new_sl, new_tp) if adjustment is needed, else (None, None).
                                              Returns current values if no change, new values if changed.
    """
    if entry_price == 0: # Avoid division by zero
        print("Error: Entry price is zero, cannot calculate profit percentage for dynamic adjustment.")
        return None, None

    profit_percentage = (current_price - entry_price) / entry_price if side == "LONG" else (entry_price - current_price) / entry_price

    new_sl, new_tp = current_sl, current_tp  # Start with current values, adjust if conditions met
    made_adjustment = False

    # Rule 1: If price moves +0.5% in favor, move SL to +0.2% profit (breakeven plus)
    if profit_percentage >= 0.005:  # +0.5% profit
        target_sl_profit_percentage = 0.002  # +0.2% profit lock
        if side == "LONG":
            potential_new_sl = entry_price * (1 + target_sl_profit_percentage)
            # Only adjust if new SL is an improvement (higher for LONG)
            if potential_new_sl > new_sl:
                new_sl = potential_new_sl
                print(f"Dynamic SL Adjustment (Profit Lock): Moved SL for LONG to {new_sl:.4f} (+0.2% profit).")
                made_adjustment = True
        else:  # SHORT
            potential_new_sl = entry_price * (1 - target_sl_profit_percentage)
            # Only adjust if new SL is an improvement (lower for SHORT)
            if potential_new_sl < new_sl:
                new_sl = potential_new_sl
                print(f"Dynamic SL Adjustment (Profit Lock): Moved SL for SHORT to {new_sl:.4f} (+0.2% profit).")
                made_adjustment = True

    # Rule 2: If price moves -0.5% against, move TP to +0.2% profit
    # Interpretation: If trade experiences a 0.5% drawdown, the TP target is reduced
    # from the initial +1% to a more conservative +0.2% if the trade recovers.
    if profit_percentage <= -0.005:  # -0.5% loss (drawdown)
        target_tp_profit_percentage = 0.002  # New TP target is +0.2%
        if side == "LONG":
            potential_new_tp = entry_price * (1 + target_tp_profit_percentage)
            # Only adjust if new TP is more conservative (lower for LONG) than current TP,
            # and still represents a profit.
            if potential_new_tp < new_tp and potential_new_tp > entry_price:
                new_tp = potential_new_tp
                print(f"Dynamic TP Adjustment (Drawdown Response): Moved TP for LONG to {new_tp:.4f} (+0.2% target).")
                made_adjustment = True
        else:  # SHORT
            potential_new_tp = entry_price * (1 - target_tp_profit_percentage)
            # Only adjust if new TP is more conservative (higher for SHORT) than current TP,
            # and still represents a profit.
            if potential_new_tp > new_tp and potential_new_tp < entry_price:
                new_tp = potential_new_tp
                print(f"Dynamic TP Adjustment (Drawdown Response): Moved TP for SHORT to {new_tp:.4f} (+0.2% target).")
                made_adjustment = True

    if made_adjustment:
        return new_sl, new_tp
    return None, None # No adjustment made based on rules

def get_symbol_info(client, symbol):
    """
    Fetches exchange information for a specific symbol from Binance Futures.
    This info includes precision rules for price and quantity, filters like LOT_SIZE, MIN_NOTIONAL.

    Args:
        client (binance.client.Client): Initialized Binance client.
        symbol (str): Trading symbol (e.g., "BTCUSDT").

    Returns:
        dict or None: Dictionary containing symbol information, or None on failure.
    """
    try:
        exchange_info = client.futures_exchange_info()
        for s_info in exchange_info['symbols']:
            if s_info['symbol'] == symbol:
                return s_info
        print(f"Warning: Could not find exchange information for symbol {symbol}.")
    except BinanceAPIException as e:
        print(f"Binance API Exception fetching exchange info for {symbol}: {e}")
    except KeyError: # If 'symbols' key is missing or structure is unexpected
        print(f"Error: Unexpected structure in exchange info response for {symbol}.")
    except Exception as e:
        print(f"An unexpected error occurred fetching exchange info for {symbol}: {e}")
    return None

def calculate_position_size(account_balance, risk_percent_per_trade, entry_price, sl_price, symbol_info):
    """
    Calculates the position size for a trade based on account balance, risk percentage,
    entry/SL prices, and symbol-specific trading rules (precision, min quantity/notional).

    Args:
        account_balance (float): Current available account balance for trading.
        risk_percent_per_trade (float): Account risk percentage per trade (e.g., 0.01 for 1%).
        entry_price (float): Proposed entry price for the trade.
        sl_price (float): Proposed stop-loss price for the trade.
        symbol_info (dict): Exchange information for the symbol.

    Returns:
        float or None: Calculated position quantity, or None if sizing fails or rules are violated.
    """
    if not symbol_info:
        print("Error: Cannot calculate position size without symbol info (for precision/filters).")
        return None
    if account_balance <= 0:
        print("Error: Account balance is zero or negative. Cannot calculate position size.")
        return None
    if entry_price <= 0 or sl_price <= 0:
        print(f"Error: Entry price ({entry_price}) or SL price ({sl_price}) is zero or negative.")
        return None

    # Extract quantity and price precision from symbol_info
    qty_precision = int(symbol_info.get('quantityPrecision', 0)) # Number of decimal places for quantity

    # Extract LOT_SIZE filter details (minQty, maxQty, stepSize)
    lot_size_filter = next((f for f in symbol_info.get('filters', []) if f.get('filterType') == 'LOT_SIZE'), None)
    if not lot_size_filter:
        print(f"Warning: LOT_SIZE filter not found for {symbol_info['symbol']}. Cannot accurately determine min quantity or step size.")
        return None
    min_trade_qty = float(lot_size_filter.get('minQty', 0))
    step_size = float(lot_size_filter.get('stepSize', 0))
    if step_size == 0: # step_size must be positive for calculations
        print(f"Error: step_size is zero for {symbol_info['symbol']}. Cannot calculate position size.")
        return None


    # Ensure SL and entry prices are different to avoid division by zero
    if abs(entry_price - sl_price) < 1e-9: # Check for near-equality for floating point numbers
        print("Error: SL price is too close or equal to entry price. Cannot calculate position size due to zero risk range.")
        return None

    risk_amount_usd = account_balance * risk_percent_per_trade
    price_risk_per_unit = abs(entry_price - sl_price) # Difference in price per unit of the asset

    position_size_raw = risk_amount_usd / price_risk_per_unit

    # Adjust for step_size: quantity must be a multiple of step_size.
    # We floor the quantity to the nearest valid multiple of step_size.
    position_size_adjusted = math.floor(position_size_raw / step_size) * step_size
    position_size_adjusted = round(position_size_adjusted, qty_precision) # Final round based on precision

    if position_size_adjusted < min_trade_qty:
        print(f"Warning: Risk-calculated position size {position_size_adjusted} is less than min trade quantity {min_trade_qty} for {symbol_info['symbol']}.")
        # Option: could try to use min_trade_qty and check if risk is acceptable, but prompt implies fixed risk %.
        # For now, if calculated size is too small based on risk, we don't trade.
        print("No trade possible with current risk settings for this SL distance. Consider adjusting risk % or widening SL (if appropriate).")
        return None

    # Check MIN_NOTIONAL filter
    min_notional_filter = next((f for f in symbol_info.get('filters', []) if f.get('filterType') == 'MIN_NOTIONAL'), None)
    if min_notional_filter:
        min_notional_value = float(min_notional_filter.get('notional', 0))
        current_notional = position_size_adjusted * entry_price
        if current_notional < min_notional_value:
            print(f"Warning: Calculated position's notional value ({current_notional:.2f} USDT) is less than min notional {min_notional_value} USDT for {symbol_info['symbol']}.")
            # Attempt to adjust quantity to meet min_notional, then re-check risk.
            qty_for_min_notional = (min_notional_value / entry_price)
            # Adjust this quantity to be a multiple of step_size (ceiling)
            qty_for_min_notional = math.ceil(qty_for_min_notional / step_size) * step_size
            qty_for_min_notional = round(qty_for_min_notional, qty_precision)

            if qty_for_min_notional < min_trade_qty: # Ensure it also meets LOT_SIZE minQty
                qty_for_min_notional = min_trade_qty

            # Recalculate risk if we use this adjusted quantity
            new_risk_amount_usd_if_adjusted = qty_for_min_notional * price_risk_per_unit
            new_risk_percent_if_adjusted = (new_risk_amount_usd_if_adjusted / account_balance)

            print(f"To meet min notional, quantity would be ~{qty_for_min_notional:.{qty_precision}f}. This would risk ~${new_risk_amount_usd_if_adjusted:.2f} ({new_risk_percent_if_adjusted*100:.2f}% of account).")

            # Allow some leeway for this adjustment, e.g., up to 1.5 times the configured risk percentage
            if new_risk_percent_if_adjusted <= (risk_percent_per_trade * 1.5):
                print(f"Adjusting position size to {qty_for_min_notional:.{qty_precision}f} to meet MIN_NOTIONAL filter. New risk: {new_risk_percent_if_adjusted*100:.2f}%.")
                position_size_adjusted = qty_for_min_notional
            else:
                print(f"Risk with min_notional_qty ({new_risk_percent_if_adjusted*100:.2f}%) exceeds allowed max adjustment ({risk_percent_per_trade*100*1.5:.2f}%). Cannot place trade.")
                return None

    if position_size_adjusted <= 0: # Should be caught by min_trade_qty check, but as a safeguard
        print(f"Error: Final calculated position size is zero or negative ({position_size_adjusted}). Cannot place trade.")
        return None

    print(f"Calculated position size: {position_size_adjusted:.{qty_precision}f} for {symbol_info['symbol']} (Risking ${risk_amount_usd:.2f})")
    return position_size_adjusted

# --- Main Trading Logic Functions ---

def manage_trade_entry(client, configs, symbol, klines_df):
    """
    Manages the logic for evaluating a new trade entry for a given symbol.
    This includes checking strategy conditions, calculating SL/TP, position size,
    setting leverage/margin, and placing orders.
    """
    global active_trades # Allow modification of the global active_trades dict

    if symbol in active_trades and active_trades[symbol]:
        print(f"Already in an active trade for {symbol}. Skipping new entry check.")
        return

    if len(active_trades) >= configs["max_concurrent_positions"]:
        print(f"Reached max concurrent positions ({configs['max_concurrent_positions']}). Cannot open new trade for {symbol}.")
        return

    # Calculate indicators
    klines_df['EMA100'] = calculate_ema(klines_df, 100)
    klines_df['EMA200'] = calculate_ema(klines_df, 200)

    # Ensure EMAs are calculated and enough data points exist
    min_candles_needed = 202 # Approx: 200 for longest EMA, 20 for validation, 2 for crossover detection
    if klines_df['EMA100'].isnull().any() or klines_df['EMA200'].isnull().any() or len(klines_df) < min_candles_needed:
        print(f"Not enough data or EMA calculation failed for {symbol}. Required at least {min_candles_needed} candles with valid EMAs.")
        return

    # Check for trading signal
    signal = check_ema_crossover_conditions(klines_df, short_ema_col='EMA100', long_ema_col='EMA200', validation_candles=20)

    if signal: # "LONG" or "SHORT"
        print(f"\n--- New Trade Signal for {symbol}: {signal} ---")

        symbol_info = get_symbol_info(client, symbol)
        if not symbol_info:
            print(f"Could not get symbol info for {symbol}. Aborting trade.")
            return

        # Assume entry on the close of the last completed candle (signal candle)
        entry_price = klines_df['close'].iloc[-1]
        ema100_value_at_entry = klines_df['EMA100'].iloc[-1]

        # Set leverage and margin type for the symbol BEFORE placing orders
        print(f"Preparing to trade {symbol}. Setting leverage and margin type...")
        if not set_leverage_on_symbol(client, symbol, configs['leverage']):
            print(f"Failed to set leverage for {symbol}. Aborting trade entry.")
            return
        if not set_margin_type_on_symbol(client, symbol, configs['margin_type']):
            print(f"Failed to set margin type for {symbol}. Aborting trade entry.")
            return

        # Calculate SL and TP prices
        sl_price, tp_price = calculate_sl_tp_values(entry_price, signal, ema100_value_at_entry, klines_df, current_candle_idx=-1)
        if sl_price is None or tp_price is None:
            print(f"Could not calculate SL/TP values for {symbol}. Aborting trade entry.")
            return

        # Fetch current account balance for position sizing
        account_balance = get_account_balance(client, asset="USDT")
        if account_balance <= 0: # Handles None or 0
            print("Account balance is zero, not available, or error fetching. Cannot calculate position size.")
            return

        # Calculate position quantity
        quantity = calculate_position_size(account_balance, configs['risk_percent'], entry_price, sl_price, symbol_info)
        if quantity is None or quantity <= 0:
            print(f"Could not calculate a valid position size for {symbol}. Aborting trade entry.")
            return

        qty_precision = int(symbol_info.get('quantityPrecision', 0))
        if round(quantity, qty_precision) == 0.0: # Check if rounding makes it zero
             print(f"Calculated quantity for {symbol} ({quantity}) rounded to 0.0 due to precision. Aborting trade.")
             return

        print(f"Attempting to place {signal} order for {quantity:.{qty_precision}f} {symbol} at market (approx. entry: {entry_price:.{int(symbol_info.get('pricePrecision',2))}f}). SL: {sl_price:.{int(symbol_info.get('pricePrecision',2))}f}, TP: {tp_price:.{int(symbol_info.get('pricePrecision',2))}f}")

        # Place MARKET order for entry
        entry_order_side = "BUY" if signal == "LONG" else "SELL"
        entry_order = place_new_order(client, symbol_info, entry_order_side, "MARKET", quantity)

        if entry_order and entry_order.get('status') == 'FILLED':
            # Use actual filled price if available and significantly different, otherwise use signal candle close.
            # For MARKET orders, 'avgPrice' is the fill price.
            original_signal_entry_price = entry_price # Save the original signal price for percentage calculation
            actual_entry_price = float(entry_order.get('avgPrice', original_signal_entry_price))

            # Calculate percentage distances for SL and TP from the original signal entry price
            # These percentages define the intended risk/reward profile of the signal
            # Ensure original_signal_entry_price is not zero to avoid DivisionByZeroError
            if original_signal_entry_price == 0:
                print(f"Error: Original signal entry price for {symbol} is zero. Cannot accurately adjust SL/TP for slippage. Using originally calculated SL/TP.")
            else:
                sl_percentage_distance = abs(original_signal_entry_price - sl_price) / original_signal_entry_price
                tp_percentage_distance = abs(original_signal_entry_price - tp_price) / original_signal_entry_price

                # Recalculate sl_price and tp_price based on actual_entry_price to maintain the same % distance
                if signal == "LONG":
                    new_sl_price = actual_entry_price * (1 - sl_percentage_distance)
                    new_tp_price = actual_entry_price * (1 + tp_percentage_distance)
                else: # SHORT
                    new_sl_price = actual_entry_price * (1 + sl_percentage_distance)
                    new_tp_price = actual_entry_price * (1 - tp_percentage_distance)

                print(f"Original SL/TP based on signal price {original_signal_entry_price:.{int(symbol_info.get('pricePrecision',2))}f}: SL {sl_price:.{int(symbol_info.get('pricePrecision',2))}f}, TP {tp_price:.{int(symbol_info.get('pricePrecision',2))}f}")
                sl_price, tp_price = new_sl_price, new_tp_price # Update SL/TP to be based on actual fill
                print(f"Adjusted SL/TP based on actual fill price {actual_entry_price:.{int(symbol_info.get('pricePrecision',2))}f}: SL {sl_price:.{int(symbol_info.get('pricePrecision',2))}f}, TP {tp_price:.{int(symbol_info.get('pricePrecision',2))}f}")

            print(f"{signal} order for {quantity:.{qty_precision}f} {symbol} filled. Actual avg entry price: {actual_entry_price:.{int(symbol_info.get('pricePrecision',2))}f}.")

            initial_sl_price, initial_tp_price = sl_price, tp_price # Store for dynamic adjustment reference

            # Place SL order (STOP_MARKET, reduceOnly)
            sl_order_side = "SELL" if signal == "LONG" else "BUY"
            sl_order = place_new_order(client, symbol_info, sl_order_side, "STOP_MARKET", quantity, stop_price=sl_price, reduce_only=True)
            if not sl_order:
                print(f"CRITICAL: Failed to place SL order for {symbol} after market entry. Position is unprotected by SL!")
                # Consider automatically closing the market position if SL order fails critically.
                # Example: emergency_close_order = place_new_order(client, symbol_info, sl_order_side, "MARKET", quantity, reduce_only=True)
                # if emergency_close_order: print(f"Emergency closed position for {symbol} due to SL failure.")
                return # Abort further actions for this trade if SL fails

            # Place TP order (TAKE_PROFIT_MARKET, reduceOnly)
            tp_order_side = "SELL" if signal == "LONG" else "BUY"
            tp_order = place_new_order(client, symbol_info, tp_order_side, "TAKE_PROFIT_MARKET", quantity, stop_price=tp_price, reduce_only=True)
            if not tp_order:
                print(f"Warning: Failed to place TP order for {symbol}. Position is open with SL only.")
                # Trade can continue, but TP is not automatically managed by an order.

            # Record active trade
            active_trades[symbol] = {
                "entry_order_id": entry_order['orderId'],
                "sl_order_id": sl_order.get('orderId') if sl_order else None,
                "tp_order_id": tp_order.get('orderId') if tp_order else None,
                "entry_price": actual_entry_price,
                "current_sl_price": sl_price,
                "current_tp_price": tp_price,
                "initial_sl_price": initial_sl_price, # For reference in dynamic adjustments
                "initial_tp_price": initial_tp_price, # For reference
                "quantity": quantity,
                "side": signal,
                "symbol_info": symbol_info # Store for precision details in adjustments
            }
            print(f"Trade entry for {symbol} successful and recorded. Monitoring will begin.")
            # Display current positions and orders for this symbol post-trade
            get_open_positions(client)
            get_open_orders(client, symbol)

        else:
            print(f"Failed to fill {signal} market order for {symbol}. Order response: {entry_order}")
    # else: (No signal)
        # print(f"No valid signal for {symbol} at this time based on EMA crossover and validation.")


def monitor_active_trades(client, configs):
    """
    Monitors active trades for dynamic SL/TP adjustments or closure confirmation.
    Checks if positions still exist and if SL/TP orders need modification.
    """
    global active_trades
    if not active_trades:
        return # No trades to monitor

    print("\n--- Monitoring Active Trades ---")
    symbols_to_remove_from_active = [] # List to collect symbols of closed trades

    for symbol, trade_details in list(active_trades.items()): # Iterate on a copy for safe deletion
        print(f"Checking active trade for {symbol} (Side: {trade_details['side']}, Entry: {trade_details['entry_price']:.4f})...")

        # 1. Verify if the position still exists on Binance
        current_pos_qty_on_exchange = 0.0
        position_exists_on_exchange = False
        try:
            position_info_list = client.futures_position_information(symbol=symbol)
            # Ensure position_info_list is not None and is a list
            if position_info_list and isinstance(position_info_list, list):
                 current_position_data = next((p for p in position_info_list if p['symbol'] == symbol), None)
                 if current_position_data:
                    current_pos_qty_on_exchange = float(current_position_data.get('positionAmt', 0.0))
                    if abs(current_pos_qty_on_exchange) > 1e-9: # Check if position amount is non-zero (within float tolerance)
                        position_exists_on_exchange = True

                        # Log current position details if it exists
                        current_pnl = float(current_position_data.get('unRealizedProfit', 0))
                        print(f"Position for {symbol} exists on exchange. Qty: {current_pos_qty_on_exchange}, PnL: {current_pnl:.2f} USDT")

                        # Check for quantity mismatch with bot's records
                        expected_bot_qty_abs = trade_details['quantity']
                        if abs(abs(current_pos_qty_on_exchange) - expected_bot_qty_abs) > (expected_bot_qty_abs * 0.001) and abs(abs(current_pos_qty_on_exchange) - expected_bot_qty_abs) > float(trade_details['symbol_info'].get('filters', [{}])[0].get('stepSize', 1e-8)): # Min step size or 0.1% diff
                            print(f"Warning: Position quantity mismatch for {symbol}. Exchange Qty: {abs(current_pos_qty_on_exchange):.{trade_details['symbol_info'].get('quantityPrecision', 8)}f}, Bot Expected Qty: {expected_bot_qty_abs:.{trade_details['symbol_info'].get('quantityPrecision', 8)}f}. "
                                  f"This could be due to manual intervention or unhandled partial fills. SL/TP orders from bot will use bot's expected quantity.")
            else:
                print(f"Warning: Received unexpected data type for position_information for {symbol}: {type(position_info_list)}")

        except BinanceAPIException as e:
            print(f"Binance API Exception fetching position info for {symbol}: {e}. Skipping monitoring for this cycle.")
            continue
        except ValueError as ve: # Error converting positionAmt or PnL to float
            print(f"ValueError parsing position data for {symbol}: {ve}. Skipping monitoring for this cycle.")
            continue
        except Exception as e:
            print(f"Unexpected error fetching position info for {symbol}: {e}. Skipping monitoring for this cycle.")
            continue

        if not position_exists_on_exchange:
            print(f"Position for {symbol} no longer exists or is zero on Binance (Exchange Qty: {current_pos_qty_on_exchange}). Assuming SL/TP hit or manually closed.")
            # Attempt to cancel any lingering SL/TP orders for this symbol, just in case.
            for order_type_key in ['sl_order_id', 'tp_order_id']:
                order_id_to_cancel = trade_details.get(order_type_key)
                if order_id_to_cancel:
                    try:
                        print(f"Attempting to cancel residual {order_type_key.split('_')[0].upper()} order {order_id_to_cancel} for closed {symbol} trade.")
                        client.futures_cancel_order(symbol=symbol, orderId=order_id_to_cancel)
                        print(f"Successfully cancelled residual order {order_id_to_cancel} for {symbol}.")
                    except BinanceAPIException as e_cancel:
                        if e_cancel.code == -2011: # "Unknown order sent." - order already filled/cancelled
                            print(f"Residual order {order_id_to_cancel} for {symbol} was likely already executed or cancelled.")
                        else:
                            print(f"Could not cancel residual order {order_id_to_cancel} for {symbol}: {e_cancel}")
                    except Exception as e_general_cancel:
                         print(f"Unexpected error cancelling residual order {order_id_to_cancel} for {symbol}: {e_general_cancel}")
            symbols_to_remove_from_active.append(symbol)
            continue # Move to the next active trade

        # 2. If position exists, check for dynamic SL/TP adjustments
        try:
            ticker = client.futures_ticker(symbol=symbol)
            current_market_price = float(ticker['lastPrice'])
        except BinanceAPIException as e:
            print(f"Binance API Exception fetching ticker for {symbol}: {e}. Cannot perform dynamic adjustment check this cycle.")
            continue
        except (ValueError, KeyError) as e: # Error parsing ticker data
            print(f"Error parsing ticker data for {symbol}: {e}. Cannot perform dynamic adjustment check this cycle.")
            continue
        except Exception as e:
            print(f"Unexpected error fetching ticker for {symbol}: {e}. Cannot perform dynamic adjustment check this cycle.")
            continue

        # Call the dynamic adjustment logic
        adjusted_sl, adjusted_tp = check_and_adjust_sl_tp_dynamic(
            current_market_price,
            trade_details['entry_price'],
            trade_details['initial_sl_price'],
            trade_details['initial_tp_price'],
            trade_details['current_sl_price'],
            trade_details['current_tp_price'],
            trade_details['side']
        )

        symbol_info = trade_details['symbol_info'] # Already stored

        # Handle SL adjustment if needed
        if adjusted_sl is not None and abs(adjusted_sl - trade_details['current_sl_price']) > 1e-9 : # Check for actual change
            print(f"Attempting to update SL for {symbol} from {trade_details['current_sl_price']:.4f} to {adjusted_sl:.4f}")
            if trade_details.get('sl_order_id'): # Cancel existing SL order first
                try:
                    client.futures_cancel_order(symbol=symbol, orderId=trade_details['sl_order_id'])
                    print(f"Cancelled old SL order {trade_details['sl_order_id']} for {symbol}.")
                except BinanceAPIException as e:
                    if e.code == -2011: # Order already gone
                        print(f"Old SL order {trade_details['sl_order_id']} for {symbol} already filled/cancelled.")
                    else:
                        print(f"Could not cancel old SL order {trade_details['sl_order_id']} for {symbol}: {e}. SL adjustment aborted.")
                        continue # Skip SL update if old one cannot be cancelled

            # Place new SL order
            sl_order_side = "SELL" if trade_details['side'] == "LONG" else "BUY"
            new_sl_order = place_new_order(client, symbol_info, sl_order_side, "STOP_MARKET", trade_details['quantity'], stop_price=adjusted_sl, reduce_only=True)
            if new_sl_order:
                active_trades[symbol]['current_sl_price'] = adjusted_sl
                active_trades[symbol]['sl_order_id'] = new_sl_order.get('orderId')
                print(f"Successfully updated SL for {symbol} to {adjusted_sl:.4f}. New SL Order ID: {new_sl_order.get('orderId')}")
            else:
                print(f"CRITICAL: Failed to place new SL order for {symbol} after cancelling old one. Position might be unprotected or mismanaged.")

        # Handle TP adjustment if needed
        if adjusted_tp is not None and abs(adjusted_tp - trade_details['current_tp_price']) > 1e-9: # Check for actual change
            print(f"Attempting to update TP for {symbol} from {trade_details['current_tp_price']:.4f} to {adjusted_tp:.4f}")
            if trade_details.get('tp_order_id'): # Cancel existing TP order first
                try:
                    client.futures_cancel_order(symbol=symbol, orderId=trade_details['tp_order_id'])
                    print(f"Cancelled old TP order {trade_details['tp_order_id']} for {symbol}.")
                except BinanceAPIException as e:
                    if e.code == -2011: # Order already gone
                        print(f"Old TP order {trade_details['tp_order_id']} for {symbol} already filled/cancelled.")
                    else:
                        print(f"Could not cancel old TP order {trade_details['tp_order_id']} for {symbol}: {e}. TP adjustment aborted.")
                        continue # Skip TP update

            # Place new TP order
            tp_order_side = "SELL" if trade_details['side'] == "LONG" else "BUY"
            new_tp_order = place_new_order(client, symbol_info, tp_order_side, "TAKE_PROFIT_MARKET", trade_details['quantity'], stop_price=adjusted_tp, reduce_only=True)
            if new_tp_order:
                active_trades[symbol]['current_tp_price'] = adjusted_tp
                active_trades[symbol]['tp_order_id'] = new_tp_order.get('orderId')
                print(f"Successfully updated TP for {symbol} to {adjusted_tp:.4f}. New TP Order ID: {new_tp_order.get('orderId')}")
            else:
                print(f"Warning: Failed to place new TP order for {symbol} after cancelling old one.")

    # Clean up trades that were marked for removal
    for symbol in symbols_to_remove_from_active:
        if symbol in active_trades:
            del active_trades[symbol]
            print(f"Removed {symbol} from internal active trades list.")

def trading_loop(client, configs):
    """
    Main trading loop for the bot.
    Continuously fetches data, checks for signals, manages trades, and monitors positions.
    """
    print("\n--- Starting Trading Loop ---")

    symbols_input = input("Enter symbols to monitor (comma-separated, e.g., BTCUSDT,ETHUSDT): ").upper()
    monitored_symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
    if not monitored_symbols:
        print("No symbols provided by user. Exiting trading loop.")
        return

    print(f"Monitoring symbols: {monitored_symbols}")

    while True:
        iteration_timestamp = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"\n--- Loop Iteration: {iteration_timestamp} ---")

        try:
            # Display general account status at the start of each major loop
            get_account_balance(client)
            get_open_positions(client) # Shows all open positions, not just bot's

            for symbol in monitored_symbols:
                print(f"\nProcessing symbol: {symbol}")
                # Fetch fresh kline data for the symbol
                # Strategy needs 15-min candles, fetch enough for EMA200 + validation (e.g., 550)
                klines_df = get_historical_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=550)

                min_hist_candles = 202 # EMA200 + validation period + crossover check
                if klines_df.empty or len(klines_df) < min_hist_candles:
                    print(f"Not enough kline data for {symbol} to proceed (fetched: {len(klines_df)}, need min: {min_hist_candles}). Skipping this symbol for now.")
                    time.sleep(configs.get("api_delay_short", 2)) # Short delay before trying next symbol
                    continue

                # Check for new trade entries only if no active trade for this symbol managed by the bot
                if symbol not in active_trades or not active_trades[symbol]:
                     manage_trade_entry(client, configs, symbol, klines_df.copy()) # Pass copy to avoid modification by reference
                else:
                    print(f"Skipping new entry check for {symbol}; already in an active trade managed by the bot.")

                time.sleep(configs.get("api_delay_symbol_processing", 2)) # Small delay after processing each symbol

            # Monitor all existing active trades managed by the bot
            monitor_active_trades(client, configs)

        except BinanceAPIException as e:
            print(f"A Binance API error occurred during the trading loop: {e}. Continuing...")
        except Exception as e:
            print(f"An unexpected error occurred during the trading loop: {e}. Attempting to continue...")
            # For critical unknown errors, you might want to implement a more robust alert or shutdown.

        # Loop delay: 15-minute candles mean we don't need to check every second.
        # A 5-minute check interval allows catching new candle data promptly.
        loop_delay_seconds = configs.get("loop_delay_minutes", 5) * 60
        print(f"\n--- End of Loop Iteration. Waiting for {configs.get('loop_delay_minutes', 5)} minutes... ---")
        time.sleep(loop_delay_seconds)

# --- Main Execution ---

def main():
    """
    Main function to run the Binance Trading Bot.
    Handles initial configuration, client setup, and starts the trading loop.
    Includes error handling and cleanup on exit.
    """
    print("Initializing Binance Trading Bot - Advance EMA Cross Strategy (ID: 8)")

    # Get user configurations (API keys, risk, leverage, etc.)
    configs = get_user_configurations()

    print("\nLoaded Configurations:")
    # Display configurations (excluding API keys for security)
    displayed_configs = {k: v for k, v in configs.items() if k not in ["api_key", "api_secret"]}
    for key, value in displayed_configs.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    # Initialize Binance client
    client = initialize_binance_client(configs)
    if not client:
        print("Exiting bot due to Binance client initialization failure.")
        sys.exit(1)

    # Add some configurable delays to configs if not present
    configs.setdefault("api_delay_short", 2) # seconds, for minor pauses
    configs.setdefault("api_delay_symbol_processing", 2) # seconds, after each symbol
    configs.setdefault("loop_delay_minutes", 5) # minutes, for main loop cycle

    # Start the main trading loop with robust error handling
    try:
        trading_loop(client, configs)
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user (Ctrl+C). Performing cleanup...")
    except Exception as e:
        print(f"\nAn critical unexpected error occurred, forcing bot to stop: {e}")
        # Potentially log detailed error trace here
    finally:
        print("\n--- Trading Bot Shutting Down ---")
        if client and active_trades: # Check if client was initialized and there are trades
            print("Attempting to cancel any open SL/TP orders for trades managed by this bot...")
            for symbol, trade_details in list(active_trades.items()): # Iterate on copy
                for order_type_key in ['sl_order_id', 'tp_order_id']:
                    order_id = trade_details.get(order_type_key)
                    if order_id:
                        try:
                            print(f"Cancelling {order_type_key.split('_')[0].upper()} order {order_id} for {symbol}...")
                            client.futures_cancel_order(symbol=symbol, orderId=order_id)
                            print(f"Successfully cancelled order {order_id} for {symbol}.")
                        except BinanceAPIException as e_cancel:
                             if e_cancel.code == -2011: # "Unknown order sent."
                                print(f"Order {order_id} for {symbol} was likely already executed or cancelled.")
                             else:
                                print(f"Could not cancel order {order_id} for {symbol} on exit: {e_cancel}")
                        except Exception as e_general_cancel:
                            print(f"Unexpected error cancelling order {order_id} for {symbol} on exit: {e_general_cancel}")
        print("Bot shutdown sequence complete.")

if __name__ == "__main__":
    main()
