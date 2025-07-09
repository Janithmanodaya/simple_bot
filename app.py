import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import lightgbm as lgb
from lightgbm.callback import early_stopping
import optuna
import joblib
from binance.client import Client # Assuming you'll use python-binance
from binance.exceptions import BinanceAPIException, BinanceRequestException
import time
import importlib.util # For loading keys.py
import sys # For sys.exit
import os # For file existence checks
import pandas as pd # For pd.Timestamp in initialize_binance_client, already imported but good to note

import threading # For app_active_trades_lock
import asyncio # For Telegram sending
import telegram # For Telegram bot
from telegram.ext import Application # If app.py were to run its own listener
from concurrent.futures import TimeoutError as FutureTimeoutError # For Telegram sending

# --- Global Binance Client ---
# This will be initialized by initialize_binance_client function
app_binance_client = None

# --- Global Telegram Loop for app.py ---
app_ptb_event_loop = None

# --- Global App Trading Configurations ---
# This will be populated by load_app_trading_configs
app_trading_configs = {}
APP_TRADE_CONFIG_FILE = "app_trade_config.csv"

# --- Global App Active Trades ---
# Stores details of trades initiated and managed by app.py
app_active_trades = {}
app_active_trades_lock = threading.Lock()

# --- ML Training Configuration (existing globals) ---
# These are primarily for the ML training part of app.py
PIVOT_N_LEFT = 3
PIVOT_N_RIGHT = 3
ATR_PERIOD = 14 # This is the default ATR period for feature engineering in app.py
MIN_ATR_DISTANCE = 1.0
MIN_BAR_GAP = 8
FIB_LEVEL_ENTRY = 0.618
FIB_LEVEL_TP1 = 0.382
FIB_LEVEL_TP2 = 0.618
FIB_LEVEL_TP3_EXTENSION = 1.618 # Example extension level

# --- Binance API Integration Functions ---

def load_app_api_keys(env="mainnet"): # Default to mainnet for app.py trading
    """
    Loads API keys from keys.py based on the specified environment for app.py.
    Exits if keys.py is not found or keys are not configured.
    """
    try:
        spec = importlib.util.spec_from_file_location("keys", "keys.py")
        if spec is None:
            print("Error (app.py): Could not prepare to load keys.py. File might be missing.")
            sys.exit(1)
        keys_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            print("Error (app.py): Could not get a loader for keys.py.")
            sys.exit(1)
        spec.loader.exec_module(keys_module)

        api_key, api_secret = None, None
        if env == "testnet":
            api_key = getattr(keys_module, "api_testnet", None)
            api_secret = getattr(keys_module, "secret_testnet", None)
        elif env == "mainnet":
            api_key = getattr(keys_module, "api_mainnet", None)
            api_secret = getattr(keys_module, "secret_mainnet", None)
        else:
            raise ValueError("Invalid environment specified for loading API keys in app.py.")

        placeholders_binance = ["<your-testnet-api-key>", "<your-testnet-secret>",
                                "<your-mainnet-api-key>", "<your-mainnet-secret>"]
        if not api_key or not api_secret or api_key in placeholders_binance or api_secret in placeholders_binance:
            print(f"Error (app.py): Binance API key/secret for {env} not found or not configured in keys.py.")
            sys.exit(1)
        
        # Telegram keys can also be loaded here if app.py will send its own messages
        telegram_token = getattr(keys_module, "telegram_bot_token", None)
        telegram_chat_id = getattr(keys_module, "telegram_chat_id", None)
        
        return api_key, api_secret, telegram_token, telegram_chat_id
    except FileNotFoundError:
        print("Error (app.py): keys.py not found. Please create it.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading keys in app.py: {e}")
        sys.exit(1)

def initialize_app_binance_client(env="mainnet"): # Removed app_configs parameter
    """
    Initializes the Binance client for app.py.
    Stores it in the global `app_binance_client`.
    If Telegram alerts on init failure are needed, it will use global `app_trading_configs`.
    """
    global app_binance_client, app_trading_configs # Ensure app_trading_configs is accessible if needed for alerts
    api_key, api_secret, _, _ = load_app_api_keys(env) # Telegram keys not used directly in init

    try:
        recv_window_ms = 20000
        requests_timeout_seconds = 15
        
        temp_client = Client(api_key, api_secret, testnet=(env == "testnet"), requests_params={'timeout': requests_timeout_seconds})
        temp_client.RECV_WINDOW = recv_window_ms
        
        server_time_info = temp_client.get_server_time()
        server_timestamp_ms = server_time_info['serverTime']
        local_timestamp_ms = int(time.time() * 1000)
        time_offset_ms = local_timestamp_ms - server_timestamp_ms

        print(f"app.py Binance Client: Server Time: {pd.Timestamp(server_timestamp_ms, unit='ms', tz='UTC')}")
        print(f"app.py Binance Client: Local System Time: {pd.Timestamp(local_timestamp_ms, unit='ms', tz='UTC')}")
        print(f"app.py Binance Client: Time Offset (Local - Server): {time_offset_ms} ms")

        if abs(time_offset_ms) > 1000:
            warning_message = (
                f"⚠️ WARNING (app.py Client): System clock out of sync with Binance by {time_offset_ms} ms.\n"
                f"This can lead to API errors. Ensure system time is synchronized."
            )
            print(warning_message)
            # Optionally send Telegram if app_configs and send_telegram_message are available

        temp_client.ping()
        app_binance_client = temp_client # Assign to global client
        print(f"app.py: Successfully connected to Binance {env.title()} API.")
        return True
        
    except BinanceAPIException as e:
        print(f"app.py Binance API Exception (client init): {e}")
        if e.code == -1021:
             print("Timestamp error during app.py client initialization. Check system time.")
        app_binance_client = None
        return False
    except Exception as e:
        print(f"app.py Error initializing Binance client: {e}")
        app_binance_client = None
        return False

def get_app_symbol_info(symbol: str):
    """Fetches symbol information using the app_binance_client."""
    global app_binance_client
    if app_binance_client is None:
        print("Error (app.py): Binance client not initialized. Call initialize_app_binance_client first.")
        return None
    try:
        exchange_info = app_binance_client.futures_exchange_info()
        for s_info in exchange_info['symbols']:
            if s_info['symbol'] == symbol:
                return s_info
        print(f"app.py: No symbol info found for {symbol}.")
        return None
    except Exception as e:
        print(f"app.py: Error getting symbol info for {symbol}: {e}")
        return None

def get_app_account_balance(asset="USDT"):
    """Fetches account balance using the app_binance_client."""
    global app_binance_client
    if app_binance_client is None:
        print("Error (app.py): Binance client not initialized.")
        return None
    try:
        balances = app_binance_client.futures_account_balance()
        for b_info in balances:
            if b_info['asset'] == asset:
                return float(b_info['balance'])
        print(f"app.py: {asset} not found in futures balance.")
        return 0.0
    except BinanceAPIException as e:
        if e.code == -2015: # IP whitelist or permissions
            print(f"app.py CRITICAL: API key/IP issue getting balance: {e}")
            # Add Telegram alert here if needed
            return None
        print(f"app.py API Error getting balance: {e}")
        return 0.0 # Fallback for other API errors
    except Exception as e:
        print(f"app.py Unexpected error getting balance: {e}")
        return 0.0

# --- 1. Data Collection & Labeling (Modified to use app_binance_client) ---

def get_historical_bars(symbol, interval, start_str, end_str=None):
    """
    Pull historical OHLCV bars from Binance using the global app_binance_client.
    """
    global app_binance_client
    if app_binance_client is None:
        print("Error (get_historical_bars in app.py): Binance client not initialized.")
        # Attempt to initialize if not already (e.g. if running app.py standalone for data fetching)
        if not initialize_app_binance_client(): # Uses default 'mainnet'
            print("Failed to auto-initialize client in get_historical_bars.")
            return pd.DataFrame() # Return empty DataFrame on failure
        # If successful, app_binance_client is now set.

    klines = app_binance_client.get_historical_klines(symbol, interval, start_str, end_str)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
        df[col] = pd.to_numeric(df[col])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

# --- Configuration Management for App Trading ---
def load_app_trading_configs(filepath=APP_TRADE_CONFIG_FILE):
    """
    Loads trading configurations for app.py from a CSV file.
    Uses defaults if the file or specific settings are not found.
    Populates the global `app_trading_configs` dictionary.
    """
    global app_trading_configs

    defaults = {
        "app_trading_environment": "mainnet", # 'mainnet' or 'testnet'
        "app_operational_mode": "signal", # 'live' or 'signal'
        "app_risk_percent": 0.01, # e.g., 0.01 for 1%
        "app_leverage": 20,
        "app_allow_exceed_risk_for_min_notional": False,
        "app_telegram_bot_token": None, # To be loaded from keys.py if needed by app.py independently
        "app_telegram_chat_id": None,   # To be loaded from keys.py
        "app_tp1_qty_pct": 0.25, # Default TP1 quantity percentage
        "app_tp2_qty_pct": 0.50, # Default TP2 quantity percentage
        # TP3 is remainder
    }

    loaded_csv_configs = {}
    if os.path.exists(filepath):
        try:
            df_config = pd.read_csv(filepath)
            if 'name' in df_config.columns and 'value' in df_config.columns:
                # Convert to dictionary, handling potential NaN values
                loaded_csv_configs = pd.Series(df_config.value.values, index=df_config.name).dropna().to_dict()
                print(f"app.py: Trading configurations loaded from '{filepath}'.")
            else:
                print(f"app.py: Trading config file '{filepath}' is missing 'name' or 'value' columns. Using defaults.")
        except pd.errors.EmptyDataError:
            print(f"app.py: Trading config file '{filepath}' is empty. Using defaults.")
        except Exception as e:
            print(f"app.py: Error loading trading configs from '{filepath}': {e}. Using defaults.")
    else:
        print(f"app.py: Trading config file '{filepath}' not found. Using default trading configurations.")

    # Merge loaded CSV configs with defaults (defaults take precedence if key not in CSV or CSV load failed)
    # No, it should be: CSV overrides defaults if key exists and is valid.
    
    final_configs = defaults.copy() # Start with defaults
    
    # Type conversion and validation for loaded CSV values
    for key, str_val in loaded_csv_configs.items():
        if key in final_configs: # Only process keys that are expected (defined in defaults)
            expected_type = type(defaults[key])
            try:
                if expected_type == bool:
                    if str(str_val).lower() in ['true', '1', 'yes', 'y']: final_configs[key] = True
                    elif str(str_val).lower() in ['false', '0', 'no', 'n']: final_configs[key] = False
                    else: print(f"app.py config: Invalid boolean value '{str_val}' for '{key}'. Using default.")
                elif expected_type == int:
                    final_configs[key] = int(float(str_val)) # float first to handle "20.0"
                elif expected_type == float:
                    final_configs[key] = float(str_val)
                elif defaults[key] is None and isinstance(str_val, str): # For string types that might be None by default (like telegram keys)
                    final_configs[key] = str_val if str_val.lower() != 'none' else None
                elif isinstance(defaults[key], str): # For string types that have a string default
                     final_configs[key] = str(str_val)
                # Add more type handlers if needed
            except ValueError:
                print(f"app.py config: Could not convert value '{str_val}' for '{key}' to {expected_type}. Using default.")
        # else: Key from CSV is not in defaults, ignore it for app_trading_configs

    # Specific validation for critical numeric values
    if not (0 < final_configs["app_risk_percent"] <= 1.0): # Risk percent should be decimal
        print(f"app.py config: Invalid app_risk_percent ({final_configs['app_risk_percent']}). Resetting to default {defaults['app_risk_percent']}.")
        final_configs["app_risk_percent"] = defaults["app_risk_percent"]
    if not (1 <= final_configs["app_leverage"] <= 125):
        print(f"app.py config: Invalid app_leverage ({final_configs['app_leverage']}). Resetting to default {defaults['app_leverage']}.")
        final_configs["app_leverage"] = defaults["app_leverage"]

    # Populate global app_trading_configs
    app_trading_configs.update(final_configs)
    
    # Load API keys and Telegram details from keys.py if not overridden by config file
    # Note: load_app_api_keys exits on failure, so these will be valid or script stops.
    # The env for API keys should come from app_trading_configs.
    app_env = app_trading_configs.get("app_trading_environment", "mainnet")
    api_k, api_s, tele_token, tele_chat_id = load_app_api_keys(env=app_env)
    
    # Store actual API keys in a way that they are not directly in app_trading_configs if it's printed/logged broadly
    # For now, let's add them, but be mindful if app_trading_configs is dumped.
    # These are needed for initialize_app_binance_client.
    app_trading_configs["api_key"] = api_k
    app_trading_configs["api_secret"] = api_s
    
    # Only update telegram from keys.py if not set by app_trade_config.csv
    if app_trading_configs.get("app_telegram_bot_token") is None and tele_token:
        app_trading_configs["app_telegram_bot_token"] = tele_token
    if app_trading_configs.get("app_telegram_chat_id") is None and tele_chat_id:
        app_trading_configs["app_telegram_chat_id"] = tele_chat_id

    print("app.py: Final trading configurations applied:", {k:v for k,v in app_trading_configs.items() if k not in ['api_key', 'api_secret']})


# --- Order Placement and Sizing Functions (Adapted from main.py) ---

def calculate_app_position_size(balance, risk_pct, entry_price, sl_price, symbol_info_app, app_configs_local=None): # Renamed parameter
    """
    Calculates position size for app.py. Adapted from main.py's calculate_position_size.
    `app_configs_local` (if provided) or global `app_trading_configs` should contain risk management settings like 'allow_exceed_risk_for_min_notional'.
    """
    if not symbol_info_app or balance <= 0 or entry_price <= 0 or sl_price <= 0 or abs(entry_price - sl_price) < 1e-9:
        print("app.py pos_size: Invalid inputs.")
        return None

    q_prec = int(symbol_info_app.get('quantityPrecision', 0))
    lot_size_filter = next((f for f in symbol_info_app.get('filters', []) if f.get('filterType') == 'LOT_SIZE'), None)
    
    if not lot_size_filter or float(lot_size_filter.get('stepSize', 0)) == 0:
        print(f"app.py pos_size: No LOT_SIZE/stepSize for {symbol_info_app.get('symbol', 'Unknown')}")
        return None
        
    min_qty = float(lot_size_filter.get('minQty', 0.001)) # Default min_qty if not found
    step_size = float(lot_size_filter.get('stepSize', 0.001)) # Default step_size

    # Ideal size based on risk_pct
    pos_size_ideal = (balance * risk_pct) / abs(entry_price - sl_price)
    
    # Adjust for step size
    adj_size = np.floor(pos_size_ideal / step_size) * step_size if step_size > 0 else pos_size_ideal
    adj_size = round(adj_size, q_prec)

    # Ensure minimum quantity
    if adj_size < min_qty:
        print(f"app.py pos_size: Initial calc {adj_size} for {symbol_info_app.get('symbol')} < min_qty {min_qty}.")
        risk_for_min_qty = (min_qty * abs(entry_price - sl_price)) / balance
        
        # Use local parameter if available, else global app_trading_configs
        current_configs_to_use = app_configs_local if app_configs_local is not None else app_trading_configs
        allow_exceed = current_configs_to_use.get('allow_exceed_risk_for_min_notional', False)

        if risk_for_min_qty > risk_pct:
            if allow_exceed:
                print(f"app.py pos_size: Warning - Using min_qty {min_qty} results in risk {risk_for_min_qty*100:.2f}%, exceeding target {risk_pct*100:.2f}%. Allowed by config.")
                adj_size = min_qty
            else: # Not allowed to exceed, or stricter cap if risk is too high
                # Maintain a hard cap (e.g., 1.5x configured risk) if not explicitly allowed to exceed any risk
                if risk_for_min_qty > (risk_pct * 1.5): 
                     print(f"app.py pos_size: Risk for min_qty {min_qty} ({risk_for_min_qty*100:.2f}%) is too high (>{risk_pct*1.5*100:.2f}%). No trade.")
                     return None
                else:
                     print(f"app.py pos_size: Warning - Using min_qty {min_qty} results in risk {risk_for_min_qty*100:.2f}%. This is > target {risk_pct*100:.2f}% but within 1.5x limit. Proceeding.")
                     adj_size = min_qty
        else: # min_qty is within risk target
            adj_size = min_qty
        
        if adj_size < min_qty: # Final check after min_qty logic
            print(f"app.py pos_size: Final size {adj_size} after min_qty check is still < min_qty {min_qty}. No trade.")
            return None

    # Check MIN_NOTIONAL filter
    min_notional_filter = next((f for f in symbol_info_app.get('filters', []) if f.get('filterType') == 'MIN_NOTIONAL'), None)
    if min_notional_filter:
        min_notional_val = float(min_notional_filter.get('notional', 0))
        if adj_size * entry_price < min_notional_val:
            print(f"app.py pos_size: Notional for size {adj_size} ({adj_size * entry_price:.2f}) < MIN_NOTIONAL ({min_notional_val:.2f}) for {symbol_info_app.get('symbol')}.")
            
            qty_for_min_notional = np.ceil((min_notional_val / entry_price) / step_size) * step_size if step_size > 0 else (min_notional_val / entry_price)
            qty_for_min_notional = round(max(qty_for_min_notional, min_qty), q_prec) # Ensure meets min_qty too

            risk_for_min_notional_qty = (qty_for_min_notional * abs(entry_price - sl_price)) / balance
            print(f"app.py pos_size: Qty for MIN_NOTIONAL: {qty_for_min_notional}. Implied risk: {risk_for_min_notional_qty*100:.2f}%. Target: {risk_pct*100:.2f}%.")

            # Use local parameter if available, else global app_trading_configs (already defined as current_configs_to_use)
            allow_exceed_mn = current_configs_to_use.get('allow_exceed_risk_for_min_notional', False)

            if risk_for_min_notional_qty > risk_pct:
                if allow_exceed_mn:
                    print(f"app.py pos_size: Warning - Risk increased to {risk_for_min_notional_qty*100:.2f}% for {symbol_info_app.get('symbol')} to meet MIN_NOTIONAL. Allowed by config.")
                    adj_size = qty_for_min_notional
                else:
                    if risk_for_min_notional_qty > (risk_pct * 1.5):
                        print(f"app.py pos_size: Risk for MIN_NOTIONAL ({risk_for_min_notional_qty*100:.2f}%) is too high. No trade.")
                        return None
                    else:
                        print(f"app.py pos_size: Warning - Risk increased to {risk_for_min_notional_qty*100:.2f}% for {symbol_info_app.get('symbol')} to meet MIN_NOTIONAL. Within 1.5x limit.")
                        adj_size = qty_for_min_notional
            else: # Risk for min_notional_qty is within target
                adj_size = qty_for_min_notional
    
    if adj_size <= 0:
        print(f"app.py pos_size: Final calculated size {adj_size} is zero or negative. No trade.")
        return None
    if adj_size < min_qty: # Should be caught by earlier logic, but as a safeguard
        print(f"app.py pos_size: Final size {adj_size} after MIN_NOTIONAL is < min_qty {min_qty}. No trade.")
        return None

    print(f"app.py pos_size: Calculated Position Size: {adj_size} for {symbol_info_app.get('symbol')} (Risk: ${(adj_size*abs(entry_price-sl_price)):.2f}, Notional: ${(adj_size*entry_price):.2f})")
    return adj_size

def place_app_new_order(symbol_info_app, side, order_type, quantity, price=None, stop_price=None, position_side=None, is_closing_order=False):
    """
    Places a new order using app_binance_client. Adapted from main.py's place_new_order.
    """
    global app_binance_client
    if app_binance_client is None:
        print("Error (app.py place_order): Binance client not initialized.")
        return None, "Client not initialized"

    symbol = symbol_info_app.get('symbol')
    p_prec = int(symbol_info_app.get('pricePrecision', 2))
    q_prec = int(symbol_info_app.get('quantityPrecision', 0))
    
    params = {"symbol": symbol, "side": side.upper(), "type": order_type.upper(), "quantity": f"{quantity:.{q_prec}f}"}

    if position_side:
        params["positionSide"] = position_side.upper()

    if order_type.upper() in ["LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if price is None:
            print(f"app.py place_order: Price needed for {order_type} on {symbol}")
            return None, "Price missing for limit-type order"
        params.update({"price": f"{price:.{p_prec}f}", "timeInForce": "GTC"})
    
    if order_type.upper() in ["STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if stop_price is None:
            print(f"app.py place_order: Stop price needed for {order_type} on {symbol}")
            return None, "Stop price missing for stop-type order"
        params["stopPrice"] = f"{stop_price:.{p_prec}f}"
        if is_closing_order:
            params["closePosition"] = "true"
            if "reduceOnly" in params: del params["reduceOnly"] # Avoid conflict
    
    try:
        print(f"app.py place_order: Attempting to place order with params: {params}")
        order = app_binance_client.futures_create_order(**params)
        print(f"app.py place_order: Order PLACED: {order.get('symbol')} ID {order.get('orderId')} {order.get('positionSide','N/A')} {order.get('side')} {order.get('type')} Qty:{order.get('origQty')} @ {order.get('price','MARKET')} SP:{order.get('stopPrice','N/A')} Status:{order.get('status')}")
        return order, None
    except BinanceAPIException as e_api:
        error_msg = f"app.py place_order: API Error for {symbol} {side} {quantity} {order_type}: {e_api}"
        print(error_msg)
        return None, str(e_api)
    except Exception as e_gen:
        error_msg = f"app.py place_order: General Error for {symbol} {side} {quantity} {order_type}: {e_gen}"
        print(error_msg)
        return None, str(e_gen)

# --- End Order Placement and Sizing Functions ---

# --- Initial Trade Execution Logic Structure ---

def execute_app_trade_signal(symbol: str, side: str, 
                             sl_price: float, tp1_price: float, 
                             tp2_price: float = None, tp3_price: float = None, 
                             entry_price_target: float = None, order_type: str = "MARKET"):
    """
    Initial structure for executing a trade signal from within app.py.
    Currently places a market order for entry. SL/TP placement will be added later.
    
    Args:
        symbol (str): The trading symbol (e.g., "BTCUSDT").
        side (str): "LONG" or "SHORT".
        sl_price (float): Stop loss price.
        tp1_price (float): Take profit 1 price.
        tp2_price (float, optional): Take profit 2 price.
        tp3_price (float, optional): Take profit 3 price.
        entry_price_target (float, optional): Target entry price if order_type is "LIMIT".
        order_type (str, optional): "MARKET" or "LIMIT". Defaults to "MARKET".
    """
    global app_binance_client, app_trading_configs, app_active_trades, app_active_trades_lock

    current_operational_mode = app_trading_configs.get("app_operational_mode", "signal")
    log_prefix = f"[AppTradeExec ({current_operational_mode.upper()}) - {symbol}]"
    print(f"{log_prefix} Received trade signal: {side} {symbol}, Order: {order_type}, Entry Target: {entry_price_target}, SL: {sl_price}, TP1: {tp1_price}")

    if app_binance_client is None:
        print(f"{log_prefix} Binance client not initialized. Attempting to initialize...")
        if not initialize_app_binance_client(env=app_trading_configs.get("app_trading_environment", "mainnet")):
            print(f"{log_prefix} Critical: Failed to initialize Binance client. Cannot execute trade.")
            return False
        print(f"{log_prefix} Binance client initialized successfully.")

    # 1. Get Symbol Info
    symbol_info_app = get_app_symbol_info(symbol)
    if not symbol_info_app:
        print(f"{log_prefix} Failed to get symbol info for {symbol}. Cannot execute trade.")
        return False
    
    p_prec = int(symbol_info_app.get('pricePrecision', 2))

    # 2. Get Account Balance
    balance = get_app_account_balance()
    if balance is None or balance <= 0:
        print(f"{log_prefix} Invalid account balance ({balance}). Cannot execute trade.")
        return False
    print(f"{log_prefix} Current balance: {balance:.2f} USDT.")

    # 3. Determine Entry Price for Sizing and Order
    # For market orders, we need a reference price for position sizing. Use current market price.
    # For limit orders, the entry_price_target is used.
    effective_entry_price_for_sizing = entry_price_target
    if order_type.upper() == "MARKET":
        try:
            ticker = app_binance_client.futures_ticker(symbol=symbol)
            effective_entry_price_for_sizing = float(ticker['lastPrice'])
            print(f"{log_prefix} Using current market price for MARKET order sizing: {effective_entry_price_for_sizing:.{p_prec}f}")
        except Exception as e:
            print(f"{log_prefix} Could not fetch current market price for MARKET order sizing: {e}. Cannot execute.")
            return False
    elif order_type.upper() == "LIMIT" and entry_price_target is None:
        print(f"{log_prefix} Entry price target required for LIMIT order. Cannot execute.")
        return False
        
    if effective_entry_price_for_sizing is None: # Should be caught above, but as a safeguard
        print(f"{log_prefix} Effective entry price for sizing could not be determined. Cannot execute.")
        return False

    # 4. Calculate Position Size
    # Ensure app_trading_configs is loaded and available
    if not app_trading_configs:
        print(f"{log_prefix} App trading configurations not loaded. Attempting to load defaults...")
        load_app_trading_configs() # Load with defaults if not already loaded
        if not app_trading_configs: # Still not loaded
            print(f"{log_prefix} Critical: Failed to load app trading configurations. Cannot size trade.")
            return False
            
    risk_percentage = app_trading_configs.get("app_risk_percent", 0.01) # Default 1%
    
    quantity = calculate_app_position_size(balance, risk_percentage,
                                           effective_entry_price_for_sizing, sl_price,
                                           symbol_info_app, app_configs_local=app_trading_configs) # Pass global app_trading_configs
    if quantity is None or quantity <= 0:
        print(f"{log_prefix} Position size calculation failed or resulted in zero/negative quantity ({quantity}). Cannot execute trade.")
        return False
    print(f"{log_prefix} Calculated position size: {quantity}")

    # 5. Place Entry Order
    api_order_side = "BUY" if side.upper() == "LONG" else "SELL"
    api_position_side = side.upper()

    # Ensure leverage and margin type are set before placing order
    # These would typically be set once per symbol at startup or if changed by dynamic logic.
    # For now, let's assume they are managed elsewhere or set it here if needed.
    # Example:
    # target_leverage = app_trading_configs.get("app_leverage", 20)
    # if not app_binance_client.futures_change_leverage(symbol=symbol, leverage=target_leverage): ... error ...
    # For simplicity, this part is omitted for now but is crucial in a full system.

    entry_order_result, error_msg = place_app_new_order(
        symbol_info_app, api_order_side, order_type.upper(), quantity,
        price=entry_price_target if order_type.upper() == "LIMIT" else None,
        position_side=api_position_side
    )

    if entry_order_result:
        entry_order_id = entry_order_result.get('orderId')
        actual_filled_price = effective_entry_price_for_sizing # Default for LIMIT or if avgPrice not available
        total_filled_quantity = quantity # Assume full quantity filled for now

        if order_type.upper() == "MARKET" and entry_order_result.get('status') == 'FILLED':
            actual_filled_price = float(entry_order_result.get('avgPrice', effective_entry_price_for_sizing))
            total_filled_quantity = float(entry_order_result.get('executedQty', quantity))
            print(f"{log_prefix} Market Entry order {entry_order_id} FILLED. AvgPrice: {actual_filled_price:.{p_prec}f}, ExecutedQty: {total_filled_quantity}")
        elif order_type.upper() == "LIMIT":
            print(f"{log_prefix} LIMIT Entry order {entry_order_id} PLACED. Status: {entry_order_result.get('status')}. Waiting for fill to place SL/TP (manual or via monitor).")
            # For LIMIT orders not immediately filled, SL/TP placement is deferred.
            # A monitoring function will need to pick this up.
            # For now, we can store it as pending SL/TP placement.
            with app_active_trades_lock:
                app_active_trades[symbol] = {
                    "entry_order_id": entry_order_id,
                    "entry_price_target": entry_price_target, # The limit price
                    "status": "PENDING_FILL_FOR_SLTP", # Custom status
                    "total_quantity": total_filled_quantity, # Expected quantity
                    "side": side.upper(),
                    "symbol_info": symbol_info_app,
                    "open_timestamp": pd.Timestamp.now(tz='UTC'), # Time limit order was placed
                    "strategy_type": "APP_ML_TRADE", # Example strategy type
                    "intended_sl": sl_price, "intended_tp1": tp1_price, 
                    "intended_tp2": tp2_price, "intended_tp3": tp3_price,
                    "order_type": "LIMIT"
                }
            return True # Indicate limit order placed, but SL/TP pending


        # Proceed with SL/TP placement only if entry is confirmed (e.g., MARKET order filled)
        # For LIMIT orders, this block would be skipped if not filled, handled by a monitor.
        # The current structure assumes MARKET order or immediate LIMIT fill for SL/TP placement.
        # Let's refine this to only proceed if we have a filled market order.
        if not (order_type.upper() == "MARKET" and entry_order_result.get('status') == 'FILLED'):
            print(f"{log_prefix} Entry order ({order_type}) not immediately confirmed as FILLED. SL/TP placement deferred.")
            # If it was a LIMIT order, it's already stored as PENDING_FILL_FOR_SLTP.
            # If it was MARKET but somehow not FILLED, that's an issue.
            if order_type.upper() == "MARKET":
                 print(f"{log_prefix} CRITICAL: Market order {entry_order_id} did not return FILLED status immediately. Status: {entry_order_result.get('status')}. Manual check required.")
                 # Store it with a special status for review
                 with app_active_trades_lock:
                    app_active_trades[symbol] = {
                        "entry_order_id": entry_order_id, "status": "MARKET_ORDER_UNCONFIRMED_FILL",
                        "total_quantity": total_filled_quantity, "side": side.upper(), "symbol_info": symbol_info_app,
                        "open_timestamp": pd.Timestamp.now(tz='UTC'), "strategy_type": "APP_ML_TRADE_ERROR",
                        "intended_sl": sl_price, "intended_tp1": tp1_price,
                    }
            return False # Indicate that full SL/TP setup is not complete

        print(f"{log_prefix} Placing SL/TP orders for filled entry {entry_order_id} (Qty: {total_filled_quantity}).")

        # Quantity Distribution for TPs
        q_prec = int(symbol_info_app.get('quantityPrecision', 0))
        min_qty_val = 0.001 # Default, should get from symbol_info_app filters
        lot_size_filter = next((f for f in symbol_info_app.get('filters', []) if f.get('filterType') == 'LOT_SIZE'), None)
        if lot_size_filter:
            min_qty_val = float(lot_size_filter.get('minQty', 0.001))

        tp1_pct = app_trading_configs.get("app_tp1_qty_pct", 0.25)
        tp2_pct = app_trading_configs.get("app_tp2_qty_pct", 0.50)
        # TP3 pct is remainder

        qty_tp1 = round(total_filled_quantity * tp1_pct, q_prec)
        qty_tp2 = round(total_filled_quantity * tp2_pct, q_prec)
        
        # Ensure minQty if percentage > 0
        if tp1_pct > 0 and qty_tp1 < min_qty_val: qty_tp1 = min_qty_val
        if tp2_pct > 0 and qty_tp2 < min_qty_val: qty_tp2 = min_qty_val

        # Cap at total_filled_quantity
        qty_tp1 = min(qty_tp1, total_filled_quantity)
        qty_tp2 = min(qty_tp2, round(total_filled_quantity - qty_tp1, q_prec))
        if qty_tp2 < 0: qty_tp2 = 0.0
        
        qty_tp3 = round(total_filled_quantity - qty_tp1 - qty_tp2, q_prec)
        if qty_tp3 < 0: qty_tp3 = 0.0
        
        # Handle TP3 dust if it was meant to have quantity
        if (1 - tp1_pct - tp2_pct) > 1e-5 and 0 < qty_tp3 < min_qty_val: # If TP3 was intended but is dust
            if qty_tp2 > 0 : qty_tp2 = round(qty_tp2 + qty_tp3, q_prec) # Add to TP2
            elif qty_tp1 > 0 : qty_tp1 = round(qty_tp1 + qty_tp3, q_prec) # Or TP1
            else: # If TP1,TP2 are 0, TP3 gets all (already handled by remainder calc)
                  pass 
            if not (qty_tp1 == 0 and qty_tp2 == 0): qty_tp3 = 0.0 # Zero out TP3 if reallocated

        print(f"{log_prefix} TP Quantities: TP1={qty_tp1}, TP2={qty_tp2}, TP3={qty_tp3}")

        sl_order_obj, tp_orders_details_list = None, []

        # Place SL for the total quantity
        sl_order_obj, sl_err_msg = place_app_new_order(symbol_info_app,
                                     "SELL" if side.upper() == "LONG" else "BUY", "STOP_MARKET", 
                                     total_filled_quantity, stop_price=sl_price, 
                                     position_side=api_position_side, is_closing_order=True)
        if not sl_order_obj:
            print(f"{log_prefix} CRITICAL: FAILED TO PLACE SL! Error: {sl_err_msg}")
            # Decide handling: close position? For now, log and continue to TPs.

        # Place TP orders
        tp_targets = [
            {"price": tp1_price, "quantity": qty_tp1, "name": "TP1"},
            {"price": tp2_price, "quantity": qty_tp2, "name": "TP2"},
            {"price": tp3_price, "quantity": qty_tp3, "name": "TP3"}
        ]

        for tp_info in tp_targets:
            current_tp_price = tp_info["price"]
            current_tp_qty = tp_info["quantity"]
            tp_name = tp_info["name"]

            if current_tp_price is not None and current_tp_qty > 0:
                tp_ord_obj, tp_err_msg = place_app_new_order(symbol_info_app,
                                             "SELL" if side.upper() == "LONG" else "BUY", 
                                             "TAKE_PROFIT_MARKET", current_tp_qty,
                                             stop_price=current_tp_price, 
                                             position_side=api_position_side, is_closing_order=True)
                if not tp_ord_obj:
                    print(f"{log_prefix} WARNING: Failed to place {tp_name}. Error: {tp_err_msg}")
                    tp_orders_details_list.append({"id": None, "price": current_tp_price, "quantity": current_tp_qty, "status": "FAILED", "name": tp_name})
                else:
                    tp_orders_details_list.append({"id": tp_ord_obj.get('orderId'), "price": current_tp_price, "quantity": current_tp_qty, "status": "OPEN", "name": tp_name})
            elif current_tp_price is None and current_tp_qty > 0:
                 print(f"{log_prefix} Skipping {tp_name} as price is None, but quantity {current_tp_qty} was assigned.")
        
        # Store in app_active_trades
        with app_active_trades_lock:
            app_active_trades[symbol] = {
                "entry_order_id": entry_order_id,
                "sl_order_id": sl_order_obj.get('orderId') if sl_order_obj else None,
                "tp_orders": tp_orders_details_list,
                "entry_price": actual_filled_price,
                "current_sl_price": sl_price, "initial_sl_price": sl_price, # Initial SL
                "total_quantity": total_filled_quantity, 
                "side": side.upper(),
                "symbol_info": symbol_info_app,
                "open_timestamp": pd.Timestamp.now(tz='UTC'), # Time trade became active with SL/TP
                "strategy_type": "APP_ML_TRADE", # Example
                "sl_management_stage": "initial"
            }
        print(f"{log_prefix} Trade {entry_order_id} with SL/TPs stored in app_active_trades.")
        return True
    else:
        print(f"{log_prefix} Failed to place entry order. Error: {error_msg}")
        return False

# --- End Initial Trade Execution Logic Structure ---

# --- Trade Monitoring Logic ---
def monitor_app_trades():
    """
    Monitors active trades initiated by app.py.
    Handles limit order fills, SL/TP hits, and staged SL adjustments.
    """
    global app_binance_client, app_trading_configs, app_active_trades, app_active_trades_lock

    if not app_active_trades:
        return

    current_operational_mode = app_trading_configs.get("app_operational_mode", "signal") # Get mode for monitor
    log_prefix_monitor = f"[AppTradeMonitor ({current_operational_mode.upper()})]"
    # print(f"\n{log_prefix_monitor} Checking {len(app_active_trades)} app-managed trade(s)/signal(s)...")

    trades_to_remove = []
    active_trades_snapshot = {}
    with app_active_trades_lock:
        active_trades_snapshot = app_active_trades.copy()

    for symbol, trade_details in active_trades_snapshot.items():
        log_sym_prefix = f"{log_prefix_monitor} [{symbol} ID:{trade_details.get('entry_order_id','N/A')}]"
        
        # Ensure client is initialized
        if app_binance_client is None:
            print(f"{log_sym_prefix} Binance client not available. Skipping monitoring for this cycle.")
            continue # Skip this trade if client is down

        s_info = trade_details.get('symbol_info')
        if not s_info: # Should always have symbol_info if trade was stored
            print(f"{log_sym_prefix} Missing symbol_info. Skipping trade.")
            trades_to_remove.append(symbol)
            continue
        
        p_prec = int(s_info.get('pricePrecision', 2))
        q_prec = int(s_info.get('quantityPrecision', 0))

        # --- Stage 1: Handle Limit Orders Pending SL/TP Placement ---
        if trade_details.get('status') == "PENDING_FILL_FOR_SLTP":
            entry_order_id = trade_details.get('entry_order_id')
            if not entry_order_id:
                print(f"{log_sym_prefix} PENDING_FILL_FOR_SLTP status but no entry_order_id. Removing.")
                trades_to_remove.append(symbol)
                continue
            
            try:
                limit_order_status = app_binance_client.futures_get_order(symbol=symbol, orderId=entry_order_id)
                
                if limit_order_status['status'] == 'FILLED':
                    actual_filled_price = float(limit_order_status.get('avgPrice', trade_details.get('entry_price_target')))
                    total_filled_quantity = float(limit_order_status.get('executedQty', trade_details.get('total_quantity')))
                    print(f"{log_sym_prefix} LIMIT entry order {entry_order_id} FILLED. AvgPrice: {actual_filled_price:.{p_prec}f}, Qty: {total_filled_quantity}")

                    # Retrieve intended SL/TPs
                    sl_price = trade_details['intended_sl']
                    tp1_price = trade_details['intended_tp1']
                    tp2_price = trade_details.get('intended_tp2')
                    tp3_price = trade_details.get('intended_tp3')
                    
                    # --- Place SL/TPs (logic copied and adapted from execute_app_trade_signal) ---
                    api_order_side = "BUY" if trade_details['side'].upper() == "LONG" else "SELL" # This is for SL/TP orders, so opposite of trade side
                    api_position_side = trade_details['side'].upper()

                    # Quantity Distribution
                    tp1_pct = app_trading_configs.get("app_tp1_qty_pct", 0.25)
                    tp2_pct = app_trading_configs.get("app_tp2_qty_pct", 0.50)
                    min_qty_val = float(next((f['minQty'] for f in s_info.get('filters', []) if f['filterType'] == 'LOT_SIZE'), '0.001'))

                    qty_tp1 = round(total_filled_quantity * tp1_pct, q_prec)
                    qty_tp2 = round(total_filled_quantity * tp2_pct, q_prec)
                    if tp1_pct > 0 and qty_tp1 < min_qty_val: qty_tp1 = min_qty_val
                    if tp2_pct > 0 and qty_tp2 < min_qty_val: qty_tp2 = min_qty_val
                    qty_tp1 = min(qty_tp1, total_filled_quantity)
                    qty_tp2 = min(qty_tp2, round(total_filled_quantity - qty_tp1, q_prec))
                    if qty_tp2 < 0: qty_tp2 = 0.0
                    qty_tp3 = round(total_filled_quantity - qty_tp1 - qty_tp2, q_prec)
                    if qty_tp3 < 0: qty_tp3 = 0.0
                    if (1 - tp1_pct - tp2_pct) > 1e-5 and 0 < qty_tp3 < min_qty_val:
                        if qty_tp2 > 0: qty_tp2 = round(qty_tp2 + qty_tp3, q_prec)
                        elif qty_tp1 > 0: qty_tp1 = round(qty_tp1 + qty_tp3, q_prec)
                        if not (qty_tp1 == 0 and qty_tp2 == 0): qty_tp3 = 0.0
                    
                    print(f"{log_sym_prefix} TP Qtys for filled limit: TP1={qty_tp1}, TP2={qty_tp2}, TP3={qty_tp3}")

                    placed_sl_order_obj, placed_tp_orders_list = None, []
                    placed_sl_order_obj, sl_err_msg = place_app_new_order(s_info, "SELL" if api_position_side == "LONG" else "BUY", 
                                                                        "STOP_MARKET", total_filled_quantity, 
                                                                        stop_price=sl_price, position_side=api_position_side, 
                                                                        is_closing_order=True)
                    if not placed_sl_order_obj: print(f"{log_sym_prefix} CRITICAL: FAILED SL for filled LIMIT! Err: {sl_err_msg}")

                    tp_targets_filled_limit = [
                        {"price": tp1_price, "quantity": qty_tp1, "name": "TP1"},
                        {"price": tp2_price, "quantity": qty_tp2, "name": "TP2"},
                        {"price": tp3_price, "quantity": qty_tp3, "name": "TP3"}
                    ]
                    for tp_info_fill in tp_targets_filled_limit:
                        if tp_info_fill["price"] is not None and tp_info_fill["quantity"] > 0:
                            tp_ord_obj_fill, tp_err_msg_fill = place_app_new_order(s_info, "SELL" if api_position_side == "LONG" else "BUY",
                                                                         "TAKE_PROFIT_MARKET", tp_info_fill["quantity"],
                                                                         stop_price=tp_info_fill["price"], position_side=api_position_side,
                                                                         is_closing_order=True)
                            if not tp_ord_obj_fill: print(f"{log_sym_prefix} WARN: Failed {tp_info_fill['name']} for filled LIMIT. Err: {tp_err_msg_fill}")
                            placed_tp_orders_list.append({"id": tp_ord_obj_fill.get('orderId') if tp_ord_obj_fill else None, 
                                                          "price": tp_info_fill["price"], "quantity": tp_info_fill["quantity"], 
                                                          "status": "OPEN" if tp_ord_obj_fill else "FAILED", "name": tp_info_fill["name"]})
                    
                    # Update app_active_trades
                    with app_active_trades_lock:
                        if symbol in app_active_trades: # Ensure it wasn't removed by another thread
                            app_active_trades[symbol].update({
                                "status": "ACTIVE", # Trade is now fully active
                                "entry_price": actual_filled_price, # Actual fill price
                                "total_quantity": total_filled_quantity, # Actual filled quantity
                                "sl_order_id": placed_sl_order_obj.get('orderId') if placed_sl_order_obj else None,
                                "tp_orders": placed_tp_orders_list,
                                "current_sl_price": sl_price, "initial_sl_price": sl_price,
                                "open_timestamp": pd.Timestamp(limit_order_status['updateTime'], unit='ms', tz='UTC'), # Time of fill
                                "sl_management_stage": "initial"
                            })
                            print(f"{log_sym_prefix} Limit order filled, SL/TPs placed. Trade now ACTIVE.")
                        else:
                             print(f"{log_sym_prefix} Limit order filled, but trade was removed from app_active_trades concurrently. SL/TPs may be orphaned.")
                             # TODO: Cancel newly placed SL/TPs if trade entry vanished. This is an edge case.

                elif limit_order_status['status'] in ['CANCELED', 'EXPIRED', 'REJECTED', 'PENDING_CANCEL']:
                    print(f"{log_sym_prefix} Limit entry order {entry_order_id} is {limit_order_status['status']}. Removing from app_active_trades.")
                    trades_to_remove.append(symbol)
                # else: Order is still 'NEW' or 'PARTIALLY_FILLED' (partially filled needs more complex handling not in scope for this step)
                # Timeout check for limit orders can also be added here.
                
            except BinanceAPIException as e_api:
                if e_api.code == -2013: # Order does not exist
                    print(f"{log_sym_prefix} Limit entry order {entry_order_id} NOT FOUND on exchange. Removing from app_active_trades.")
                    trades_to_remove.append(symbol)
                else:
                    print(f"{log_sym_prefix} API Error checking limit order {entry_order_id}: {e_api}")
            except Exception as e_gen:
                print(f"{log_sym_prefix} Unexpected error checking limit order {entry_order_id}: {e_gen}")
            
            continue # Move to the next symbol after processing a PENDING_FILL_FOR_SLTP trade (live mode)

        # --- Stage 1b: Handle VIRTUAL Limit Signals Pending Fill (Signal Mode) ---
        if trade_details.get('mode') == "signal" and trade_details.get('status') == "SIGNAL_PENDING_LIMIT_FILL":
            limit_price_signal = trade_details.get('entry_price_target')
            signal_side_virtual = trade_details.get('side')
            
            current_market_price_virtual = None
            try:
                ticker_virtual = app_binance_client.futures_ticker(symbol=symbol)
                current_market_price_virtual = float(ticker_virtual['lastPrice'])
            except Exception as e_fetch_signal:
                print(f"{log_sym_prefix} Could not fetch market price for virtual signal trigger: {e_fetch_signal}")
                continue

            price_triggered_virtual_limit = False
            if signal_side_virtual == "LONG" and current_market_price_virtual <= limit_price_signal:
                price_triggered_virtual_limit = True
            elif signal_side_virtual == "SHORT" and current_market_price_virtual >= limit_price_signal:
                price_triggered_virtual_limit = True

            if price_triggered_virtual_limit:
                print(f"{log_sym_prefix} Virtual LIMIT signal TRIGGERED. Limit: {limit_price_signal:.{p_prec}f}, Market: {current_market_price_virtual:.{p_prec}f}")
                with app_active_trades_lock:
                    if symbol in app_active_trades:
                        app_active_trades[symbol]['status'] = "SIGNAL_ACTIVE"
                        app_active_trades[symbol]['entry_price'] = current_market_price_virtual # Effective entry for simulation
                        app_active_trades[symbol]['open_timestamp'] = pd.Timestamp.now(tz='UTC')
                # TODO: Send Telegram update for virtual limit fill
            # TODO: Add timeout for virtual pending limit signals
            continue # Move to next symbol

        # --- Stage 2: Monitor Active Trades/Signals ---
        trade_is_live = trade_details.get('mode') == "live" and trade_details.get('status') == "ACTIVE"
        trade_is_signal_active = trade_details.get('mode') == "signal" and trade_details.get('status') == "SIGNAL_ACTIVE"

        if not (trade_is_live or trade_is_signal_active):
            if trade_details.get('status') not in ["PENDING_FILL_FOR_SLTP", "SIGNAL_PENDING_LIMIT_FILL"]: # Avoid logging for already handled pending states
                 print(f"{log_sym_prefix} Trade status is '{trade_details.get('status')}', not ACTIVE or SIGNAL_ACTIVE. Monitor skipping detailed checks.")
            continue
            
        current_market_price_for_active = None
        try:
            ticker_active = app_binance_client.futures_ticker(symbol=symbol)
            current_market_price_for_active = float(ticker_active['lastPrice'])
        except Exception as e_active:
            print(f"{log_sym_prefix} Could not fetch market price for active trade/signal monitoring: {e_active}. Skipping.")
            continue

        # Check SL Hit (Virtual or Real)
        current_sl_price = trade_details.get('current_sl_price')
        sl_order_id = trade_details.get('sl_order_id') # Will be None for signals
        trade_side = trade_details.get('side')

        sl_hit_flag = False
        if current_sl_price:
            if trade_side == "LONG" and current_market_price_for_active <= current_sl_price: sl_hit_flag = True
            elif trade_side == "SHORT" and current_market_price_for_active >= current_sl_price: sl_hit_flag = True
        
        if sl_hit_flag:
            print(f"{log_sym_prefix} STOP LOSS HIT ({trade_details.get('mode')}) at ~{current_sl_price:.{p_prec}f} (Market: {current_market_price_for_active:.{p_prec}f}).")
            if trade_is_live and sl_order_id:
                # For live trades, SL order fill confirmation would ideally come from checking the SL order status.
                # This market price check is a proactive measure.
                # If SL order is confirmed filled, then proceed to cancel TPs.
                print(f"{log_sym_prefix} Live SL hit. Assuming SL order {sl_order_id} will fill/has filled.")
                for tp_order_info in trade_details.get('tp_orders', []):
                    if tp_order_info.get('id') and tp_order_info.get('status') == 'OPEN':
                        try:
                            app_binance_client.futures_cancel_order(symbol=symbol, orderId=tp_order_info['id'])
                            print(f"{log_sym_prefix} Canceled TP order {tp_order_info['id']} due to SL hit.")
                        except Exception as e_cancel_tp_live:
                            print(f"{log_sym_prefix} Failed to cancel TP order {tp_order_info['id']} (live SL hit): {e_cancel_tp_live}")
            # TODO: Log P&L for live trade SL hit
            # TODO: Send Telegram for SL hit (live or signal)
            trades_to_remove.append(symbol)
            continue

        # Check TP Hits (Virtual or Real)
        # For live trades, this means checking status of actual TP orders.
        # For signal trades, this means checking market price against virtual TP levels.
        # This will be complex and is a placeholder for now.
        # The logic needs to iterate `trade_details['tp_orders']`, check status/price,
        # handle partial fills, and trigger SL adjustments.
        
        if trade_is_live:
            # print(f"{log_sym_prefix} Monitoring LIVE trade TPs. SL: {current_sl_price}, Market: {current_market_price_for_active:.{p_prec}f}")
            # TODO: Logic to check actual TP order statuses for live trades
            pass
        elif trade_is_signal_active:
            # print(f"{log_sym_prefix} Monitoring VIRTUAL signal TPs. SL: {current_sl_price}, Market: {current_market_price_for_active:.{p_prec}f}")
            # TODO: Logic to check market price against virtual TP levels for signals
            # And simulate SL adjustments (BE, TP1 trail)
            pass


    # Remove trades marked for deletion
    if trades_to_remove:
        with app_active_trades_lock:
            for sym_remove in trades_to_remove:
                if sym_remove in app_active_trades:
                    print(f"{log_prefix_monitor} Removing trade for {sym_remove} from app_active_trades.")
                    del app_active_trades[sym_remove]

# --- End Trade Monitoring Logic ---

# --- Telegram Integration for app.py ---
def send_app_telegram_message(message: str):
    """
    Sends a Telegram message using app.py's configured bot token and chat ID.
    Uses the global `app_ptb_event_loop` if available for async sending.
    """
    global app_trading_configs, app_ptb_event_loop

    bot_token = app_trading_configs.get("app_telegram_bot_token")
    chat_id = app_trading_configs.get("app_telegram_chat_id")

    if not bot_token or not chat_id:
        print(f"APP_TELEGRAM_SKIPPED: Token or Chat ID not configured in app_trading_configs. Message: '{message[:100]}...'")
        return False

    async def _send_async():
        try:
            bot = telegram.Bot(token=bot_token)
            await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
            print(f"app.py: Telegram message sent successfully to chat ID {chat_id}.")
            return True
        except telegram.error.TelegramError as e:
            print(f"app.py: Error sending Telegram message: {e}")
            return False
        except Exception as e_unexp:
            print(f"app.py: Unexpected error sending Telegram message: {e_unexp}")
            return False

    if app_ptb_event_loop and app_ptb_event_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_send_async(), app_ptb_event_loop)
        try:
            return future.result(timeout=10) # Wait up to 10 seconds
        except FutureTimeoutError:
            print(f"app.py: Timeout sending Telegram message via event loop to {chat_id}.")
            return False # Indicate timeout, though it might still send
        except Exception as e_rcs:
            print(f"app.py: Error scheduling Telegram message with run_coroutine_threadsafe: {e_rcs}")
            return False
    else:
        # Fallback: run in a new thread with its own loop (less efficient but robust)
        # print("app.py: Telegram event loop not available or not running. Sending in new thread.")
        try:
            return asyncio.run(_send_async()) # This will block until send is complete
        except RuntimeError as e_runtime: # Can happen if another asyncio loop is running in the main thread already
            print(f"app.py: Runtime error trying to send telegram message with asyncio.run (possibly due to existing loop): {e_runtime}. Trying threaded approach.")
            sender_thread = threading.Thread(target=lambda: asyncio.run(_send_async()), daemon=True)
            sender_thread.start()
            return True # Assume submitted to thread is success for this fallback

def escape_app_markdown_v1(text: str) -> str:
    """Escapes characters for Telegram Markdown V1 (app.py version)."""
    if not isinstance(text, str): return ""
    text = text.replace('_', r'\_').replace('*', r'\*').replace('`', r'\`').replace('[', r'\[')
    return text

# Placeholder for starting a dedicated Telegram listener for app.py commands
# def start_app_telegram_listener():
#     global app_ptb_event_loop, app_trading_configs
#     token = app_trading_configs.get("app_telegram_bot_token")
#     if not token:
#         print("app.py: Telegram bot token not configured. Cannot start listener.")
#         return
# 
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     app_ptb_event_loop = loop
# 
#     application = Application.builder().token(token).build()
#     # Add command handlers here:
#     # application.add_handler(CommandHandler("app_status", app_status_command_handler))
#     print("app.py: Telegram command listener (placeholder) starting...")
#     application.run_polling() # This would block if called in main thread
# --- End Telegram Integration ---

# --- Dynamic Parameter Selection Structure ---
def get_dynamic_trade_parameters(symbol: str, current_market_data_df: pd.DataFrame = None, account_balance: float = None):
    """
    Placeholder function to determine dynamic trade parameters.
    Currently returns static values from app_trading_configs or defaults.
    Future: This function will house ML models or complex heuristics to adapt parameters.

    Args:
        symbol (str): The trading symbol.
        current_market_data_df (pd.DataFrame, optional): Recent market data for the symbol.
        account_balance (float, optional): Current account balance.

    Returns:
        dict: A dictionary of trade parameters.
            Example: {
                "risk_percentage": 0.01,
                "sl_atr_multiplier": 2.0, # Example if SL is ATR based
                "tp_rr_ratio": 1.5,       # Example if TP is R:R based
                "tp1_qty_pct": 0.25,
                "tp2_qty_pct": 0.50,
                # ... other parameters like breakeven_buffer_r, etc.
            }
    """
    global app_trading_configs
    log_prefix = f"[DynamicParams - {symbol}]"

    # For now, fetch directly from app_trading_configs or use hardcoded defaults
    # In the future, ML logic would go here.
    
    # Ensure configs are loaded if empty (e.g., if app.py is run in a way that skips initial load)
    if not app_trading_configs:
        print(f"{log_prefix} app_trading_configs is empty. Loading defaults.")
        load_app_trading_configs() # This will populate app_trading_configs

    # These are examples; the actual parameters needed will depend on the strategy logic
    # that calls this function.
    params = {
        "risk_percentage": app_trading_configs.get("app_risk_percent", 0.01),
        "tp1_qty_pct": app_trading_configs.get("app_tp1_qty_pct", 0.25),
        "tp2_qty_pct": app_trading_configs.get("app_tp2_qty_pct", 0.50),
        # Add other parameters that might become dynamic:
        # "sl_atr_multiplier": app_trading_configs.get("app_sl_atr_multiplier", 2.0), # Example
        # "tp_rr_ratio": app_trading_configs.get("app_tp_rr_ratio", 1.5), # Example
        # "breakeven_buffer_r": app_trading_configs.get("app_breakeven_buffer_r", 0.1), # Example
    }

    # Example of how ML/dynamic logic could be integrated later:
    # if current_market_data_df is not None and not current_market_data_df.empty:
    #     volatility = current_market_data_df['close'].pct_change().rolling(window=20).std().iloc[-1]
    #     if volatility > some_threshold:
    #         params["risk_percentage"] *= 0.8 # Reduce risk in high volatility
    #         params["tp_rr_ratio"] *= 0.9     # Aim for quicker TPs
    
    # if account_balance is not None:
    #     if account_balance < 1000:
    #         params["risk_percentage"] = max(0.005, params["risk_percentage"] * 0.5) # Smaller risk for small accounts
    
    print(f"{log_prefix} Using parameters: Risk={params['risk_percentage']:.4f}, TP1%={params['tp1_qty_pct']:.2f}, TP2%={params['tp2_qty_pct']:.2f}")
    return params

# --- End Dynamic Parameter Selection Structure ---


def calculate_atr(df, period=ATR_PERIOD):
    """Calculates Average True Range."""
    """Calculates Average True Range."""
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = np.abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    df[f'atr_{period}'] = df['tr'].rolling(window=period).mean()
    return df

def generate_candidate_pivots(df, n_left=PIVOT_N_LEFT, n_right=PIVOT_N_RIGHT):
    """
    Generates candidate swing highs and lows based on local max/min.
    Loose rule-based swing finder.
    """
    df['is_candidate_high'] = False
    df['is_candidate_low'] = False

    for i in range(n_left, len(df) - n_right):
        # Candidate High
        is_high = True
        for j in range(1, n_left + 1):
            if df['high'].iloc[i] <= df['high'].iloc[i-j]:
                is_high = False
                break
        if not is_high:
            continue
        for j in range(1, n_right + 1):
            if df['high'].iloc[i] < df['high'].iloc[i+j]: # Use < for right side to allow for equal highs
                is_high = False
                break
        if is_high:
            df.loc[df.index[i], 'is_candidate_high'] = True

        # Candidate Low
        is_low = True
        for j in range(1, n_left + 1):
            if df['low'].iloc[i] >= df['low'].iloc[i-j]:
                is_low = False
                break
        if not is_low:
            continue
        for j in range(1, n_right + 1):
            if df['low'].iloc[i] > df['low'].iloc[i+j]: # Use > for right side
                is_low = False
                break
        if is_low:
            df.loc[df.index[i], 'is_candidate_low'] = True
    return df

def prune_and_label_pivots(df, atr_col_name, atr_distance_factor=MIN_ATR_DISTANCE, min_bar_gap=MIN_BAR_GAP): # Added atr_col_name
    """
    Prunes candidate pivots and labels them.
    - Enforce ATR-distance >= 1 * ATR14 from the previous pivot.
    - Enforce bar gap >= 8 bars since the last confirmed pivot.
    - Label each pivot as is_swing_high, is_swing_low, or neither.
    """
    df['is_swing_high'] = 0 # 0: neither, 1: swing high, 2: swing low (for multiclass target)
    df['is_swing_low'] = 0
    df['pivot_label'] = 0 # 0: none, 1: high, 2: low

    last_confirmed_pivot_idx = -1
    last_confirmed_pivot_price = 0
    last_confirmed_pivot_type = None

    # atr_col_name (e.g., 'atr_14') must be passed if ATR_PERIOD is dynamic
    # atr_distance_factor and min_bar_gap are now parameters
    
    # Ensure ATR column exists; if not, calculate it using the provided (or default) period.
    # This internal call to calculate_atr needs the correct period if atr_col_name implies it.
    # For simplicity, assume atr_col_name is passed and exists.
    # If it doesn't, the caller should have ensured it.
    # However, for robustness, if we only have atr_period, we can derive atr_col_name.
    # Let's assume atr_col_name is like 'atr_XX' and we can parse XX or it's passed.
    # For this refactor, the function will expect atr_col_name to be valid.
    
    # The original code used global ATR_PERIOD if f'atr_{ATR_PERIOD}' was not in df.columns.
    # Now, it relies on the caller to ensure the correct ATR column (based on tuned atr_period) is present.
    # We will pass atr_col_name to this function.

    if atr_col_name not in df.columns:
        print(f"Warning (prune_and_label_pivots): ATR column '{atr_col_name}' not found. Pivots might be incorrect if ATR is needed and not present.")
        # Or, if an atr_period was also passed, it could calculate it:
        # current_atr_period = int(atr_col_name.split('_')[-1]) # Simple parse
        # df = calculate_atr(df, period=current_atr_period)
        # For now, assume caller provides df with the correct atr_col_name.


    for i in range(len(df)):
        # Use the dynamic atr_col_name
        atr_val_at_pivot = df.get(atr_col_name, pd.Series(np.nan, index=df.index)).iloc[i] # Graceful fetch
        if pd.isna(atr_val_at_pivot) and atr_distance_factor > 0 : # Only skip if ATR is needed for distance check
            # If atr_distance_factor is 0, ATR is not needed for distance pruning.
            # print(f"Debug: ATR NaN at index {i} for {atr_col_name}, but atr_distance_factor is {atr_distance_factor}")
            pass # Allow processing if ATR not strictly needed for this pivot

        is_ch = df['is_candidate_high'].iloc[i]
        is_cl = df['is_candidate_low'].iloc[i]

        if not is_ch and not is_cl:
            continue

        # Check bar gap
        if last_confirmed_pivot_idx != -1 and (i - last_confirmed_pivot_idx) < min_bar_gap:
            continue

        current_price = 0
        current_type = None

        if is_ch:
            current_price = df['high'].iloc[i]
            current_type = 'high'
        elif is_cl: # Check is_cl, can't be both ch and cl with current logic, but good practice
            current_price = df['low'].iloc[i]
            current_type = 'low'

        # Check ATR distance (if there's a previous pivot)
        if last_confirmed_pivot_idx != -1 and atr_distance_factor > 0: # Only check if factor > 0
            # Use ATR at the time of the last confirmed pivot for distance check
            atr_at_last_pivot = df.get(atr_col_name, pd.Series(np.nan, index=df.index)).iloc[last_confirmed_pivot_idx]
            if pd.notna(atr_at_last_pivot) and atr_at_last_pivot > 0:
                price_diff = abs(current_price - last_confirmed_pivot_price)
                if price_diff < (atr_distance_factor * atr_at_last_pivot):
                    # Too close to the last pivot in terms of price * ATR
                    continue
            # else: Cannot perform ATR distance check if ATR at last pivot is NaN/zero, proceed without this specific pruning for this candidate.

        # Confirm pivot
        if current_type == 'high':
            # Ensure it's not immediately followed by a higher high or preceded by a higher high (within candidate window)
            # This is somewhat handled by generate_candidate_pivots, but an extra check can be useful
            # For simplicity, we'll rely on the candidate generation for now.
            df.loc[df.index[i], 'is_swing_high'] = 1
            df.loc[df.index[i], 'pivot_label'] = 1
            last_confirmed_pivot_idx = i
            last_confirmed_pivot_price = current_price
            last_confirmed_pivot_type = 'high'
        elif current_type == 'low':
            df.loc[df.index[i], 'is_swing_low'] = 1
            df.loc[df.index[i], 'pivot_label'] = 2
            last_confirmed_pivot_idx = i
            last_confirmed_pivot_price = current_price
            last_confirmed_pivot_type = 'low'

    return df


def simulate_fib_entries(df, atr_col_name): # Added atr_col_name
    """
    For each confirmed pivot, simulate entry at chosen Fib level, SL, and TPs.
    Records outcome classes: 0=stopped-out, 1=hit TP1 only, 2=hit TP2+, 3=hit TP3.
    This is a complex function and will require careful implementation.
    We'll store results in new columns.
    """
    df['trade_outcome'] = -1 # -1: No trade, 0: SL, 1: TP1, 2: TP2, 3: TP3
    df['entry_price_sim'] = np.nan
    df['sl_price_sim'] = np.nan
    df['tp1_price_sim'] = np.nan
    df['tp2_price_sim'] = np.nan
    df['tp3_price_sim'] = np.nan

    # Ensure the dynamic ATR column (atr_col_name) is present.
    # The caller (objective_optuna or get_processed_data_for_symbol) is responsible for ensuring
    # df contains an ATR column that corresponds to the atr_period used in this context.
    if atr_col_name not in df.columns:
        print(f"Error (simulate_fib_entries): Required ATR column '{atr_col_name}' not found in DataFrame. Cannot simulate entries.")
        # Fallback or error: If an atr_period was also passed, could calculate it.
        # For now, returning df as is, which means 'trade_outcome' will remain -1.
        return df

    pivots = df[(df['is_swing_high'] == 1) | (df['is_swing_low'] == 1)].copy()

    for i, pivot_row in pivots.iterrows():
        # Use the dynamic atr_col_name
        atr_at_pivot = df.loc[i, atr_col_name]
        if pd.isna(atr_at_pivot): # If ATR is NaN for this pivot, cannot reliably set SL
            continue

        pivot_price = 0
        is_long_trade = False

        if pivot_row['is_swing_low'] == 1: # Confirmed swing low, looking for long entry
            pivot_price = pivot_row['low']
            is_long_trade = True
            entry_price = pivot_price + (pivot_row['high'] - pivot_price) * (1 - FIB_LEVEL_ENTRY) # Retrace from high after low
            sl_price = pivot_price - atr_at_pivot # Simple ATR based SL
            tp1_price = entry_price + (pivot_row['high'] - entry_price) * FIB_LEVEL_TP1 # Project from entry to pivot high
            tp2_price = entry_price + (pivot_row['high'] - entry_price) * FIB_LEVEL_TP2
            tp3_price = entry_price + (pivot_row['high'] - entry_price) * FIB_LEVEL_TP3_EXTENSION
        elif pivot_row['is_swing_high'] == 1: # Confirmed swing high, looking for short entry
            pivot_price = pivot_row['high']
            is_long_trade = False
            entry_price = pivot_price - (pivot_price - pivot_row['low']) * (1 - FIB_LEVEL_ENTRY) # Retrace from low after high
            sl_price = pivot_price + atr_at_pivot
            tp1_price = entry_price - (entry_price - pivot_row['low']) * FIB_LEVEL_TP1
            tp2_price = entry_price - (entry_price - pivot_row['low']) * FIB_LEVEL_TP2
            tp3_price = entry_price - (entry_price - pivot_row['low']) * FIB_LEVEL_TP3_EXTENSION
        else:
            continue

        df.loc[i, 'entry_price_sim'] = entry_price
        df.loc[i, 'sl_price_sim'] = sl_price
        df.loc[i, 'tp1_price_sim'] = tp1_price
        df.loc[i, 'tp2_price_sim'] = tp2_price
        df.loc[i, 'tp3_price_sim'] = tp3_price

        # Simulate trade progression in subsequent bars
        outcome = 0 # Default to stopped-out
        entered_trade = False
        for k in range(i + 1, len(df)):
            bar_low = df['low'].iloc[k]
            bar_high = df['high'].iloc[k]

            if is_long_trade:
                if not entered_trade and bar_low <= entry_price: # Entry triggered
                    entered_trade = True
                if entered_trade:
                    if bar_low <= sl_price: # Stop loss hit
                        outcome = 0
                        break
                    if bar_high >= tp3_price: # TP3 hit
                        outcome = 3
                        break
                    if bar_high >= tp2_price and outcome < 2: # TP2 hit
                        outcome = 2
                    if bar_high >= tp1_price and outcome < 1: # TP1 hit
                        outcome = 1
            else: # Short trade
                if not entered_trade and bar_high >= entry_price: # Entry triggered
                    entered_trade = True
                if entered_trade:
                    if bar_high >= sl_price: # Stop loss hit
                        outcome = 0
                        break
                    if bar_low <= tp3_price: # TP3 hit
                        outcome = 3
                        break
                    if bar_low <= tp2_price and outcome < 2: # TP2 hit
                        outcome = 2
                    if bar_low <= tp1_price and outcome < 1: # TP1 hit
                        outcome = 1
            # What if trade doesn't hit SL or TP within a certain number of bars?
            # For now, assume it resolves or we'd need a max trade duration.
            if k > i + 100 and entered_trade and outcome == 0 : # Max holding period, still SL if no TP
                 # If no TP hit after X bars, and not SL, consider it unresolved or some default.
                 # For this labeling, if it hasn't hit SL or TP, it might be considered SL if not exited.
                 # This part needs refinement based on strategy rules (e.g., time-based exit).
                 # For now, if it doesn't hit a TP and is still open, it's not counted as a win.
                 # If it hits SL, it's 0. If it hits TPs, it's 1,2,3.
                 # If it's still open after many bars without hitting SL, it's still outcome 0 for this labeling.
                 pass


        if entered_trade: # Only record outcome if trade was entered
            df.loc[i, 'trade_outcome'] = outcome
        else: # Trade not triggered
            df.loc[i, 'trade_outcome'] = -1 # Reset to -1 if entry not triggered

    return df

# --- 2. Feature Engineering ---

def calculate_ema(df, period, column='close'):
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_rsi(df, period=14, column='close'):
    delta = df[column].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def engineer_pivot_features(df, atr_col_name): # Added atr_col_name
    """
    Engineers features for the pivot detection model.
    `atr_col_name` should be the name of the ATR column to use (e.g., 'atr_14').
    """
    # The caller is responsible for ensuring df contains the correct atr_col_name.
    # This function will use the provided atr_col_name.
    if atr_col_name not in df.columns:
        print(f"Error (engineer_pivot_features): Required ATR column '{atr_col_name}' not found. Feature engineering may be incomplete or fail.")
        # As a fallback, create a dummy ATR column with NaNs if not present, so code doesn't break, but features will be NaN.
        # This is not ideal; caller should provide it.
        df[atr_col_name] = np.nan 
        # A better fallback might be to calculate it using a default period if an atr_period param was also passed.

    # Volatility & Range
    df['range_atr_norm'] = (df['high'] - df['low']) / df[atr_col_name]

    # Trend & Momentum
    df['ema12'] = calculate_ema(df, 12)
    df['ema26'] = calculate_ema(df, 26)
    df['macd_line'] = df['ema12'] - df['ema26']
    # Use dynamic atr_col_name for normalization
    df['macd_slope_atr_norm'] = df['macd_line'].diff() / df[atr_col_name]

    for n in [1, 3, 5]:
        # Use dynamic atr_col_name for normalization
        df[f'return_{n}b_atr_norm'] = df['close'].pct_change(n) / df[atr_col_name]

    # Local Structure
    df['high_rank_7'] = df['high'].rolling(window=7).rank(pct=True)
    # Bars since last candidate pivot (requires iterating or more complex logic)
    # This is tricky to do vectorized efficiently for *any* candidate.
    # For now, let's use bars since last *confirmed* pivot as a proxy, though the request was candidate.
    # This might be better done during the iteration when confirming pivots or in a post-processing step.
    # Placeholder:
    df['bars_since_last_pivot'] = 0
    last_pivot_idx = -1
    for i in range(len(df)):
        if df['is_swing_high'].iloc[i] == 1 or df['is_swing_low'].iloc[i] == 1:
            last_pivot_idx = i
        if last_pivot_idx != -1:
            df.loc[df.index[i], 'bars_since_last_pivot'] = i - last_pivot_idx
        else:
            df.loc[df.index[i], 'bars_since_last_pivot'] = i # Or a large number

    # Volume
    df['volume_rolling_avg_20'] = df['volume'].rolling(window=20).mean()
    df['volume_spike_vs_avg'] = df['volume'] / df['volume_rolling_avg_20']

    # Add more features as needed: RSI, other indicators
    df['rsi_14'] = calculate_rsi(df, 14)

    # Target: 'pivot_label' (0: none, 1: high, 2: low)
    feature_cols = [
        atr_col_name, 'range_atr_norm', 'macd_slope_atr_norm', # Use dynamic atr_col_name
        'return_1b_atr_norm', 'return_3b_atr_norm', 'return_5b_atr_norm',
        'high_rank_7', 'bars_since_last_pivot', 'volume_spike_vs_avg', 'rsi_14'
    ]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df, feature_cols

def engineer_entry_features(df, atr_col_name, entry_features_base_list_arg=None): # Added atr_col_name, made list an arg
    """
    Engineers features for the entry evaluation model.
    `atr_col_name` is the name of the ATR column to use.
    `entry_features_base_list_arg` can be passed if a specific list is desired, otherwise uses default.
    """
    # The caller is responsible for ensuring df contains the correct atr_col_name.
    if atr_col_name not in df.columns:
        print(f"Error (engineer_entry_features): Required ATR column '{atr_col_name}' not found. Feature engineering may be incomplete or fail.")
        df[atr_col_name] = np.nan # Fallback dummy column

    # Features will be calculated at the time of the pivot.
    # `norm_dist_entry_pivot` and `norm_dist_entry_sl` are calculated *outside* this function,
    # typically in objective_optuna or full_backtest, because they depend on simulated trade prices
    # and the specific pivot point being evaluated. This function engineers general market context features.
    # `P_swing` is also a meta-feature added externally.
    # These would typically be available from the `simulate_fib_entries` output or live calculations.

    # Placeholder columns - these should be populated based on the detected pivot and entry simulation
    # df['entry_price_actual'] = np.nan # To be filled
    # df['pivot_price_actual'] = np.nan # To be filled
    # df['sl_price_actual'] = np.nan    # To be filled
    # df['P_swing'] = np.nan           # To be filled (meta-feature)

    # Normalized Distances (calculated for rows that are pivots)
    # (entry_price - pivot_price)/ATR14
    # (entry_price - SL_price)/ATR14
    # These require the context of a specific pivot and its simulated entry.
    # We will add these features to the rows identified as pivots.

    # Extended Trend
    df['ema20'] = calculate_ema(df, 20)
    df['ema50'] = calculate_ema(df, 50)
    # Use dynamic atr_col_name for normalization
    df['ema20_ema50_norm_atr'] = (df['ema20'] - df['ema50']) / df[atr_col_name]

    # Recent Behavior
    for n in [1, 3, 5]: # Returns *before* entry
        df[f'return_entry_{n}b'] = df['close'].pct_change(n) 
    # Use dynamic atr_col_name for ATR change feature
    df[f'{atr_col_name}_change'] = df[atr_col_name].pct_change()

    # Contextual Flags
    # Ensure 'timestamp' column exists and is datetime type
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    else: # Fallback if timestamp column is missing or not datetime
        print("Warning (engineer_entry_features): 'timestamp' column missing or not datetime. Time features will be NaN.")
        df['hour_of_day'] = np.nan
        df['day_of_week'] = np.nan


    # Regime cluster label (e.g. low/high vol from K-means) - Placeholder
    # This would require a separate clustering step on volatility or other features.
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Regime cluster label (e.g. low/high vol from K-means) - Placeholder
    # This would require a separate clustering step on volatility or other features.
    df['vol_regime'] = 0 # Example: 0 for low, 1 for high. Needs actual implementation.

    # Meta-Feature: P_swing (This will be added when preparing data for the entry model)

    # Target: 'trade_outcome' (0=SL, 1=TP1, 2=TP2, 3=TP3)
    # The base features engineered by this function.
    # Specific features like 'norm_dist_entry_pivot', 'norm_dist_entry_sl', 'P_swing'
    # are added externally where pivot context is available.
    if entry_features_base_list_arg is None: # Use default if not provided
        _entry_feature_cols_base = [
            'ema20_ema50_norm_atr',
            'return_entry_1b', 'return_entry_3b', 'return_entry_5b',
            f'{atr_col_name}_change', # Dynamic ATR column name
            'hour_of_day', 'day_of_week', 'vol_regime'
        ]
    else:
        _entry_feature_cols_base = entry_features_base_list_arg

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df, _entry_feature_cols_base


# --- 3. Model Training & Validation ---

# Modified to accept pivot_max_depth for RF, and num_leaves/learning_rate for LGBM
def train_pivot_model(X_train, y_train, X_val, y_val, model_type='lgbm', 
                      num_leaves=31, learning_rate=0.05, max_depth=7):
    """Trains pivot detection model."""
    if model_type == 'lgbm':
        # Target: multiclass {none, high, low}
        # pivot_label: 0=none, 1=high, 2=low
        model = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(stopping_rounds=10, verbose=-1)])
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
    else:
        raise ValueError("Unsupported model type for pivot detection.")

    # Evaluate (example)
    preds_val = model.predict(X_val)
    # For multiclass:
    # Precision/Recall for high (label 1) and low (label 2) swings
    precision_high = precision_score(y_val, preds_val, labels=[1], average='micro', zero_division=0) # Adjust labels/average as needed
    recall_high = recall_score(y_val, preds_val, labels=[1], average='micro', zero_division=0)
    precision_low = precision_score(y_val, preds_val, labels=[2], average='micro', zero_division=0)
    recall_low = recall_score(y_val, preds_val, labels=[2], average='micro', zero_division=0)

    print(f"Pivot Model ({model_type}) Validation:")
    print(f"  Precision (High): {precision_high:.3f}, Recall (High): {recall_high:.3f}")
    print(f"  Precision (Low): {precision_low:.3f}, Recall (Low): {recall_low:.3f}")
    # print(confusion_matrix(y_val, preds_val))
    return model

def train_entry_model(X_train, y_train, X_val, y_val, model_type='lgbm'):
    """Trains entry profitability model."""
    # Target: binary (profit > 1R vs not) or multiclass (TP1/TP2/TP3/loss)
    # For this example, let's use multiclass trade_outcome (0,1,2,3)
    # We might want to transform this to binary: profitable (1,2,3) vs loss (0)
    y_train_binary = (y_train > 0).astype(int)
    y_val_binary = (y_val > 0).astype(int)

    if model_type == 'lgbm':
        model = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train_binary, eval_set=[(X_val, y_val_binary)], callbacks=[early_stopping(stopping_rounds=10, verbose=-1)])
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train_binary)
    # TODO: Add MLP option
    else:
        raise ValueError("Unsupported model type for entry evaluation.")

    # Evaluate (example for binary profitable vs not)
    preds_val = model.predict(X_val)
    precision = precision_score(y_val_binary, preds_val, zero_division=0)
    recall = recall_score(y_val_binary, preds_val, zero_division=0)
    print(f"Entry Model ({model_type}) Validation (Binary Profitable):")
    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")
    # print(confusion_matrix(y_val_binary, preds_val))
    return model


def objective_optuna(trial, df_processed, pivot_features, entry_features_base):
    """Optuna objective function."""
    # Hyperparameters to tune
    # Swing model
    pivot_model_type = trial.suggest_categorical('pivot_model_type', ['lgbm']) # Could add 'rf'
    pivot_num_leaves = trial.suggest_int('pivot_num_leaves', 20, 50) if pivot_model_type == 'lgbm' else None
    pivot_learning_rate = trial.suggest_float('pivot_learning_rate', 0.01, 0.1) if pivot_model_type == 'lgbm' else None
    pivot_max_depth = trial.suggest_int('pivot_max_depth', 5, 10) # Also for RF
    # Entry model
    entry_model_type = trial.suggest_categorical('entry_model_type', ['lgbm'])
    entry_num_leaves = trial.suggest_int('entry_num_leaves', 20, 50) if entry_model_type == 'lgbm' else None
    entry_learning_rate = trial.suggest_float('entry_learning_rate', 0.01, 0.1) if entry_model_type == 'lgbm' else None
    entry_max_depth = trial.suggest_int('entry_max_depth', 5, 10)
    # Thresholds
    p_swing_threshold = trial.suggest_float('p_swing_threshold', 0.5, 0.9)
    profit_threshold = trial.suggest_float('profit_threshold', 0.5, 0.9) # For P_profit

    # --- NEW: Strategy/Data Processing Parameters to Tune ---
    current_atr_period = trial.suggest_int('atr_period_opt', 10, 24) # Example range for ATR period
    current_pivot_n_left = trial.suggest_int('pivot_n_left_opt', 2, 7)
    current_pivot_n_right = trial.suggest_int('pivot_n_right_opt', 2, 7)
    current_min_atr_distance = trial.suggest_float('min_atr_distance_opt', 0.5, 2.5) # Factor for ATR distance
    current_min_bar_gap = trial.suggest_int('min_bar_gap_opt', 5, 15) # Min bars between pivots

    atr_col_name_optuna = f'atr_{current_atr_period}'

    # --- Data Prep for Optuna Trial ---
    # The crucial part for the *next* step is that df_processed would be re-processed here
    # using current_atr_period, current_pivot_n_left, etc.
    # For *this* step, we acknowledge these parameters are suggested, but the functions
    # below still use the globally defined ATR_PERIOD for feature names, etc.
    # This will be rectified in the "Refactor Data Processing" step.
    print(f"Optuna Trial Suggests: ATR_P={current_atr_period}, Pivot_L={current_pivot_n_left}, Pivot_R={current_pivot_n_right}, MinATR_D={current_min_atr_distance:.2f}, MinBar_G={current_min_bar_gap}")

    train_size = int(0.7 * len(df_processed))
    val_size = int(0.15 * len(df_processed))
    # test_size = len(df_processed) - train_size - val_size

    df_train = df_processed.iloc[:train_size].copy()
    df_val = df_processed.iloc[train_size:train_size + val_size].copy()

    # Prepare data for pivot model
    X_pivot_train = df_train[pivot_features].fillna(-1) # Simple imputation
    y_pivot_train = df_train['pivot_label']
    X_pivot_val = df_val[pivot_features].fillna(-1)
    y_pivot_val = df_val['pivot_label']

    # Train Pivot Model
    if pivot_model_type == 'lgbm':
        pivot_model = lgb.LGBMClassifier(num_leaves=pivot_num_leaves, learning_rate=pivot_learning_rate,
                                         max_depth=pivot_max_depth, class_weight='balanced', random_state=42, n_estimators=100)
        pivot_model.fit(X_pivot_train, y_pivot_train, eval_set=[(X_pivot_val, y_pivot_val)], callbacks=[early_stopping(stopping_rounds=5, verbose=-1)])
    else: # rf
        pivot_model = RandomForestClassifier(n_estimators=100, max_depth=pivot_max_depth, class_weight='balanced', random_state=42)
        pivot_model.fit(X_pivot_train, y_pivot_train)

    # Predict P_swing on validation set (and train for entry model training)
    # We need probabilities for high (1) and low (2)
    p_swing_train_all_classes = pivot_model.predict_proba(X_pivot_train)
    p_swing_val_all_classes = pivot_model.predict_proba(X_pivot_val)

    # P_swing is max prob of being a high or low pivot
    df_train['P_swing'] = np.max(p_swing_train_all_classes[:, 1:], axis=1) # Prob of class 1 (high) or 2 (low)
    df_val['P_swing'] = np.max(p_swing_val_all_classes[:, 1:], axis=1)


    # Filter validation data based on P_swing_threshold for entry model evaluation
    # Entry model is trained on *actual* pivots from training data that would have passed the threshold
    # This is a bit tricky: for training entry model, we should use actual pivots.
    # For evaluating the *pipeline*, we use predicted pivots.

    # Let's train entry model on actual pivots from df_train that *also* have a high P_swing
    # This ensures the entry model learns from reasonably good pivot candidates.
    entry_train_candidates = df_train[
        (df_train['pivot_label'].isin([1, 2])) & # Is an actual pivot
        (df_train['trade_outcome'] != -1) &      # Trade was simulated
        (df_train['P_swing'] >= p_swing_threshold) # Pivot model would have picked it
    ].copy()


    if len(entry_train_candidates) < 50: # Not enough samples to train entry model
        return -1.0 # Bad score if no trades

    # Add specific entry features for these candidates
    entry_train_candidates['norm_dist_entry_pivot'] = (entry_train_candidates['entry_price_sim'] - entry_train_candidates.apply(lambda r: r['low'] if r['is_swing_low'] else r['high'], axis=1)) / entry_train_candidates[atr_col_name_optuna]
    entry_train_candidates['norm_dist_entry_sl'] = (entry_train_candidates['entry_price_sim'] - entry_train_candidates['sl_price_sim']).abs() / entry_train_candidates[atr_col_name_optuna]


    X_entry_train = entry_train_candidates[entry_features_base + ['P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl']].fillna(-1)
    y_entry_train = (entry_train_candidates['trade_outcome'] > 0).astype(int) # Binary profitable

    if len(X_entry_train['P_swing'].unique()) < 2 or len(y_entry_train.unique()) < 2 : # Check for variance
        return -1.0


    # Train Entry Model
    if entry_model_type == 'lgbm':
        entry_model = lgb.LGBMClassifier(num_leaves=entry_num_leaves, learning_rate=entry_learning_rate,
                                         max_depth=entry_max_depth, class_weight='balanced', random_state=42, n_estimators=100)
        # Need a small val set for early stopping if used, from entry_train_candidates
        if len(entry_train_candidates) > 20:
             X_entry_train_sub, X_entry_val_sub, y_entry_train_sub, y_entry_val_sub = train_test_split(X_entry_train, y_entry_train, test_size=0.2, stratify=y_entry_train if len(y_entry_train.unique()) > 1 else None, random_state=42)
             if len(X_entry_val_sub) > 0 and len(y_entry_val_sub.unique()) > 1:
                entry_model.fit(X_entry_train_sub, y_entry_train_sub, eval_set=[(X_entry_val_sub, y_entry_val_sub)], callbacks=[early_stopping(stopping_rounds=5, verbose=-1)])
             else:
                entry_model.fit(X_entry_train, y_entry_train) # No early stopping
        else:
            entry_model.fit(X_entry_train, y_entry_train)

    else: # rf
        entry_model = RandomForestClassifier(n_estimators=100, max_depth=entry_max_depth, class_weight='balanced', random_state=42)
        entry_model.fit(X_entry_train, y_entry_train)


    # --- Mini Backtest on df_val ---
    # 1. Identify pivots in df_val using the trained pivot_model and p_swing_threshold
    potential_pivots_val = df_val[df_val['P_swing'] >= p_swing_threshold].copy()
    potential_pivots_val = potential_pivots_val[potential_pivots_val['trade_outcome'] != -1] # Ensure these are rows where a trade could happen

    if len(potential_pivots_val) == 0:
        return -0.5 # No trades triggered by pivot model

    # 2. For these pivots, compute entry features (including the P_swing from pivot_model)
    potential_pivots_val['norm_dist_entry_pivot'] = (potential_pivots_val['entry_price_sim'] - potential_pivots_val.apply(lambda r: r['low'] if r['is_swing_low'] else r['high'], axis=1)) / potential_pivots_val[atr_col_name_optuna]
    potential_pivots_val['norm_dist_entry_sl'] = (potential_pivots_val['entry_price_sim'] - potential_pivots_val['sl_price_sim']).abs() / potential_pivots_val[atr_col_name_optuna]

    X_entry_eval = potential_pivots_val[entry_features_base + ['P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl']].fillna(-1)

    if len(X_entry_eval) == 0: return -0.5

    # 3. Predict P_profit using the trained entry_model
    p_profit_val = entry_model.predict_proba(X_entry_eval)[:, 1] # Probability of class 1 (profitable)

    # 4. Filter trades based on profit_threshold
    final_trades_val = potential_pivots_val[p_profit_val >= profit_threshold]

    if len(final_trades_val) == 0:
        return 0.0 # No trades made it through the full pipeline

    # 5. Calculate backtest metric (e.g., Sharpe, Profit Factor)
    # Using a simplified profit factor: sum of profits / sum of losses
    # Assuming 1R loss for SL, and R for TPs (e.g. TP1=1R, TP2=2R, TP3=3R approx)
    # This is a simplification; actual R would depend on SL placement relative to entry.
    # For now, use outcome: 0=loss (-1R), 1=TP1 (1R), 2=TP2 (2R), 3=TP3 (3R)
    # This needs to be linked to the actual Fib structure for R values.
    # Simplified:
    profit_sum = 0
    loss_sum = 0
    for idx, trade in final_trades_val.iterrows():
        outcome = trade['trade_outcome']
        if outcome == 0: # SL
            loss_sum += 1
        elif outcome == 1: # TP1
            profit_sum += 1 # Simplified R value
        elif outcome == 2: # TP2
            profit_sum += 2
        elif outcome == 3: # TP3
            profit_sum += 3

    if loss_sum == 0 and profit_sum > 0:
        return profit_sum # High score for no losses
    if loss_sum == 0 and profit_sum == 0: # No trades or all scratch
        return 0.0
    profit_factor = profit_sum / loss_sum

    # Objective: Maximize profit factor (or Sharpe)
    # Number of trades penalty could be added if too few trades.
    # return profit_factor * np.log1p(len(final_trades_val)) # Penalize if too few trades
    return profit_factor if profit_factor > 0 else -1.0 * (1/ (profit_factor -0.001)) # Penalize negative PFs heavily


# Modified to accept df_raw (less processed) and static_entry_features_base
def objective_optuna(trial, df_raw, static_entry_features_base): # Removed pivot_features, entry_features_base
    """Optuna objective function. Performs data processing per trial."""
    # Hyperparameters to tune for models
    pivot_model_type = trial.suggest_categorical('pivot_model_type', ['lgbm'])
    pivot_num_leaves = trial.suggest_int('pivot_num_leaves', 20, 50) if pivot_model_type == 'lgbm' else None
    pivot_learning_rate = trial.suggest_float('pivot_learning_rate', 0.01, 0.1) if pivot_model_type == 'lgbm' else None
    pivot_max_depth = trial.suggest_int('pivot_max_depth', 5, 10)
    entry_model_type = trial.suggest_categorical('entry_model_type', ['lgbm'])
    entry_num_leaves = trial.suggest_int('entry_num_leaves', 20, 50) if entry_model_type == 'lgbm' else None
    entry_learning_rate = trial.suggest_float('entry_learning_rate', 0.01, 0.1) if entry_model_type == 'lgbm' else None
    entry_max_depth = trial.suggest_int('entry_max_depth', 5, 10)
    # Thresholds
    p_swing_threshold = trial.suggest_float('p_swing_threshold', 0.5, 0.9)
    profit_threshold = trial.suggest_float('profit_threshold', 0.5, 0.9)

    # --- Strategy/Data Processing Parameters to Tune ---
    current_atr_period = trial.suggest_int('atr_period_opt', 10, 24)
    current_pivot_n_left = trial.suggest_int('pivot_n_left_opt', 2, 7)
    current_pivot_n_right = trial.suggest_int('pivot_n_right_opt', 2, 7)
    current_min_atr_distance = trial.suggest_float('min_atr_distance_opt', 0.5, 2.5)
    current_min_bar_gap = trial.suggest_int('min_bar_gap_opt', 5, 15)

    atr_col_name_optuna = f'atr_{current_atr_period}'
    print(f"Optuna Trial - Params: ATR_P={current_atr_period}, Pivot_L={current_pivot_n_left}, Pivot_R={current_pivot_n_right}, MinATR_D={current_min_atr_distance:.2f}, MinBar_G={current_min_bar_gap}")

    # --- Per-Trial Data Processing ---
    df_trial_processed = df_raw.copy() # Start with a fresh copy of the raw data for each trial

    # 1. Calculate ATR for the current trial's period
    df_trial_processed = calculate_atr(df_trial_processed, period=current_atr_period)
    if atr_col_name_optuna not in df_trial_processed.columns:
        print(f"Error: ATR column '{atr_col_name_optuna}' not created in trial. Skipping trial.")
        return -100.0 # Penalize heavily

    # 2. Generate candidate pivots
    df_trial_processed = generate_candidate_pivots(df_trial_processed, n_left=current_pivot_n_left, n_right=current_pivot_n_right)

    # 3. Prune and label pivots
    df_trial_processed = prune_and_label_pivots(df_trial_processed, atr_col_name=atr_col_name_optuna, 
                                                atr_distance_factor=current_min_atr_distance, 
                                                min_bar_gap=current_min_bar_gap)

    # 4. Simulate Fibonacci entries
    df_trial_processed = simulate_fib_entries(df_trial_processed, atr_col_name=atr_col_name_optuna)
    
    # Drop rows with NaN in critical columns that might have been introduced or not handled by ATR/simulation
    # This is important before feature engineering
    df_trial_processed.dropna(subset=[atr_col_name_optuna, 'low', 'high', 'close'], inplace=True) # Add other critical columns if necessary
    df_trial_processed.reset_index(drop=True, inplace=True)

    if len(df_trial_processed) < 100: # Check if enough data remains after initial processing
        print(f"Warning: Not enough data ({len(df_trial_processed)} rows) after initial trial processing. Skipping trial.")
        return -99.0


    # 5. Engineer pivot features (returns DataFrame and pivot_feature_names for this trial)
    df_trial_processed, trial_pivot_features = engineer_pivot_features(df_trial_processed, atr_col_name=atr_col_name_optuna)

    # 6. Engineer entry features (returns DataFrame and entry_feature_names_base for this trial)
    # Pass the static_entry_features_base list which engineer_entry_features will use and potentially extend
    # with atr_col_name_optuna related features.
    df_trial_processed, trial_entry_features_base = engineer_entry_features(df_trial_processed, atr_col_name=atr_col_name_optuna, 
                                                                          entry_features_base_list_arg=static_entry_features_base)

    df_trial_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    # It's crucial to handle NaNs from feature engineering before splitting
    df_trial_processed.dropna(subset=trial_pivot_features, inplace=True) # Drop rows where pivot features are NaN
    # For entry features, NaNs are typically handled on the subset of data used for entry model training.
    df_trial_processed.reset_index(drop=True, inplace=True)
    
    if len(df_trial_processed) < 100: # Check if enough data remains after feature engineering
        print(f"Warning: Not enough data ({len(df_trial_processed)} rows) after trial feature engineering. Skipping trial.")
        return -98.0

    # --- Data Splitting ---
    # The rest of the function (splitting, training, eval) remains largely the same,
    # but uses df_trial_processed and trial_pivot_features / trial_entry_features_base.
    train_size = int(0.7 * len(df_trial_processed))
    val_size = int(0.15 * len(df_trial_processed))

    df_train = df_trial_processed.iloc[:train_size].copy()
    df_val = df_trial_processed.iloc[train_size:train_size + val_size].copy()

    if len(df_train) < 50 or len(df_val) < 20: # Ensure enough data for train/val
        print(f"Warning: Not enough data for training/validation sets in trial. Train: {len(df_train)}, Val: {len(df_val)}. Skipping.")
        return -97.0

    # Prepare data for pivot model
    X_pivot_train = df_train[trial_pivot_features].fillna(-1) 
    y_pivot_train = df_train['pivot_label']
    X_pivot_val = df_val[trial_pivot_features].fillna(-1)
    y_pivot_val = df_val['pivot_label']

    if X_pivot_train.empty or X_pivot_val.empty:
        print("Warning: Pivot training or validation features are empty. Skipping trial.")
        return -96.0

    # Train Pivot Model
    if pivot_model_type == 'lgbm':
        pivot_model = lgb.LGBMClassifier(num_leaves=pivot_num_leaves, learning_rate=pivot_learning_rate,
                                         max_depth=pivot_max_depth, class_weight='balanced', random_state=42, n_estimators=100, verbosity=-1)
        pivot_model.fit(X_pivot_train, y_pivot_train, eval_set=[(X_pivot_val, y_pivot_val)], callbacks=[early_stopping(stopping_rounds=5, verbose=False)])
    else: # rf
        pivot_model = RandomForestClassifier(n_estimators=100, max_depth=pivot_max_depth, class_weight='balanced', random_state=42)
        pivot_model.fit(X_pivot_train, y_pivot_train)

    p_swing_train_all_classes = pivot_model.predict_proba(X_pivot_train)
    p_swing_val_all_classes = pivot_model.predict_proba(X_pivot_val)
    df_train['P_swing'] = np.max(p_swing_train_all_classes[:, 1:], axis=1)
    df_val['P_swing'] = np.max(p_swing_val_all_classes[:, 1:], axis=1)

    entry_train_candidates = df_train[
        (df_train['pivot_label'].isin([1, 2])) &
        (df_train['trade_outcome'] != -1) &
        (df_train['P_swing'] >= p_swing_threshold)
    ].copy()

    if len(entry_train_candidates) < 50:
        return -1.0 

    entry_train_candidates['norm_dist_entry_pivot'] = (entry_train_candidates['entry_price_sim'] - entry_train_candidates.apply(lambda r: r['low'] if r['is_swing_low'] == 1 else r['high'], axis=1)) / entry_train_candidates[atr_col_name_optuna]
    entry_train_candidates['norm_dist_entry_sl'] = (entry_train_candidates['entry_price_sim'] - entry_train_candidates['sl_price_sim']).abs() / entry_train_candidates[atr_col_name_optuna]
    
    # Construct the full list of entry features for this trial
    current_trial_full_entry_features = trial_entry_features_base + ['P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl']
    
    X_entry_train = entry_train_candidates[current_trial_full_entry_features].fillna(-1)
    y_entry_train = (entry_train_candidates['trade_outcome'] > 0).astype(int)

    if len(X_entry_train['P_swing'].unique()) < 2 or len(y_entry_train.unique()) < 2 or X_entry_train.empty:
        return -1.0

    if entry_model_type == 'lgbm':
        entry_model = lgb.LGBMClassifier(num_leaves=entry_num_leaves, learning_rate=entry_learning_rate,
                                         max_depth=entry_max_depth, class_weight='balanced', random_state=42, n_estimators=100, verbosity=-1)
        if len(entry_train_candidates) > 20:
             X_entry_train_sub, X_entry_val_sub, y_entry_train_sub, y_entry_val_sub = train_test_split(X_entry_train, y_entry_train, test_size=0.2, stratify=y_entry_train if len(y_entry_train.unique()) > 1 else None, random_state=42)
             if len(X_entry_val_sub) > 0 and len(y_entry_val_sub.unique()) > 1:
                entry_model.fit(X_entry_train_sub, y_entry_train_sub, eval_set=[(X_entry_val_sub, y_entry_val_sub)], callbacks=[early_stopping(stopping_rounds=5, verbose=False)])
             else:
                entry_model.fit(X_entry_train, y_entry_train) 
        else:
            entry_model.fit(X_entry_train, y_entry_train)
    else: # rf
        entry_model = RandomForestClassifier(n_estimators=100, max_depth=entry_max_depth, class_weight='balanced', random_state=42)
        entry_model.fit(X_entry_train, y_entry_train)

    potential_pivots_val = df_val[df_val['P_swing'] >= p_swing_threshold].copy()
    potential_pivots_val = potential_pivots_val[potential_pivots_val['trade_outcome'] != -1]

    if len(potential_pivots_val) == 0:
        return -0.5

    potential_pivots_val['norm_dist_entry_pivot'] = (potential_pivots_val['entry_price_sim'] - potential_pivots_val.apply(lambda r: r['low'] if r['is_swing_low'] == 1 else r['high'], axis=1)) / potential_pivots_val[atr_col_name_optuna]
    potential_pivots_val['norm_dist_entry_sl'] = (potential_pivots_val['entry_price_sim'] - potential_pivots_val['sl_price_sim']).abs() / potential_pivots_val[atr_col_name_optuna]

    X_entry_eval = potential_pivots_val[current_trial_full_entry_features].fillna(-1)

    if len(X_entry_eval) == 0: return -0.5

    p_profit_val = entry_model.predict_proba(X_entry_eval)[:, 1]
    final_trades_val = potential_pivots_val[p_profit_val >= profit_threshold]

    if len(final_trades_val) == 0:
        return 0.0

    profit_sum = 0; loss_sum = 0
    for idx, trade in final_trades_val.iterrows():
        outcome = trade['trade_outcome']
        if outcome == 0: loss_sum += 1
        elif outcome == 1: profit_sum += 1
        elif outcome == 2: profit_sum += 2
        elif outcome == 3: profit_sum += 3
    if loss_sum == 0 and profit_sum > 0: return profit_sum
    if loss_sum == 0 and profit_sum == 0: return 0.0
    profit_factor = profit_sum / loss_sum
    return profit_factor if profit_factor > 0 else -1.0 * (1/ (profit_factor -0.001))


def run_optuna_tuning(df_universal_raw, static_entry_features_base_list, n_trials=50): # Renamed df_processed to df_universal_raw, changed features params
    """Runs Optuna hyperparameter tuning."""
    study = optuna.create_study(direction='maximize')
    # Pass the raw DataFrame and the list of static base entry features to objective_optuna
    study.optimize(lambda trial: objective_optuna(trial, df_universal_raw, static_entry_features_base_list),
                   n_trials=n_trials,
                   # Consider using a time-series aware sampler/pruner if available and suitable
                   # For now, default sampler.
                   )
    print("Best Optuna trial:", study.best_trial.params)

    # Save best_params to a JSON file
    try:
        import json
        # Include feature names and ATR period used for training if they are fixed or derived
        # For now, just saving Optuna's direct output.
        # Consider adding 'pivot_feature_names': pivot_features, 'entry_features_base': entry_features_base,
        # 'model_atr_period': ATR_PERIOD to this dictionary before saving.
        params_to_save = study.best_trial.params.copy()
        # Example of adding more info:
        # params_to_save['_comment'] = "Add feature names and other relevant training settings here"
        # params_to_save['pivot_feature_names_example'] = ['atr_14', 'range_atr_norm', ...] # Replace with actual list
        # params_to_save['model_training_atr_period'] = ATR_PERIOD


        with open("best_model_params.json", 'w') as f:
            json.dump(params_to_save, f, indent=4)
        print(f"Best Optuna parameters saved to best_model_params.json")
    except Exception as e:
        print(f"Error saving Optuna best parameters: {e}")

    return study.best_trial.params


def process_dataframe_with_params(df_initial, params, static_entry_features_base_list_arg=None):
    """
    Processes a DataFrame using a given set of parameters (typically best_hyperparams from Optuna).
    This function mirrors the per-trial processing logic in objective_optuna.
    """
    print(f"Processing DataFrame with params: {params}")
    df_processed = df_initial.copy()

    # Extract parameters, providing defaults if not all are in 'params' (e.g. if Optuna didn't tune some)
    atr_period = params.get('atr_period_opt', ATR_PERIOD)
    pivot_n_left = params.get('pivot_n_left_opt', PIVOT_N_LEFT)
    pivot_n_right = params.get('pivot_n_right_opt', PIVOT_N_RIGHT)
    min_atr_distance = params.get('min_atr_distance_opt', MIN_ATR_DISTANCE)
    min_bar_gap = params.get('min_bar_gap_opt', MIN_BAR_GAP)
    
    atr_col_name = f'atr_{atr_period}'

    # 1. Calculate ATR
    df_processed = calculate_atr(df_processed, period=atr_period)
    if atr_col_name not in df_processed.columns:
        print(f"Error processing with params: ATR column '{atr_col_name}' not created.")
        return None, None, None # Or raise an error

    # 2. Generate candidate pivots
    df_processed = generate_candidate_pivots(df_processed, n_left=pivot_n_left, n_right=pivot_n_right)

    # 3. Prune and label pivots
    df_processed = prune_and_label_pivots(df_processed, atr_col_name=atr_col_name, 
                                          atr_distance_factor=min_atr_distance, 
                                          min_bar_gap=min_bar_gap)

    # 4. Simulate Fibonacci entries
    df_processed = simulate_fib_entries(df_processed, atr_col_name=atr_col_name)
    
    df_processed.dropna(subset=[atr_col_name, 'low', 'high', 'close'], inplace=True)
    df_processed.reset_index(drop=True, inplace=True)

    if len(df_processed) < 30: # Min data check
        print(f"Warning: Not enough data ({len(df_processed)} rows) after initial processing with best_params.")
        return None, None, None

    # 5. Engineer pivot features
    df_processed, final_pivot_features = engineer_pivot_features(df_processed, atr_col_name=atr_col_name)

    # 6. Engineer entry features
    df_processed, final_entry_features_base = engineer_entry_features(
        df_processed, 
        atr_col_name=atr_col_name, 
        entry_features_base_list_arg=static_entry_features_base_list_arg # This should be the static list
    )

    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed.dropna(subset=final_pivot_features, inplace=True)
    df_processed.reset_index(drop=True, inplace=True)
    
    if len(df_processed) < 30:
        print(f"Warning: Not enough data ({len(df_processed)} rows) after feature engineering with best_params.")
        return None, None, None
        
    return df_processed, final_pivot_features, final_entry_features_base


# --- Model Artifacts & Backtesting ---
def save_model(model, filename="model.joblib"):
    """Saves a model to disk."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename="model.joblib"):
    """Loads a model from disk."""
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def full_backtest(df_processed, pivot_model, entry_model, best_params, pivot_features, entry_features_base):
    """
    Performs a full backtest on the (hold-out) test set.
    Uses the best models and thresholds found by Optuna.
    """
    print("\n--- Starting Full Backtest ---")
    # Determine ATR column name for backtest based on best_params or default
    backtest_atr_period = best_params.get('atr_period_opt', ATR_PERIOD) # Get tuned ATR period or default
    atr_col_name_backtest = f'atr_{backtest_atr_period}'
    print(f"Full Backtest using ATR column: {atr_col_name_backtest}")

    # Split: e.g., last 15% for test
    train_val_size = int(0.85 * len(df_processed))
    df_test = df_processed.iloc[train_val_size:].copy()

    if len(df_test) == 0:
        print("No data in test set for backtest.")
        return

    # 1. Pivot Predictions
    X_pivot_test = df_test[pivot_features].fillna(-1)
    p_swing_test_all_classes = pivot_model.predict_proba(X_pivot_test)
    df_test['P_swing'] = np.max(p_swing_test_all_classes[:, 1:], axis=1)
    df_test['predicted_pivot_class'] = np.argmax(p_swing_test_all_classes, axis=1)


    # 2. Filter by P_swing threshold
    p_swing_threshold = best_params['p_swing_threshold']
    potential_pivots_test = df_test[
        (df_test['P_swing'] >= p_swing_threshold) &
        (df_test['trade_outcome'] != -1) # Has valid simulated outcome
    ].copy()

    if len(potential_pivots_test) == 0:
        print("No pivots passed P_swing threshold in test set.")
        return None, None, None, None

    # 3. Entry Features for these pivots
    potential_pivots_test['norm_dist_entry_pivot'] = (potential_pivots_test['entry_price_sim'] - potential_pivots_test.apply(lambda r: r['low'] if r['predicted_pivot_class'] == 2 else r['high'], axis=1)) / potential_pivots_test[atr_col_name_backtest]
    potential_pivots_test['norm_dist_entry_sl'] = (potential_pivots_test['entry_price_sim'] - potential_pivots_test['sl_price_sim']).abs() / potential_pivots_test[atr_col_name_backtest]

    X_entry_test = potential_pivots_test[entry_features_base + ['P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl']].fillna(-1)

    if len(X_entry_test) == 0:
        print("No data for entry model evaluation in test set after feature engineering.")
        return None, None, None, None

    # 4. Entry Predictions
    p_profit_test = entry_model.predict_proba(X_entry_test)[:, 1] # Prob of being profitable

    # 5. Filter by P_profit threshold
    profit_threshold = best_params['profit_threshold']
    final_trades_test = potential_pivots_test[p_profit_test >= profit_threshold].copy()

    if len(final_trades_test) == 0:
        print("No trades passed P_profit threshold in test set.")
        return 0, 0, 0, 0 # Trades, Win Rate, Avg R, Profit Factor

    # 6. Calculate Metrics
    num_trades = len(final_trades_test)
    wins = final_trades_test[final_trades_test['trade_outcome'] > 0]
    num_wins = len(wins)
    win_rate = num_wins / num_trades if num_trades > 0 else 0

    # Calculate R values (simplified)
    total_r = 0
    profit_sum_bt = 0
    loss_sum_bt = 0
    for idx, trade in final_trades_test.iterrows():
        outcome = trade['trade_outcome']
        # This R calculation is simplified. True R needs entry, SL, TP prices.
        # For now, use the 1,2,3 mapping as approximate R.
        if outcome == 0: # SL
            total_r -= 1
            loss_sum_bt += 1
        elif outcome == 1: # TP1
            total_r += 1 # Example: TP1 = 1R
            profit_sum_bt +=1
        elif outcome == 2: # TP2
            total_r += 2 # Example: TP2 = 2R
            profit_sum_bt += 2
        elif outcome == 3: # TP3
            total_r += 3 # Example: TP3 = 3R
            profit_sum_bt += 3

    avg_r = total_r / num_trades if num_trades > 0 else 0
    profit_factor_bt = profit_sum_bt / loss_sum_bt if loss_sum_bt > 0 else (profit_sum_bt if profit_sum_bt > 0 else 0)


    print(f"Backtest Results (Test Set):")
    print(f"  Number of Trades: {num_trades}")
    print(f"  Win Rate: {win_rate:.3f}")
    print(f"  Average R (simplified): {avg_r:.3f}")
    print(f"  Profit Factor (simplified): {profit_factor_bt:.3f}")
    # TODO: Max drawdown, trade frequency etc.

    return num_trades, win_rate, avg_r, profit_factor_bt, 0 # Placeholder for max_drawdown


def run_backtest_scenario(scenario_name: str, df_processed: pd.DataFrame,
                          pivot_model, entry_model, best_params,
                          pivot_features, entry_features_base,
                          atr_col_name=f'atr_{ATR_PERIOD}',
                          use_full_df_as_test: bool = False):
    """
    Runs a backtest for a specific scenario (Rule-based, ML Stage 1, Full ML).
    If use_full_df_as_test is True, df_processed is considered the entire test set.
    """
    print(f"\n--- Starting Backtest Scenario: {scenario_name} ---")

    if use_full_df_as_test:
        df_test = df_processed.copy()
        print(f"Using full provided DataFrame ({len(df_test)} rows) as test set for {scenario_name}.")
    else:
        # Determine test set (e.g., last 15% of data from df_processed)
        train_val_size = int(0.85 * len(df_processed))
        df_test = df_processed.iloc[train_val_size:].copy()
        print(f"Using last 15% of provided DataFrame ({len(df_test)} rows) as test set for {scenario_name}.")


    if df_test.empty:
        print(f"No data in (derived) test set for {scenario_name} backtest.")
        return {"scenario": scenario_name, "trades": 0, "win_rate": 0, "avg_r": 0, "profit_factor": 0, "max_dd_r": 0, "trade_frequency":0}

    # Ensure df_test has a DatetimeIndex from 'timestamp' column for period calculation
    if 'timestamp' in df_test.columns:
        df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
        df_test.set_index('timestamp', inplace=True)
    else:
        print(f"Warning: 'timestamp' column not found in df_test for scenario {scenario_name}. Cannot accurately calculate test_period_days based on dates.")
        # Fallback or alternative calculation might be needed if 'timestamp' is missing.
        # For now, this might lead to 'days' attribute error if not handled before calculation.

    final_trades_for_metrics = pd.DataFrame()

    if scenario_name == "Rule-Based Baseline":
        potential_trades_df = df_test[
            (df_test['pivot_label'].isin([1,2])) & (df_test['trade_outcome'] != -1)
        ].copy()
        final_trades_for_metrics = potential_trades_df
        print(f"Rule-Based: Identified {len(final_trades_for_metrics)} rule-based trade setups.")

    elif scenario_name == "ML Stage 1 (Pivot Filter)":
        if pivot_model is None:
            print("ML Stage 1: Pivot model not available. Skipping scenario.")
            return {"scenario": scenario_name, "trades": 0, "win_rate": 0, "avg_r": 0, "profit_factor": 0, "max_dd_r": 0, "trade_frequency":0}

        X_pivot_test = df_test[pivot_features].fillna(-1)
        p_swing_test_all_classes = pivot_model.predict_proba(X_pivot_test)
        df_test['P_swing'] = np.max(p_swing_test_all_classes[:, 1:], axis=1)

        p_swing_threshold = best_params.get('p_swing_threshold', 0.6)
        potential_trades_df = df_test[
            (df_test['P_swing'] >= p_swing_threshold) &
            (df_test['pivot_label'].isin([1,2])) &
            (df_test['trade_outcome'] != -1)
        ].copy()
        final_trades_for_metrics = potential_trades_df
        print(f"ML Stage 1: Identified {len(final_trades_for_metrics)} ML-filtered pivot trade setups.")

    elif scenario_name == "Full ML Pipeline":
        if pivot_model is None or entry_model is None:
            print("Full ML Pipeline: Pivot or Entry model not available. Skipping scenario.")
            return {"scenario": scenario_name, "trades": 0, "win_rate": 0, "avg_r": 0, "profit_factor": 0, "max_dd_r": 0, "trade_frequency":0}

        X_pivot_test = df_test[pivot_features].fillna(-1)
        p_swing_test_all_classes = pivot_model.predict_proba(X_pivot_test)
        df_test['P_swing'] = np.max(p_swing_test_all_classes[:, 1:], axis=1)
        df_test['predicted_pivot_class_ml'] = np.argmax(p_swing_test_all_classes, axis=1)

        p_swing_threshold = best_params.get('p_swing_threshold', 0.6)
        profit_threshold = best_params.get('profit_threshold', 0.6)

        potential_pivots_ml = df_test[
            (df_test['P_swing'] >= p_swing_threshold) &
            (df_test['predicted_pivot_class_ml'].isin([1,2])) &
            (df_test['trade_outcome'] != -1)
        ].copy()

        if potential_pivots_ml.empty:
            print("Full ML: No pivots passed P_swing threshold in test set.")
            return {"scenario": scenario_name, "trades": 0, "win_rate": 0, "avg_r": 0, "profit_factor": 0, "max_dd_r": 0, "trade_frequency":0}

        potential_pivots_ml['norm_dist_entry_pivot'] = (potential_pivots_ml['entry_price_sim'] - potential_pivots_ml.apply(lambda r: r['low'] if r['predicted_pivot_class_ml'] == 2 else r['high'], axis=1)) / potential_pivots_ml[atr_col_name]
        potential_pivots_ml['norm_dist_entry_sl'] = (potential_pivots_ml['entry_price_sim'] - potential_pivots_ml['sl_price_sim']).abs() / potential_pivots_ml[atr_col_name]

        X_entry_test_ml = potential_pivots_ml[entry_features_base + ['P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl']].fillna(-1)

        if X_entry_test_ml.empty:
            print("Full ML: No data for entry model evaluation.")
            return {"scenario": scenario_name, "trades": 0, "win_rate": 0, "avg_r": 0, "profit_factor": 0, "max_dd_r": 0, "trade_frequency":0}

        p_profit_test_ml = entry_model.predict_proba(X_entry_test_ml)[:, 1]
        final_trades_for_metrics = potential_pivots_ml[p_profit_test_ml >= profit_threshold].copy()
        print(f"Full ML: Identified {len(final_trades_for_metrics)} full ML pipeline trade setups.")
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    if final_trades_for_metrics.empty:
        print(f"{scenario_name}: No trades to evaluate metrics for.")
        return {"scenario": scenario_name, "trades": 0, "win_rate": 0, "avg_r": 0, "profit_factor": 0, "max_dd_r": 0, "trade_frequency":0}

    num_trades = len(final_trades_for_metrics)
    wins = final_trades_for_metrics[final_trades_for_metrics['trade_outcome'] > 0]
    num_wins = len(wins)
    win_rate = num_wins / num_trades if num_trades > 0 else 0

    r_values = []
    equity_curve_r = [0.0]
    current_equity_r = 0.0
    peak_equity_r = 0.0
    max_drawdown_r = 0.0

    for _, trade in final_trades_for_metrics.iterrows():
        entry_p = trade['entry_price_sim']
        sl_p = trade['sl_price_sim']

        if pd.isna(entry_p) or pd.isna(sl_p) or entry_p == sl_p: continue
        initial_risk_r = abs(entry_p - sl_p)
        if initial_risk_r == 0: continue

        trade_profit_r = 0; outcome = trade['trade_outcome']
        is_long_trade_sim = trade['is_swing_low'] == 1 # Assuming is_swing_low means a long trade was simulated

        if outcome == 0: trade_profit_r = -1.0
        elif outcome == 1:
            tp1_p = trade['tp1_price_sim']
            if pd.notna(tp1_p): trade_profit_r = abs(tp1_p - entry_p) / initial_risk_r if is_long_trade_sim else abs(entry_p - tp1_p) / initial_risk_r
            else: trade_profit_r = 1.0
        elif outcome == 2:
            tp2_p = trade['tp2_price_sim']
            if pd.notna(tp2_p): trade_profit_r = abs(tp2_p - entry_p) / initial_risk_r if is_long_trade_sim else abs(entry_p - tp2_p) / initial_risk_r
            else: trade_profit_r = 2.0
        elif outcome == 3:
            tp3_p = trade['tp3_price_sim']
            if pd.notna(tp3_p): trade_profit_r = abs(tp3_p - entry_p) / initial_risk_r if is_long_trade_sim else abs(entry_p - tp3_p) / initial_risk_r
            else: trade_profit_r = 3.0

        r_values.append(trade_profit_r)
        current_equity_r += trade_profit_r
        equity_curve_r.append(current_equity_r)
        peak_equity_r = max(peak_equity_r, current_equity_r)
        drawdown = peak_equity_r - current_equity_r
        if drawdown > max_drawdown_r: max_drawdown_r = drawdown

    avg_r = np.mean(r_values) if r_values else 0
    profit_sum_r = sum(r for r in r_values if r > 0)
    loss_sum_r_abs = sum(abs(r) for r in r_values if r < 0)
    profit_factor = profit_sum_r / loss_sum_r_abs if loss_sum_r_abs > 0 else (profit_sum_r if profit_sum_r > 0 else 0)

    # Trade frequency: trades per day. Test period duration needed.
    if not df_test.empty and isinstance(df_test.index, pd.DatetimeIndex):
        test_period_days = (df_test.index.max() - df_test.index.min()).days + 1
    elif not df_test.empty:
        # Fallback if not a DatetimeIndex (e.g., 'timestamp' was missing)
        # Estimate based on number of data points and an assumed interval (e.g., 15min)
        # This is less accurate but prevents a crash.
        # Assuming KLINE_INTERVAL_15MINUTE -> 96 intervals per day.
        # This requires KLINE_INTERVAL to be known here or passed. For now, a rough estimate.
        print(f"Warning: Could not determine test period in days accurately for {scenario_name} due to non-DatetimeIndex. Estimating based on row count.")
        # A more robust solution would be to pass the interval to this function.
        # For now, if we know the interval is 15min:
        # num_intervals_per_day = 24 * (60 / 15)
        # test_period_days = len(df_test) / num_intervals_per_day
        # Simplified: assume at least 1 day if there's data.
        test_period_days = max(1, len(df_test) / 96) # Rough estimate for 15min interval
    else:
        test_period_days = 1
    
    trade_frequency = num_trades / test_period_days if test_period_days > 0 else 0


    print(f"Results for {scenario_name}:")
    print(f"  Number of Trades: {num_trades} (Frequency: {trade_frequency:.2f} trades/day over {test_period_days} days)")
    print(f"  Win Rate: {win_rate:.3f}")
    print(f"  Average R: {avg_r:.3f}")
    print(f"  Profit Factor: {profit_factor:.3f}")
    print(f"  Max Drawdown (in R units): {max_drawdown_r:.3f}R")

    return {
        "scenario": scenario_name, "trades": num_trades, "win_rate": win_rate,
        "avg_r": avg_r, "profit_factor": profit_factor, "max_dd_r": max_drawdown_r,
        "trade_frequency": trade_frequency
    }

# --- Main Orchestration ---
def get_processed_data_for_symbol(symbol_ticker, kline_interval, start_date, end_date):
    """
    Fetches, preprocesses, and engineers features for a single symbol.
    Returns a processed DataFrame, pivot feature names, and entry feature names.
    """
    print(f"\n--- Initial Data Processing for Symbol: {symbol_ticker} ---")
    # 1. Data Collection
    print(f"Fetching historical data for {symbol_ticker}...")
    try:
        historical_df = get_historical_bars(symbol_ticker, kline_interval, start_date, end_date)
        if historical_df.empty:
            print(f"No data fetched for {symbol_ticker} (empty DataFrame). Skipping.")
            return None, None, None
    except BinanceAPIException as e:
        print(f"Binance API Exception for {symbol_ticker}: {e}. Skipping symbol.")
        return None, None, None
    except BinanceRequestException as e:
        print(f"Binance Request Exception for {symbol_ticker}: {e}. Skipping symbol.")
        return None, None, None
    except Exception as e: # Catch any other unforeseen errors during data fetching
        print(f"An unexpected error occurred fetching data for {symbol_ticker}: {e}. Skipping symbol.")
        import traceback
        traceback.print_exc()
        return None, None, None

    print(f"Data fetched for {symbol_ticker}: {len(historical_df)} bars")
    if 'timestamp' not in historical_df.columns:
         historical_df.reset_index(inplace=True)
    if 'timestamp' not in historical_df.columns and 'index' in historical_df.columns and pd.api.types.is_datetime64_any_dtype(historical_df['index']):
        historical_df.rename(columns={'index':'timestamp'}, inplace=True)
    
    historical_df['symbol'] = symbol_ticker # Add symbol identifier

    # 2. Preprocessing & Labeling
    historical_df = calculate_atr(historical_df, period=ATR_PERIOD) # Ensure ATR_PERIOD is used
    atr_col_name_dynamic = f'atr_{ATR_PERIOD}'
    historical_df = generate_candidate_pivots(historical_df)
    historical_df = prune_and_label_pivots(historical_df, atr_col_name=atr_col_name_dynamic)
    historical_df = simulate_fib_entries(historical_df, atr_col_name=atr_col_name_dynamic)
    
    if 'timestamp' not in historical_df.columns:
        if pd.api.types.is_datetime64_any_dtype(historical_df.index):
            historical_df.reset_index(inplace=True)
            if 'index' in historical_df.columns and 'timestamp' not in historical_df.columns:
                 historical_df.rename(columns={'index':'timestamp'}, inplace=True)

    historical_df.dropna(subset=[f'atr_{ATR_PERIOD}'], inplace=True)
    historical_df.reset_index(drop=True, inplace=True)

    # 3. Feature Engineering
    # atr_col_name_dynamic was defined above
    historical_df, pivot_feature_names = engineer_pivot_features(historical_df, atr_col_name=atr_col_name_dynamic)
    historical_df, entry_feature_names_base = engineer_entry_features(historical_df, atr_col_name=atr_col_name_dynamic, entry_features_base_list_arg=None)

    historical_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    min_required_data_for_features = 30
    if len(historical_df) < min_required_data_for_features:
        print(f"Not enough data after initial processing for {symbol_ticker} ({len(historical_df)} rows). Skipping.")
        return None, None, None

    df_processed = historical_df.iloc[min_required_data_for_features:].copy()
    
    if 'timestamp' not in df_processed.columns:
        print(f"CRITICAL: 'timestamp' column lost for {symbol_ticker} during processing.")
        return None, None, None # Cannot proceed without timestamp

    df_processed.dropna(subset=pivot_feature_names, inplace=True) # Ensure core features are present
    df_processed.reset_index(drop=True, inplace=True)

    if len(df_processed) < 100: # Minimum data for meaningful processing/splitting
        print(f"Not enough final processed data for {symbol_ticker} ({len(df_processed)} rows). Skipping.")
        return None, None, None
        
    print(f"Initial data processing complete for {symbol_ticker}: {len(df_processed)} rows")
    return df_processed, pivot_feature_names, entry_feature_names_base


if __name__ == '__main__':
    symbols_df = pd.read_csv("symbols.csv")
    KLINE_INTERVAL = Client.KLINE_INTERVAL_15MINUTE
    START_DATE = "1 Jan, 2023" # Adjust as needed
    END_DATE = "1 May, 2023"   # Adjust as needed, ensure enough data

    all_symbols_train_data_list = []
    all_symbols_test_data_map = {} # Store test data per symbol
    processed_pivot_feature_names = None
    processed_entry_feature_names_base = None

    for index, row in symbols_df.iterrows():
        symbol_ticker = row['symbol']
        processed_df, pivot_feats, entry_feats_base = get_processed_data_for_symbol(
            symbol_ticker, KLINE_INTERVAL, START_DATE, END_DATE
        )

        if processed_df is not None and not processed_df.empty:
            if processed_pivot_feature_names is None: # Store feature names from the first successful processing
                processed_pivot_feature_names = pivot_feats
                processed_entry_feature_names_base = entry_feats_base

            # Split data for this symbol: 85% train (for universal model), 15% test (for symbol-specific backtest)
            # Ensure 'timestamp' is available for splitting if needed, or use frac.
            # Since data is time-series, chronological split is important.
            train_size = int(0.85 * len(processed_df))
            symbol_train_df = processed_df.iloc[:train_size].copy()
            symbol_test_df = processed_df.iloc[train_size:].copy()

            if not symbol_train_df.empty:
                all_symbols_train_data_list.append(symbol_train_df)
            if not symbol_test_df.empty:
                all_symbols_test_data_map[symbol_ticker] = symbol_test_df
        else:
            print(f"No processed data for {symbol_ticker}, skipping.")

    if not all_symbols_train_data_list:
        print("No training data collected from any symbol. Exiting.")
        exit()

    # Combine all training data (these are initially processed with default params)
    universal_df_initial_processed = pd.concat(all_symbols_train_data_list, ignore_index=True)
    universal_df_initial_processed.reset_index(drop=True, inplace=True)

    if len(universal_df_initial_processed) < 200:
        print(f"Combined initial training data is too small ({len(universal_df_initial_processed)} rows). Exiting.")
        exit()
    
    # The processed_pivot_feature_names and processed_entry_feature_names_base from initial default processing
    # are no longer strictly needed for Optuna or final model training's feature lists, as those are
    # dynamically generated or determined by process_dataframe_with_params using best_hyperparams.
    # The variable static_base_entry_features_for_final is removed.

    universal_train_df = universal_df_initial_processed.copy() # Define universal_train_df

    # 4. Optuna Hyperparameter Tuning for Universal Model
    print("Running Optuna for universal model...")
    try:
        best_hyperparams = run_optuna_tuning(
            universal_df_initial_processed.copy(), 
            static_entry_features_base_list=None, # Pass None, objective_optuna will handle it
            n_trials=20 
        )
        print("Best Universal Hyperparameters from Optuna:", best_hyperparams)
    except Exception as e:
        print(f"Universal Optuna tuning failed: {e}. Using default parameters.")
        import traceback
        traceback.print_exc()
        best_hyperparams = {
            'pivot_model_type': 'lgbm', 'pivot_num_leaves': 30, 'pivot_learning_rate': 0.05,
            'pivot_max_depth': 7, 'entry_model_type': 'lgbm', 'entry_num_leaves': 30,
            'entry_learning_rate': 0.05, 'entry_max_depth': 7,
            'p_swing_threshold': 0.6, 'profit_threshold': 0.6
        }

    # 5. Train Universal Final Models
    # For final models, usually train on the whole available dataset (here, universal_train_df)
    # or a large portion if a final validation set is held out from it.
    # The objective_optuna already does a train/val split internally for evaluation during tuning.
    # So, we can use the full universal_train_df for training the final model.
    print("Training universal final models...")

    # Re-process the universal_train_df with the best hyperparameters
    print(f"Re-processing universal training data with best_hyperparams: {best_hyperparams}")
    df_final_processed_for_training, final_pivot_features, final_entry_features_base = process_dataframe_with_params(
        universal_train_df.copy(), # Pass the initially processed universal data
        best_hyperparams,
        static_entry_features_base_list_arg=processed_entry_feature_names_base # Use base features from initial processing
    )

    if df_final_processed_for_training is None or df_final_processed_for_training.empty:
        print("CRITICAL: Failed to re-process universal training data with best_hyperparams or data became empty. Exiting.")
        exit()
    
    print(f"Universal training data re-processed: {len(df_final_processed_for_training)} rows. Using {len(final_pivot_features)} pivot features and {len(final_entry_features_base)} base entry features.")


    # --- Universal Pivot Model ---
    # Use df_final_processed_for_training and final_pivot_features
    X_p_universal_train = df_final_processed_for_training[final_pivot_features].fillna(-1)
    y_p_universal_train = df_final_processed_for_training['pivot_label']

    if best_hyperparams['pivot_model_type'] == 'lgbm':
        universal_pivot_model = lgb.LGBMClassifier(
            num_leaves=best_hyperparams['pivot_num_leaves'],
            learning_rate=best_hyperparams['pivot_learning_rate'],
            max_depth=best_hyperparams['pivot_max_depth'],
            class_weight='balanced', random_state=42, n_estimators=150
        )
    else: # rf
        universal_pivot_model = RandomForestClassifier(
            n_estimators=150, max_depth=best_hyperparams['pivot_max_depth'],
            class_weight='balanced', random_state=42
        )
    universal_pivot_model.fit(X_p_universal_train, y_p_universal_train)
    print("Universal Pivot Model trained.")
    save_model(universal_pivot_model, "pivot_detector_model.joblib")

    # --- Universal Entry Model ---
    # Use the final processed DataFrame for P_swing calculation and candidate selection
    p_swing_universal_train_all_classes = universal_pivot_model.predict_proba(X_p_universal_train) # X_p_universal_train is from df_final_processed_for_training
    df_final_processed_for_training['P_swing'] = np.max(p_swing_universal_train_all_classes[:,1:], axis=1)

    entry_universal_train_candidates = df_final_processed_for_training[
        (df_final_processed_for_training['pivot_label'].isin([1, 2])) &
        (df_final_processed_for_training['trade_outcome'] != -1) &
        (df_final_processed_for_training['P_swing'] >= best_hyperparams['p_swing_threshold'])
    ].copy()

    universal_entry_model = None
    if len(entry_universal_train_candidates) < 50:
        print("Not enough candidates for universal entry model training. Skipping entry model.")
    else:
        # ATR column name based on best_hyperparams
        final_model_atr_period = best_hyperparams.get('atr_period_opt', ATR_PERIOD)
        atr_col_name_final_model_train = f'atr_{final_model_atr_period}'
        print(f"Universal Entry Model training features using ATR column: {atr_col_name_final_model_train}")

        # Ensure the ATR column exists in entry_universal_train_candidates (it should, from df_final_processed_for_training)
        if atr_col_name_final_model_train not in entry_universal_train_candidates.columns:
            print(f"ERROR: ATR column {atr_col_name_final_model_train} missing in entry_universal_train_candidates. Skipping entry model.")
        else:
            entry_universal_train_candidates['norm_dist_entry_pivot'] = (entry_universal_train_candidates['entry_price_sim'] - entry_universal_train_candidates.apply(lambda r: r['low'] if r['is_swing_low'] == 1 else r['high'], axis=1)) / entry_universal_train_candidates[atr_col_name_final_model_train]
            entry_universal_train_candidates['norm_dist_entry_sl'] = (entry_universal_train_candidates['entry_price_sim'] - entry_universal_train_candidates['sl_price_sim']).abs() / entry_universal_train_candidates[atr_col_name_final_model_train]
            
            # Use final_entry_features_base from the re-processing step
            current_final_full_entry_features = final_entry_features_base + ['P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl']
            X_e_universal_train = entry_universal_train_candidates[current_final_full_entry_features].fillna(-1)
            y_e_universal_train = (entry_universal_train_candidates['trade_outcome'] > 0).astype(int)

        if len(X_e_universal_train) > 0 and len(y_e_universal_train.unique()) > 1:
            if best_hyperparams['entry_model_type'] == 'lgbm':
                universal_entry_model = lgb.LGBMClassifier(
                    num_leaves=best_hyperparams['entry_num_leaves'],
                    learning_rate=best_hyperparams['entry_learning_rate'],
                    max_depth=best_hyperparams['entry_max_depth'],
                    class_weight='balanced', random_state=42, n_estimators=150
                )
            else: # rf
                universal_entry_model = RandomForestClassifier(
                    n_estimators=150, max_depth=best_hyperparams['entry_max_depth'],
                    class_weight='balanced', random_state=42
                )
            universal_entry_model.fit(X_e_universal_train, y_e_universal_train)
            print("Universal Entry Model trained.")
            save_model(universal_entry_model, "entry_evaluator_model.joblib")
        else:
            print("Not enough data or variance to train universal entry model.")
    
    # 6. Backtesting each symbol with Universal Models
    all_symbols_backtest_results = []
    print("\n--- Backtesting Symbols with Universal Models (using best_hyperparams) ---")
    for symbol_ticker, symbol_test_df_initial_processed in all_symbols_test_data_map.items():
        print(f"\nBacktesting for symbol: {symbol_ticker}")
        if symbol_test_df_initial_processed.empty:
            print(f"No initial test data for {symbol_ticker}, skipping backtest.")
            continue

        # Re-process this symbol's test data using best_hyperparams
        print(f"Re-processing test data for {symbol_ticker} with best_hyperparams...")
        symbol_test_df_final_processed, symbol_final_pivot_features, symbol_final_entry_features_base = process_dataframe_with_params(
            symbol_test_df_initial_processed.copy(), # Important: use the initial version for reprocessing
            best_hyperparams,
            static_entry_features_base_list_arg=processed_entry_feature_names_base # Corrected: Use defined variable
        )

        if symbol_test_df_final_processed is None or symbol_test_df_final_processed.empty:
            print(f"Failed to re-process test data for {symbol_ticker} or data became empty. Skipping backtest.")
            continue
        
        # Determine the ATR column name used for this backtest (from best_hyperparams)
        backtest_atr_period_for_symbol = best_hyperparams.get('atr_period_opt', ATR_PERIOD)
        atr_col_name_for_symbol_backtest = f'atr_{backtest_atr_period_for_symbol}'
        print(f"Symbol {symbol_ticker} backtest using ATR column: {atr_col_name_for_symbol_backtest} and re-processed data.")

        symbol_backtest_summary = []
        # Scenario 1: Rule-Based Baseline (on re-processed test data)
        baseline_res = run_backtest_scenario(
            scenario_name="Rule-Based Baseline", 
            df_processed=symbol_test_df_final_processed.copy(), # Use re-processed data
            pivot_model=None, entry_model=None, best_params=best_hyperparams, 
            pivot_features=symbol_final_pivot_features, # Use features from re-processing
            entry_features_base=symbol_final_entry_features_base, # Use features from re-processing
            atr_col_name=atr_col_name_for_symbol_backtest,
            use_full_df_as_test=True 
        )
        if baseline_res: symbol_backtest_summary.append(baseline_res)

        # Scenario 2: Stage 1 ML Only (Universal Pivot Filter)
        if universal_pivot_model:
            stage1_res = run_backtest_scenario(
                scenario_name="ML Stage 1 (Pivot Filter)", 
                df_processed=symbol_test_df_final_processed.copy(), # Use re-processed data
                pivot_model=universal_pivot_model, entry_model=None, best_params=best_hyperparams,
                pivot_features=symbol_final_pivot_features, # Use features from re-processing
                entry_features_base=symbol_final_entry_features_base, # Use features from re-processing
                atr_col_name=atr_col_name_for_symbol_backtest,
                use_full_df_as_test=True
            )
            if stage1_res: symbol_backtest_summary.append(stage1_res)
        
        # Scenario 3: Full ML Pipeline (Universal Models)
        if universal_pivot_model and universal_entry_model:
            full_ml_res = run_backtest_scenario(
                scenario_name="Full ML Pipeline", 
                df_processed=symbol_test_df_final_processed.copy(), # Use re-processed data
                pivot_model=universal_pivot_model, entry_model=universal_entry_model, best_params=best_hyperparams,
                pivot_features=symbol_final_pivot_features, # Use features from re-processing
                entry_features_base=symbol_final_entry_features_base, # Use features from re-processing
                atr_col_name=atr_col_name_for_symbol_backtest,
                use_full_df_as_test=True
            )
            if full_ml_res: symbol_backtest_summary.append(full_ml_res)

        for res_dict in symbol_backtest_summary:
            combined_data = {'symbol': symbol_ticker, **res_dict, **best_hyperparams}
            all_symbols_backtest_results.append(combined_data)

    if all_symbols_backtest_results:
        results_df = pd.DataFrame(all_symbols_backtest_results)
        cols_ordered = ['symbol', 'scenario', 'trades', 'win_rate', 'avg_r', 'profit_factor', 'max_dd_r', 'trade_frequency']
        hyperparam_cols = [col for col in results_df.columns if col not in cols_ordered]
        final_cols = cols_ordered + sorted(list(set(hyperparam_cols)))
        final_cols_existing = [col for col in final_cols if col in results_df.columns]
        results_df = results_df[final_cols_existing]
        results_df.to_csv("universal_model_backtest_summary.csv", index=False)
        print("\nConsolidated universal model backtest summary saved to universal_model_backtest_summary.csv")
    else:
        print("\nNo universal model backtest results were aggregated to save to CSV.")

    print("\nAll symbols processed with universal models. app.py script finished.")
