import math
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

import json # For JSON operations

# --- Global App Settings ---
# This will be populated by load_app_settings
app_settings = {}
APP_CONFIG_FILE = "app_settings.json" # Changed from APP_TRADE_CONFIG_FILE and to JSON

# --- Global App Trading Configs ---
# This will be populated by load_app_trading_configs, likely from app_settings
app_trading_configs = {}
# No separate file, will draw from app_settings

# --- Global App Active Trades ---
# Stores details of trades initiated and managed by app.py
app_active_trades = {}
app_active_trades_lock = threading.Lock()

# --- Global Fibonacci Pre-Order Proposals ---
# Stores details of proposed Fibonacci limit orders before they meet secondary conditions.
# Key: symbol (str), Value: dict of proposal details
app_fib_proposals = {}
app_fib_proposals_lock = threading.Lock()

# --- Global ML Models & Parameters ---
# These will be populated by training or loading from disk.
universal_pivot_model = None
universal_entry_model = None
best_hyperparams = {}

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
    global app_binance_client # Ensure we are modifying the global client instance
    # global app_trading_configs # app_trading_configs is already global, typically accessed directly if needed for alerts
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
    if not klines: # Check if klines list is empty
        print(f"Warning (get_historical_bars): No klines returned from API for {symbol} {interval} {start_str} {end_str}.")
        # Return DataFrame with expected columns but no data
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
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


def get_or_download_historical_data(symbol: str, interval: str, 
                                    start_date_str: str, end_date_str: str = None,
                                    data_directory: str = "historical_data", 
                                    force_redownload_all: bool = False) -> pd.DataFrame:
    """
    Manages fetching historical kline data for a symbol.
    - Loads existing data from a Parquet file if available.
    - Downloads new data since the last record or full data if no existing file/force_redownload_all.
    - Appends new data to existing data and saves the combined data back to Parquet.

    Args:
        symbol (str): Trading symbol (e.g., "BTCUSDT").
        interval (str): Kline interval (e.g., Client.KLINE_INTERVAL_15MINUTE).
        start_date_str (str): The overall start date for data fetching if no existing data.
        end_date_str (str, optional): The overall end date. Defaults to None (fetch up to current).
        data_directory (str, optional): Directory to store/load Parquet files. Defaults to "historical_data".
        force_redownload_all (bool, optional): If True, ignore existing file and redownload all data.

    Returns:
        pd.DataFrame: DataFrame containing the historical klines.
    """
    os.makedirs(data_directory, exist_ok=True)
    # Sanitize interval string for filename (e.g. "15m" from "15minute")
    # This is a basic sanitization; might need adjustment based on actual interval string formats
    safe_interval_str = interval.replace(" ", "").lower()
    if "minute" in safe_interval_str: safe_interval_str = safe_interval_str.replace("minute", "m")
    elif "hour" in safe_interval_str: safe_interval_str = safe_interval_str.replace("hour", "h")
    elif "day" in safe_interval_str: safe_interval_str = safe_interval_str.replace("day", "d")
    
    file_path = os.path.join(data_directory, f"{symbol}_{safe_interval_str}_data.parquet")
    log_prefix_parquet = f"[ParquetIO-{symbol}]" # Logger prefix for this function

    existing_df = None
    last_timestamp_ms = None

    if os.path.exists(file_path) and not force_redownload_all:
        print(f"{log_prefix_parquet} Loading existing data for {symbol} from {file_path}...")
        try:
            existing_df = pd.read_parquet(file_path)
            if existing_df is not None and not existing_df.empty and 'timestamp' in existing_df.columns:
                print(f"{log_prefix_parquet} Existing data loaded. Shape: {existing_df.shape}")
                # Ensure timestamp is datetime and sorted
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                existing_df.sort_values('timestamp', inplace=True)
                existing_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True) # Deduplicate just in case
                
                # Get the last timestamp in milliseconds for Binance API
                last_timestamp_dt = existing_df['timestamp'].iloc[-1]
                last_timestamp_ms = int(last_timestamp_dt.timestamp() * 1000)
                print(f"{log_prefix_parquet} Last record in existing data: {last_timestamp_dt}")
                
                # New start date for download is 1 millisecond after the last record's timestamp
                effective_start_str_for_new_data = str(last_timestamp_ms + 1) # Binance uses ms timestamp string
            elif existing_df is not None: # Loaded but empty or no timestamp
                print(f"{log_prefix_parquet} Existing file {file_path} is empty or missing 'timestamp' column. Will perform a full download.")
                existing_df = None # Treat as no existing data
            else: # existing_df is None from read_parquet
                 print(f"{log_prefix_parquet} read_parquet returned None for {file_path}. Will perform a full download.")
                 existing_df = None # Ensure it's None
        except Exception as e:
            print(f"{log_prefix_parquet} ERROR: Error loading or processing existing Parquet file {file_path}: {e}. Will attempt full redownload.")
            existing_df = None
            # Optionally, delete or move the corrupted file here
            # os.remove(file_path) 
    else:
        print(f"{log_prefix_parquet} No existing data file found at {file_path} (or force_redownload_all=True). Performing full download.")

    if existing_df is None or force_redownload_all: # Full download needed
        print(f"{log_prefix_parquet} Fetching full historical data for {symbol} from {start_date_str} to {end_date_str or 'latest'}...")
        global app_binance_client
        if app_binance_client is None:
            print(f"{log_prefix_parquet} Warning: app_binance_client not initialized. Attempting default init.")
            if not initialize_app_binance_client(): # Default env
                 print(f"{log_prefix_parquet} CRITICAL: Failed to initialize Binance client. Cannot download.")
                 return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']) # Return empty with headers
        
        downloaded_df = get_historical_bars(symbol, interval, start_date_str, end_str=end_date_str)
        if downloaded_df is None or downloaded_df.empty:
            print(f"{log_prefix_parquet} No data downloaded for {symbol} (full download attempt). Returning empty DataFrame with headers.")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        print(f"{log_prefix_parquet} Downloaded full data. Shape: {downloaded_df.shape}. Sorting and saving...")
        downloaded_df.sort_values('timestamp', inplace=True)
        downloaded_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        
        try:
            print(f"{log_prefix_parquet} Attempting to write full data to Parquet. Shape: {downloaded_df.shape}. Path: {file_path}")
            downloaded_df.to_parquet(file_path, index=False)
            print(f"{log_prefix_parquet} INFO: Wrote full data: {file_path}")
        except Exception as e:
            print(f"{log_prefix_parquet} ERROR: Failed writing full data to Parquet {file_path}: {e}")
        return downloaded_df

    else: # We have existing_df, try to download new data
        if last_timestamp_ms is None: 
            print(f"{log_prefix_parquet} Error: last_timestamp_ms is None despite having existing_df. Aborting update.")
            return existing_df 

        print(f"{log_prefix_parquet} Fetching new data for {symbol} since {pd.to_datetime(last_timestamp_ms, unit='ms')} (exclusive)...")
        
        if app_binance_client is None:
            print(f"{log_prefix_parquet} Warning (update): app_binance_client not initialized. Attempting default init.")
            if not initialize_app_binance_client():
                 print(f"{log_prefix_parquet} CRITICAL: Failed to initialize Binance client (update). Cannot download new data.")
                 return existing_df 
        
        new_data_df = get_historical_bars(symbol, interval, 
                                          start_str=str(last_timestamp_ms + 1), 
                                          end_str=end_date_str)

        if new_data_df is not None and not new_data_df.empty:
            print(f"{log_prefix_parquet} Downloaded {len(new_data_df)} new bars for {symbol}. Shape: {new_data_df.shape}")
            new_data_df.sort_values('timestamp', inplace=True)
            new_data_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)

            combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            print(f"{log_prefix_parquet} Combined old and new data. Shape before final sort/dedup: {combined_df.shape}")
            combined_df.sort_values('timestamp', inplace=True)
            combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            print(f"{log_prefix_parquet} Shape after final sort/dedup: {combined_df.shape}")
            
            try:
                print(f"{log_prefix_parquet} Attempting to write appended data to Parquet. Shape: {combined_df.shape}. Path: {file_path}")
                combined_df.to_parquet(file_path, index=False)
                print(f"{log_prefix_parquet} INFO: Wrote appended data: {file_path}")
            except Exception as e:
                print(f"{log_prefix_parquet} ERROR: Failed writing appended data to Parquet {file_path}: {e}")
            
            overall_start_dt = pd.to_datetime(start_date_str)
            combined_df = combined_df[combined_df['timestamp'] >= overall_start_dt]
            return combined_df
        else:
            print(f"No new data found for {symbol} since last record.")
            # Filter existing_df to respect original overall start_date_str
            overall_start_dt = pd.to_datetime(start_date_str)
            existing_df = existing_df[existing_df['timestamp'] >= overall_start_dt]
            return existing_df

# --- Configuration Management ---
def get_user_input(prompt, default_value, value_type=str, choices=None):
    """Helper function to get validated user input."""
    while True:
        user_val_str = input(f"{prompt} (default: {default_value}): ").strip()
        if not user_val_str:
            user_val_str = str(default_value) # Use default if empty

        try:
            if value_type == bool:
                if user_val_str.lower() in ['true', 't', 'yes', 'y', '1']:
                    converted_value = True
                elif user_val_str.lower() in ['false', 'f', 'no', 'n', '0']:
                    converted_value = False
                else:
                    raise ValueError("Invalid boolean value.")
            else:
                converted_value = value_type(user_val_str)

            if choices and converted_value not in choices:
                raise ValueError(f"Invalid choice. Must be one of {choices}.")
            
            # Specific range checks can be added here if needed, e.g., for risk_percent
            if prompt.startswith("Risk percent per trade") and not (0 < converted_value <= 1.0): # Assuming input is decimal
                 raise ValueError("Risk percent must be between 0.0 (exclusive) and 1.0 (inclusive). E.g., 0.01 for 1%.")
            if prompt.startswith("Leverage") and not (1 <= converted_value <= 125):
                 raise ValueError("Leverage must be between 1 and 125.")

            return converted_value
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")


def load_app_settings(filepath=APP_CONFIG_FILE):
    """
    Loads app settings from a JSON file.
    Uses defaults and prompts user if the file or specific settings are not found/incomplete.
    Populates the global `app_settings` dictionary.
    """
    global app_settings # Use the renamed global dict

    defaults = {
        "app_trading_environment": "mainnet",
        "app_operational_mode": "signal", # 'live', 'signal', or 'train_only'
        "app_risk_percent": 0.01,
        "app_leverage": 20,
        "app_allow_exceed_risk_for_min_notional": False,
        "app_telegram_bot_token": None,
        "app_telegram_chat_id": None,
        "app_tp1_qty_pct": 0.25,
        "app_tp2_qty_pct": 0.50,
        "app_pivot_model_path": "pivot_detector_model.joblib",
        "app_entry_model_path": "entry_evaluator_model.joblib",
        "app_model_params_path": "best_model_params.json",
        "app_auto_start_trading_after_train": False,
        "app_force_retrain_on_startup": False,
        # Add other ML training specific defaults if needed (e.g., Optuna trials)
        "app_optuna_trials": 20, # Default Optuna trials for app.py training
        "app_use_dynamic_threshold": True, # New: Toggle for dynamic/fixed pivot threshold
        "app_fixed_pivot_threshold": 0.7,  # New: Fixed pivot threshold value
        
        # New settings for the trading/signal loop
        "app_trading_symbols": "BTCUSDT", # Comma-separated list of symbols
        "app_trading_kline_interval": Client.KLINE_INTERVAL_15MINUTE, # Kline interval for live processing
        "app_scan_interval_seconds": 60,    # How often the main loop runs
        "app_delay_between_symbols_seconds": 2, # Delay between processing multiple symbols in a cycle
        "app_sim_sl_atr_multiplier": 1.0,   # ATR multiplier for SL simulation for entry feature calculation
        "app_sl_atr_multiplier": 2.0,       # ATR multiplier for actual trade SL
        "app_tp_rr_ratio": 1.5,             # R:R ratio for actual trade TP
        "app_symbols_csv_path": "app_symbols.csv", # Default path for symbols CSV
        "app_training_symbols_source_type": "list", # "list" or "csv"
        "app_training_symbols_list_str": "BTCUSDT,ETHUSDT", # Comma-separated list if source_type is "list"
        "app_notify_on_trade": True, # New: Toggle for trade/signal execution messages
        "app_min_volume_spike_ratio": 1.5, # New setting for Volume-Based Trend-Strength Filter
        "app_atr_period_sl_tp": 14, # ATR period for SL/TP lifecycle management (also used for RSI period for now)
        "app_tp3_atr_multiplier": 2.0, # ATR multiplier for floating TP3
        "app_breakeven_buffer_pct": 0.001, # 0.1% buffer for breakeven SL
        "app_max_trade_duration_hours": 24, # Max hours a trade can remain open
        "app_rsi_max_long": 70.0, # Max RSI for LONG entry
        "app_rsi_min_long": 0.0,  # Min RSI for LONG entry (0 effectively means no lower bound)
        "app_rsi_min_short": 30.0, # Min RSI for SHORT entry
        "app_rsi_max_short": 100.0, # Max RSI for SHORT entry (100 effectively means no upper bound)
        "app_reject_notify_telegram": True, # Toggle for sending Telegram notifications for pre-check rejections
        "use_fib_preorder": False, # Default for the new Fibonacci pre-order logic
        "fib_proposal_validity_minutes": 15, # Default validity for Fib proposals
        "fib_leg_lookback_candles": 50, # Default lookback for finding the other end of Fib leg
        "app_simulate_limit_orders": False # Default for dry-run/paper trading mode for Fib limits
    }

    loaded_json_settings = {}
    config_file_exists = os.path.exists(filepath)

    if config_file_exists:
        try:
            with open(filepath, 'r') as f:
                loaded_json_settings = json.load(f)
            print(f"app.py: Settings loaded from '{filepath}'.")
        except json.JSONDecodeError:
            print(f"app.py: Error decoding JSON from '{filepath}'. File might be corrupted. Using defaults and prompting.")
            # config_file_exists = False # Treat as if file doesn't exist for prompting
        except Exception as e:
            print(f"app.py: Error loading settings from '{filepath}': {e}. Using defaults and prompting.")
            # config_file_exists = False
    else:
        print(f"app.py: Settings file '{filepath}' not found. Using default settings and prompting for essentials.")

    # Start with defaults, then override with loaded JSON settings
    current_settings = defaults.copy()
    current_settings.update(loaded_json_settings) # Loaded values override defaults

    # --- Conditional prompting for essential/missing settings ---

    # Environment
    app_trading_env_valid = False
    if "app_trading_environment" in loaded_json_settings:
        env_val = loaded_json_settings["app_trading_environment"]
        if env_val in ['mainnet', 'testnet']:
            current_settings["app_trading_environment"] = env_val
            app_trading_env_valid = True
    if not app_trading_env_valid:
        print("app.py: 'app_trading_environment' missing or invalid in settings file. Prompting user.")
        current_settings["app_trading_environment"] = get_user_input(
            "Trading environment ('mainnet' or 'testnet')",
            current_settings.get("app_trading_environment", defaults["app_trading_environment"]),
            str, choices=['mainnet', 'testnet']
        )

    # Risk Percent
    app_risk_percent_valid = False
    if "app_risk_percent" in loaded_json_settings:
        try:
            risk_val = float(loaded_json_settings["app_risk_percent"])
            if 0 < risk_val <= 1.0: # Validation: e.g., 0.01 for 1%
                current_settings["app_risk_percent"] = risk_val
                app_risk_percent_valid = True
        except ValueError:
            pass # Invalid float, will prompt
    if not app_risk_percent_valid:
        print("app.py: 'app_risk_percent' missing or invalid in settings file. Prompting user.")
        current_settings["app_risk_percent"] = get_user_input(
            "Risk percent per trade (e.g., 0.01 for 1%)",
            current_settings.get("app_risk_percent", defaults["app_risk_percent"]),
            float
        )
        # Re-validate after input if get_user_input doesn't do it sufficiently
        while not (0 < current_settings["app_risk_percent"] <= 1.0):
            print("Invalid input: Risk percent must be between 0.0 (exclusive) and 1.0 (inclusive).")
            current_settings["app_risk_percent"] = get_user_input(
                "Risk percent per trade (e.g., 0.01 for 1%)",
                defaults["app_risk_percent"], # Offer default again
                float
            )

    # Leverage
    app_leverage_valid = False
    if "app_leverage" in loaded_json_settings:
        try:
            lev_val = int(loaded_json_settings["app_leverage"])
            if 1 <= lev_val <= 125: # Validation
                current_settings["app_leverage"] = lev_val
                app_leverage_valid = True
        except ValueError:
            pass # Invalid int, will prompt
    if not app_leverage_valid:
        print("app.py: 'app_leverage' missing or invalid in settings file. Prompting user.")
        current_settings["app_leverage"] = get_user_input(
            "Leverage (e.g., 20)",
            current_settings.get("app_leverage", defaults["app_leverage"]),
            int
        )
        # Re-validate after input
        while not (1 <= current_settings["app_leverage"] <= 125):
            print("Invalid input: Leverage must be between 1 and 125.")
            current_settings["app_leverage"] = get_user_input(
                "Leverage (e.g., 20)",
                defaults["app_leverage"], # Offer default again
                int
            )

    # Force retrain on startup
    app_force_retrain_valid = False
    if "app_force_retrain_on_startup" in loaded_json_settings:
        retrain_val = loaded_json_settings["app_force_retrain_on_startup"]
        if isinstance(retrain_val, bool):
            current_settings["app_force_retrain_on_startup"] = retrain_val
            app_force_retrain_valid = True
    if not app_force_retrain_valid:
        print("app.py: 'app_force_retrain_on_startup' missing or invalid in settings file. Prompting user.")
        current_settings["app_force_retrain_on_startup"] = get_user_input(
            "Force retrain models on startup (true/false)?",
            current_settings.get("app_force_retrain_on_startup", defaults["app_force_retrain_on_startup"]),
            bool
        )

    # Dynamic/Fixed Threshold Mode
    app_use_dynamic_thresh_valid = False
    if "app_use_dynamic_threshold" in loaded_json_settings:
        use_dyn_val = loaded_json_settings["app_use_dynamic_threshold"]
        if isinstance(use_dyn_val, bool):
            current_settings["app_use_dynamic_threshold"] = use_dyn_val
            app_use_dynamic_thresh_valid = True
    if not app_use_dynamic_thresh_valid:
        print("app.py: 'app_use_dynamic_threshold' missing or invalid. Prompting.")
        current_settings["app_use_dynamic_threshold"] = get_user_input(
            "Use dynamic pivot threshold (true/false)?",
            current_settings.get("app_use_dynamic_threshold", defaults["app_use_dynamic_threshold"]),
            bool
        )

    # Fixed Pivot Threshold Value
    app_fixed_thresh_valid = False
    if "app_fixed_pivot_threshold" in loaded_json_settings:
        try:
            fixed_thresh_val = float(loaded_json_settings["app_fixed_pivot_threshold"])
            if 0.0 < fixed_thresh_val < 1.0: # Threshold should be a probability
                current_settings["app_fixed_pivot_threshold"] = fixed_thresh_val
                app_fixed_thresh_valid = True
        except ValueError:
            pass # Invalid float, will prompt
    if not app_fixed_thresh_valid:
        print("app.py: 'app_fixed_pivot_threshold' missing or invalid. Prompting.")
        current_settings["app_fixed_pivot_threshold"] = get_user_input(
            "Fixed pivot threshold value (e.g., 0.7)",
            current_settings.get("app_fixed_pivot_threshold", defaults["app_fixed_pivot_threshold"]),
            float
        )
        while not (0.0 < current_settings["app_fixed_pivot_threshold"] < 1.0):
            print("Invalid input: Fixed pivot threshold must be between 0.0 and 1.0 (exclusive).")
            current_settings["app_fixed_pivot_threshold"] = get_user_input(
                "Fixed pivot threshold value (e.g., 0.7)",
                defaults["app_fixed_pivot_threshold"],
                float
            )
        
    # Operational mode (this is often set interactively later in start_app_main_flow)
    # If missing or invalid in JSON, it will use the default. No interactive prompt here
    # as it's typically selected from a menu later if needed.
    if "app_operational_mode" not in loaded_json_settings or \
       loaded_json_settings.get("app_operational_mode") not in ['live', 'signal', 'train_only']:
        # Use default if missing or invalid from JSON
        current_settings["app_operational_mode"] = defaults["app_operational_mode"]
        if "app_operational_mode" not in loaded_json_settings:
             print(f"app.py: 'app_operational_mode' missing from settings file. Using default: {defaults['app_operational_mode']}")
        else: # Was present but invalid
             print(f"app.py: 'app_operational_mode' ('{loaded_json_settings.get('app_operational_mode')}') was invalid. Using default: {defaults['app_operational_mode']}")
    else: # Valid and present in JSON
        current_settings["app_operational_mode"] = loaded_json_settings["app_operational_mode"]


    # Populate global app_settings with the processed current_settings
    # This ensures app_settings reflects the file-loaded or prompted values.
    app_settings.update(current_settings)

    # Load API keys and Telegram details from keys.py
    # These are NOT saved in app_settings.json but are part of the runtime app_settings dict.
    # Use the now-set app_settings["app_trading_environment"]
    app_env_for_keys = app_settings.get("app_trading_environment", defaults["app_trading_environment"])
    api_k, api_s, tele_token, tele_chat_id = load_app_api_keys(env=app_env_for_keys)

    app_settings["api_key"] = api_k # Runtime only
    app_settings["api_secret"] = api_s # Runtime only

    # Update current_settings with Telegram details from keys.py ONLY if not already set by app_settings.json
    # This ensures that if Telegram details are in app_settings.json, they take precedence.
    # If they are missing from JSON (i.e., current_settings still has them as None from defaults),
    # then values from keys.py are used for both runtime (app_settings) and for saving (current_settings).
    if current_settings.get("app_telegram_bot_token") is None and tele_token:
        current_settings["app_telegram_bot_token"] = tele_token
        app_settings["app_telegram_bot_token"] = tele_token # Ensure runtime also gets it
    if current_settings.get("app_telegram_chat_id") is None and tele_chat_id:
        current_settings["app_telegram_chat_id"] = tele_chat_id
        app_settings["app_telegram_chat_id"] = tele_chat_id # Ensure runtime also gets it
    
    # At this point, current_settings should contain the full desired state to be saved (if changed).
    # And app_settings contains the full runtime state including API keys.

    # Ensure all default keys are present in current_settings before saving.
    # This handles cases where a new default is added but not yet in an old JSON file.
    for key, default_value in defaults.items():
        if key not in current_settings:
            current_settings[key] = default_value # Add missing keys with their defaults

    print("app.py: Final application settings applied (API keys hidden from print):",
          {k: v for k, v in app_settings.items() if k not in ['api_key', 'api_secret']})

    # Save the final state of current_settings (which includes prompted values,
    # validated JSON values, and defaults for anything missing) back to the file.
    save_app_settings(filepath, settings_dict_to_save=current_settings)

def load_app_trading_configs():
    """
    Populates the global `app_trading_configs` dictionary.
    It sources its values primarily from the already loaded `app_settings`.
    This function ensures that trading-specific configurations are available
    under `app_trading_configs` for functions that expect them there.
    """
    global app_trading_configs, app_settings

    if app_settings is None or not app_settings: # Explicit check for None or empty
        print("Warning (load_app_trading_configs): app_settings is empty or None. Attempting to load them first.")
        # Note: load_app_settings() might prompt user if file is missing.
        # This is acceptable as app_settings are fundamental.
        load_app_settings() 
        if app_settings is None or not app_settings: # Re-check after load attempt
            print("Error (load_app_trading_configs): Failed to load app_settings. Trading configs will be empty or defaults.")
            # Populate with some very basic defaults if app_settings loading failed entirely
            app_trading_configs = {
                "app_operational_mode": "signal",
                "app_risk_percent": 0.01,
                "app_leverage": 20,
                "app_allow_exceed_risk_for_min_notional": False,
                "app_tp1_qty_pct": 0.25,
                "app_tp2_qty_pct": 0.50,
                # Add any other essential trading defaults here if app_settings might fail
            }
            return

    # Default values that might not be in app_settings or need a specific default for trading context
    default_trading_specific_configs = {
        "app_allow_exceed_risk_for_min_notional": False, # Default for this specific trading config
        # Add other trading-specific defaults here if they are not typically in app_settings.json
    }

    # Start with trading-specific defaults
    temp_trading_configs = default_trading_specific_configs.copy()

    # Override with values from app_settings if they exist
    # These are keys expected to be in app_settings and relevant for app_trading_configs
    shared_keys_from_app_settings = [
        "app_operational_mode",
        "app_risk_percent",
        "app_leverage",
        "app_telegram_bot_token", # For sending messages from trade execution logic
        "app_telegram_chat_id",   # For sending messages from trade execution logic
        "app_tp1_qty_pct",
        "app_tp2_qty_pct",
        # "app_trading_environment" is also in app_settings and used by initialize_app_binance_client
        # No need to explicitly copy if functions using app_trading_configs can also access app_settings for it.
        # Or copy it if preferred for consolidation:
        "app_trading_environment",
    ]

    for key in shared_keys_from_app_settings:
        if key in app_settings:
            temp_trading_configs[key] = app_settings[key]
        elif key not in temp_trading_configs: # If not in app_settings and no default set yet
            print(f"Warning (load_app_trading_configs): Key '{key}' not found in app_settings and no default in trading_specific_configs. It will be missing from app_trading_configs.")
            # Optionally set a fallback default here if critical
            # e.g., if key == "app_operational_mode": temp_trading_configs[key] = "signal"

    # Ensure essential keys have fallbacks if still missing (though load_app_settings should have handled most)
    if "app_operational_mode" not in temp_trading_configs: temp_trading_configs["app_operational_mode"] = "signal"
    if "app_risk_percent" not in temp_trading_configs: temp_trading_configs["app_risk_percent"] = 0.01
    if "app_leverage" not in temp_trading_configs: temp_trading_configs["app_leverage"] = 20
    if "app_tp1_qty_pct" not in temp_trading_configs: temp_trading_configs["app_tp1_qty_pct"] = 0.25
    if "app_tp2_qty_pct" not in temp_trading_configs: temp_trading_configs["app_tp2_qty_pct"] = 0.50


    app_trading_configs.update(temp_trading_configs)
    print(f"app.py: Trading configurations populated in app_trading_configs (sourced from app_settings).")
    # No saving back to a separate file, as app_settings.json is the source of truth.

def save_app_settings(filepath=APP_CONFIG_FILE, settings_dict_to_save=None):
    """
    Saves the provided settings dictionary to a JSON file, excluding sensitive keys.
    If settings_dict_to_save is None, it defaults to global app_settings.
    """
    global app_settings # Still needed if settings_dict_to_save is None
    
    source_settings_for_saving = {}
    if settings_dict_to_save is not None:
        source_settings_for_saving = settings_dict_to_save.copy()
    else:
        source_settings_for_saving = app_settings.copy() # Fallback to global if no dict provided

    # Define keys that should definitely not be written to the JSON file
    keys_to_exclude_from_file = ['api_key', 'api_secret'] 
    
    # Create a clean dictionary for JSON output, including only non-sensitive keys
    # and ensuring all default keys are considered for completeness.
    # The 'defaults' dict should be accessible here or passed if this function is more generic.
    # For now, assuming 'defaults' is the global one defined in load_app_settings.
    # However, it's better if load_app_settings passes a complete current_settings.
    
    final_json_output_dict = {}
    # Populate with keys from the source_settings_for_saving, excluding sensitive ones
    for key, value in source_settings_for_saving.items():
        if key not in keys_to_exclude_from_file:
            final_json_output_dict[key] = value
            
    # Optional: Ensure all defined default keys are in the output,
    # even if they were not in source_settings_for_saving (e.g., if a minimal dict was passed).
    # This part might be redundant if settings_dict_to_save from load_app_settings is comprehensive.
    # global defaults # Assuming 'defaults' from load_app_settings scope is available if needed.
    # for default_key, default_value in defaults.items():
    #    if default_key not in final_json_output_dict and default_key not in keys_to_exclude_from_file:
    #        final_json_output_dict[default_key] = default_value
            
    try:
        with open(filepath, 'w') as f:
            json.dump(final_json_output_dict, f, indent=4)
        print(f"app.py: Settings saved to '{filepath}'.")
    except Exception as e:
        print(f"app.py: Error saving settings to '{filepath}': {e}")

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
        error_msg_detail = f"API Error: {e_api.status_code} {e_api.message} (Code: {e_api.code})"
        error_msg_log = f"app.py place_order: {error_msg_detail} for {symbol} {side} {quantity} {order_type}"
        print(error_msg_log)
        # Send Telegram notification for trade rejection
        app_send_trade_rejection_notification(
            current_app_settings=app_trading_configs, # app_trading_configs should have telegram settings
            symbol=symbol,
            signal_type=f"{side} {order_type}", # Construct a signal type string
            reason=error_msg_detail,
            entry_price=float(params.get("price")) if params.get("price") else None,
            sl_price=float(params.get("stopPrice")) if params.get("stopPrice") else None, # Assuming stopPrice could be SL for some orders
            tp_price=None, # TP is not directly part of this order placement
            quantity=quantity,
            symbol_info=symbol_info_app
        )
        return None, str(e_api) # Return original full error string as before
    except Exception as e_gen:
        error_msg_detail = f"General Error: {str(e_gen)}"
        error_msg_log = f"app.py place_order: {error_msg_detail} for {symbol} {side} {quantity} {order_type}"
        print(error_msg_log)
        # Send Telegram notification for trade rejection
        app_send_trade_rejection_notification(
            current_app_settings=app_trading_configs,
            symbol=symbol,
            signal_type=f"{side} {order_type}",
            reason=error_msg_detail,
            entry_price=float(params.get("price")) if params.get("price") else None,
            sl_price=float(params.get("stopPrice")) if params.get("stopPrice") else None,
            tp_price=None,
            quantity=quantity,
            symbol_info=symbol_info_app
        )
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
    # Enhanced log message for trade signal trigger
    print(f"{log_prefix} ✅ Trade Signal Triggered & Processing: Side={side}, Symbol={symbol}, OrderType={order_type}, EntryTarget={entry_price_target}, SL={sl_price}, TP1={tp1_price}, TP2={tp2_price}, TP3={tp3_price}")

    if app_binance_client is None:
        print(f"{log_prefix} Binance client not initialized. Attempting to initialize...")
        if not initialize_app_binance_client(env=app_trading_configs.get("app_trading_environment", "mainnet")):
            print(f"{log_prefix} Critical: Failed to initialize Binance client. Cannot execute trade.")
            return False
        print(f"{log_prefix} Binance client initialized successfully.")

    # 1. Get Symbol Info
    symbol_info_app = get_app_symbol_info(symbol)
    if symbol_info_app is None: # Explicit check for None
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
            if ticker is None or 'lastPrice' not in ticker: # Check ticker and lastPrice
                print(f"{log_prefix} Could not fetch valid ticker or lastPrice for MARKET order sizing. Ticker: {ticker}")
                return False
            effective_entry_price_for_sizing = float(ticker['lastPrice'])
            print(f"{log_prefix} Using current market price for MARKET order sizing: {effective_entry_price_for_sizing:.{p_prec}f}")
        except Exception as e:
            print(f"{log_prefix} Could not fetch current market price for MARKET order sizing: {e}. Cannot execute.")
            return False
    elif order_type.upper() == "LIMIT" and entry_price_target is None:
        print(f"{log_prefix} Entry price target required for LIMIT order. Cannot execute.")
        return False
        
    if effective_entry_price_for_sizing is None: 
        print(f"{log_prefix} Effective entry price for sizing could not be determined. Cannot execute.")
        return False

    # 4. Calculate Position Size
    # Ensure app_trading_configs is loaded and available
    if app_trading_configs is None or not app_trading_configs: # Explicit check for None or empty
        print(f"{log_prefix} App trading configurations not loaded or empty. Attempting to load defaults...")
        load_app_trading_configs() # Load with defaults if not already loaded
        if app_trading_configs is None or not app_trading_configs: # Still not loaded or empty
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

    # --- Handle Signal Mode ---
    if current_operational_mode == "signal":
        print(f"{log_prefix} Signal Mode: Generating Telegram notification for {order_type} signal.")
        send_app_trade_execution_telegram(
            current_app_settings=app_trading_configs, # Pass app_trading_configs which has telegram settings
            symbol=symbol,
            side=side,
            entry_price=effective_entry_price_for_sizing, # For market, this is current price; for limit, it's target
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp3_price=tp3_price,
            order_type=order_type.upper(),
            executed_quantity=quantity, # Pass calculated quantity for info
            mode="signal"
        )
        # In signal mode, after sending the message, we don't proceed to place orders.
        append_decision_log({
            "symbol": symbol, "event_type": "SIGNAL_MODE_NOTIFICATION_SENT", "side": side, 
            "order_type": order_type.upper(), "entry_target": effective_entry_price_for_sizing, 
            "sl": sl_price, "tp1": tp1_price, "quantity_calculated": quantity
        })
        return True # Indicate signal processed successfully


    # --- Live Mode: Place Orders ---
    print(f"{log_prefix} Live Mode: Proceeding to place orders.")
    append_decision_log({
        "symbol": symbol, "event_type": "LIVE_MODE_PLACE_ENTRY_ATTEMPT", "side": side, 
        "order_type": order_type.upper(), "quantity": quantity, 
        "entry_target": entry_price_target if order_type.upper() == "LIMIT" else effective_entry_price_for_sizing,
        "sl": sl_price, "tp1": tp1_price
    })
    entry_order_result, error_msg = place_app_new_order(
        symbol_info_app, api_order_side, order_type.upper(), quantity,
        price=entry_price_target if order_type.upper() == "LIMIT" else None,
        position_side=api_position_side
    )

    if entry_order_result:
        entry_order_id = entry_order_result.get('orderId')
        append_decision_log({
            "symbol": symbol, "event_type": "LIVE_ENTRY_ORDER_PLACED", "order_id": entry_order_id,
            "side": side, "order_type": order_type.upper(), "status": entry_order_result.get('status')
        })
        actual_filled_price = effective_entry_price_for_sizing # Default for LIMIT or if avgPrice not available
        total_filled_quantity = quantity # Assume full quantity filled for now, will be updated for MARKET fills

        # Send Telegram notification for successful order placement (or attempt)
        # For MARKET, actual_filled_price and total_filled_quantity will be updated if FILLED
        # For LIMIT, this sends a "Limit Order Placed" message.

        if order_type.upper() == "MARKET" and entry_order_result.get('status') == 'FILLED':
            actual_filled_price = float(entry_order_result.get('avgPrice', effective_entry_price_for_sizing))
            total_filled_quantity = float(entry_order_result.get('executedQty', quantity))
            print(f"{log_prefix} Market Entry order {entry_order_id} FILLED. AvgPrice: {actual_filled_price:.{p_prec}f}, ExecutedQty: {total_filled_quantity}")
            
            # Send Telegram for FILLED market order
            send_app_trade_execution_telegram(
                current_app_settings=app_trading_configs, symbol=symbol, side=side,
                entry_price=actual_filled_price, sl_price=sl_price,
                tp1_price=tp1_price, tp2_price=tp2_price, tp3_price=tp3_price,
                order_type="MARKET", executed_quantity=total_filled_quantity, mode="live"
            )

        elif order_type.upper() == "LIMIT":
            print(f"{log_prefix} LIMIT Entry order {entry_order_id} PLACED. Status: {entry_order_result.get('status')}. Waiting for fill to place SL/TP.")
            # Send Telegram for PLACED limit order
            send_app_trade_execution_telegram(
                current_app_settings=app_trading_configs, symbol=symbol, side=side,
                entry_price=entry_price_target, # Use the target limit price for the message
                sl_price=sl_price, tp1_price=tp1_price, tp2_price=tp2_price, tp3_price=tp3_price,
                order_type="LIMIT", executed_quantity=None, mode="live" # executed_quantity is None as it's not filled yet
            )
            # Store for monitoring
            with app_active_trades_lock:
                app_active_trades[symbol] = {
                    "entry_order_id": entry_order_id,
                    "entry_price_target": entry_price_target, 
                    "status": "PENDING_FILL_FOR_SLTP", 
                    "total_quantity": total_filled_quantity, 
                    "side": side.upper(),
                    "symbol_info": symbol_info_app,
                    "open_timestamp": pd.Timestamp.now(tz='UTC'), 
                    "strategy_type": "APP_ML_TRADE", 
                    "intended_sl": sl_price, # Store intended SL for pre-fill cancellation logic
                    "intended_tp1": tp1_price, 
                    "intended_tp2": tp2_price, 
                    "intended_tp3": tp3_price,
                    "order_type": "LIMIT",
                    # Store context for trend reversal check.
                    # For an ML setup, the 'pivot_price_sim' from app_process_symbol_for_signal
                    # could be a good proxy for the swing point that defined the setup.
                    # However, execute_app_trade_signal doesn't currently receive that.
                    # For now, we'll rely on the 'intended_sl' for a simpler invalidation check.
                    # A more advanced version would pass and store more detailed setup context.
                    # "setup_pivot_price": pivot_price_sim, # Example if available
                    # "setup_expected_pullback_direction": "DOWN" if side.upper() == "LONG" else "UP" # Example
                }
            return True # Limit order placed, SL/TP pending fill


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
        # --- TP3 ATR-based calculation (if tp3_price is None) ---
        calculated_tp3_price = tp3_price # Use provided tp3_price by default
        is_tp3_floating_atr = False
        tp3_atr_multiplier_used = None

        if tp3_price is None and qty_tp3 > 0: # Only calculate if TP3 has quantity and no explicit price
            atr_period_for_tp3 = app_settings.get("app_atr_period_sl_tp", 14)
            # Need kline data to calculate ATR. Fetch recent klines.
            # This adds an API call during trade execution. Consider if ATR can be passed in.
            # For now, let's fetch.
            print(f"{log_prefix} TP3 price not provided, calculating ATR-based TP3 (ATR period: {atr_period_for_tp3}).")
            df_for_atr, atr_err = get_historical_bars(symbol, app_settings.get("app_trading_kline_interval"), f"{atr_period_for_tp3 + 50} days ago UTC"), None # Simplified fetch
            
            if atr_err is None and not df_for_atr.empty and len(df_for_atr) >= atr_period_for_tp3:
                df_for_atr = calculate_atr(df_for_atr, period=atr_period_for_tp3) # calculate_atr is from app.py
                atr_val_for_tp3 = df_for_atr[f'atr_{atr_period_for_tp3}'].iloc[-1]
                
                if pd.notna(atr_val_for_tp3) and atr_val_for_tp3 > 0:
                    tp3_atr_mult = app_settings.get('app_tp3_atr_multiplier', 2.0)
                    tp3_atr_multiplier_used = tp3_atr_mult # Store the multiplier used
                    if side.upper() == "LONG":
                        calculated_tp3_price = actual_filled_price + (atr_val_for_tp3 * tp3_atr_mult)
                    else: # SHORT
                        calculated_tp3_price = actual_filled_price - (atr_val_for_tp3 * tp3_atr_mult)
                    calculated_tp3_price = round(calculated_tp3_price, p_prec)
                    is_tp3_floating_atr = True
                    print(f"{log_prefix} ATR-based TP3 calculated: {calculated_tp3_price:.{p_prec}f} (ATR: {atr_val_for_tp3:.{p_prec}f}, Multiplier: {tp3_atr_mult})")
                else:
                    print(f"{log_prefix} Failed to get valid ATR for TP3 calculation. TP3 will not be placed if it had no explicit price.")
                    calculated_tp3_price = None # Ensure it's None if ATR calc failed
            else:
                print(f"{log_prefix} Failed to fetch data for ATR TP3 calculation. TP3 will not be placed if it had no explicit price. Error: {atr_err}")
                calculated_tp3_price = None # Ensure it's None
        # --- End TP3 ATR-based calculation ---

        tp_targets = [
            {"price": tp1_price, "quantity": qty_tp1, "name": "TP1"},
            {"price": tp2_price, "quantity": qty_tp2, "name": "TP2"},
            {"price": calculated_tp3_price, "quantity": qty_tp3, "name": "TP3"} # Use calculated_tp3_price
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
                "sl_management_stage": "initial",
                "is_tp3_floating_atr": is_tp3_floating_atr, # Store if TP3 was ATR based
                "tp3_atr_multiplier_used": tp3_atr_multiplier_used # Store multiplier if used
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
    Handles limit order fills, SL/TP hits, staged SL adjustments,
    floating TP3 updates, and timed force-closures.
    Logs decisions to the decision log.
    """
    global app_binance_client, app_trading_configs, app_active_trades, app_active_trades_lock, app_settings

    if app_active_trades is None or not app_active_trades: # Explicit check for None or empty
        return

    current_operational_mode = app_trading_configs.get("app_operational_mode", "signal")
    log_prefix_monitor = f"[AppTradeMonitor ({current_operational_mode.upper()})]"
    
    trades_to_remove = []
    active_trades_snapshot = {}
    with app_active_trades_lock:
        if app_active_trades is not None: # Check inside lock as well
            active_trades_snapshot = app_active_trades.copy()

    if active_trades_snapshot is None or not active_trades_snapshot: return # Double check after lock

    for symbol, trade_details in active_trades_snapshot.items():
        trade_id_log = trade_details.get('entry_order_id', 'N/A') # Use for logging
        log_sym_prefix = f"{log_prefix_monitor} [{symbol} ID:{trade_id_log}]"

        if app_binance_client is None:
            print(f"{log_sym_prefix} Binance client not available. Skipping monitoring cycle.")
            append_decision_log({"symbol": symbol, "trade_id": trade_id_log, "event_type": "MONITOR_SKIP_NO_CLIENT"})
            continue

        s_info = trade_details.get('symbol_info')
        if s_info is None: # Explicit check for None
            print(f"{log_sym_prefix} Missing symbol_info. Removing trade.")
            append_decision_log({"symbol": symbol, "trade_id": trade_id_log, "event_type": "MONITOR_ERROR_NO_SYM_INFO"})
            trades_to_remove.append(symbol)
            continue
        
        p_prec = int(s_info.get('pricePrecision', 2))
        q_prec = int(s_info.get('quantityPrecision', 0))

        # --- 1. Handle PENDING_FILL_FOR_SLTP (Limit Order Fill Check) ---
        if trade_details.get('status') == "PENDING_FILL_FOR_SLTP":
            entry_order_id = trade_details.get('entry_order_id')
            if not entry_order_id:
                print(f"{log_sym_prefix} PENDING_FILL_FOR_SLTP but no entry_order_id. Removing.")
                append_decision_log({"symbol": symbol, "trade_id": "N/A", "event_type": "MONITOR_ERROR_PENDING_NO_ID"})
                trades_to_remove.append(symbol)
                continue
            
            try:
                limit_order_status = app_binance_client.futures_get_order(symbol=symbol, orderId=entry_order_id)
                
                if limit_order_status['status'] == 'FILLED':
                    actual_filled_price = float(limit_order_status.get('avgPrice', trade_details.get('entry_price_target')))
                    total_filled_quantity = float(limit_order_status.get('executedQty', trade_details.get('total_quantity')))
                    fill_time_ms = limit_order_status.get('updateTime', int(time.time() * 1000))
                    fill_timestamp = pd.Timestamp(fill_time_ms, unit='ms', tz='UTC')
                    print(f"{log_sym_prefix} LIMIT entry order {entry_order_id} FILLED. AvgPrice: {actual_filled_price:.{p_prec}f}, Qty: {total_filled_quantity}")
                    append_decision_log({
                        "symbol": symbol, "trade_id": entry_order_id, "event_type": "LIMIT_ORDER_FILLED",
                        "fill_price": actual_filled_price, "fill_qty": total_filled_quantity, "fill_time": fill_timestamp.isoformat()
                    })

                    sl_price = trade_details['intended_sl']
                    tp1_price = trade_details['intended_tp1']
                    tp2_price = trade_details.get('intended_tp2')
                    tp3_price = trade_details.get('intended_tp3') # This is the initial target if set
                    
                    api_position_side = trade_details['side'].upper()
                    
                    # Calculate TP quantities
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
                    if (1 - tp1_pct - tp2_pct) > 1e-5 and 0 < qty_tp3 < min_qty_val: # TP3 intended but dust
                        if qty_tp2 > 0: qty_tp2 = round(qty_tp2 + qty_tp3, q_prec)
                        elif qty_tp1 > 0: qty_tp1 = round(qty_tp1 + qty_tp3, q_prec)
                        if not (qty_tp1 == 0 and qty_tp2 == 0): qty_tp3 = 0.0
                    
                    # Place SL
                    placed_sl_order_obj, sl_err_msg = place_app_new_order(s_info, "SELL" if api_position_side == "LONG" else "BUY", 
                                                                        "STOP_MARKET", total_filled_quantity, stop_price=sl_price, 
                                                                        position_side=api_position_side, is_closing_order=True)
                    if not placed_sl_order_obj:
                        append_decision_log({"symbol": symbol, "trade_id": entry_order_id, "event_type": "SL_PLACEMENT_FAILED_POST_LIMIT_FILL", "error": sl_err_msg})
                    else:
                        append_decision_log({"symbol": symbol, "trade_id": entry_order_id, "event_type": "SL_PLACED_POST_LIMIT_FILL", "sl_order_id": placed_sl_order_obj.get('orderId'), "sl_price": sl_price})

                    # --- TP3 ATR-based calculation (if tp3_price is None from original signal AND qty_tp3 > 0) ---
                    is_tp3_floating_atr_limit = False
                    tp3_atr_multiplier_used_limit = None
                    calculated_tp3_price_limit = tp3_price # Start with what was intended (could be None)

                    if tp3_price is None and qty_tp3 > 0: # Only if TP3 had no price and has quantity
                        atr_period_tp3 = app_settings.get("app_atr_period_sl_tp", 14)
                        df_atr, _ = get_historical_bars(symbol, app_settings.get("app_trading_kline_interval"), f"{atr_period_tp3 + 50} days ago UTC")
                        if df_atr is not None and not df_atr.empty and len(df_atr) >= atr_period_tp3:
                            df_atr = calculate_atr(df_atr, period=atr_period_tp3)
                            atr_val = df_atr[f'atr_{atr_period_tp3}'].iloc[-1]
                            if pd.notna(atr_val) and atr_val > 0:
                                mult = app_settings.get('app_tp3_atr_multiplier', 2.0)
                                tp3_atr_multiplier_used_limit = mult
                                if api_position_side == "LONG": calculated_tp3_price_limit = actual_filled_price + (atr_val * mult)
                                else: calculated_tp3_price_limit = actual_filled_price - (atr_val * mult)
                                calculated_tp3_price_limit = round(calculated_tp3_price_limit, p_prec)
                                is_tp3_floating_atr_limit = True
                                append_decision_log({"symbol":symbol, "trade_id":entry_order_id, "event_type":"TP3_ATR_CALC_POST_LIMIT_FILL", "atr_val":atr_val, "mult":mult, "calc_tp3_price":calculated_tp3_price_limit})
                            else: calculated_tp3_price_limit = None # Failed ATR calc
                        else: calculated_tp3_price_limit = None # Failed data fetch
                    
                    # Place TPs
                    placed_tp_orders_list = []
                    tp_targets_def = [
                        {"price": tp1_price, "quantity": qty_tp1, "name": "TP1"},
                        {"price": tp2_price, "quantity": qty_tp2, "name": "TP2"},
                        {"price": calculated_tp3_price_limit, "quantity": qty_tp3, "name": "TP3"} # Use potentially updated TP3 price
                    ]
                    for tp_info in tp_targets_def:
                        if tp_info["price"] is not None and tp_info["quantity"] > 0:
                            tp_ord, tp_err = place_app_new_order(s_info, "SELL" if api_position_side == "LONG" else "BUY",
                                                                 "TAKE_PROFIT_MARKET", tp_info["quantity"], stop_price=tp_info["price"],
                                                                 position_side=api_position_side, is_closing_order=True)
                            status_log = "OPEN" if tp_ord else "FAILED"
                            tp_id_log = tp_ord.get('orderId') if tp_ord else None
                            append_decision_log({"symbol":symbol, "trade_id":entry_order_id, "event_type":f"{tp_info['name']}_PLACEMENT_POST_LIMIT_FILL", "tp_order_id":tp_id_log, "price":tp_info['price'], "qty":tp_info['quantity'], "status":status_log, "error":tp_err})
                            placed_tp_orders_list.append({"id": tp_id_log, "price": tp_info["price"], "quantity": tp_info["quantity"], "status": status_log, "name": tp_info["name"]})

                    # Update app_active_trades
                    with app_active_trades_lock:
                        if symbol in app_active_trades:
                            app_active_trades[symbol].update({
                                "status": "ACTIVE", "entry_price": actual_filled_price, "total_quantity": total_filled_quantity,
                                "sl_order_id": placed_sl_order_obj.get('orderId') if placed_sl_order_obj else None,
                                "tp_orders": placed_tp_orders_list, "current_sl_price": sl_price, "initial_sl_price": sl_price,
                                "open_timestamp": fill_timestamp, "sl_management_stage": "initial",
                                "is_tp3_floating_atr": is_tp3_floating_atr_limit, # If TP3 became ATR based
                                "tp3_atr_multiplier_used": tp3_atr_multiplier_used_limit
                            })
                            print(f"{log_sym_prefix} Limit order filled, SL/TPs placed. Trade ACTIVE.")
                        else: # Edge case: trade removed concurrently
                            append_decision_log({"symbol": symbol, "trade_id": entry_order_id, "event_type": "MONITOR_ERROR_LIMIT_FILL_TRADE_GONE"})
                
                elif limit_order_status['status'] in ['CANCELED', 'EXPIRED', 'REJECTED', 'PENDING_CANCEL']:
                    print(f"{log_sym_prefix} Limit entry order {entry_order_id} is {limit_order_status['status']}. Removing.")
                    append_decision_log({"symbol": symbol, "trade_id": entry_order_id, "event_type": "LIMIT_ORDER_FINAL_STATUS_NOT_FILLED", "status": limit_order_status['status']})
                    trades_to_remove.append(symbol)
                
                elif limit_order_status['status'] in ['NEW', 'PARTIALLY_FILLED']: # Still pending, check for SL hit pre-fill
                    intended_sl = trade_details.get('intended_sl')
                    pending_side = trade_details.get('side')
                    if intended_sl and pending_side:
                        mkt_price = float(app_binance_client.futures_ticker(symbol=symbol)['lastPrice'])
                        invalidated = False
                        if pending_side == "LONG" and mkt_price <= intended_sl: invalidated = True
                        elif pending_side == "SHORT" and mkt_price >= intended_sl: invalidated = True
                        
                        if invalidated:
                            print(f"{log_sym_prefix} Pending LIMIT {entry_order_id} INVALIDATED pre-fill (SL: {intended_sl:.{p_prec}f} vs Mkt: {mkt_price:.{p_prec}f}). Cancelling.")
                            try:
                                app_binance_client.futures_cancel_order(symbol=symbol, orderId=entry_order_id)
                                send_app_telegram_message(f"⚠️ PENDING ORDER CANCELLED ⚠️\nSymbol: `{symbol}` Side: `{pending_side}`\nReason: _Intended SL hit before fill._")
                                append_decision_log({"symbol": symbol, "trade_id": entry_order_id, "event_type": "LIMIT_ORDER_CANCELLED_PRE_FILL_SL_HIT", "intended_sl": intended_sl, "market_price": mkt_price})
                                trades_to_remove.append(symbol)
                            except Exception as e_cancel:
                                append_decision_log({"symbol": symbol, "trade_id": entry_order_id, "event_type": "LIMIT_ORDER_CANCEL_FAILED_PRE_FILL_SL_HIT", "error": str(e_cancel)})
            except Exception as e_check_limit:
                print(f"{log_sym_prefix} Error checking limit order {entry_order_id}: {e_check_limit}")
                append_decision_log({"symbol": symbol, "trade_id": entry_order_id, "event_type": "MONITOR_ERROR_CHECKING_LIMIT_ORDER", "error": str(e_check_limit)})
            continue # Next symbol

        # --- 2. Monitor ACTIVE Live Trades ---
        if not (trade_details.get('status') == "ACTIVE" and current_operational_mode == "live"):
            if trade_details.get('status') != "PENDING_FILL_FOR_SLTP": # Don't log for pending, it's handled above
                 append_decision_log({"symbol": symbol, "trade_id": trade_id_log, "event_type": "MONITOR_SKIP_NOT_ACTIVE_LIVE", "current_status": trade_details.get('status')})
            continue

        # Fetch current market price for active trade checks
        current_market_price = None
        try:
            current_market_price = float(app_binance_client.futures_ticker(symbol=symbol)['lastPrice'])
        except Exception as e_fetch_active_price:
            print(f"{log_sym_prefix} Could not fetch market price for active trade: {e_fetch_active_price}")
            append_decision_log({"symbol": symbol, "trade_id": trade_id_log, "event_type": "MONITOR_ERROR_FETCH_ACTIVE_PRICE", "error": str(e_fetch_active_price)})
            continue

        # --- 2a. Timed Force-Close ---
        max_hours = app_settings.get('app_max_trade_duration_hours', 24)
        open_ts = trade_details.get('open_timestamp')
        if isinstance(open_ts, pd.Timestamp) and (pd.Timestamp.now(tz='UTC') - open_ts).total_seconds() > max_hours * 3600:
            print(f"{log_sym_prefix} Trade open > {max_hours}h. Force-closing.")
            append_decision_log({"symbol": symbol, "trade_id": trade_id_log, "event_type": "FORCE_CLOSE_INITIATED_TIMEOUT", "max_hours": max_hours, "open_since": open_ts.isoformat()})
            # Cancel SL/TPs
            if trade_details.get('sl_order_id'):
                try: app_binance_client.futures_cancel_order(symbol=symbol, orderId=trade_details['sl_order_id'])
                except Exception as e_fc_sl: append_decision_log({"symbol":symbol,"trade_id":trade_id_log,"event_type":"FORCE_CLOSE_CANCEL_SL_ERROR","error":str(e_fc_sl)})
            for tp in trade_details.get('tp_orders', []):
                if tp.get('id') and tp.get('status') == 'OPEN':
                    try: app_binance_client.futures_cancel_order(symbol=symbol, orderId=tp['id'])
                    except Exception as e_fc_tp: append_decision_log({"symbol":symbol,"trade_id":trade_id_log,"event_type":"FORCE_CLOSE_CANCEL_TP_ERROR","tp_id":tp['id'],"error":str(e_fc_tp)})
            # Market close position
            pos_amt_fc = float(app_binance_client.futures_position_information(symbol=symbol)[0].get('positionAmt',0.0)) # Simplified fetch
            if abs(pos_amt_fc) > 0:
                fc_side = "SELL" if pos_amt_fc > 0 else "BUY"
                fc_ord, fc_err = place_app_new_order(s_info, fc_side, "MARKET", abs(pos_amt_fc), position_side=trade_details['side'].upper())
                if fc_ord: send_app_telegram_message(f"✅ FORCE-CLOSED: {symbol} {trade_details['side']} after {max_hours}h. Qty: {abs(pos_amt_fc)}")
                else: send_app_telegram_message(f"🆘 FAILED Force-Close: {symbol} {trade_details['side']}. Err: {fc_err}")
                append_decision_log({"symbol":symbol,"trade_id":trade_id_log,"event_type":"FORCE_CLOSE_MARKET_ORDER_RESULT","order_id":fc_ord.get('orderId') if fc_ord else None,"error":fc_err, "qty":abs(pos_amt_fc)})
            trades_to_remove.append(symbol)
            continue

        # --- 2b. SL Hit Check (Market Price vs. Stored SL) ---
        current_sl = trade_details.get('current_sl_price')
        trade_side = trade_details.get('side')
        sl_hit = False
        if current_sl:
            if trade_side == "LONG" and current_market_price <= current_sl: sl_hit = True
            elif trade_side == "SHORT" and current_market_price >= current_sl: sl_hit = True
        
        if sl_hit:
            print(f"{log_sym_prefix} STOP LOSS HIT by market price. SL: {current_sl:.{p_prec}f}, Market: {current_market_price:.{p_prec}f}")
            append_decision_log({"symbol": symbol, "trade_id": trade_id_log, "event_type": "SL_HIT_MARKET_PRICE", "sl_price": current_sl, "market_price": current_market_price, "side": trade_side})
            # SL order should fill. Cancel remaining TPs.
            for tp in trade_details.get('tp_orders', []):
                if tp.get('id') and tp.get('status') == 'OPEN':
                    try: app_binance_client.futures_cancel_order(symbol=symbol, orderId=tp['id'])
                    except Exception as e_sl_tp_cancel: append_decision_log({"symbol":symbol,"trade_id":trade_id_log,"event_type":"SL_HIT_CANCEL_TP_ERROR","tp_id":tp['id'],"error":str(e_sl_tp_cancel)})
            send_app_telegram_message(f"❌ SL HIT: {symbol} {trade_side} @ ~{current_sl:.{p_prec}f}")
            trades_to_remove.append(symbol)
            continue
            
        # --- 2c. TP Hit Checks & SL Management ---
        sl_management_stage = trade_details.get('sl_management_stage', 'initial')
        tp_orders_snapshot = list(trade_details.get('tp_orders', [])) # Iterate over a copy
        any_tp_hit_this_cycle = False

        for idx, tp_order_info in enumerate(tp_orders_snapshot):
            if tp_order_info.get('status') == 'OPEN' and tp_order_info.get('id'):
                try:
                    tp_order_status = app_binance_client.futures_get_order(symbol=symbol, orderId=tp_order_info['id'])
                    if tp_order_status['status'] == 'FILLED':
                        any_tp_hit_this_cycle = True
                        filled_tp_name = tp_order_info['name']
                        filled_tp_price = float(tp_order_status['avgPrice'])
                        print(f"{log_sym_prefix} {filled_tp_name} FILLED @ {filled_tp_price:.{p_prec}f}")
                        append_decision_log({"symbol":symbol,"trade_id":trade_id_log,"event_type":f"{filled_tp_name}_HIT","order_id":tp_order_info['id'],"fill_price":filled_tp_price, "qty":tp_order_status['executedQty']})
                        send_app_telegram_message(f"✅ {filled_tp_name} HIT: {symbol} {trade_side} @ {filled_tp_price:.{p_prec}f}")
                        
                        with app_active_trades_lock: # Update TP status and SL stage
                            if symbol in app_active_trades and idx < len(app_active_trades[symbol]['tp_orders']):
                                app_active_trades[symbol]['tp_orders'][idx]['status'] = 'FILLED'
                                current_sl_stage_locked = app_active_trades[symbol]['sl_management_stage']
                                if filled_tp_name == 'TP1' and current_sl_stage_locked == 'initial':
                                    app_active_trades[symbol]['sl_management_stage'] = 'after_tp1'
                                elif filled_tp_name == 'TP2' and current_sl_stage_locked in ['initial', 'after_tp1']:
                                    app_active_trades[symbol]['sl_management_stage'] = 'after_tp2'
                                sl_management_stage = app_active_trades[symbol]['sl_management_stage'] # Refresh for current cycle use
                except Exception as e_tp_check:
                    append_decision_log({"symbol":symbol,"trade_id":trade_id_log,"event_type":"MONITOR_ERROR_CHECKING_TP","tp_id":tp_order_info['id'],"error":str(e_tp_check)})
        
        if any_tp_hit_this_cycle: # If a TP was hit, re-evaluate SL
            new_sl_price_target = None
            sl_adj_reason = None
            current_entry_price = trade_details['entry_price']
            initial_sl = trade_details['initial_sl_price'] # Should be current_sl_price from trade_details

            if sl_management_stage == 'after_tp1':
                buffer = app_settings.get('app_breakeven_buffer_pct', 0.001)
                new_sl_price_target = current_entry_price * (1 + buffer) if trade_side == "LONG" else current_entry_price * (1 - buffer)
                sl_adj_reason = "TP1 Hit: SL to Breakeven+"
            elif sl_management_stage == 'after_tp2':
                tp1_data = next((tp for tp in trade_details.get('tp_orders', []) if tp['name'] == 'TP1' and tp.get('price') is not None), None)
                if tp1_data: new_sl_price_target = tp1_data['price']
                sl_adj_reason = "TP2 Hit: SL to TP1 Price"
            
            if new_sl_price_target is not None:
                new_sl_price_target = round(new_sl_price_target, p_prec)
                is_improvement = (trade_side == "LONG" and new_sl_price_target > trade_details['current_sl_price']) or \
                                 (trade_side == "SHORT" and new_sl_price_target < trade_details['current_sl_price'])
                
                if is_improvement:
                    # Calculate remaining quantity for new SL order
                    remaining_qty_sl_adj = 0
                    with app_active_trades_lock:
                        if symbol in app_active_trades: # Read fresh TP states
                            for tp_o in app_active_trades[symbol]['tp_orders']:
                                if tp_o.get('status') == 'OPEN' and tp_o.get('quantity',0) > 0 : remaining_qty_sl_adj += tp_o['quantity']
                    remaining_qty_sl_adj = round(remaining_qty_sl_adj, q_prec)

                    if remaining_qty_sl_adj > 0 :
                        print(f"{log_sym_prefix} Adjusting SL: {sl_adj_reason} to {new_sl_price_target:.{p_prec}f} for Qty: {remaining_qty_sl_adj}")
                        if trade_details.get('sl_order_id'): # Cancel old SL
                            try: app_binance_client.futures_cancel_order(symbol=symbol, orderId=trade_details['sl_order_id'])
                            except Exception as e_cancel_old_sl_adj: append_decision_log({"symbol":symbol,"trade_id":trade_id_log,"event_type":"SL_ADJUST_CANCEL_OLD_ERROR","error":str(e_cancel_old_sl_adj)})
                        
                        # Place new SL
                        new_sl_obj, new_sl_err = place_app_new_order(s_info, "SELL" if trade_side == "LONG" else "BUY", "STOP_MARKET", 
                                                                     remaining_qty_sl_adj, stop_price=new_sl_price_target, 
                                                                     position_side=trade_side, is_closing_order=True)
                        if new_sl_obj:
                            with app_active_trades_lock:
                                if symbol in app_active_trades:
                                    app_active_trades[symbol]['sl_order_id'] = new_sl_obj.get('orderId')
                                    app_active_trades[symbol]['current_sl_price'] = new_sl_price_target
                            send_app_telegram_message(f"⚙️ SL ADJUSTED: {symbol} {trade_side}. {sl_adj_reason} to {new_sl_price_target:.{p_prec}f}")
                            append_decision_log({"symbol":symbol,"trade_id":trade_id_log,"event_type":"SL_ADJUSTED_SUCCESS","new_sl_price":new_sl_price_target,"new_sl_id":new_sl_obj.get('orderId'),"reason":sl_adj_reason})
                        else:
                            send_app_telegram_message(f"🆘 FAILED SL ADJUSTMENT: {symbol} {trade_side}. {sl_adj_reason}. Err: {new_sl_err}")
                            append_decision_log({"symbol":symbol,"trade_id":trade_id_log,"event_type":"SL_ADJUST_FAILED","target_price":new_sl_price_target,"error":new_sl_err})
                    else: # No remaining quantity for SL adjustment
                        append_decision_log({"symbol":symbol,"trade_id":trade_id_log,"event_type":"SL_ADJUST_NO_REMAINING_QTY","reason":"All TPs hit or quantities zeroed."})
                        if trade_details.get('sl_order_id'): # Cancel final SL if it exists
                            try: app_binance_client.futures_cancel_order(symbol=symbol, orderId=trade_details['sl_order_id'])
                            except Exception: pass # Ignore error if already gone
                        trades_to_remove.append(symbol) # Mark for removal as all TPs seem closed

        # --- 2d. Check if all TPs finalized (for trade removal) ---
        all_tps_done = True
        with app_active_trades_lock:
            if symbol in app_active_trades: # Re-check existence
                for tp_o in app_active_trades[symbol]['tp_orders']:
                    if tp_o.get('quantity', 0) > 0 and tp_o.get('status') == 'OPEN':
                        all_tps_done = False; break
        if all_tps_done:
            print(f"{log_sym_prefix} All TPs finalized or have zero quantity. Removing trade.")
            append_decision_log({"symbol": symbol, "trade_id": trade_id_log, "event_type": "ALL_TPS_FINALIZED_REMOVING_TRADE"})
            if trade_details.get('sl_order_id'): # Ensure final SL is cancelled if somehow still open
                 try: app_binance_client.futures_cancel_order(symbol=symbol, orderId=trade_details['sl_order_id'])
                 except Exception: pass
            trades_to_remove.append(symbol)

    # --- Final Removal of Trades ---
    if trades_to_remove:
        with app_active_trades_lock:
            for sym_rem in trades_to_remove:
                if sym_rem in app_active_trades:
                    removed_trade_details = app_active_trades.pop(sym_rem) # Use pop to get details if needed for final log
                    print(f"{log_prefix_monitor} Removed trade for {sym_rem} from app_active_trades. Final status: {removed_trade_details.get('status','N/A')}")
                    append_decision_log({"symbol": sym_rem, "trade_id": removed_trade_details.get('entry_order_id','N/A'), "event_type": "TRADE_REMOVED_FROM_ACTIVE_LIST", "final_recorded_status": removed_trade_details.get('status','N/A')})

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

    if bot_token is None or chat_id is None: # Explicit check for None
        print(f"APP_TELEGRAM_SKIPPED: Token or Chat ID not configured (None) in app_trading_configs. Message: '{message[:100]}...'")
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

    if app_ptb_event_loop is not None and app_ptb_event_loop.is_running(): # Explicit check for None
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

# --- Fibonacci Calculation Helper ---
def compute_fib_levels(anchor_1: float, anchor_2: float, direction: str) -> dict | None:
    """
    Calculates Fibonacci retracement levels for a given swing.

    Args:
        anchor_1 (float): Price of the first anchor point of the swing.
        anchor_2 (float): Price of the second anchor point of the swing.
        direction (str): "long" if the original move was up (expecting retracement down),
                         "short" if the original move was down (expecting retracement up).
                         This refers to the direction of the *main leg* for which fibs are drawn.
                         A "long" direction means the leg went from a low (anchor_X) to a high (anchor_Y),
                         and we expect price to retrace *down* from high towards low.
                         A "short" direction means the leg went from a high to a low,
                         and we expect price to retrace *up* from low towards high.


    Returns:
        dict | None: A dictionary with Fib levels (e.g., "0.618": price), or None if inputs invalid.
    """
    if anchor_1 is None or anchor_2 is None:
        print("compute_fib_levels: One or both anchor prices are None.")
        return None

    swing_high_price = max(anchor_1, anchor_2)
    swing_low_price = min(anchor_1, anchor_2)

    if swing_high_price == swing_low_price:
        print(f"compute_fib_levels: Swing high and low are identical ({swing_high_price}). Cannot calculate Fib levels.")
        return None

    price_range = swing_high_price - swing_low_price
    # Standard Fibonacci retracement levels often used for entries.
    # Extensions (e.g., 1.272, 1.618) are typically for targets, not handled here.
    fib_ratios = {"0.236": 0.236, "0.382": 0.382, "0.500": 0.500, "0.618": 0.618, "0.786": 0.786}
    calculated_levels = {}

    # The 'direction' parameter determines how retracement is viewed.
    # If direction == "long", it implies the primary move was UP (from swing_low_price to swing_high_price),
    # and we are looking for a retracement DOWNWARDS from swing_high_price.
    # If direction == "short", it implies the primary move was DOWN (from swing_high_price to swing_low_price),
    # and we are looking for a retracement UPWARDS from swing_low_price.

    for level_name, ratio_val in fib_ratios.items():
        if direction.lower() == "long": # Primary move was up, expect retrace down from swing_high_price
            calculated_levels[level_name] = swing_high_price - (price_range * ratio_val)
        elif direction.lower() == "short": # Primary move was down, expect retrace up from swing_low_price
            calculated_levels[level_name] = swing_low_price + (price_range * ratio_val)
        else:
            print(f"compute_fib_levels: Invalid direction '{direction}'. Must be 'long' or 'short'.")
            return None
            
    # print(f"compute_fib_levels ({direction}): Leg [{swing_low_price} - {swing_high_price}]. Levels: {calculated_levels}")
    return calculated_levels
# --- End Fibonacci Calculation Helper ---

def escape_app_markdown_v1(text: str) -> str:
    """Escapes characters for Telegram Markdown V1 (app.py version)."""
    if not isinstance(text, str): return ""
    text = text.replace('_', r'\_').replace('*', r'\*').replace('`', r'\`').replace('[', r'\[')
    return text

def app_send_entry_signal_telegram(current_app_settings: dict, symbol: str, signal_type_display: str, 
                                   leverage: int, entry_price: float, 
                                   tp1_price: float, tp2_price: float | None, tp3_price: float | None, 
                                   sl_price: float, risk_percentage_config: float, 
                                   est_pnl_tp1: float | None, est_pnl_sl: float | None,
                                   symbol_info: dict, strategy_name_display: str = "App ML Signal",
                                   signal_timestamp: pd.Timestamp = None, signal_order_type: str = "N/A",
                                   p_swing_score: float = None, p_profit_score: float = None):
    """
    Formats and sends a new trade signal notification via app.py's Telegram sender.
    Mirrors main.py's send_entry_signal_telegram formatting.
    """
    log_prefix = f"[AppSendEntrySignal-{symbol}]"
    bot_token = current_app_settings.get("app_telegram_bot_token")
    chat_id = current_app_settings.get("app_telegram_chat_id")
    notify_on_trade = current_app_settings.get("app_notify_on_trade", True) # Check the toggle

    if not notify_on_trade: # This is a boolean check, not an object check, so it's fine.
        print(f"{log_prefix} Telegram notifications for trade execution are disabled by 'app_notify_on_trade' setting. Skipping signal message.")
        return

    if bot_token is None or chat_id is None: # Explicit check for None
        print(f"{log_prefix} Telegram token/chat_id not configured (None) in app_settings. Cannot send signal.")
        return

    if signal_timestamp is None:
        signal_timestamp = pd.Timestamp.now(tz='UTC')
    
    p_prec = int(symbol_info.get('pricePrecision', 2)) if symbol_info is not None else 2 # Check symbol_info

    tp1_str = f"{tp1_price:.{p_prec}f}" if tp1_price is not None else "N/A"
    tp2_str = f"{tp2_price:.{p_prec}f}" if tp2_price is not None else "N/A"
    tp3_str = f"{tp3_price:.{p_prec}f}" if tp3_price is not None else "N/A"
    sl_str = f"{sl_price:.{p_prec}f}" if sl_price is not None else "N/A"
    
    # P&L estimations are passed in, so no need for calculate_pnl_for_fixed_capital here
    # unless we want to standardize that helper within app.py too. For now, assume they are provided.
    pnl_tp1_str = f"{est_pnl_tp1:.2f} USDT" if est_pnl_tp1 is not None else "Not Calculated"
    pnl_sl_str = f"{est_pnl_sl:.2f} USDT" if est_pnl_sl is not None else "Not Calculated"

    side_emoji = "🔼" if "LONG" in signal_type_display.upper() else "🔽" if "SHORT" in signal_type_display.upper() else "↔️"
    signal_side_text = "LONG" if "LONG" in signal_type_display.upper() else "SHORT" if "SHORT" in signal_type_display.upper() else "N/A"
    formatted_timestamp = signal_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')

    # Escape dynamic string parts
    escaped_symbol = escape_app_markdown_v1(symbol)
    escaped_strategy_name = escape_app_markdown_v1(strategy_name_display)
    escaped_signal_type = escape_app_markdown_v1(signal_type_display)
    escaped_order_type = escape_app_markdown_v1(signal_order_type)
    
    message = (
        f"🔔 *NEW TRADE SIGNAL* | {escaped_strategy_name} {side_emoji}\n\n"
        f"🗓️ Time: `{formatted_timestamp}`\n"
        f"📈 Symbol: `{escaped_symbol}`\n"
        f"SIDE: *{signal_side_text}*\n"
        f"🔩 Strategy: `{escaped_signal_type}`\n"
        f"📊 Order Type: `{escaped_order_type}`\n"
        f"Leverage: `{leverage}x`\n\n"
        f"➡️ Entry Price: `{entry_price:.{p_prec}f}`\n"
        f"🛡️ Stop Loss: `{sl_str}`\n"
    )
    
    tps_message_part = ""
    if tp1_price is not None: tps_message_part += f"🎯 Take Profit 1: `{tp1_str}`\n"
    if tp2_price is not None: tps_message_part += f"🎯 Take Profit 2: `{tp2_str}`\n"
    if tp3_price is not None: tps_message_part += f"🎯 Take Profit 3: `{tp3_str}`\n"
    if not tps_message_part and tp1_price is None: tps_message_part = "🎯 Take Profit Levels: `N/A`\n"
    message += tps_message_part
    
    message += f"\n📊 Configured Risk: `{risk_percentage_config * 100:.2f}%`\n"

    if p_swing_score is not None and p_profit_score is not None:
        message += f"📈 ML Scores: P_Swing=`{p_swing_score:.3f}`, P_Profit=`{p_profit_score:.3f}`\n"

    # P&L estimation based on $100 capital is often part of main.py's message.
    # If est_pnl_tp1 and est_pnl_sl are provided, we use them.
    message += (
        f"\n💰 *Est. P&L ($100 Capital Trade):*\n"
        f"  - TP1 Hit: `{pnl_tp1_str}`\n"
        f"  - SL Hit: `{pnl_sl_str}`\n\n"
    )

    message += f"⚠️ _This is a signal only. No order has been placed by app.py in 'signal' mode._"
    
    print(f"{log_prefix} Formatted entry signal message for Telegram.")
    send_app_telegram_message(message) # Uses app.py's sender

def app_send_fib_proposal_telegram(current_app_settings: dict, symbol: str, direction: str,
                                   pivot_price: float, leg_start_price: float, leg_end_price: float,
                                   proposed_limit_price: float, fib_ratio_used: float,
                                   valid_until_timestamp: pd.Timestamp,
                                   p_swing_score: float, p_profit_score_initial: float,
                                   symbol_info_for_prec: dict):
    """
    Formats and sends a Telegram message for a new Fibonacci limit order proposal.
    Indicates if it's a paper trade proposal based on app_settings.
    """
    log_prefix = f"[AppSendFibProposal-{symbol}]"
    bot_token = current_app_settings.get("app_telegram_bot_token")
    chat_id = current_app_settings.get("app_telegram_chat_id")
    notify_on_trade = current_app_settings.get("app_notify_on_trade", True) # Use general notification toggle

    if not notify_on_trade:
        print(f"{log_prefix} Telegram notifications are disabled. Skipping Fib proposal message.")
        return
    if bot_token is None or chat_id is None:
        print(f"{log_prefix} Telegram token/chat_id not configured. Cannot send Fib proposal.")
        return

    p_prec = int(symbol_info_for_prec.get('pricePrecision', 2)) if symbol_info_for_prec else 2
    
    direction_upper = direction.upper()
    side_emoji = "🔼" if direction_upper == "LONG" else "🔽" if direction_upper == "SHORT" else "↔️"
    
    proposal_time_str = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')
    expiry_time_str = valid_until_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')

    title = "🚧 *Fibonacci Limit Order PROPOSED*"
    if current_app_settings.get("app_simulate_limit_orders", False):
        title = "PAPER TRADE: " + title
    
    message = (
        f"{title} {side_emoji}\n\n"
        f"🗓️ Proposal Time: `{proposal_time_str}`\n"
        f"📈 Symbol: `{escape_app_markdown_v1(symbol)}`\n"
        f"Direction: *{direction_upper}*\n\n"
        f"🔍 **Basis:**\n"
        f"  - ML Pivot Price ({'Low' if direction_upper == 'LONG' else 'High'}): `{pivot_price:.{p_prec}f}`\n"
        f"  - Fib Leg Start: `{leg_start_price:.{p_prec}f}`\n"
        f"  - Fib Leg End: `{leg_end_price:.{p_prec}f}`\n"
        f"  - P_Swing Score: `{p_swing_score:.3f}`\n"
        f"  - P_Profit (Initial): `{p_profit_score_initial:.3f}`\n\n"
        f"🎯 **Proposal:**\n"
        f"  - Target Fib Level: `{fib_ratio_used*100:.1f}%`\n"
        f"  - Proposed Limit Price: `{proposed_limit_price:.{p_prec}f}`\n"
        f"  - Valid Until: `{expiry_time_str}`\n\n"
        f"⏳ _Waiting for price to touch limit and secondary conditions to be met..._"
    )
    
    print(f"{log_prefix} Formatted Fib proposal message for Telegram.")
    send_app_telegram_message(message)

def app_send_fib_proposal_update_telegram(current_app_settings: dict, proposal_data: dict, 
                                          update_type: str, message_detail: str):
    """
    Formats and sends a Telegram message for an update on a Fibonacci proposal 
    (e.g., expired, error, conditions met for simulated execution).
    """
    symbol = proposal_data.get('symbol', 'N/A')
    log_prefix = f"[AppSendFibUpdate-{symbol}]"
    bot_token = current_app_settings.get("app_telegram_bot_token")
    chat_id = current_app_settings.get("app_telegram_chat_id")
    notify_on_trade = current_app_settings.get("app_notify_on_trade", True)

    if not notify_on_trade:
        print(f"{log_prefix} Telegram notifications disabled. Skipping Fib proposal update.")
        return
    if bot_token is None or chat_id is None:
        print(f"{log_prefix} Telegram token/chat_id not configured. Cannot send Fib update.")
        return

    s_info = get_app_symbol_info(symbol) # Fetch fresh for precision
    p_prec = int(s_info.get('pricePrecision', 2)) if s_info else 2
    
    proposal_id_display = proposal_data.get('proposal_id', 'N/A')
    side_display = proposal_data.get('side', 'N/A').upper()
    limit_price_display = f"{proposal_data.get('proposed_limit_price', 0.0):.{p_prec}f}"

    title_emoji = "ℹ️" # Default
    if update_type == "EXPIRED": title_emoji = "⏱️"
    elif update_type == "CANCELLED_MANUAL": title_emoji = "❌" # Example for future use
    elif "ERROR" in update_type.upper(): title_emoji = "🆘"

    message = (
        f"{title_emoji} *Fibonacci Proposal Update* {title_emoji}\n\n"
        f"Symbol: `{escape_app_markdown_v1(symbol)}`\n"
        f"Side: `{side_display}`\n"
        f"Proposal ID: `{proposal_id_display}`\n"
        f"Original Limit: `{limit_price_display}`\n\n"
        f"Update Type: *{escape_app_markdown_v1(update_type)}*\n"
        f"Details: _{escape_app_markdown_v1(message_detail)}_"
    )
    
    print(f"{log_prefix} Formatted Fib proposal update message for Telegram.")
    send_app_telegram_message(message)


def send_app_trade_execution_telegram(current_app_settings: dict, symbol: str, side: str,
                                      entry_price: float, sl_price: float,
                                      tp1_price: float, tp2_price: float | None, tp3_price: float | None,
                                      order_type: str, executed_quantity: float | None,
                                      mode: str = "live"):
    """
    Formats and sends a Telegram message for a trade execution or confirmed signal.
    """
    log_prefix = f"[AppSendTradeExec-{symbol}]"
    bot_token = current_app_settings.get("app_telegram_bot_token")
    chat_id = current_app_settings.get("app_telegram_chat_id")
    notify_on_trade = current_app_settings.get("app_notify_on_trade", True)

    if not notify_on_trade: # Boolean check, fine as is
        print(f"{log_prefix} Telegram notifications for trade execution are disabled. Skipping.")
        return

    if bot_token is None or chat_id is None: # Explicit check for None
        print(f"{log_prefix} Telegram token/chat_id not configured (None). Cannot send trade execution message.")
        return

    s_info = get_app_symbol_info(symbol) # Fetch fresh symbol info for precision
    p_prec = int(s_info.get('pricePrecision', 2)) if s_info is not None else 2 # Check s_info
    q_prec = int(s_info.get('quantityPrecision', 0)) if s_info is not None else 0 # Check s_info


    side_upper = side.upper()
    action_title = "📈 Trade Executed" if mode == "live" else "🎯 Trade Signal Confirmed"
    if order_type.upper() == "LIMIT" and mode == "live" and executed_quantity is None: # Limit order placed, not yet filled
        action_title = "⏳ Limit Order Placed"
    elif order_type.upper() == "LIMIT" and mode == "signal":
        action_title = "🎯 Limit Signal Confirmed"


    message = f"{action_title}\n\n"
    message += f"🪙 Symbol: `{escape_app_markdown_v1(symbol)}`\n"
    message += f"📊 Side: *{escape_app_markdown_v1(side_upper)}*\n"
    if order_type.upper() == "MARKET" or (order_type.upper() == "LIMIT" and executed_quantity is not None):
        message += f"🎯 Entry: `{entry_price:.{p_prec}f}`\n"
    elif order_type.upper() == "LIMIT": # Limit order not yet filled, or signal mode for limit
         message += f"⏳ Limit Entry Target: `{entry_price:.{p_prec}f}`\n"

    message += f"🛡️ SL: `{sl_price:.{p_prec}f}`\n"
    
    if tp1_price is not None:
        message += f"🏁 TP1: `{tp1_price:.{p_prec}f}`\n"
    if tp2_price is not None:
        message += f"🏁 TP2: `{tp2_price:.{p_prec}f}`\n"
    if tp3_price is not None:
        message += f"🏁 TP3: `{tp3_price:.{p_prec}f}`\n"

    if executed_quantity is not None and mode == "live":
        message += f"📦 Quantity: `{executed_quantity:.{q_prec}f}`\n"
    
    if mode == "signal":
        message += f"\nℹ️ _This is a signal notification ({order_type.upper()} type). No live trade was placed by the bot._"
    elif order_type.upper() == "LIMIT" and executed_quantity is None and mode == "live":
        message += f"\nℹ️ _This LIMIT order has been placed. SL/TP will be set upon fill._"


    print(f"{log_prefix} Formatted trade execution message for Telegram.")
    send_app_telegram_message(message)


def app_send_signal_update_telegram(current_app_settings: dict, signal_details: dict, update_type: str, 
                                    message_detail: str, current_market_price: float, 
                                    pnl_estimation_fixed_capital: float | None = None):
    """
    Formats and sends updates on an existing signal (e.g., SL/TP hit) via app.py's sender.
    Mirrors main.py's send_signal_update_telegram formatting.
    `signal_details` should be a dictionary from `app_active_trades` or similar structure.
    """
    log_prefix = f"[AppSendSignalUpdate-{signal_details.get('symbol', 'N/A')}]"
    bot_token = current_app_settings.get("app_telegram_bot_token")
    chat_id = current_app_settings.get("app_telegram_chat_id")

    if bot_token is None or chat_id is None: # Explicit check for None
        print(f"{log_prefix} Telegram token/chat_id not configured (None). Cannot send signal update.")
        return

    symbol = signal_details.get('symbol', 'N/A')
    side = signal_details.get('side', 'N/A')
    entry_price = signal_details.get('entry_price', 0.0) # This should be the actual entry price
    s_info = signal_details.get('symbol_info', {}) # s_info can be an empty dict, that's fine
    p_prec = int(s_info.get('pricePrecision', 2)) if s_info is not None and s_info else 2 # Check s_info is not None and not empty
    strategy_type_display = escape_app_markdown_v1(signal_details.get('strategy_type', "App Signal"))
    escaped_symbol = escape_app_markdown_v1(symbol)
    escaped_update_type = escape_app_markdown_v1(update_type)
    escaped_message_detail = escape_app_markdown_v1(message_detail)

    title_emoji = "⚙️" # Default
    if update_type.startswith("TP"): title_emoji = "✅"
    elif "SL_HIT" in update_type.upper(): title_emoji = "❌" # More robust check
    elif "SL_ADJUSTED" in update_type.upper(): title_emoji = "🛡️"
    elif "CLOSED" in update_type.upper(): title_emoji = "🎉" # For general closures like all TPs hit
    
    pnl_info_str = ""
    if pnl_estimation_fixed_capital is not None:
        pnl_info_str = f"\nEst. P&L ($100 Capital): `{pnl_estimation_fixed_capital:.2f} USDT`"

    message = (
        f"{title_emoji} *SIGNAL UPDATE* ({strategy_type_display}) {title_emoji}\n\n"
        f"Symbol: `{escaped_symbol}` ({side})\n"
        f"Entry: `{entry_price:.{p_prec}f}`\n"
        f"Update Type: `{escaped_update_type}`\n"
        f"Details: _{escaped_message_detail}_\n"
        f"Current Market Price: `{current_market_price:.{p_prec}f}`"
        f"{pnl_info_str}"
    )
    
    # Simple spam prevention (can be enhanced if signal_details is mutable and tracks last message)
    # For now, this function doesn't modify signal_details.
    
    print(f"{log_prefix} Formatted signal update message for Telegram.")
    send_app_telegram_message(message)

def app_send_trade_rejection_notification(current_app_settings: dict, symbol: str, signal_type: str, 
                                          reason: str, entry_price: float | None, sl_price: float | None, 
                                          tp_price: float | None, quantity: float | None, 
                                          symbol_info: dict | None):
    """
    Formats and sends a trade rejection notification via app.py's sender.
    Mirrors main.py's send_trade_rejection_notification formatting.
    """
    log_prefix = f"[AppSendTradeRejection-{symbol}]"
    bot_token = current_app_settings.get("app_telegram_bot_token")
    chat_id = current_app_settings.get("app_telegram_chat_id")
    notify_on_trade = current_app_settings.get("app_notify_on_trade", True) # Check the toggle

    if not notify_on_trade: # Boolean check, fine
        print(f"{log_prefix} Telegram notifications for trade execution are disabled by 'app_notify_on_trade' setting. Skipping.")
        return

    if bot_token is None or chat_id is None: # Explicit check for None
        print(f"{log_prefix} Telegram token/chat_id not configured (None). Cannot send rejection notification.")
        return

    p_prec = int(symbol_info.get('pricePrecision', 2)) if symbol_info is not None else 2 # Check symbol_info
    q_prec = int(symbol_info.get('quantityPrecision', 0)) if symbol_info is not None else 0 # Check symbol_info

    entry_price_str = f"{entry_price:.{p_prec}f}" if entry_price is not None else "N/A"
    sl_price_str = f"{sl_price:.{p_prec}f}" if sl_price is not None else "N/A"
    tp_price_str = f"{tp_price:.{p_prec}f}" if tp_price is not None else "N/A"
    quantity_str = f"{quantity:.{q_prec}f}" if quantity is not None else "N/A"

    escaped_symbol = escape_app_markdown_v1(symbol)
    escaped_signal_type = escape_app_markdown_v1(signal_type)
    escaped_reason = escape_app_markdown_v1(reason)

    message = (
        f"⚠️ TRADE REJECTED (App.py) ⚠️\n\n"
        f"Symbol: `{escaped_symbol}`\n"
        f"Signal Type: `{escaped_signal_type}`\n"
        f"Reason: _{escaped_reason}_\n\n"
        f"*Attempted Parameters:*\n"
        f"Entry: `{entry_price_str}`\n"
        f"SL: `{sl_price_str}`\n"
        f"TP: `{tp_price_str}`\n"
        f"Qty: `{quantity_str}`"
    )

    print(f"{log_prefix} Formatted trade rejection message for Telegram.")
    send_app_telegram_message(message)

def app_calculate_pnl_for_fixed_capital(entry_price: float, exit_price: float, side: str, 
                                        leverage: int, fixed_capital_usdt: float = 100.0, 
                                        symbol_info: dict = None) -> float | None:
    """
    Calculates estimated P&L for a trade based on a fixed capital amount (e.g., $100) for app.py.
    """
    log_prefix = "[AppCalcPnlFixedCap]"
    if entry_price is None or exit_price is None:
        print(f"{log_prefix} Entry price ({entry_price}) or Exit price ({exit_price}) is None.")
        return None
    if not all([isinstance(entry_price, (int,float)) and entry_price > 0, 
                isinstance(exit_price, (int,float)) and exit_price > 0, 
                leverage > 0, fixed_capital_usdt > 0]):
        print(f"{log_prefix} Invalid inputs (entry_price:{entry_price}, exit_price:{exit_price}, leverage:{leverage}, capital:{fixed_capital_usdt})")
        return None
    if side.upper() not in ["LONG", "SHORT"]:
        print(f"{log_prefix} Invalid side '{side}'")
        return None
    if entry_price == exit_price:
        return 0.0

    position_value_usdt = fixed_capital_usdt * leverage
    quantity_base_asset = position_value_usdt / entry_price # Ideal quantity

    if symbol_info is not None: # Explicit check for None
        q_prec = int(symbol_info.get('quantityPrecision', 8))
        lot_size_filter = next((f for f in symbol_info.get('filters', []) if f.get('filterType') == 'LOT_SIZE'), None)
        
        min_qty_val = 0.0
        if lot_size_filter is not None: # Check lot_size_filter
            min_qty_val = float(lot_size_filter.get('minQty', 0.0))
            step_size = float(lot_size_filter.get('stepSize', 0.0))
            if step_size > 0:
                 quantity_base_asset = math.floor(quantity_base_asset / step_size) * step_size
        
        quantity_base_asset = round(quantity_base_asset, q_prec)
        
        if quantity_base_asset < min_qty_val and min_qty_val > 0 : # If calculated qty is less than min_qty, PNL is effectively 0 for this estimation
            print(f"{log_prefix} Calculated quantity {quantity_base_asset} for ${fixed_capital_usdt} is less than min_qty {min_qty_val} for {symbol_info.get('symbol', 'N/A')}. Estimating PNL as 0.")
            return 0.0
    
    if quantity_base_asset == 0:
        print(f"{log_prefix} Calculated quantity for ${fixed_capital_usdt} is zero for {symbol_info.get('symbol', 'N/A') if symbol_info is not None else 'N/A'}. PNL is 0.") # Check symbol_info here too
        return 0.0

    pnl = 0.0
    if side.upper() == "LONG":
        pnl = (exit_price - entry_price) * quantity_base_asset
    elif side.upper() == "SHORT":
        pnl = (entry_price - exit_price) * quantity_base_asset
    
    return pnl

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

# --- ML Model File Utilities ---
def check_ml_models_exist():
    """Checks if the ML model files specified in app_settings exist."""
    global app_settings
    pivot_model_path = app_settings.get("app_pivot_model_path", "app_pivot_model.joblib")
    entry_model_path = app_settings.get("app_entry_model_path", "app_entry_model.joblib")
    models_params_path = app_settings.get("app_model_params_path", "best_model_params.json") # Corrected default

    pivot_exists = os.path.exists(pivot_model_path)
    entry_exists = os.path.exists(entry_model_path)
    params_exist = os.path.exists(models_params_path) # Also check for model params file

    if pivot_exists and entry_exists and params_exist:
        print(f"ML Models and params found: Pivot='{pivot_model_path}', Entry='{entry_model_path}', Params='{models_params_path}'")
        return True, pivot_model_path, entry_model_path, models_params_path
    else:
        missing_files = []
        if not pivot_exists: missing_files.append(pivot_model_path)
        if not entry_exists: missing_files.append(entry_model_path)
        if not params_exist: missing_files.append(models_params_path)
        print(f"ML Model(s) or params file NOT found. Missing: {', '.join(missing_files)}")
        return False, pivot_model_path, entry_model_path, models_params_path

# --- End ML Model File Utilities ---

# --- Logging Utilities ---
APP_DECISION_LOG_FILE = "app_decision_log.csv"
app_decision_log_entries = [] # Global list to store log entries
app_decision_log_lock = threading.Lock()

def append_decision_log(log_entry: dict):
    """Appends a structured log entry to the global list."""
    global app_decision_log_entries, app_decision_log_lock
    with app_decision_log_lock:
        log_entry['timestamp_utc'] = pd.Timestamp.now(tz='UTC').isoformat()
        app_decision_log_entries.append(log_entry)

def save_decision_log_to_csv(force_save=False):
    """Saves the collected decision log entries to a CSV file.
    Can be forced or save based on buffer size."""
    global app_decision_log_entries, app_decision_log_lock, APP_DECISION_LOG_FILE
    
    # Define a buffer size, e.g., save every 100 entries or if forced
    LOG_BUFFER_SIZE_TO_SAVE = 100 
    log_prefix_decision = "[AppDecisionLog]" # Logger prefix
    
    with app_decision_log_lock:
        if not app_decision_log_entries:
            return # Nothing to save

        if not force_save and len(app_decision_log_entries) < LOG_BUFFER_SIZE_TO_SAVE:
            return # Wait for more entries unless forced

        try:
            log_df = pd.DataFrame(app_decision_log_entries)
            if log_df.empty: # Should be caught by app_decision_log_entries check, but safeguard
                print(f"{log_prefix_decision} Decision log DataFrame is empty after creation. Skipping save.")
                app_decision_log_entries.clear() # Clear if it was somehow non-empty but produced empty df
                return

            print(f"{log_prefix_decision} Attempting to write decision log. Shape: {log_df.shape}. Entries: {len(app_decision_log_entries)}. Path: {APP_DECISION_LOG_FILE}")
            
            file_exists = os.path.exists(APP_DECISION_LOG_FILE)
            log_df.to_csv(APP_DECISION_LOG_FILE, mode='a' if file_exists else 'w', header=not file_exists, index=False)
            
            print(f"{log_prefix_decision} INFO: Wrote {len(app_decision_log_entries)} entries to {APP_DECISION_LOG_FILE}")
            app_decision_log_entries.clear() # Clear after saving
        except Exception as e:
            print(f"{log_prefix_decision} ERROR: Failed writing decision log {APP_DECISION_LOG_FILE}: {e}")
# --- End Logging Utilities ---


# --- Live Feature Calculation Functions ---
def app_calculate_live_pivot_features(df_live: pd.DataFrame, atr_period: int, pivot_feature_names: list, current_app_settings: dict):
    """
    Calculates features for the pivot detection model on live data.
    `df_live` should be a DataFrame of historical klines ending with the current (or last closed) candle.
    `atr_period` must match the period used during training.
    `pivot_feature_names` is the list of feature names the model expects.
    `current_app_settings` provides access to app configurations if needed (e.g. for logging or specific parameters).
    Returns a pd.Series of features for the latest candle, or None if an error occurs.
    """
    log_prefix = "[AppLivePivotFeatures]"
    if df_live.empty or len(df_live) < atr_period + 50: # Min data for ATR, EMAs, rolling features
        print(f"{log_prefix} Insufficient data ({len(df_live)} candles) for live pivot feature calculation. Need at least {atr_period + 50}.")
        return None

    df = df_live.copy() # Work on a copy

    # ATR Calculation (consistent with app.py's training method)
    # The `calculate_atr` function in app.py uses rolling().mean() for TR.
    # It adds a column like 'atr_14' if period is 14.
    df = calculate_atr(df, period=atr_period) # This should add f'atr_{atr_period}'
    atr_col_name = f'atr_{atr_period}'

    if atr_col_name not in df.columns or df[atr_col_name].iloc[-1] is None or pd.isna(df[atr_col_name].iloc[-1]):
        print(f"{log_prefix} ATR calculation failed or resulted in NaN for the latest candle. ATR Column: '{atr_col_name}'.")
        # Attempt to log more details about the ATR series end
        if atr_col_name in df.columns:
            print(f"{log_prefix} Last few ATR values: {df[atr_col_name].tail().to_dict()}")
        else:
            print(f"{log_prefix} ATR column '{atr_col_name}' was not even created.")
        return None
    
    # Ensure ATR is not zero to prevent division by zero errors for normalized features
    if df[atr_col_name].iloc[-1] == 0:
        print(f"{log_prefix} Warning: ATR for the latest candle is zero. Normalizing features might lead to inf. Using a small epsilon for ATR.")
        # Replace zero ATR with a very small number to avoid division by zero
        # This should only affect the last row for feature calculation.
        # A more robust way might be to check and adjust df[atr_col_name].iloc[-1] before it's used in divisions.
        # However, feature engineering functions below might use the whole column.
        # For safety, let's ensure the last ATR value used in division is non-zero.
        # The functions like engineer_pivot_features might need to be aware of this.
        # The current engineer_pivot_features in app.py uses the atr_col_name directly.
        # We will rely on it to handle or propagate NaNs/Infs if ATR is zero.

    # Re-call app.py's engineer_pivot_features function.
    # It's designed to add features to the DataFrame.
    # We need to ensure it's using the correct atr_col_name.
    # The existing `engineer_pivot_features` in `app.py` takes `atr_col_name` as an argument.
    
    # The global PIVOT_N_LEFT, PIVOT_N_RIGHT are used by prune_and_label_pivots,
    # which is part of the training data generation. For live feature calculation for a *pre-trained model*,
    # we only need to generate the input features the model expects.
    # The `engineer_pivot_features` function calculates features like 'bars_since_last_pivot'.
    # This specific feature's live calculation needs careful consideration if it relied on labels.
    # Looking at app.py's `engineer_pivot_features`:
    # `bars_since_last_pivot` is calculated based on `is_swing_high` or `is_swing_low` columns.
    # These columns are typically generated by `prune_and_label_pivots` during training.
    # For live data, these labels won't exist.
    # Solution: `engineer_pivot_features` needs to be adapted or a live-specific version created
    # that does not depend on pre-existing labels for features like `bars_since_last_pivot`.
    # For now, let's assume `engineer_pivot_features` can run and we'll address `bars_since_last_pivot` if it causes issues.
    # A simple fix for 'bars_since_last_pivot' if labels are missing: set it to 0 or a high number.
    # Add dummy label columns if engineer_pivot_features strictly requires them, but they won't be used for actual labeling.
    # The modified engineer_pivot_features will handle this internally.
    # if 'is_swing_high' not in df.columns: df['is_swing_high'] = 0 # No longer needed here
    # if 'is_swing_low' not in df.columns: df['is_swing_low'] = 0 # No longer needed here
    
    df_with_features, calculated_feature_names = engineer_pivot_features(df, atr_col_name=atr_col_name)
    
    # Verify that the `calculated_feature_names` from `engineer_pivot_features` match the expected `pivot_feature_names`.
    # This is a sanity check. The `pivot_feature_names` list (from best_hyperparams) is the source of truth.

    # Select the required features for the latest candle
    live_features_series = df_with_features.iloc[-1][pivot_feature_names].copy()
    
    # Replace Inf values (e.g., from division by zero if ATR was zero and not handled robustly)
    live_features_series = live_features_series.replace([np.inf, -np.inf], np.nan) # Addressed FutureWarning

    if live_features_series.isnull().any():
        print(f"{log_prefix} NaN values found in live pivot features for the latest candle:")
        print(live_features_series[live_features_series.isnull()])
        # Impute NaNs with -1 (consistent with Optuna trial feature preparation in app.py)
        live_features_series.fillna(-1, inplace=True)
        print(f"{log_prefix} NaNs filled with -1.")

    print(f"{log_prefix} Successfully calculated live pivot features for the latest candle.")
    
    # --- Enhanced Logging for Final Selected Live Features ---
    print(f"\n--- Final Selected Live Pivot Feature Values (Series sent to model) ---")
    if live_features_series is not None and not live_features_series.empty:
        for idx, val in live_features_series.items():
            print(f"  {idx}: {val:.4f}")
    else:
        print("  Live features series is None or empty.")
    print(f"--- End Final Selected Live Pivot Feature Values ---\n")
    # --- End Enhanced Logging ---

    return live_features_series

def app_calculate_live_entry_features(df_live: pd.DataFrame, atr_period: int, entry_feature_names_base: list,
                                      p_swing_score: float, simulated_entry_price: float, simulated_sl_price: float,
                                      pivot_price_for_dist: float, current_app_settings: dict):
    """
    Calculates features for the entry evaluation model on live data.
    `df_live` should be klines ending with the current (pivot) candle.
    `atr_period` must match training.
    `entry_feature_names_base` are base features the model expects.
    `p_swing_score`, `simulated_entry_price`, `simulated_sl_price`, `pivot_price_for_dist` are contextual.
    Returns a pd.Series of features for the current context, or None.
    """
    log_prefix = "[AppLiveEntryFeatures]"
    if df_live.empty or len(df_live) < atr_period + 50: # Min data for ATR, EMAs, etc.
        print(f"{log_prefix} Insufficient data ({len(df_live)} candles) for live entry feature calculation. Need {atr_period + 50}.")
        return None

    df = df_live.copy()
    atr_col_name = f'atr_{atr_period}'

    # Ensure ATR column from pivot feature calculation is present or recalculate
    if atr_col_name not in df.columns or df[atr_col_name].iloc[-1] is None or pd.isna(df[atr_col_name].iloc[-1]):
        print(f"{log_prefix} ATR column '{atr_col_name}' missing or NaN. Attempting recalculation for entry features.")
        df = calculate_atr(df, period=atr_period) # Use app.py's version
        if atr_col_name not in df.columns or df[atr_col_name].iloc[-1] is None or pd.isna(df[atr_col_name].iloc[-1]):
            print(f"{log_prefix} Critical: ATR recalculation failed or resulted in NaN. Cannot proceed.")
            return None

    current_atr_value = df[atr_col_name].iloc[-1]
    if current_atr_value == 0:
        print(f"{log_prefix} Warning: Current ATR for entry features is zero. Using small epsilon.")
        current_atr_value = 1e-9 # Avoid division by zero

    # Add contextual features to the last row of the DataFrame (current candle)
    # These are calculated based on the current event/pivot.
    last_candle_idx = df.index[-1]
    df.loc[last_candle_idx, 'P_swing'] = p_swing_score
    
    # Normalized distance features
    # norm_dist_entry_pivot = (simulated_entry_price - pivot_price_for_dist) / ATR
    # norm_dist_entry_sl = abs(simulated_entry_price - simulated_sl_price) / ATR
    # Ensure pivot_price_for_dist is not None
    if pivot_price_for_dist is None:
        print(f"{log_prefix} pivot_price_for_dist is None. Cannot calculate norm_dist_entry_pivot.")
        return None # Critical feature missing

    df.loc[last_candle_idx, 'norm_dist_entry_pivot'] = (simulated_entry_price - pivot_price_for_dist) / current_atr_value
    df.loc[last_candle_idx, 'norm_dist_entry_sl'] = abs(simulated_entry_price - simulated_sl_price) / current_atr_value

    # Call app.py's engineer_entry_features to calculate the base features
    # It takes `entry_features_base_list_arg` which should be `entry_feature_names_base`
    df_with_base_features, calculated_base_feature_names = engineer_entry_features(
        df, 
        atr_col_name=atr_col_name,
        entry_features_base_list_arg=entry_feature_names_base
    )

    # The full list of features the entry model expects
    # (base features + P_swing + normalized distances)
    full_entry_feature_list = entry_feature_names_base + ['P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl']
    
    # Ensure all features in full_entry_feature_list are present in df_with_base_features.
    # engineer_entry_features adds its columns. P_swing, norm_dist_... were added above.
    missing_cols = [col for col in full_entry_feature_list if col not in df_with_base_features.columns]
    if missing_cols:
        print(f"{log_prefix} Error: The following expected entry features are missing from the DataFrame after calculation: {missing_cols}")
        # This indicates a discrepancy between `entry_feature_names_base` (from params) and what `engineer_entry_features` produced,
        # or an issue with adding the contextual features.
        return None

    # Select the full feature set for the latest candle
    live_features_series = df_with_base_features.iloc[-1][full_entry_feature_list].copy()

    # Replace Inf values
    live_features_series = live_features_series.replace([np.inf, -np.inf], np.nan) # Addressed FutureWarning

    if live_features_series.isnull().any():
        print(f"{log_prefix} NaN values found in live entry features for the latest candle:")
        print(live_features_series[live_features_series.isnull()])
        live_features_series.fillna(-1, inplace=True) # Impute with -1
        print(f"{log_prefix} NaNs filled with -1.")
    
    print(f"{log_prefix} Successfully calculated live entry features.")
    return live_features_series

# --- Symbol Loading Utilities ---
def app_load_symbols_from_csv(filepath: str) -> list[str]:
    """Loads symbols from a CSV file, expecting a single 'symbol' column."""
    log_prefix = "[AppLoadSymbols]"
    if not os.path.exists(filepath):
        print(f"{log_prefix} Info: Symbol CSV file '{filepath}' not found.")
        return []
    try:
        df = pd.read_csv(filepath)
        if 'symbol' not in df.columns:
            print(f"{log_prefix} Error: Symbol CSV file '{filepath}' must contain a 'symbol' column.")
            return []
        
        symbols_list = sorted(list(set(df['symbol'].dropna().astype(str).str.upper().tolist())))
        
        if not symbols_list:
            print(f"{log_prefix} Info: Symbol CSV file '{filepath}' is empty or contains no valid symbols.")
            return []
        print(f"{log_prefix} Loaded {len(symbols_list)} unique symbols from '{filepath}'.")
        return symbols_list
    except pd.errors.EmptyDataError:
        print(f"{log_prefix} Info: Symbol CSV file '{filepath}' is empty.")
        return []
    except Exception as e:
        print(f"{log_prefix} Error loading symbols from CSV '{filepath}': {e}")
        return []

# --- End Symbol Loading Utilities ---

def app_process_symbol_for_signal(symbol: str, client, current_app_settings: dict, 
                                  pivot_model_loaded, entry_model_loaded, current_best_hyperparams: dict):
    """
    Processes a single symbol to detect trading signals using loaded ML models.
    Fetches data, calculates features, gets predictions, and then either sends
    a Telegram notification (signal mode) or executes a trade (live mode).
    """
    log_prefix = f"[AppProcessSymbol-{symbol}]"
    print(f"{log_prefix} Starting processing for symbol.")

    if not pivot_model_loaded or not entry_model_loaded:
        print(f"{log_prefix} Models not loaded. Skipping.")
        return

    if not current_best_hyperparams:
        print(f"{log_prefix} Best hyperparameters not loaded. Skipping.")
        return

    # --- Parameters from best_hyperparams and app_settings ---
    if current_best_hyperparams is None: # Explicit check for None
        print(f"{log_prefix} Best hyperparameters not loaded (None). Skipping.")
        return

    atr_period_for_features = current_best_hyperparams.get('model_training_atr_period_used')
    pivot_feature_names_list = current_best_hyperparams.get('pivot_feature_names_used')
    entry_feature_names_base_list = current_best_hyperparams.get('entry_feature_names_base_used')
    
    # Dynamic threshold is the p_swing_threshold from Optuna/best_params.json
    dynamic_pivot_threshold = current_best_hyperparams.get('p_swing_threshold')
    # Fixed threshold is from app_settings.json
    fixed_pivot_threshold = current_app_settings.get('app_fixed_pivot_threshold', 0.7) # Default if not in settings
    use_dynamic_threshold_mode = current_app_settings.get('app_use_dynamic_threshold', True) # Default to dynamic

    p_profit_thresh = current_best_hyperparams.get('profit_threshold') # For entry model

    # Validate dynamic_pivot_threshold
    if dynamic_pivot_threshold is None or not (0.0 < dynamic_pivot_threshold < 1.0):
        print(f"{log_prefix} WARNING: Dynamic pivot threshold (p_swing_threshold from best_params.json: {dynamic_pivot_threshold}) is missing or invalid.")
        if use_dynamic_threshold_mode:
            print(f"{log_prefix} Dynamic mode was enabled but dynamic threshold is invalid. Falling back to fixed threshold: {fixed_pivot_threshold}.")
            # Fallback to fixed threshold if dynamic was selected but invalid
            p_swing_thresh_to_use = fixed_pivot_threshold
            threshold_mode_in_use = f"Fixed (Fallback from invalid Dynamic: {dynamic_pivot_threshold})"
            # Log the original dynamic threshold for reference if it was just out of range but not None
            reference_threshold_val = dynamic_pivot_threshold if dynamic_pivot_threshold is not None else "N/A (Missing)"
            reference_threshold_type = "Dynamic (Invalid)"
        else: # Fixed mode was already selected, dynamic threshold invalidity is just for logging comparison
            p_swing_thresh_to_use = fixed_pivot_threshold
            threshold_mode_in_use = "Fixed"
            reference_threshold_val = dynamic_pivot_threshold if dynamic_pivot_threshold is not None else "N/A (Missing)"
            reference_threshold_type = "Dynamic (Invalid)"
    elif use_dynamic_threshold_mode:
        p_swing_thresh_to_use = dynamic_pivot_threshold
        threshold_mode_in_use = "Dynamic"
        reference_threshold_val = fixed_pivot_threshold
        reference_threshold_type = "Fixed"
    else: # Using fixed threshold mode, and dynamic_pivot_threshold is valid (for comparison logging)
        p_swing_thresh_to_use = fixed_pivot_threshold
        threshold_mode_in_use = "Fixed"
        reference_threshold_val = dynamic_pivot_threshold
        reference_threshold_type = "Dynamic"

    print(f"{log_prefix} Pivot Threshold Selection: Mode Active='{threshold_mode_in_use}', Threshold Value Used={p_swing_thresh_to_use:.3f}")
    print(f"{log_prefix}   Reference Threshold: Type='{reference_threshold_type}', Value={reference_threshold_val if isinstance(reference_threshold_val, str) else f'{reference_threshold_val:.3f}'}")

    # Check for None or empty for list parameters
    missing_params = False
    if atr_period_for_features is None:
        print(f"{log_prefix} Missing: model_training_atr_period_used")
        missing_params = True
    if pivot_feature_names_list is None or not pivot_feature_names_list: # Check for None or empty list
        print(f"{log_prefix} Missing or empty: pivot_feature_names_used")
        missing_params = True
    if entry_feature_names_base_list is None or not entry_feature_names_base_list: # Check for None or empty list
        print(f"{log_prefix} Missing or empty: entry_feature_names_base_used")
        missing_params = True
    if p_profit_thresh is None:
        print(f"{log_prefix} Missing: profit_threshold (for entry model)")
        missing_params = True
    
    if missing_params:
        print(f"{log_prefix} Critical parameters missing or empty from best_hyperparams or app_settings. Skipping.")
        return

    # --- Data Fetching ---
    # Determine kline interval (e.g., from app_settings or fixed based on training)
    # Assuming training used 15-minute interval as per app.py's get_processed_data_for_symbol defaults
    kline_interval_live = current_app_settings.get("app_trading_kline_interval", Client.KLINE_INTERVAL_15MINUTE) # Add this setting
    
    # Fetch enough klines for feature calculations (e.g., ATR period + EMA periods + buffer)
    # A common need is around 200-250 candles for 15-min interval.
    # Max lookback for features in app.py: EMA50, rolling volume 20, ATR period (e.g. 14-20).
    # So, atr_period + 50 (for EMA50) + some buffer = e.g., 20 + 50 + 30 = 100. Let's use more to be safe.
    num_klines_to_fetch = atr_period_for_features + 50 + 150 # ~200-250 depending on ATR period
    
    print(f"{log_prefix} Fetching {num_klines_to_fetch} klines at {kline_interval_live} for {symbol}...")
    try:
        # Assuming get_historical_bars is suitable for fetching live-ish data by not passing end_str
        df_live_raw = get_historical_bars(symbol, kline_interval_live, start_str=f"{num_klines_to_fetch + 5} days ago UTC") # Fetch a bit more initially
        if df_live_raw.empty:
            print(f"{log_prefix} No kline data returned for {symbol}. Skipping.")
            return
        # Ensure we have exactly num_klines_to_fetch of the most recent data
        df_live_raw = df_live_raw.iloc[-num_klines_to_fetch:]
        if len(df_live_raw) < num_klines_to_fetch * 0.9: # If significantly less data than requested
             print(f"{log_prefix} Insufficient klines ({len(df_live_raw)}/{num_klines_to_fetch}) after fetch for {symbol}. Skipping.")
             return

    except Exception as e:
        print(f"{log_prefix} Error fetching kline data for {symbol}: {e}")
        return
    
    print(f"{log_prefix} Kline data fetched. Length: {len(df_live_raw)}. Last candle time: {df_live_raw.index[-1]}")

    # --- Stage 1: Pivot Model Prediction ---
    live_pivot_features = app_calculate_live_pivot_features(df_live_raw, atr_period_for_features, pivot_feature_names_list, current_app_settings)
    if live_pivot_features is None:
        print(f"{log_prefix} Failed to calculate live pivot features. Skipping.")
        return

    try:
        # Convert Series to DataFrame with named columns for robust prediction
        live_pivot_df = pd.DataFrame([live_pivot_features.to_dict()])
        pivot_probs = pivot_model_loaded.predict_proba(live_pivot_df)[0]
        # Assuming class 0: none, 1: high, 2: low (from app.py's pivot_label)
        p_swing_high = pivot_probs[1] if len(pivot_probs) > 1 else 0
        p_swing_low = pivot_probs[2] if len(pivot_probs) > 2 else 0
        p_swing_score_live = max(p_swing_high, p_swing_low)
        predicted_pivot_class_live = np.argmax(pivot_probs)
        p_none_pivot = pivot_probs[0] if len(pivot_probs) > 0 else 0
    except Exception as e:
        print(f"{log_prefix} Error during pivot model prediction: {e}")
        return

    print(f"{log_prefix} Pivot Model Output: P_None={p_none_pivot:.3f}, P_High={p_swing_high:.3f}, P_Low={p_swing_low:.3f} -> P_Swing_Score={p_swing_score_live:.3f} (Threshold: {p_swing_thresh_to_use:.2f}), PredictedClass={predicted_pivot_class_live}")

    # Detailed log of features fed to pivot model
    if live_pivot_features is not None:
        feature_log_str = f"{log_prefix} Features for Pivot Model ({symbol} @ {df_live_raw.index[-1]}):\n"
        for feature_name, value in live_pivot_features.items():
            feature_log_str += f"  - {feature_name}: {value:.4f}\n"
        print(feature_log_str)
    else:
        print(f"{log_prefix} live_pivot_features is None, cannot log them.")

    # Log the decision based on the *chosen* threshold
    pivot_decision_log_message = f"Pivot Score: {p_swing_score_live:.3f}. Threshold Used ({threshold_mode_in_use}): {p_swing_thresh_to_use:.3f}. "
    
    # Determine if trade would be taken under dynamic and fixed for logging
    # Ensure dynamic_pivot_threshold and fixed_pivot_threshold are valid floats for comparison
    # If dynamic_pivot_threshold was invalid, its comparison result is not meaningful for "would it have taken trade"
    
    # Dynamic perspective
    would_trade_dynamic = False
    if dynamic_pivot_threshold is not None and (0.0 < dynamic_pivot_threshold < 1.0): # Only if valid dynamic threshold
        if p_swing_score_live >= dynamic_pivot_threshold and predicted_pivot_class_live != 0:
            would_trade_dynamic = True
    
    # Fixed perspective (fixed_pivot_threshold is validated during load_app_settings)
    would_trade_fixed = False
    if p_swing_score_live >= fixed_pivot_threshold and predicted_pivot_class_live != 0:
        would_trade_fixed = True

    pivot_decision_log_message += f"Trade Signal (Dynamic: {'YES' if would_trade_dynamic else 'NO'}, Fixed: {'YES' if would_trade_fixed else 'NO'})."


    if p_swing_score_live < p_swing_thresh_to_use or predicted_pivot_class_live == 0:
        print(f"{log_prefix} No significant pivot based on {threshold_mode_in_use} threshold. {pivot_decision_log_message}")
        return
    
    # If pivot passes, log the full decision message
    print(f"{log_prefix} Pivot passes {threshold_mode_in_use} threshold. {pivot_decision_log_message}")
    append_decision_log({
        "symbol": symbol, "event_type": "PIVOT_MODEL_PASS",
        "p_swing_score": p_swing_score_live, "threshold_used": p_swing_thresh_to_use,
        "predicted_pivot_class": predicted_pivot_class_live, "details": pivot_decision_log_message
    })

    # --- Volume Spike Check ---
    min_volume_spike_ratio = current_app_settings.get("app_min_volume_spike_ratio", 1.5) # Default 1.5 if not set
    volume_spike_feature_name = 'volume_spike_vs_avg' # This must match the name in PIVOT_FEATURE_NAMES and live_pivot_features
    
    current_volume_spike_value = np.nan # Default to NaN
    if live_pivot_features is not None and volume_spike_feature_name in live_pivot_features:
        current_volume_spike_value = live_pivot_features[volume_spike_feature_name]
    
    if pd.isna(current_volume_spike_value):
        print(f"{log_prefix} Volume spike value is NaN. Skipping volume check or treating as insufficient.")
        # Depending on strictness, could reject here. For now, let's say if it's NaN, it doesn't pass.
        # Or, if a very low default (e.g. 0) is used for min_volume_spike_ratio, NaN < 0 would be false.
        # Let's assume NaN fails the check if min_volume_spike_ratio > 0
        if min_volume_spike_ratio > 0: # Only fail if a positive threshold is expected
            print(f"{log_prefix} Signal REJECTED due to NaN volume spike value (threshold: {min_volume_spike_ratio}).")
            append_decision_log({"symbol": symbol, "event_type": "REJECT_VOLUME_SPIKE_NAN", "min_ratio": min_volume_spike_ratio})
            return # Reject signal
    elif current_volume_spike_value < min_volume_spike_ratio:
        print(f"{log_prefix} Signal REJECTED: Volume spike ({current_volume_spike_value:.2f}) is below threshold ({min_volume_spike_ratio}).")
        append_decision_log({"symbol": symbol, "event_type": "REJECT_VOLUME_SPIKE_LOW", "volume_spike": current_volume_spike_value, "min_ratio": min_volume_spike_ratio})
        # Optionally send a specific Telegram rejection message for volume
        # app_send_trade_rejection_notification(...)
        return # Reject signal
    
    print(f"{log_prefix} Volume Spike Check PASSED: Value {current_volume_spike_value:.2f} >= Threshold {min_volume_spike_ratio}")
    append_decision_log({"symbol": symbol, "event_type": "PASS_VOLUME_SPIKE", "volume_spike": current_volume_spike_value, "min_ratio": min_volume_spike_ratio})
    # --- End Volume Spike Check ---

    # --- Debug message for Pivot Signal Passed ---
    print(f"{log_prefix} ✅ Pivot Signal Passed (Score & Volume) - PScore: {p_swing_score_live:.3f}, VolSpike: {current_volume_spike_value:.2f}, Class: {predicted_pivot_class_live}")
    
    # --- Perform Additional Pre-Checks (RSI, etc.) ---
    # trade_side_potential is "long" or "short"
    preconditions_ok, reject_reason = check_signal_preconditions(
        symbol, 
        trade_side_potential,  # "long" or "short"
        df_live_raw, # Full klines DataFrame for RSI calculation
        current_app_settings, 
        live_pivot_features # Pass the already calculated pivot features
    )

    if not preconditions_ok:
        print(f"{log_prefix} Signal REJECTED by pre-conditions: {reject_reason}")
        if current_app_settings.get('app_reject_notify_telegram', True):
            send_app_telegram_message(
                f"⚠️ [REJECT] {symbol} {trade_side_potential.upper()} – Reason: {escape_app_markdown_v1(reject_reason)}"
            )
        append_decision_log({"symbol": symbol, "event_type": "REJECT_PRECONDITION", "side": trade_side_potential, "reason": reject_reason})
        return # Reject signal
    print(f"{log_prefix} Additional Pre-checks (RSI) PASSED.")
    append_decision_log({"symbol": symbol, "event_type": "PASS_PRECONDITION", "side": trade_side_potential, "reason": "RSI OK"})
    # --- End Additional Pre-Checks ---

    # Optional: Send Telegram alert for pivot detection (for testing, can be removed or made configurable)
    # send_app_telegram_message(f"⚠️ DEBUG PIVOT: {symbol} Pivot detected. Score: {p_swing_score_live:.3f} (Class: {predicted_pivot_class_live})")

    # --- Pivot Detected, Proceed to Entry Model ---
    current_candle = df_live_raw.iloc[-1]
    pivot_price_sim = current_candle['high'] if predicted_pivot_class_live == 1 else current_candle['low'] # High for short, Low for long
    trade_side_potential = "short" if predicted_pivot_class_live == 1 else "long"
    print(f"{log_prefix} Pivot detected: {trade_side_potential.upper()} setup. PivotPrice_Sim={pivot_price_sim}, P_Swing={p_swing_score_live:.3f}")

    # Simulate entry/SL for entry feature calculation
    # This simulation is only for generating features, not actual trade parameters yet.
    # Uses ATR at the current candle.
    atr_val_current = df_live_raw[f'atr_{atr_period_for_features}'].iloc[-1]
    if pd.isna(atr_val_current) or atr_val_current == 0:
        print(f"{log_prefix} Invalid ATR ({atr_val_current}) for simulating entry/SL for features. Skipping.")
        return
        
    # Simplified simulation for feature calculation (e.g., entry at close, SL is ATR based from pivot_price_sim)
    # This might differ from actual trade SL calculation later.
    # The objective_optuna uses specific fib-based simulation. Here, we need a generic way for live.
    # Let's use a simple ATR-based SL from the pivot price for feature input.
    # `model_best_params` might have `sl_atr_multiplier_opt` or similar if SL calculation was tuned.
    # For now, use a generic multiplier or one from general app_settings if available.
    # `app.py`'s `objective_optuna` uses `simulate_fib_entries` which is complex.
    # `main.py`'s `process_symbol_adv_fib_ml_task` uses `sim_sl_dist_feat = atr_val_for_sim_features * sl_mult_feat`
    # where `sl_mult_feat` is `configs.get("fib_sl_atr_multiplier_exec")`.
    # Let's assume `app_settings` or `best_hyperparams` has a suitable `sim_sl_atr_multiplier`.
    sim_sl_atr_mult = current_app_settings.get('app_sim_sl_atr_multiplier', 1.0) # Add to settings if needed
    
    sim_entry_for_features = current_candle['close']
    sim_sl_dist_for_features = atr_val_current * sim_sl_atr_mult
    sim_sl_for_features = (pivot_price_sim - sim_sl_dist_for_features) if trade_side_potential == "long" else (pivot_price_sim + sim_sl_dist_for_features)
    
    live_entry_features = app_calculate_live_entry_features(
        df_live_raw, atr_period_for_features, entry_feature_names_base_list,
        p_swing_score_live, sim_entry_for_features, sim_sl_for_features,
        pivot_price_sim, current_app_settings
    )

    if live_entry_features is None:
        print(f"{log_prefix} Failed to calculate live entry features. Skipping.")
        return

    try:
        # Convert Series to DataFrame with named columns for robust prediction
        live_entry_df = pd.DataFrame([live_entry_features.to_dict()])
        entry_probs = entry_model_loaded.predict_proba(live_entry_df)[0]
        # Assuming binary classification: class 1 is "profitable"
        p_profit_live = entry_probs[1] if len(entry_probs) > 1 else 0
    except Exception as e:
        print(f"{log_prefix} Error during entry model prediction: {e}")
        return
    
    # Assuming binary classification for entry: class 0 (not profitable), class 1 (profitable)
    p_not_profit_live = entry_probs[0] if len(entry_probs) > 0 else 0
    print(f"{log_prefix} Entry Model Output: P_NotProfit={p_not_profit_live:.3f}, P_Profit={p_profit_live:.3f} (Threshold: {p_profit_thresh:.2f})")

    if p_profit_live < p_profit_thresh:
        print(f"{log_prefix} Entry signal not strong enough (P_Profit {p_profit_live:.3f} < {p_profit_thresh:.2f}).")
        append_decision_log({"symbol": symbol, "event_type": "REJECT_ENTRY_MODEL_LOW_PROFIT", "p_profit": p_profit_live, "threshold": p_profit_thresh})
        return

    # --- Entry Confirmed by Both Models ---
    print(f"{log_prefix} ✅ ENTRY SIGNAL CONFIRMED: {trade_side_potential.upper()} for {symbol}. P_Swing={p_swing_score_live:.2f}, P_Profit={p_profit_live:.2f}")
    append_decision_log({
        "symbol": symbol, "event_type": "ENTRY_MODEL_PASS", 
        "p_swing": p_swing_score_live, "p_profit": p_profit_live, 
        "side": trade_side_potential
    })

    # Determine actual entry price, SL, TP for trade/signal
    # Entry: Current market price (last close for simulation, or fetch live ticker for live)
    actual_entry_price = current_candle['close'] # For now, use close of the signal candle
    
    # SL: Could be ATR-based from the `pivot_price_sim` or from `actual_entry_price`.
    # Let's use `pivot_price_sim` as the anchor for SL, consistent with how features might interpret it.
    # The multiplier for actual SL can come from `app_settings` (e.g., `app_sl_atr_multiplier`)
    actual_sl_atr_mult = current_app_settings.get('app_sl_atr_multiplier', 2.0) # Example setting
    actual_sl_distance = atr_val_current * actual_sl_atr_mult
    actual_sl_price = (pivot_price_sim - actual_sl_distance) if trade_side_potential == "long" else (pivot_price_sim + actual_sl_distance)
    
    # TP: Could be fixed R:R from `actual_sl_price`
    tp_rr_ratio = current_app_settings.get('app_tp_rr_ratio', 1.5) # Example setting
    risk_amount_per_unit = abs(actual_entry_price - actual_sl_price)
    if risk_amount_per_unit == 0: # Should be handled by sanity checks later too
        print(f"{log_prefix} Risk amount is zero. Cannot calculate TP. Skipping.")
        return
        
    actual_tp_price = (actual_entry_price + (risk_amount_per_unit * tp_rr_ratio)) if trade_side_potential == "long" else \
                      (actual_entry_price - (risk_amount_per_unit * tp_rr_ratio))

    # Round prices to symbol precision (fetch from symbol_info if available, or use a default)
    # This should ideally be done by execute_app_trade_signal or before sending telegram.
    # For now, raw calculation.
    # Example: p_prec = get_app_symbol_info(symbol).get('pricePrecision', 2)
    # actual_sl_price = round(actual_sl_price, p_prec)
    # actual_tp_price = round(actual_tp_price, p_prec)

    # --- Perform Action based on Operational Mode ---

    # --- NEW: Fibonacci Pre-Order Logic ---
    use_fib_logic = current_app_settings.get("use_fib_preorder", False)
    fib_proposal_details_for_log = None # For decision log

    if use_fib_logic:
        print(f"{log_prefix} Fibonacci pre-order logic is ENABLED.")
        # 1. Determine Preceding Swing for Fib Leg
        fib_leg_lookback = current_app_settings.get("fib_leg_lookback_candles", 50)
        
        # Data for leg anchor should be before the current pivot candle (df_live_raw.iloc[-1])
        # So, look at df_live_raw up to index -2, then take a tail of fib_leg_lookback.
        if len(df_live_raw) > fib_leg_lookback + 1:
            data_for_leg_anchor = df_live_raw.iloc[-(fib_leg_lookback + 1) : -1]
        else: # Not enough history for full lookback, use what's available before current candle
            data_for_leg_anchor = df_live_raw.iloc[:-1]

        preceding_swing_price = None
        if not data_for_leg_anchor.empty:
            if trade_side_potential == "long": # Current pivot is a low, need preceding high
                preceding_swing_price = data_for_leg_anchor['high'].max()
            else: # Current pivot is a high, need preceding low
                preceding_swing_price = data_for_leg_anchor['low'].min()
        
        if preceding_swing_price is None or pd.isna(preceding_swing_price):
            print(f"{log_prefix} Could not determine valid preceding swing price for Fib leg. Skipping Fib proposal.")
            append_decision_log({"symbol": symbol, "event_type": "FIB_PROPOSAL_SKIP_NO_PRECEDING_SWING", "side": trade_side_potential})
            # Fall through to original logic if Fib proposal cannot be made
            use_fib_logic = False # Temporarily disable for this instance
        else:
            print(f"{log_prefix} Fib Leg Anchors: Pivot_Price_Sim={pivot_price_sim}, Preceding_Swing_Price={preceding_swing_price}")
            
            # 2. Calculate Fibonacci Levels
            # The `direction` for compute_fib_levels refers to the direction of the main leg.
            # If trade_side_potential is "long", the ML pivot was a LOW, so the main leg was UP to a preceding_swing_price (which was a high).
            # So, fib_direction should be "long".
            # If trade_side_potential is "short", ML pivot was a HIGH, main leg was DOWN to a preceding_swing_price (which was a low).
            # So, fib_direction should be "short".
            # This seems correct with compute_fib_levels definition.
            fib_levels = compute_fib_levels(pivot_price_sim, preceding_swing_price, direction=trade_side_potential)

            if fib_levels and "0.618" in fib_levels:
                proposed_limit_price_raw = fib_levels["0.618"]
                s_info_fib = get_app_symbol_info(symbol)
                p_prec_fib = int(s_info_fib.get('pricePrecision', 2)) if s_info_fib else 2
                proposed_limit_price = round(proposed_limit_price_raw, p_prec_fib)

                # Validate proposed_limit_price (must be between pivot_price_sim and preceding_swing_price)
                # For long: pivot_low < proposed_limit < preceding_high
                # For short: preceding_low < proposed_limit < pivot_high
                valid_limit = False
                if trade_side_potential == "long" and pivot_price_sim < proposed_limit_price < preceding_swing_price:
                    valid_limit = True
                elif trade_side_potential == "short" and preceding_swing_price < proposed_limit_price < pivot_price_sim:
                    valid_limit = True
                
                if not valid_limit:
                    print(f"{log_prefix} Proposed Fib limit price {proposed_limit_price} is outside the leg [{min(pivot_price_sim, preceding_swing_price)}, {max(pivot_price_sim, preceding_swing_price)}]. Skipping Fib proposal.")
                    append_decision_log({"symbol": symbol, "event_type": "FIB_PROPOSAL_SKIP_LIMIT_OUT_OF_LEG", "side": trade_side_potential, "limit": proposed_limit_price})
                    use_fib_logic = False # Fall through
                else:
                    proposal_validity_minutes = current_app_settings.get("fib_proposal_validity_minutes", 15)
                    from datetime import datetime, timedelta # Ensure timedelta is available
                    expiry_timestamp = pd.Timestamp.now(tz='UTC') + timedelta(minutes=proposal_validity_minutes)

                    # Store proposal
                    proposal_id = f"fib_{symbol}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S%f')}"
                    proposal_data = {
                        "proposal_id": proposal_id,
                        "symbol": symbol, 
                        "side": trade_side_potential,
                        "proposed_limit_price": proposed_limit_price, 
                        "fib_ratio": 0.618,
                        "pivot_score": p_swing_score_live, 
                        "p_profit_score_initial": p_profit_live, # Store initial P_profit
                        "timestamp_created": pd.Timestamp.now(tz='UTC'), 
                        "valid_until": expiry_timestamp,
                        "pivot_price_anchor": pivot_price_sim,
                        "leg_other_anchor": preceding_swing_price,
                        "original_sl_price": actual_sl_price, # SL based on original ML signal logic
                        "original_tp1_price": actual_tp_price, # TP1 based on original ML signal logic
                        "original_tp2_price": None, # Placeholder for potential multi-TP from original logic
                        "original_tp3_price": None, # Placeholder
                        "status": "PROPOSED",
                        "last_market_price_checked": None, # For secondary condition checker
                        "secondary_p_profit_score": None # For secondary condition checker
                    }
                    
                    with app_fib_proposals_lock:
                        app_fib_proposals[symbol] = proposal_data # Overwrites if symbol already has a proposal
                    
                    print(f"{log_prefix} Stored Fibonacci Limit Proposal for {symbol}: ID {proposal_id}, Limit {proposed_limit_price}")
                    append_decision_log({"symbol": symbol, "event_type": "FIB_PROPOSAL_STORED", "details": proposal_data})
                    
                    # Send Telegram Message for Fib Proposal
                    app_send_fib_proposal_telegram(
                        current_app_settings=current_app_settings,
                        symbol=symbol,
                        direction=trade_side_potential,
                        pivot_price=pivot_price_sim,
                        leg_start_price=min(pivot_price_sim, preceding_swing_price), # For display
                        leg_end_price=max(pivot_price_sim, preceding_swing_price),   # For display
                        proposed_limit_price=proposed_limit_price,
                        fib_ratio_used=0.618,
                        valid_until_timestamp=expiry_timestamp,
                        p_swing_score=p_swing_score_live,
                        p_profit_score_initial=p_profit_live,
                        symbol_info_for_prec=s_info_fib
                    )
                    return # IMPORTANT: Return here to prevent original execution logic
            else:
                print(f"{log_prefix} Failed to calculate 0.618 Fib level. Skipping Fib proposal.")
                append_decision_log({"symbol": symbol, "event_type": "FIB_PROPOSAL_SKIP_LEVEL_CALC_FAIL", "side": trade_side_potential})
                use_fib_logic = False # Fall through

    # --- Original Execution Logic (if Fib logic not used or skipped) ---
    if not use_fib_logic: # This condition is now re-evaluated
        print(f"{log_prefix} Proceeding with original execution logic (Fib pre-order not used or skipped).")
        if current_app_settings.get('app_operational_mode') == 'signal':
            # Send Telegram signal using the new rich wrapper
            symbol_info_for_signal = get_app_symbol_info(symbol) # Fetch for precision and other details
            if not symbol_info_for_signal:
                print(f"{log_prefix} Could not get symbol_info for Telegram message. Sending basic notification.")
                send_app_telegram_message(f"Basic ML Signal: {symbol} {trade_side_potential.upper()} @ {actual_entry_price}, SL {actual_sl_price}, TP {actual_tp_price}")
            else:
                # Placeholder for P&L estimation - ideally, a helper function would calculate this.
                # For now, passing None for est_pnl_tp1 and est_pnl_sl.
                # These would require a fixed capital assumption (e.g., $100) and leverage.
                # Leverage can be taken from current_app_settings.get('app_leverage', 20)
                # Calculate P&L estimations for the signal message
                current_leverage_for_signal = current_app_settings.get('app_leverage', 20)
                fixed_capital_for_est = 100.0 # Standard $100 assumption

                est_pnl_at_tp = app_calculate_pnl_for_fixed_capital(
                    entry_price=actual_entry_price,
                    exit_price=actual_tp_price,
                    side=trade_side_potential.upper(),
                    leverage=current_leverage_for_signal,
                    fixed_capital_usdt=fixed_capital_for_est,
                    symbol_info=symbol_info_for_signal
                )
                est_pnl_at_sl = app_calculate_pnl_for_fixed_capital(
                    entry_price=actual_entry_price,
                    exit_price=actual_sl_price,
                    side=trade_side_potential.upper(),
                    leverage=current_leverage_for_signal,
                    fixed_capital_usdt=fixed_capital_for_est,
                    symbol_info=symbol_info_for_signal
                )

                app_send_entry_signal_telegram(
                    current_app_settings=current_app_settings,
                    symbol=symbol,
                    signal_type_display=f"ML_PIVOT_{trade_side_potential.upper()}", # More specific type
                    leverage=current_leverage_for_signal, 
                    entry_price=actual_entry_price,
                    tp1_price=actual_tp_price, # Assuming single TP from ML strategy for now
                    tp2_price=None,
                    tp3_price=None,
                    sl_price=actual_sl_price,
                    risk_percentage_config=current_app_settings.get('app_risk_percent', 0.01),
                    est_pnl_tp1=est_pnl_at_tp, 
                    est_pnl_sl=est_pnl_at_sl,
                    symbol_info=symbol_info_for_signal,
                    strategy_name_display="App ML Model",
                    signal_timestamp=current_candle.name, # Timestamp of the signal candle
                    signal_order_type="MARKET", # Assuming signal implies market execution
                    p_swing_score=p_swing_score_live,
                    p_profit_score=p_profit_live
                )

    elif current_app_settings.get('app_operational_mode') == 'live':
        print(f"{log_prefix} Executing LIVE trade for {symbol} {trade_side_potential.upper()}.")
        # Ensure client is initialized and available
        if client is None:
            print(f"{log_prefix} Binance client not available for live trade. Skipping.")
            return

        # The execute_app_trade_signal function expects entry_price_target.
        # For market orders, it fetches current price. We can pass our calculated actual_entry_price.
        # It also needs SL, TP1, TP2, TP3. We have one TP.
        execute_app_trade_signal(
            symbol=symbol,
            side=trade_side_potential.upper(),
            sl_price=actual_sl_price,
            tp1_price=actual_tp_price, 
            tp2_price=None, # ML model currently gives one overall profit signal
            tp3_price=None,
            entry_price_target=actual_entry_price, # Pass our calculated entry
            order_type="MARKET" # Assume market execution for now
        )
    else:
        print(f"{log_prefix} Unknown operational mode: {current_app_settings.get('app_operational_mode')}")

    print(f"{log_prefix} Finished processing for symbol.")

# --- Fibonacci Pre-Order Proposal Checker ---
def app_check_fib_proposals(client, current_app_settings: dict, 
                            pivot_model_loaded, entry_model_loaded, # Pass models if needed for re-evaluation
                            current_best_hyperparams: dict):
    """
    Checks active Fibonacci proposals against current market conditions.
    - Manages proposal expiry.
    - Checks for price touch of the proposed limit.
    - If price touched, re-evaluates entry signal using the ML entry model.
    - If all conditions met, triggers trade execution (live or simulated) or sends signal.
    Called periodically by the main trading loop.
    """
    global app_fib_proposals, app_fib_proposals_lock, app_active_trades_lock # app_active_trades used by execute_app_trade_signal

    if not app_fib_proposals:
        return

    log_prefix_main_checker = "[AppFibChecker]"
    print(f"{log_prefix_main_checker} Checking {len(app_fib_proposals)} active Fibonacci proposal(s)...")

    proposals_to_remove = [] # List of symbols whose proposals should be removed
    proposals_to_execute = [] # List of (symbol, proposal_data) for execution

    # Iterate over a copy for safe modification of the original dictionary
    with app_fib_proposals_lock:
        current_proposals_snapshot = list(app_fib_proposals.items())

    for symbol, proposal_data in current_proposals_snapshot:
        log_prefix_checker = f"[AppFibChecker-{symbol}-{proposal_data.get('proposal_id','N/A')}]"
        
        # 1. Expiry Check
        if pd.Timestamp.now(tz='UTC') > proposal_data['valid_until']:
            print(f"{log_prefix_checker} Proposal for {symbol} has EXPIRED at {proposal_data['valid_until']}.")
            proposals_to_remove.append(symbol)
            append_decision_log({"symbol": symbol, "event_type": "FIB_PROPOSAL_EXPIRED", "proposal_id": proposal_data.get('proposal_id')})
            # Optionally send Telegram notification for expiry
            app_send_fib_proposal_update_telegram(current_app_settings, proposal_data, "EXPIRED", 
                                                  f"Proposal expired at {proposal_data['valid_until'].strftime('%Y-%m-%d %H:%M:%S %Z')}.")
            continue

        # 2. Fetch Latest Market Data for Price Touch Check
        try:
            # Fetch last few candles to check high/low for price touch
            # Kline interval for this check should be quick, e.g., 1m or same as proposal trigger (15m)
            # For now, let's use the main trading kline interval for simplicity
            kline_interval_check = current_app_settings.get("app_trading_kline_interval", Client.KLINE_INTERVAL_15MINUTE)
            # Fetch enough for ATR calc if needed by entry features + a few for price touch
            num_klines_for_check = current_best_hyperparams.get('model_training_atr_period_used', 14) + 55 
            
            df_check_raw = get_historical_bars(symbol, kline_interval_check, start_str=f"{num_klines_for_check + 5} days ago UTC")
            if df_check_raw.empty or len(df_check_raw) < 3: # Need at least a few candles
                print(f"{log_prefix_checker} Insufficient kline data for price touch check for {symbol}. Skipping.")
                continue
            
            # Use the most recent closed candles for price touch logic
            # For example, check the high/low of the last closed candle
            last_closed_candle = df_check_raw.iloc[-2] # Second to last is the most recently fully closed
            current_candle_for_touch = df_check_raw.iloc[-1] # Current, potentially incomplete candle

            # Price touch logic:
            price_touched = False
            proposed_limit = proposal_data['proposed_limit_price']
            
            # Check if the proposed limit was touched by the low of the last closed candle OR current candle's low (for longs)
            # or by the high of the last closed candle OR current candle's high (for shorts)
            if proposal_data['side'] == "long":
                if last_closed_candle['low'] <= proposed_limit or current_candle_for_touch['low'] <= proposed_limit:
                    price_touched = True
            elif proposal_data['side'] == "short":
                if last_closed_candle['high'] >= proposed_limit or current_candle_for_touch['high'] >= proposed_limit:
                    price_touched = True
            
            # Update last market price checked in proposal_data (even if not touched)
            # This is a bit tricky as we are modifying a copy.
            # We should update the main app_fib_proposals if we decide to keep the proposal.
            # For now, this field is not critical path.

        except Exception as e_fetch:
            print(f"{log_prefix_checker} Error fetching market data for {symbol}: {e_fetch}. Skipping this check cycle.")
            continue

        if not price_touched:
            # print(f"{log_prefix_checker} Price condition not met for {symbol}. Limit: {proposed_limit}, LastClosedLow: {last_closed_candle['low']}, CurrLow: {current_candle_for_touch['low']} (if long).")
            continue # Move to next proposal

        print(f"{log_prefix_checker} Price condition MET for {symbol}. Limit: {proposed_limit}. Checking secondary conditions...")
        append_decision_log({"symbol": symbol, "event_type": "FIB_PROPOSAL_PRICE_TOUCHED", "proposal_id": proposal_data.get('proposal_id'), "limit": proposed_limit})

        # 3. Secondary Gating Condition (Entry Model Profit Score)
        # Re-evaluate entry model based on current conditions (or use stored initial p_profit)
        # For this implementation, let's re-evaluate using current data (df_check_raw)
        
        atr_period_secondary = current_best_hyperparams.get('model_training_atr_period_used')
        entry_features_base_secondary = current_best_hyperparams.get('entry_feature_names_base_used')
        profit_thresh_secondary = current_best_hyperparams.get('profit_threshold')

        # Features for entry model:
        p_swing_secondary = proposal_data['pivot_score'] # Use initial pivot score
        sim_entry_secondary = proposal_data['proposed_limit_price'] # Entry is the fib limit
        # SL for feature calculation: Use the original SL calculated when proposal was made
        sim_sl_secondary = proposal_data['original_sl_price']
        pivot_price_anchor_secondary = proposal_data['pivot_price_anchor']

        # Ensure df_check_raw is long enough for feature calculation
        if len(df_check_raw) < atr_period_secondary + 50 :
            print(f"{log_prefix_checker} Not enough data in df_check_raw ({len(df_check_raw)}) for secondary entry feature calculation. Skipping this check.")
            continue

        entry_features_secondary = app_calculate_live_entry_features(
            df_check_raw, atr_period_secondary, entry_features_base_secondary,
            p_swing_secondary, sim_entry_secondary, sim_sl_secondary,
            pivot_price_anchor_secondary, current_app_settings
        )

        if entry_features_secondary is None:
            print(f"{log_prefix_checker} Failed to calculate secondary entry features for {symbol}. Skipping.")
            continue
        
        try:
            entry_probs_secondary = entry_model_loaded.predict_proba(pd.DataFrame([entry_features_secondary.to_dict()]))[0]
            p_profit_secondary_live = entry_probs_secondary[1]
        except Exception as e_predict_secondary:
            print(f"{log_prefix_checker} Error during secondary entry model prediction for {symbol}: {e_predict_secondary}")
            continue
        
        print(f"{log_prefix_checker} Secondary Entry Model: P_Profit={p_profit_secondary_live:.3f} (Thresh: {profit_thresh_secondary:.2f})")

        if p_profit_secondary_live >= profit_thresh_secondary:
            print(f"{log_prefix_checker} ✅ All gating conditions MET for {symbol}!")
            append_decision_log({"symbol": symbol, "event_type": "FIB_PROPOSAL_CONDITIONS_MET", 
                                 "proposal_id": proposal_data.get('proposal_id'), "p_profit_secondary": p_profit_secondary_live})
            proposals_to_execute.append((symbol, proposal_data))
            proposals_to_remove.append(symbol) # Mark for removal after queuing for execution
        else:
            print(f"{log_prefix_checker} Secondary profit score ({p_profit_secondary_live:.3f}) for {symbol} did not meet threshold ({profit_thresh_secondary:.2f}). Proposal remains active.")
            # Update proposal with this score for future reference if needed (optional)
            # with app_fib_proposals_lock:
            #    if symbol in app_fib_proposals:
            #        app_fib_proposals[symbol]['last_market_price_checked'] = current_market_price # Example
            #        app_fib_proposals[symbol]['secondary_p_profit_score'] = p_profit_secondary_live
            append_decision_log({"symbol": symbol, "event_type": "FIB_PROPOSAL_SECONDARY_PROFIT_FAIL", 
                                 "proposal_id": proposal_data.get('proposal_id'), "p_profit_secondary": p_profit_secondary_live})


    # --- Process Removals ---
    if proposals_to_remove:
        with app_fib_proposals_lock:
            for sym_rem in set(proposals_to_remove): # Use set to avoid issues if symbol added multiple times
                if sym_rem in app_fib_proposals:
                    print(f"{log_prefix_main_checker} Removing proposal for {sym_rem} (reason: executed, expired, or error).")
                    del app_fib_proposals[sym_rem]
    
    # --- Process Executions (occurs after lock release for removals to avoid deadlock with execute_app_trade_signal) ---
    for symbol_to_exec, proposal_data_to_exec in proposals_to_execute:
        log_prefix_exec = f"[AppFibExecute-{symbol_to_exec}-{proposal_data_to_exec.get('proposal_id','N/A')}]"
        print(f"{log_prefix_exec} Initiating execution for Fib proposal.")
        
        # Retrieve original SL/TP from proposal
        exec_sl = proposal_data_to_exec['original_sl_price']
        exec_tp1 = proposal_data_to_exec['original_tp1_price']
        exec_tp2 = proposal_data_to_exec.get('original_tp2_price') # Might be None
        exec_tp3 = proposal_data_to_exec.get('original_tp3_price') # Might be None

        if current_app_settings.get('app_operational_mode') == 'signal':
            # Send "Market Order Execution" style Telegram signal
            # This indicates conditions were met and a paper trade would occur.
            # Need to fetch current market price for this message.
            s_info_exec = get_app_symbol_info(symbol_to_exec)
            p_prec_exec = int(s_info_exec.get('pricePrecision',2)) if s_info_exec else 2
            temp_mkt_price = get_app_symbol_info(symbol_to_exec) # This is wrong, need ticker
            try:
                ticker_exec = app_binance_client.futures_ticker(symbol=symbol_to_exec)
                price_for_signal_msg = float(ticker_exec['lastPrice'])
            except Exception: price_for_signal_msg = proposal_data_to_exec['proposed_limit_price'] # Fallback

            # We don't have actual executed quantity for signal mode, can estimate or omit
            # For now, let's use a placeholder for quantity in signal message or calculate based on fixed capital.
            # Using the `send_app_trade_execution_telegram` which expects executed_quantity.
            # Let's pass None for quantity in signal mode for Fib execution message.
            send_app_trade_execution_telegram(
                current_app_settings=current_app_settings,
                symbol=symbol_to_exec,
                side=proposal_data_to_exec['side'],
                entry_price=price_for_signal_msg, # Effective "execution" price for signal
                sl_price=exec_sl, tp1_price=exec_tp1, tp2_price=exec_tp2, tp3_price=exec_tp3,
                order_type="MARKET (Fib Conditions Met)", # Indicate it's from Fib
                executed_quantity=None, # No real quantity in signal mode for this step
                mode="signal" 
            )
            append_decision_log({"symbol": symbol_to_exec, "event_type": "FIB_PROPOSAL_SIGNAL_MODE_EXECUTE_MSG_SENT", 
                                 "proposal_id": proposal_data_to_exec.get('proposal_id')})

        elif current_app_settings.get('app_operational_mode') == 'live':
            # Call execute_app_trade_signal to place live market order
            # `execute_app_trade_signal` will determine its own market entry price.
            
            is_simulation_mode = current_app_settings.get("app_simulate_limit_orders", False)

            if is_simulation_mode:
                print(f"{log_prefix_exec} SIMULATE MODE: Conditions met for Fib proposal. Logging simulated market execution.")
                # In simulation mode, we don't call execute_app_trade_signal.
                # We log and send a specific Telegram message.
                
                # Fetch current market price for the simulation log/message
                sim_exec_price = proposal_data_to_exec['proposed_limit_price'] # Default to proposed
                try:
                    ticker_sim_exec = app_binance_client.futures_ticker(symbol=symbol_to_exec)
                    sim_exec_price = float(ticker_sim_exec['lastPrice'])
                except Exception as e_sim_tick:
                    print(f"{log_prefix_exec} SIMULATE MODE: Error fetching ticker for sim exec price: {e_sim_tick}. Using proposed limit.")

                s_info_sim_exec = get_app_symbol_info(symbol_to_exec)
                p_prec_sim_exec = int(s_info_sim_exec.get('pricePrecision',2)) if s_info_sim_exec else 2
                
                # For simulated quantity, we'd need to calculate it as if a trade was happening.
                # This requires balance, risk %, etc. For simplicity in this message, we can omit qty
                # or state that it would be calculated based on standard risk params.
                # Let's omit specific quantity from this "simulated execution" message for now,
                # as it's about the *trigger* of the Fib condition.
                
                sim_exec_message = (
                    f"PAPER TRADE: ✅ Fib Limit Conditions Met - Simulated Market Order ✅\n\n"
                    f"Symbol: `{escape_app_markdown_v1(symbol_to_exec)}`\n"
                    f"Direction: *{proposal_data_to_exec['side'].upper()}*\n"
                    f"Original Proposed Limit: `{proposal_data_to_exec['proposed_limit_price']:.{p_prec_sim_exec}f}` (0.618 Fib)\n"
                    f"Simulated Market Entry Price: `~{sim_exec_price:.{p_prec_sim_exec}f}`\n"
                    f"Planned SL: `{exec_sl:.{p_prec_sim_exec}f}`, Planned TP1: `{exec_tp1:.{p_prec_sim_exec}f}`\n"
                    f"_(This is a paper trade notification. No real order was placed.)_"
                )
                send_app_telegram_message(sim_exec_message)
                append_decision_log({"symbol": symbol_to_exec, "event_type": "FIB_PROPOSAL_SIMULATED_EXECUTION", 
                                     "proposal_id": proposal_data_to_exec.get('proposal_id'), 
                                     "sim_exec_price": sim_exec_price, "side": proposal_data_to_exec['side']})
            else: # Live mode
                print(f"{log_prefix_exec} Live mode: Calling execute_app_trade_signal.")
                execute_app_trade_signal(
                    symbol=symbol_to_exec,
                    side=proposal_data_to_exec['side'],
                    sl_price=exec_sl,
                    tp1_price=exec_tp1,
                    tp2_price=exec_tp2,
                    tp3_price=exec_tp3,
                    entry_price_target=None, # Market order, execute_app_trade_signal will use current price
                    order_type="MARKET" 
                )
            # `execute_app_trade_signal` already appends to decision log and sends Telegram for live.
        
        # Note: Proposal was already marked for removal.

# --- End Fibonacci Pre-Order Proposal Checker ---


# --- Pre-Signal Condition Checks ---
def check_signal_preconditions(symbol: str, side: str, klines_df: pd.DataFrame, 
                               current_app_settings: dict, live_pivot_features: pd.Series) -> tuple[bool, str]:
    """
    Checks pre-conditions like RSI and Volume before proceeding with entry model evaluation.
    Args:
        symbol (str): Trading symbol.
        side (str): "long" or "short".
        klines_df (pd.DataFrame): DataFrame with historical klines for RSI calculation.
        current_app_settings (dict): Current application settings.
        live_pivot_features (pd.Series): Series containing calculated pivot features, including 'volume_spike_vs_avg'.

    Returns:
        tuple[bool, str]: (True, "OK") if all checks pass, otherwise (False, "Reason for rejection").
    """
    log_prefix = f"[PreCheck-{symbol}-{side.upper()}]"

    # --- RSI Check ---
    rsi_period = current_app_settings.get('app_atr_period_sl_tp', 14) # Using existing ATR period for RSI for now
    rsi_values = calculate_rsi(klines_df.copy(), period=rsi_period, column='close') # calculate_rsi is from app.py
    
    if rsi_values is None or rsi_values.empty or pd.isna(rsi_values.iloc[-1]):
        reason = "RSI calculation failed or unavailable."
        print(f"{log_prefix} {reason}")
        return False, reason
    
    current_rsi = rsi_values.iloc[-1]
    
    if side == "long":
        min_rsi_long = current_app_settings.get('app_rsi_min_long', 0.0)
        max_rsi_long = current_app_settings.get('app_rsi_max_long', 70.0)
        if not (min_rsi_long <= current_rsi <= max_rsi_long):
            reason = f"RSI ({current_rsi:.2f}) out of bounds for LONG (Min: {min_rsi_long}, Max: {max_rsi_long})."
            print(f"{log_prefix} {reason}")
            return False, reason
    elif side == "short":
        min_rsi_short = current_app_settings.get('app_rsi_min_short', 30.0)
        max_rsi_short = current_app_settings.get('app_rsi_max_short', 100.0)
        if not (min_rsi_short <= current_rsi <= max_rsi_short):
            reason = f"RSI ({current_rsi:.2f}) out of bounds for SHORT (Min: {min_rsi_short}, Max: {max_rsi_short})."
            print(f"{log_prefix} {reason}")
            return False, reason
    
    print(f"{log_prefix} RSI Check PASSED: {current_rsi:.2f}")

    # --- Volume Check (already done in app_process_symbol_for_signal, but can be centralized here) ---
    # This part is redundant if the call to this function is placed after the volume check in app_process_symbol_for_signal.
    # However, for true centralization, it should be here. Assuming it might be called independently or volume check moved here.
    min_vol_spike_ratio_setting = current_app_settings.get('app_min_volume_spike_ratio', 1.5)
    vol_spike_feat_name = 'volume_spike_vs_avg'
    
    current_vol_spike = np.nan
    if live_pivot_features is not None and vol_spike_feat_name in live_pivot_features:
        current_vol_spike = live_pivot_features[vol_spike_feat_name]
    
    if pd.isna(current_vol_spike):
        if min_vol_spike_ratio_setting > 0: # Only fail if a positive threshold is expected
            reason = f"Volume spike value is NaN (Threshold: {min_vol_spike_ratio_setting})."
            print(f"{log_prefix} {reason}")
            return False, reason
    elif current_vol_spike < min_vol_spike_ratio_setting:
        reason = f"Volume spike ({current_vol_spike:.2f}) below threshold ({min_vol_spike_ratio_setting})."
        print(f"{log_prefix} {reason}")
        return False, reason
        
    print(f"{log_prefix} Volume Spike Check PASSED (or already passed): {current_vol_spike:.2f} >= {min_vol_spike_ratio_setting}")
    
    # --- Trend Check ---
    # The primary trend/setup confirmation comes from ML model scores (p_swing_score_live, p_profit_live).
    # This function is called *after* pivot model score is checked.
    # If p_swing_score_live was too low, app_process_symbol_for_signal would have already returned.
    # So, "Trend couldn't be identified (ML score pending/low)" due to pivot score is not applicable here.
    # If this function is called, it means the pivot model part of the "trend" was okay.
    # We could add a placeholder reason if we expect more rule-based trend checks here later.
    # For now, if RSI and Volume pass, this pre-check is okay regarding trend from its perspective.
    # print(f"{log_prefix} Trend Check (ML Pivot score already passed). OK.")

    return True, "OK"
# --- End Pre-Signal Condition Checks ---


def app_trading_signal_loop(current_app_settings: dict, pivot_model_loaded, entry_model_loaded, current_best_hyperparams: dict):
    """
    Main continuous loop for live trading or signal generation.
    """
    log_prefix = "[AppTradingLoop]"
    print(f"{log_prefix} Starting trading/signal loop...")

    # Send startup Telegram message
    startup_message = (
        f"✅ *App Trading Bot STARTED*\n\n"
        f"Operational Mode: `{current_app_settings.get('app_operational_mode', 'N/A').upper()}`\n"
        f"Environment: `{current_app_settings.get('app_trading_environment', 'N/A').upper()}`\n"
        f"Initial Symbols: `{current_app_settings.get('app_trading_symbols', 'N/A')}`\n\n"
        f"Bot is now operational and scanning for signals."
    )
    send_app_telegram_message(startup_message)
    
    global app_binance_client # Ensure global client is accessible

    # Ensure client is initialized (should be by start_app_main_flow, but double check)
    if app_binance_client is None:
        print(f"{log_prefix} Binance client not initialized. Attempting to initialize...")
        if not initialize_app_binance_client(env=current_app_settings.get("app_trading_environment")):
            print(f"{log_prefix} CRITICAL: Failed to initialize Binance client. Exiting loop.")
            return
        print(f"{log_prefix} Binance client initialized successfully within loop startup.")

    try:
        while True:
            loop_start_time = time.time()
            print(f"\n{log_prefix} --- New Scan Cycle --- | Mode: {current_app_settings.get('app_operational_mode', 'UNKNOWN')}")

            trading_symbols = []
            symbols_source_config = current_app_settings.get("app_trading_symbols", "BTCUSDT") # Default to BTCUSDT string
            
            if isinstance(symbols_source_config, str) and symbols_source_config.lower().endswith(".csv"):
                # Attempt to load from CSV file specified in app_trading_symbols
                print(f"{log_prefix} Attempting to load symbols from CSV: {symbols_source_config}")
                trading_symbols = app_load_symbols_from_csv(symbols_source_config)
                if not trading_symbols:
                    print(f"{log_prefix} Failed to load symbols from {symbols_source_config} or file is empty. Defaulting to BTCUSDT.")
                    trading_symbols = ["BTCUSDT"]
            elif isinstance(symbols_source_config, str):
                # Parse as comma-separated string
                trading_symbols = [s.strip().upper() for s in symbols_source_config.split(',') if s.strip()]
                if not trading_symbols: # If string was empty or only commas
                    print(f"{log_prefix} 'app_trading_symbols' string ('{symbols_source_config}') resulted in no symbols. Defaulting to BTCUSDT.")
                    trading_symbols = ["BTCUSDT"]
            else: # Fallback if config is not a string (e.g. error or unexpected type)
                print(f"{log_prefix} 'app_trading_symbols' has unexpected type or value. Defaulting to BTCUSDT.")
                trading_symbols = ["BTCUSDT"]

            if not trading_symbols: # Final fallback, though logic above should ensure trading_symbols is not empty
                print(f"{log_prefix} No trading symbols configured after all checks. Skipping cycle.")
            else:
                print(f"{log_prefix} Processing symbols: {trading_symbols}")
                for symbol_to_trade in trading_symbols:
                    print(f"{log_prefix} Processing symbol: {symbol_to_trade}")
                    app_process_symbol_for_signal(
                        symbol=symbol_to_trade,
                        client=app_binance_client, # Use the global client
                        current_app_settings=current_app_settings,
                        pivot_model_loaded=pivot_model_loaded,
                        entry_model_loaded=entry_model_loaded,
                        current_best_hyperparams=current_best_hyperparams
                    )
                    # Optional: Short delay between processing symbols if multiple are listed
                    time.sleep(current_app_settings.get("app_delay_between_symbols_seconds", 2)) # Default 2s

            # --- Check Fibonacci Pre-Order Proposals ---
            if current_app_settings.get("use_fib_preorder", False): # Only run if feature is enabled
                print(f"{log_prefix} Checking Fibonacci pre-order proposals...")
                app_check_fib_proposals(
                    client=app_binance_client,
                    current_app_settings=current_app_settings,
                    pivot_model_loaded=pivot_model_loaded, # Pass loaded models
                    entry_model_loaded=entry_model_loaded,
                    current_best_hyperparams=current_best_hyperparams
                )
            # --- End Fibonacci Pre-Order Proposal Check ---

            # Monitor existing trades/signals (if any were placed)
            # monitor_app_trades uses the global app_binance_client and app_active_trades
            if current_app_settings.get('app_operational_mode') == 'live':
                 print(f"{log_prefix} Monitoring active live trades...")
                 monitor_app_trades() 
            elif current_app_settings.get('app_operational_mode') == 'signal':
                 # If signal mode also needs monitoring (e.g. for virtual SL/TP hits), call a similar function.
                 # For now, monitor_app_trades is primarily for live order management.
                 # We might need a separate `monitor_app_virtual_signals` or adapt `monitor_app_trades`.
                 # The current `monitor_app_trades` has logic for "signal" mode signals if they are in `app_active_trades`.
                 # The `execute_app_trade_signal` does not add to `app_active_trades` in signal mode.
                 # This part needs refinement if signal mode requires active monitoring beyond just sending initial signal.
                 # For now, let's assume `monitor_app_trades` is primarily for live logic.
                 # If `app_process_symbol_for_signal` were to add signals to a list for tracking,
                 # a `monitor_app_signals` function would go here.
                 pass
            
            # Save decision log periodically (e.g., every cycle if new entries, or based on buffer)
            save_decision_log_to_csv() # Will save if buffer is full or if forced (not forced here)


            # Loop delay
            scan_interval_seconds = current_app_settings.get("app_scan_interval_seconds", 60) # Default to 60 seconds
            loop_duration = time.time() - loop_start_time
            sleep_time = max(0, scan_interval_seconds - loop_duration)
            
            if sleep_time > 0:
                print(f"{log_prefix} Cycle completed in {loop_duration:.2f}s. Sleeping for {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            else:
                print(f"{log_prefix} Cycle completed in {loop_duration:.2f}s. Starting next cycle immediately (scan interval too short or processing too long).")

    except KeyboardInterrupt:
        print(f"\n{log_prefix} Loop interrupted by user (Ctrl+C). Shutting down...")
    except Exception as e:
        print(f"\n{log_prefix} CRITICAL UNEXPECTED ERROR in trading loop: {e}")
        import traceback
        traceback.print_exc()
        # Optionally send a Telegram alert about the critical error
        error_message = f"🆘 CRITICAL ERROR in AppTradingLoop 🆘\nSymbol: {symbol_to_trade if 'symbol_to_trade' in locals() else 'N/A'}\nError: {str(e)[:1000]}"
        send_app_telegram_message(error_message) # Uses global app_trading_configs for token/chat_id
    finally:
        print(f"{log_prefix} Trading/signal loop stopped. Saving final decision log...")
        save_decision_log_to_csv(force_save=True) # Force save any remaining entries on shutdown
        # Potential cleanup logic here if needed

# --- Main Orchestration & Startup Logic ---
def start_app_main_flow():
    """Orchestrates the main application flow based on settings and user input."""
    log_prefix = "[AppMainFlow]"
    global app_settings, universal_pivot_model, universal_entry_model, best_hyperparams # Ensure best_hyperparams is global if modified

    # 1. Load settings
    load_app_settings() # Loads or prompts for settings and saves them
    load_app_trading_configs() # Ensure trading configs are populated from app_settings

    # 2. Check for ML Models
    models_exist, pivot_model_path, entry_model_path, params_path = check_ml_models_exist()
    
    force_retrain = app_settings.get("app_force_retrain_on_startup", False)

    # Main application loop / decision tree
    while True: # Loop to allow retraining and then returning to choices
        if not models_exist or force_retrain:
            if force_retrain:
                print("Configuration set to force retrain models.")
                force_retrain = False # Reset flag after use
                app_settings["app_force_retrain_on_startup"] = False # Also reset in runtime settings
                save_app_settings() # Persist the reset flag state
            else: # models_exist is False
                print("ML models not found. Starting training process...")
            
            # --- Training Process ---
            # Determine training symbols based on new settings
            training_symbols_list = []
            source_type = app_settings.get("app_training_symbols_source_type", "list")
            
            if source_type.lower() == "csv":
                csv_path = app_settings.get("app_symbols_csv_path", "app_symbols.csv")
                print(f"{log_prefix} Attempting to load training symbols from CSV: {csv_path}")
                training_symbols_list = app_load_symbols_from_csv(csv_path)
                if training_symbols_list is None or not training_symbols_list: # Check None or empty
                    print(f"{log_prefix} WARN: CSV specified for training symbols ('{csv_path}') load failed or empty. Falling back to list string.")
                    source_type = "list" # Force fallback to list

            if source_type.lower() == "list" or training_symbols_list is None or not training_symbols_list: # Fallback or primary if "list"
                list_str = app_settings.get("app_training_symbols_list_str", "BTCUSDT,ETHUSDT")
                print(f"{log_prefix} Loading training symbols from list string: '{list_str}'")
                training_symbols_list = [s.strip().upper() for s in list_str.split(',') if s.strip()]

            if training_symbols_list is None or not training_symbols_list: # Check None or empty
                print(f"{log_prefix} CRITICAL: No training symbols loaded from any source. Defaulting to BTCUSDT for training.")
                training_symbols_list = ["BTCUSDT"]
            
            kline_interval_train = app_settings.get("app_training_kline_interval", Client.KLINE_INTERVAL_15MINUTE)
            start_date_train = app_settings.get("app_training_start_date", "1 Jan, 2023")
            end_date_train = app_settings.get("app_training_end_date", "1 May, 2023")
            optuna_trials_app = app_settings.get("app_optuna_trials", 20)

            print(f"Training with: Symbols={training_symbols_list}, Interval={kline_interval_train}, Start={start_date_train}, End={end_date_train}, Trials={optuna_trials_app}")

            all_symbols_train_data_list_app = []
            processed_pivot_feature_names_app = None # Initialize for this training run
            processed_entry_feature_names_base_app = None # Initialize for this training run


            for symbol_ticker_app in training_symbols_list:
                processed_df_app, pivot_feats_app, entry_feats_base_app = get_processed_data_for_symbol(
                    symbol_ticker_app, kline_interval_train, start_date_train, end_date_train
                )
                if processed_df_app is not None and not processed_df_app.empty:
                    if processed_pivot_feature_names_app is None: # Capture feature names from first successful processing
                        processed_pivot_feature_names_app = pivot_feats_app
                    if processed_entry_feature_names_base_app is None: # Capture from first
                        processed_entry_feature_names_base_app = entry_feats_base_app
                    all_symbols_train_data_list_app.append(processed_df_app)
            
            if not all_symbols_train_data_list_app: print("CRITICAL: No training data. Cannot train models."); sys.exit(1) # Empty list check
            universal_df_initial_processed_app = pd.concat(all_symbols_train_data_list_app, ignore_index=True)
            if len(universal_df_initial_processed_app) < 200: print(f"CRITICAL: Combined training data too small ({len(universal_df_initial_processed_app)})."); sys.exit(1)

            # Ensure feature names are available before Optuna if they were determined
            if processed_pivot_feature_names_app is None or processed_entry_feature_names_base_app is None:
                print("CRITICAL: Feature names for training not determined (no symbol data processed successfully?). Exiting.")
                sys.exit(1)

            try:
                current_best_hyperparams = run_optuna_tuning(
                    universal_df_initial_processed_app.copy(),
                    static_entry_features_base_list=processed_entry_feature_names_base_app, # Pass determined base names
                    n_trials=optuna_trials_app
                )
                if current_best_hyperparams is None: # Check if Optuna returned None
                    print(f"Optuna tuning did not return valid parameters. Exiting."); sys.exit(1)


                # --- Correctly define feature names based on Optuna's tuned ATR period ---
                tuned_atr_period = current_best_hyperparams.get('atr_period_opt', ATR_PERIOD)
                atr_col_name_tuned = f'atr_{tuned_atr_period}'

                # Define pivot feature names consistent with engineer_pivot_features and tuned ATR
                pivot_feature_names_list_tuned = [
                    atr_col_name_tuned, 'range_atr_norm', 'macd_slope_atr_norm',
                    'return_1b_atr_norm', 'return_3b_atr_norm', 'return_5b_atr_norm',
                    'high_rank_7', 'bars_since_last_pivot', 'volume_spike_vs_avg', 'rsi_14'
                ]

                # Define entry base feature names consistent with engineer_entry_features and tuned ATR
                entry_feature_names_base_list_tuned = [
                    'ema20_ema50_norm_atr',
                    'return_entry_1b', 'return_entry_3b', 'return_entry_5b',
                    f'{atr_col_name_tuned}_change', # Uses tuned ATR column name
                    'hour_of_day', 'day_of_week' # Removed 'vol_regime'
                ]
                # Note: 'P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl' are added later to form full entry features.

                current_best_hyperparams['pivot_feature_names_used'] = pivot_feature_names_list_tuned
                current_best_hyperparams['entry_feature_names_base_used'] = entry_feature_names_base_list_tuned
                current_best_hyperparams['model_training_atr_period_used'] = tuned_atr_period
                # --- End feature name correction ---

                with open(params_path, 'w') as f: json.dump(current_best_hyperparams, f, indent=4)
                print(f"Best Optuna parameters (including dynamically generated feature names and ATR period) saved to {params_path}")
                best_hyperparams = current_best_hyperparams # Update global
            except Exception as e: print(f"Optuna tuning failed: {e}. Exiting."); import traceback; traceback.print_exc(); sys.exit(1) # Added traceback

            # Use feature names directly from the now-corrected best_hyperparams for final model training
            final_pivot_feats_to_use = best_hyperparams.get('pivot_feature_names_used') # Use .get()
            final_entry_base_feats_to_use = best_hyperparams.get('entry_feature_names_base_used') # Use .get()
            if final_pivot_feats_to_use is None or final_entry_base_feats_to_use is None:
                print("CRITICAL: Feature names not found in best_hyperparams after Optuna. Exiting."); sys.exit(1)

            
            # Pass the correctly defined final_entry_base_feats_to_use to process_dataframe_with_params
            df_final_train, processed_pivot_features_final, processed_entry_features_base_final = process_dataframe_with_params(
                universal_df_initial_processed_app.copy(), 
                best_hyperparams, # Contains the correct ATR period and feature name lists
                static_entry_features_base_list_arg=final_entry_base_feats_to_use # Pass the list based on tuned ATR
            )
            # df_final_train should now be processed using the tuned ATR period, and its features should align.
            # processed_pivot_features_final and processed_entry_features_base_final from this function
            # should match what's in best_hyperparams if process_dataframe_with_params is consistent.

            if df_final_train is None: print("CRITICAL: Failed to process data with best params. Exiting."); sys.exit(1)

            # Ensure X_p_train uses the feature list that the model will expect, which is from best_hyperparams
            X_p_train = df_final_train[final_pivot_feats_to_use].fillna(-1)
            y_p_train = df_final_train['pivot_label']

            # Log class distribution for pivot labels
            print(f"\n{log_prefix} --- Pivot Label Distribution BEFORE Resampling (y_p_train) ---")
            print(f"{log_prefix} Raw Value Counts:")
            print(y_p_train.value_counts())
            print(f"{log_prefix} Normalized Value Counts (Percentages):")
            print(y_p_train.value_counts(normalize=True) * 100)
            print(f"{log_prefix} --- End Pivot Label Distribution ---\n")

            # --- Implement Oversampling for Pivot Model Training Data ---
            from sklearn.utils import resample

            df_pivot_train_temp = pd.concat([X_p_train, y_p_train], axis=1)
            df_majority = df_pivot_train_temp[df_pivot_train_temp['pivot_label'] == 0]
            df_minority1 = df_pivot_train_temp[df_pivot_train_temp['pivot_label'] == 1] # Swing High
            df_minority2 = df_pivot_train_temp[df_pivot_train_temp['pivot_label'] == 2] # Swing Low

            # Determine target sample size for minority classes
            # Example: Oversample to be 30% of the majority class size for each minority class
            # Or a fixed number, e.g., min(len(df_majority) * 0.3, 20000) to avoid excessive memory usage
            target_minority_samples = int(len(df_majority) * 0.30) 
            if target_minority_samples == 0 and len(df_majority) > 0: # Handle cases where 30% is less than 1
                target_minority_samples = 1
            
            print(f"{log_prefix} Resampling: Majority size: {len(df_majority)}, Target for each minority: {target_minority_samples}")

            df_minority1_resampled = pd.DataFrame()
            if not df_minority1.empty:
                df_minority1_resampled = resample(df_minority1, 
                                                  replace=True, # Oversample with replacement
                                                  n_samples=target_minority_samples if target_minority_samples > 0 else len(df_minority1), 
                                                  random_state=42)
            
            df_minority2_resampled = pd.DataFrame()
            if not df_minority2.empty:
                df_minority2_resampled = resample(df_minority2,
                                                  replace=True,
                                                  n_samples=target_minority_samples if target_minority_samples > 0 else len(df_minority2),
                                                  random_state=42)

            df_resampled = pd.concat([df_majority, df_minority1_resampled, df_minority2_resampled])
            
            X_p_train_resampled = df_resampled[final_pivot_feats_to_use]
            y_p_train_resampled = df_resampled['pivot_label']
            
            print(f"\n{log_prefix} --- Pivot Label Distribution AFTER Resampling ---")
            print(f"{log_prefix} Raw Value Counts:")
            print(y_p_train_resampled.value_counts())
            print(f"{log_prefix} Normalized Value Counts (Percentages):")
            print(y_p_train_resampled.value_counts(normalize=True) * 100)
            print(f"{log_prefix} --- End Resampled Pivot Label Distribution ---\n")
            # --- End Resampling ---

            # Define valid keys for train_pivot_model based on its signature
            valid_pivot_model_arg_names = ['model_type', 'num_leaves', 'learning_rate', 'max_depth']
            raw_pivot_args = {k.replace('pivot_', ''):v for k,v in best_hyperparams.items() if k.startswith('pivot_')}
            pivot_model_args = {k: v for k, v in raw_pivot_args.items() if k in valid_pivot_model_arg_names}
            
            print(f"DEBUG: Args passed to train_pivot_model: {pivot_model_args}") # Log the actual args
            # Train pivot model with resampled data
            temp_pivot_model, _ = train_pivot_model(X_p_train_resampled, y_p_train_resampled, 
                                                    X_p_train_resampled, y_p_train_resampled, # Using resampled for validation during this fit
                                                    **pivot_model_args)
            if temp_pivot_model is None: print("CRITICAL: Pivot model training failed. Exiting."); sys.exit(1)
            save_model(temp_pivot_model, pivot_model_path)
            universal_pivot_model = temp_pivot_model

            # For calculating P_swing on the original (non-resampled) df_final_train for entry candidate filtering:
            # This is important because we want to evaluate P_swing on the original data distribution.
            p_swing_on_original_train_data = temp_pivot_model.predict_proba(X_p_train) # X_p_train is original
            df_final_train['P_swing'] = np.max(p_swing_on_original_train_data[:,1:], axis=1)
            
            # Filter for entry candidates
            p_swing_thresh_for_entry_candidates = best_hyperparams.get('p_swing_threshold', 0.5) # Default if not in params
            entry_candidates_condition = (
                (df_final_train['pivot_label'].isin([1,2])) &
                (df_final_train['trade_outcome'] != -1) &
                (df_final_train['P_swing'] >= p_swing_thresh_for_entry_candidates)
            )
            entry_candidates = df_final_train[entry_candidates_condition].copy()
            
            MIN_ENTRY_CANDIDATES = 10 # Reduced from 20
            
            if len(entry_candidates) >= MIN_ENTRY_CANDIDATES:
                print(f"{log_prefix} Sufficient entry candidates ({len(entry_candidates)}) found with P_swing >= {p_swing_thresh_for_entry_candidates:.2f}. Proceeding with entry model training.")
                atr_col_final = f"atr_{best_hyperparams.get('atr_period_opt', ATR_PERIOD)}"
                entry_candidates['norm_dist_entry_pivot'] = (entry_candidates['entry_price_sim'] - entry_candidates.apply(lambda r: r['low'] if r['is_swing_low'] == 1 else r['high'], axis=1)) / entry_candidates[atr_col_final]
                entry_candidates['norm_dist_entry_sl'] = (entry_candidates['entry_price_sim'] - entry_candidates['sl_price_sim']).abs() / entry_candidates[atr_col_final]
                
                full_entry_feats = final_entry_base_feats_to_use + ['P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl']
                X_e_train = entry_candidates[full_entry_feats].fillna(-1)
                y_e_train = (entry_candidates['trade_outcome'] > 0).astype(int)
                
                if len(X_e_train) > 0 and len(y_e_train.unique()) > 1:
                    valid_entry_model_arg_names = ['model_type', 'num_leaves', 'learning_rate', 'max_depth', 'n_estimators']
                    raw_entry_args = {k.replace('entry_', ''):v for k,v in best_hyperparams.items() if k.startswith('entry_')}
                    entry_model_args = {k: v for k, v in raw_entry_args.items() if k in valid_entry_model_arg_names}
                    
                    print(f"DEBUG: Args passed to train_entry_model: {entry_model_args}")
                    temp_entry_model, _ = train_entry_model(X_e_train, y_e_train, X_e_train, y_e_train, **entry_model_args)
                    if temp_entry_model is None: print(f"{log_prefix} Entry model training returned None. Marking as not trained."); universal_entry_model = None
                    else: save_model(temp_entry_model, entry_model_path); universal_entry_model = temp_entry_model; print(f"{log_prefix} Entry model trained and saved.")
                else:
                    print(f"{log_prefix} Entry model training SKIPPED: Insufficient data diversity for training. X_e_train length: {len(X_e_train)}, y_e_train unique values: {y_e_train.unique() if len(X_e_train) > 0 else 'N/A'}.")
                    universal_entry_model = None # Ensure it's None if skipped
            else:
                print(f"{log_prefix} Entry model training SKIPPED: Not enough candidates ({len(entry_candidates)}) after pivot filter (P_swing >= {p_swing_thresh_for_entry_candidates:.2f}). Minimum required: {MIN_ENTRY_CANDIDATES}.")
                universal_entry_model = None # Ensure it's None if skipped
            
            print("ML Model training process complete.")
            models_exist = True

            # Generate and display backtest summary after training
            print("DEBUG: Preparing data for training summary backtest...")
            # Check all required components are not None
            if df_final_train is not None and \
               universal_pivot_model is not None and \
               universal_entry_model is not None and \
               best_hyperparams is not None and \
               final_pivot_feats_to_use is not None and \
               final_entry_base_feats_to_use is not None and \
               app_settings is not None:
                generate_training_backtest_summary(
                    df_processed_full_dataset=df_final_train, 
                    pivot_model=universal_pivot_model,
                    entry_model=universal_entry_model,
                    best_params_from_optuna=best_hyperparams, # This contains all tuned params including thresholds
                    pivot_feature_names_list=final_pivot_feats_to_use, # Features used for pivot model
                    entry_feature_names_base_list=final_entry_base_feats_to_use, # Base features for entry model
                    app_settings_dict=app_settings
                )
            else:
                print("Skipping training summary generation due to missing data or models from training process.")
                # Log which parts are missing for debugging
                if df_final_train is None: print("  - df_final_train is None")
                if universal_pivot_model is None: print("  - universal_pivot_model is None")
                if universal_entry_model is None: print("  - universal_entry_model is None")
                if best_hyperparams is None: print("  - best_hyperparams is None")
                if final_pivot_feats_to_use is None: print("  - final_pivot_feats_to_use is None")
                if final_entry_base_feats_to_use is None: print("  - final_entry_base_feats_to_use is None")
                if app_settings is None: print("  - app_settings is None")


            if not app_settings.get("app_auto_start_trading_after_train", False):
                choice_after_train = input("Training complete & summary shown. Proceed to live market trading? (yes/no) [no]: ").lower()
                if choice_after_train not in ['yes', 'y']:
                    print("Exiting after training as per user choice.")
                    sys.exit(0)
            # If auto_start or user chose yes, fall through to model loading and trading choice.
            # Note: Models are already in global vars if training just happened.

        # Models exist (either initially or after training, or loaded below)
        # Try to load models and params into global vars if they are not already 
        # (e.g. on fresh start with existing files, or if training was skipped and we fell through)
        # Also ensure best_hyperparams is not an empty dict if models are loaded
        if universal_pivot_model is None or universal_entry_model is None or best_hyperparams is None or not best_hyperparams:
            print("Loading trained ML models and parameters...")
            try:
                universal_pivot_model = load_model(pivot_model_path)
                if universal_pivot_model is None: raise FileNotFoundError("Pivot model loaded as None.")
            except Exception as e:
                print(f"[ERROR] loading pivot model from '{pivot_model_path}': {e}")
                # Decide if to offer retry_train or just exit
                retry_train = input(f"Error loading pivot model. Attempt to retrain models? (yes/no) [yes]: ").lower()
                if retry_train in ['no', 'n']: sys.exit(1)
                force_retrain = True; models_exist = False; continue
            
            try:
                universal_entry_model = load_model(entry_model_path)
                if universal_entry_model is None: raise FileNotFoundError("Entry model loaded as None.")
            except Exception as e:
                print(f"[ERROR] loading entry model from '{entry_model_path}': {e}")
                retry_train = input(f"Error loading entry model. Attempt to retrain models? (yes/no) [yes]: ").lower()
                if retry_train in ['no', 'n']: sys.exit(1)
                force_retrain = True; models_exist = False; continue

            try:
                with open(params_path, 'r') as f:
                    loaded_params = json.load(f) 
                if loaded_params is None or not loaded_params: # Check for None or empty dict
                    raise ValueError("Parameters file loaded as None or empty.")
                best_hyperparams = loaded_params # This loads params like thresholds and feature names
                
                # Add a check for essential keys after loading
                required_keys = ['pivot_feature_names_used', 'entry_feature_names_base_used', 'model_training_atr_period_used']
                missing_keys = [key for key in required_keys if key not in best_hyperparams or not best_hyperparams[key]]
                if missing_keys:
                    print(f"[ERROR] Parameters file '{params_path}' is incomplete. Missing or empty for keys: {missing_keys}.")
                    raise ValueError(f"Incomplete parameters file, missing: {missing_keys}") # This will be caught by the generic except below

            except FileNotFoundError:
                print(f"[ERROR] parameters file not found at '{params_path}'.")
                retry_train = input(f"Parameters file not found. Attempt to retrain models (which recreates this file)? (yes/no) [yes]: ").lower()
                if retry_train in ['no', 'n']: sys.exit(1)
                force_retrain = True; models_exist = False; universal_pivot_model = None; universal_entry_model = None; best_hyperparams = {}; continue
            except (json.JSONDecodeError, ValueError) as e_params_load: # Catch ValueError from our check or JSON errors
                print(f"[ERROR] loading or validating parameters file '{params_path}': {e_params_load}")
                retry_train = input(f"Error with parameters file. Attempt to retrain models? (yes/no) [yes]: ").lower()
                if retry_train in ['no', 'n']: sys.exit(1)
                force_retrain = True; models_exist = False; universal_pivot_model = None; universal_entry_model = None; best_hyperparams = {}; continue
            except Exception as e_params: # Catch any other unexpected error during params loading
                print(f"[ERROR] loading parameters file '{params_path}': {e_params}")
                retry_train = input(f"Unexpected error loading parameters file. Attempt to retrain models? (yes/no) [yes]: ").lower()
                if retry_train in ['no', 'n']: sys.exit(1)
                force_retrain = True; models_exist = False; continue
            
            print("Models and parameters loaded successfully.")

        # --- User Action Choice ---
        print("\n--- Application Menu ---")
        print("1. Start Live Market Trading / Signal Generation")
        print("2. Retrain ML Models")
        print("3. Exit")
        user_action = input("Enter choice (1-3): ")

        if user_action == '1':
            print("Starting Live Market Trading / Signal Generation mode...")
            app_settings['app_operational_mode'] = get_user_input(
                "Set operational mode ('live' or 'signal')",
                app_settings.get("app_operational_mode", "signal"), 
                str, choices=['live', 'signal']
            )
            save_app_settings() 
            print(f"Operational mode set to: {app_settings['app_operational_mode']}")
            
            if app_binance_client is None: # Initialize client if not done yet
                if not initialize_app_binance_client(env=app_settings.get("app_trading_environment")):
                    print("CRITICAL: Failed to initialize Binance client. Exiting.")
                    sys.exit(1)
            
            # --- Call the main trading/signal loop ---
            print(f"{log_prefix} Initializing trading/signal loop with mode: {app_settings['app_operational_mode']}")
            app_trading_signal_loop(
                current_app_settings=app_settings, # Pass the global app_settings
                pivot_model_loaded=universal_pivot_model,
                entry_model_loaded=universal_entry_model,
                current_best_hyperparams=best_hyperparams
            )
            # The loop will run until explicitly stopped (e.g., Ctrl+C or a shutdown command if implemented)
            # If the loop exits, the application can then terminate.
            print(f"{log_prefix} Trading/signal loop has exited. Application will now terminate.")
            sys.exit(0) # Exit after the loop finishes
        
        elif user_action == '2':
            print("User chose to retrain models.")
            force_retrain = True 
            models_exist = False 
            universal_pivot_model, universal_entry_model, best_hyperparams = None, None, {} # Clear loaded models
            continue 
        
        elif user_action == '3':
            print("Exiting application.")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")

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

    if atr_col_name not in df.columns:
        print(f"Warning (prune_and_label_pivots): ATR column '{atr_col_name}' not found. Pivots might be incorrect if ATR is needed and not present. Returning df as is.")
        return df # Cannot proceed without ATR column if it's expected

    # Parse ATR period from atr_col_name for warmup period calculation
    try:
        parsed_atr_period = int(atr_col_name.split('_')[-1])
        if parsed_atr_period <= 0:
            raise ValueError("Parsed ATR period must be positive.")
    except (ValueError, IndexError):
        print(f"Error (prune_and_label_pivots): Could not parse ATR period from '{atr_col_name}'. Assuming 14 for warmup. This may be incorrect.")
        parsed_atr_period = 14 # Fallback

    min_index_for_valid_atr = parsed_atr_period - 1

    # Iterate starting from where ATR is expected to be valid
    candidates_evaluated = 0
    pruned_by_bar_gap = 0
    pruned_by_atr_dist = 0
    pruned_by_invalid_atr_current = 0
    pruned_by_invalid_atr_last = 0
    confirmed_pivots = 0

    for i in range(len(df)):
        if i < min_index_for_valid_atr and atr_distance_factor > 0: # Skip early bars if ATR is used for pruning
            continue
        
        is_ch_initial = df['is_candidate_high'].iloc[i]
        is_cl_initial = df['is_candidate_low'].iloc[i]
        if not is_ch_initial and not is_cl_initial:
            continue
        candidates_evaluated +=1

        # ATR at the current candidate pivot `i`
        atr_val_at_current_pivot = df.loc[df.index[i], atr_col_name] 
        
        # If ATR is needed for distance factor and current ATR is invalid, skip this candidate
        if atr_distance_factor > 0 and (pd.isna(atr_val_at_current_pivot) or atr_val_at_current_pivot == 0):
            pruned_by_invalid_atr_current +=1
            continue

        is_ch = df['is_candidate_high'].iloc[i]
        is_cl = df['is_candidate_low'].iloc[i]

        if not is_ch and not is_cl:
            continue

        # Check bar gap
        if last_confirmed_pivot_idx != -1 and (df.index[i] - last_confirmed_pivot_idx) < min_bar_gap:
            # Note: This assumes df.index[i] and last_confirmed_pivot_idx are comparable.
            # If df.index is not a simple range index, direct subtraction might be an issue.
            # However, 'i' is from range(len(df)), and last_confirmed_pivot_idx stores 'i'.
            # So, this comparison is on the integer position.
            if (i - last_confirmed_pivot_idx) < min_bar_gap: # Corrected to use integer index i
                pruned_by_bar_gap += 1
                continue


        current_price = 0
        current_type = None

        if is_ch:
            current_price = df['high'].iloc[i]
            current_type = 'high'
        elif is_cl:
            current_price = df['low'].iloc[i]
            current_type = 'low'

        # Check ATR distance (if there's a previous pivot and ATR distance pruning is active)
        if last_confirmed_pivot_idx != -1 and atr_distance_factor > 0:
            # ATR at the last confirmed pivot
            # Ensure last_confirmed_pivot_idx is also >= min_index_for_valid_atr for its ATR to be valid.
            # This should be true by construction if the loop starts after warmup.
            atr_at_last_pivot = df.loc[df.index[last_confirmed_pivot_idx], atr_col_name]

            if pd.isna(atr_at_last_pivot) or atr_at_last_pivot == 0:
                pruned_by_invalid_atr_last += 1
                continue
            
            price_diff = abs(current_price - last_confirmed_pivot_price)
            if price_diff < (atr_distance_factor * atr_at_last_pivot):
                pruned_by_atr_dist += 1
                continue
        
        # Confirm pivot
        confirmed_pivots +=1
        # Note: last_confirmed_pivot_idx stores the integer position 'i', not the DataFrame label index.
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
    
    print(f"DEBUG (prune_and_label_pivots): Evaluated {candidates_evaluated} candidates. "
          f"Pruned by BarGap: {pruned_by_bar_gap}, ATRDist: {pruned_by_atr_dist}, "
          f"InvalidCurrATR: {pruned_by_invalid_atr_current}, InvalidLastATR: {pruned_by_invalid_atr_last}. "
          f"Confirmed Pivots: {confirmed_pivots}")
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
    # New columns for detailed trade data
    df['trade_direction_sim'] = None # "long" or "short"
    df['entry_bar_idx_sim'] = pd.NA # Bar index of entry
    df['exit_bar_idx_sim'] = pd.NA   # Bar index of exit
    df['exit_price_sim'] = np.nan    # Actual exit price
    df['r_multiple_sim'] = np.nan    # P&L in R-multiples
    df['duration_sim'] = pd.NA       # Duration in bars


    # --- Parameters for Time Exit ---
    # Default max_holding_bars, can be made a parameter of simulate_fib_entries if needed for tuning
    MAX_HOLDING_BARS_DEFAULT = 75 # Example: approx 3 days on 15min chart if continuous trading
                                  # This should ideally be passed as an argument or tuned.
    # MAX_HOLDING_BARS_DEFAULT = 100 # As used in loop condition below, ensure consistency or pass as param

    # Ensure the dynamic ATR column (atr_col_name) is present.
    if atr_col_name not in df.columns:
        print(f"Error (simulate_fib_entries): Required ATR column '{atr_col_name}' not found in DataFrame. Cannot simulate entries.")
        return df

    # Parse ATR period from atr_col_name (e.g., 'atr_14' -> 14)
    try:
        parsed_atr_period = int(atr_col_name.split('_')[-1])
        if parsed_atr_period <= 0:
            raise ValueError("Parsed ATR period must be positive.")
    except (ValueError, IndexError):
        print(f"Error (simulate_fib_entries): Could not parse ATR period from '{atr_col_name}'. Using a default of 14 for warmup calculation, but this may be incorrect.")
        parsed_atr_period = 14 # Fallback, though ideally this shouldn't happen

    # ATR Warmup: Only consider pivots after the initial ATR calculation period.
    # DataFrame indices must be comparable to parsed_atr_period - 1.
    # Assuming df has a simple range index [0, 1, ..., len(df)-1] or that index 'i' corresponds to row number.
    # The first valid ATR will be at index `parsed_atr_period - 1`.
    # So, pivots should be considered from this index onwards.
    min_pivot_index_for_valid_atr = parsed_atr_period -1 

    all_pivots = df[(df['is_swing_high'] == 1) | (df['is_swing_low'] == 1)].copy()
    
    # Filter pivots to only include those that occur at or after the ATR warmup period.
    # The index 'i' from iterrows() is the DataFrame index of the pivot.
    pivots = all_pivots[all_pivots.index >= min_pivot_index_for_valid_atr].copy()

    if pivots.empty:
        print(f"DEBUG (simulate_fib_entries): No pivots found after ATR warmup period ({min_pivot_index_for_valid_atr}). Original pivots: {len(all_pivots)}")
        return df # No pivots to simulate after warmup

    simulated_trades = 0
    skipped_due_to_bad_atr = 0
    entries_not_triggered = 0

    # print(f"Debug (simulate_fib_entries): Total pivots before warmup filter: {len(all_pivots)}, After filter (index >= {min_pivot_index_for_valid_atr}): {len(pivots)}")

    for i, pivot_row in pivots.iterrows():
        # Use the dynamic atr_col_name
        atr_at_pivot = df.loc[i, atr_col_name] # ATR at the time of the pivot/signal
        
        # Defensive ATR Check: Ensure ATR is valid before using it.
        if pd.isna(atr_at_pivot) or atr_at_pivot == 0:
            # This warning should ideally not occur frequently if the warmup logic above is correct,
            # but it's a good safeguard.
            # print(f"Warning (simulate_fib_entries): ATR is NaN or zero at index {i} (pivot index) for pivot despite warmup. Value: {atr_at_pivot}. Skipping simulation for this pivot.")
            df.loc[i, 'trade_outcome'] = -1 # Mark as no trade due to bad ATR
            skipped_due_to_bad_atr +=1
            continue
        
        simulated_trades += 1
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
        df.loc[i, 'trade_direction_sim'] = "long" if is_long_trade else "short"

        # Simulate trade progression
        outcome = -1 # Default to No Entry/Unresolved
        final_exit_price = np.nan
        initial_risk_per_unit = abs(entry_price - sl_price) if entry_price != sl_price else np.nan
        r_multiple = np.nan
        
        entered_trade = False
        entry_bar_k = -1
        exit_bar_k = -1

        for k in range(i + 1, len(df)):
            exit_bar_k = k # Tentative exit bar, updated if SL/TP hit earlier
            bar_low = df['low'].iloc[k]
            bar_high = df['high'].iloc[k]

            # Entry Logic
            if not entered_trade:
                if is_long_trade and bar_low <= entry_price:
                    entered_trade = True
                    entry_bar_k = k
                    # Actual entry is 'entry_price', not bar_low, for simulation consistency
                elif not is_long_trade and bar_high >= entry_price:
                    entered_trade = True
                    entry_bar_k = k
            
            if entered_trade:
                # Exit Logic
                if is_long_trade:
                    if bar_low <= sl_price: # SL Hit
                        outcome = 0
                        final_exit_price = sl_price
                        break 
                    if bar_high >= tp3_price: # TP3 Hit
                        outcome = 3
                        final_exit_price = tp3_price
                        break
                    if bar_high >= tp2_price and outcome < 2: # TP2 Hit (potentially, can still hit TP3)
                        outcome = 2
                        final_exit_price = tp2_price # Tentative exit if TP3 not hit
                    if bar_high >= tp1_price and outcome < 1: # TP1 Hit
                        outcome = 1
                        final_exit_price = tp1_price # Tentative exit
                else: # Short trade
                    if bar_high >= sl_price: # SL Hit
                        outcome = 0
                        final_exit_price = sl_price
                        break
                    if bar_low <= tp3_price: # TP3 Hit
                        outcome = 3
                        final_exit_price = tp3_price
                        break
                    if bar_low <= tp2_price and outcome < 2: # TP2 Hit
                        outcome = 2
                        final_exit_price = tp2_price
                    if bar_low <= tp1_price and outcome < 1: # TP1 Hit
                        outcome = 1
                        final_exit_price = tp1_price
                
                # Max Holding Period Check (if trade is still open)
                # Note: k is 0-indexed from start of df, i is pivot index.
                # entry_bar_k is also 0-indexed from start of df.
                # Duration is (k - entry_bar_k + 1) bars if k is current bar.
                if entry_bar_k != -1 and (k - entry_bar_k) >= MAX_HOLDING_BARS_DEFAULT:
                    if outcome <= 0: # If no TP hit yet, exit at current bar's close as SL
                        outcome = 0 # Consider it a time-based stop
                        final_exit_price = df['close'].iloc[k]
                    # If a TP was hit (outcome > 0), it would have broken or will break above.
                    # If it hit TP1/TP2 and then time exit, final_exit_price is already set to that TP.
                    break 
            
            # If trade not entered yet and current bar is too far from pivot (e.g., > N bars, maybe 5-10)
            # then consider it a missed entry.
            if not entered_trade and (k > i + 10): # Example: if entry not triggered within 10 bars of pivot
                outcome = -1 # Explicitly mark as no entry
                break

        if entered_trade:
            df.loc[i, 'trade_outcome'] = outcome
            df.loc[i, 'entry_bar_idx_sim'] = entry_bar_k
            df.loc[i, 'exit_bar_idx_sim'] = exit_bar_k # k at break or end of loop
            df.loc[i, 'exit_price_sim'] = final_exit_price
            df.loc[i, 'duration_sim'] = exit_bar_k - entry_bar_k + 1 if entry_bar_k != -1 and exit_bar_k != -1 else pd.NA
            
            if pd.notna(initial_risk_per_unit) and initial_risk_per_unit > 0 and pd.notna(final_exit_price):
                if is_long_trade:
                    r_multiple = (final_exit_price - entry_price) / initial_risk_per_unit
                else: # Short
                    r_multiple = (entry_price - final_exit_price) / initial_risk_per_unit
                df.loc[i, 'r_multiple_sim'] = r_multiple
            else: # If risk is zero or exit price is NaN (e.g. not entered)
                df.loc[i, 'r_multiple_sim'] = 0.0 if outcome == 0 else np.nan # 0R for SL if risk was 0, else NaN
        else:
            df.loc[i, 'trade_outcome'] = -1 # No entry triggered or resolved
            entries_not_triggered +=1
            # Other sim fields remain NaN/None

    print(f"DEBUG (simulate_fib_entries): Attempted {simulated_trades} simulations. "
          f"Skipped for bad ATR: {skipped_due_to_bad_atr}. Entries not triggered/resolved: {entries_not_triggered}.")
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

def engineer_pivot_features(df, atr_col_name, force_live_bars_since_pivot_calc: bool = False): # Added force_live_bars_since_pivot_calc
    """
    Engineers features for the pivot detection model.
    `atr_col_name` should be the name of the ATR column to use (e.g., 'atr_14').
    `force_live_bars_since_pivot_calc` forces the use of candidate pivots for 'bars_since_last_pivot'.
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
    # df['range_atr_norm'] = (df['high'] - df['low']) / df[atr_col_name]
    df['range_atr_norm'] = np.where(df[atr_col_name] == 0, 0, (df['high'] - df['low']) / df[atr_col_name])


    # Trend & Momentum
    df['ema12'] = calculate_ema(df, 12)
    df['ema26'] = calculate_ema(df, 26)
    df['macd_line'] = df['ema12'] - df['ema26']
    # Use dynamic atr_col_name for normalization
    # df['macd_slope_atr_norm'] = df['macd_line'].diff() / df[atr_col_name]
    macd_slope = df['macd_line'].diff()
    df['macd_slope_atr_norm'] = np.where(df[atr_col_name] == 0, 0, macd_slope / df[atr_col_name])


    for n in [1, 3, 5]:
        # Use dynamic atr_col_name for normalization
        # df[f'return_{n}b_atr_norm'] = df['close'].pct_change(n) / df[atr_col_name]
        return_n_b = df['close'].pct_change(n)
        df[f'return_{n}b_atr_norm'] = np.where(df[atr_col_name] == 0, 0, return_n_b / df[atr_col_name])


    # Local Structure
    df['high_rank_7'] = df['high'].rolling(window=7).rank(pct=True)
    
    # --- Bars Since Last Pivot Calculation (Revised) ---
    MAX_BARS_NO_PIVOT_CONSTANT = 200
    df['bars_since_last_pivot'] = MAX_BARS_NO_PIVOT_CONSTANT # Default

    has_pre_calculated_pivots = False
    # Determine if actual pre-calculated pivots exist (for the original "TrainingLabels" path)
    if 'is_swing_high' in df.columns and 'is_swing_low' in df.columns:
        if (df['is_swing_high'] == 1).any() or (df['is_swing_low'] == 1).any():
            has_pre_calculated_pivots = True # True labels exist

    # Decide which path to take for bars_since_last_pivot
    # If force_live_bars_since_pivot_calc is True, always use the candidate-based method.
    # Otherwise, use true pivots if they exist, else fall back to candidates.
    use_candidate_pivot_path = force_live_bars_since_pivot_calc or not has_pre_calculated_pivots
    path_type_for_log = "" # Will be set below

    if not use_candidate_pivot_path:
        # Path 1: Using actual pre-calculated (pruned) pivots - Original "TrainingLabels" path
        path_type_for_log = "TrainingLabels"
        pivot_indices_series = df[(df['is_swing_high'] == 1) | (df['is_swing_low'] == 1)].index
        if not pivot_indices_series.empty:
            # (Original logic for TrainingLabels path remains here)
            last_pivot_map = pd.Series(index=df.index, dtype='float64')
            current_last_pivot_idx = np.nan # Use np.nan for missing initial pivots
            for idx in df.index:
                pivots_up_to_idx = pivot_indices_series[pivot_indices_series <= idx]
                if not pivots_up_to_idx.empty:
                    current_last_pivot_idx = pivots_up_to_idx[-1]
                last_pivot_map.loc[idx] = current_last_pivot_idx
            
            df_idx_to_pos = pd.Series(range(len(df)), index=df.index)
            # Calculate bars_since: current_pos - pos_of_last_pivot
            # Ensure mapping from df.index (potentially non-integer) to integer positions
            pos_of_last_pivot = last_pivot_map.map(df_idx_to_pos).fillna(-1).astype(int) # Map pivot index labels to positions
            bars_since = df_idx_to_pos - pos_of_last_pivot
            
            # For rows before the first pivot, last_pivot_map would be NaN, pos_of_last_pivot -1
            # So bars_since would be df_idx_to_pos - (-1) = df_idx_to_pos + 1. This is correct.
            # If last_pivot_map.map(df_idx_to_pos) results in NaN (pivot index not in df_idx_to_pos), fillna(-1) handles it.
            df['bars_since_last_pivot'] = bars_since.astype(int)
            # Any remaining NaNs (e.g. if mapping failed unexpectedly) or if no pivots at all, should be caught by fillna later if not here.
            # If pivot_indices_series was empty, this block is skipped, default MAX_BARS_NO_PIVOT_CONSTANT remains.
    # else: # This 'else' corresponds to 'if not use_candidate_pivot_path'
    if use_candidate_pivot_path: # Combined condition for live path or forced live-like calculation
        # Path 2: Using candidate pivots - Original "LiveCandidates" path OR forced for training
        path_type_for_log = "LiveCandidates"
        if force_live_bars_since_pivot_calc:
            path_type_for_log = "TrainingUniversal (ForcedLiveCalc)"

        df_temp_candidates = df.copy() # Operate on a copy to avoid modifying 'is_candidate_high/low' if they already exist
        # Ensure candidate columns are fresh for this calculation if forced
        if force_live_bars_since_pivot_calc or 'is_candidate_high' not in df_temp_candidates.columns:
            df_temp_candidates = generate_candidate_pivots(df_temp_candidates, n_left=PIVOT_N_LEFT, n_right=PIVOT_N_RIGHT) 
        
        candidate_indices = df_temp_candidates[(df_temp_candidates['is_candidate_high']) | (df_temp_candidates['is_candidate_low'])].index
        
        if not candidate_indices.empty:
            last_candidate_pivot_map = pd.Series(index=df.index, dtype='float64') # Use original df.index for map
            current_last_candidate_idx = np.nan
            for idx in df.index: # Iterate using original df's index
                candidates_up_to_idx = candidate_indices[candidate_indices <= idx]
                if not candidates_up_to_idx.empty:
                    current_last_candidate_idx = candidates_up_to_idx[-1]
                last_candidate_pivot_map.loc[idx] = current_last_candidate_idx

            df_idx_to_pos = pd.Series(range(len(df)), index=df.index) # Map for original df
            pos_of_last_candidate_pivot = last_candidate_pivot_map.map(df_idx_to_pos).fillna(-1).astype(int)
            bars_since_candidate = df_idx_to_pos - pos_of_last_candidate_pivot
            df['bars_since_last_pivot'] = bars_since_candidate.astype(int)
        # If no candidates found, 'bars_since_last_pivot' remains MAX_BARS_NO_PIVOT_CONSTANT

    # Ensure any NaNs that might have slipped through (e.g. if df.index was very unusual) are defaulted
    df['bars_since_last_pivot'] = df['bars_since_last_pivot'].fillna(MAX_BARS_NO_PIVOT_CONSTANT)

    # --- Logging for bars_since_last_pivot ---
    # path_type_for_log is now set correctly based on the path taken
    print(f"\n--- `bars_since_last_pivot` Statistics (Path: {path_type_for_log}) ---")
    if 'bars_since_last_pivot' in df and not df['bars_since_last_pivot'].empty:
        print(df['bars_since_last_pivot'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))
    else:
        print("`bars_since_last_pivot` column not found or empty.")
    print(f"--- End `bars_since_last_pivot` Statistics ---\n")
    # --- End Logging ---

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
    # df.replace([np.inf, -np.inf], np.nan, inplace=True) # Moved after clipping

    # --- Clipping extreme values for specific normalized features ---
    # Define a clipping range (e.g., -5 to 5 standard deviations, or fixed values)
    # For now, let's use a fixed range that seems reasonable for normalized slopes/returns.
    CLIP_MIN = -5.0
    CLIP_MAX = 5.0

    features_to_clip = [
        'macd_slope_atr_norm',
        'return_1b_atr_norm', 'return_3b_atr_norm', 'return_5b_atr_norm'
    ]
    for feature_name in features_to_clip:
        if feature_name in df.columns:
            # Log before clipping for comparison, if needed for debugging, but can be verbose
            # print(f"Clipping {feature_name}: Min before: {df[feature_name].min()}, Max before: {df[feature_name].max()}")
            df[feature_name] = df[feature_name].clip(lower=CLIP_MIN, upper=CLIP_MAX)
            # print(f"Clipping {feature_name}: Min after: {df[feature_name].min()}, Max after: {df[feature_name].max()}")
    # --- End Clipping ---

    # --- Enhanced Logging for Feature Statistics ---
    print(f"\n--- Feature Statistics from engineer_pivot_features (ATR: {atr_col_name}) ---")
    for col in feature_cols:
        if col in df:
            stats = df[col].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
            print(f"\nFeature: {col}")
            print(stats)
        else:
            print(f"\nFeature: {col} - Not found in DataFrame after engineering.")
    print("--- End Feature Statistics from engineer_pivot_features ---\n")
    # --- End Enhanced Logging ---

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
            'hour_of_day', 'day_of_week' # Removed 'vol_regime'
        ]
    else:
        _entry_feature_cols_base = entry_features_base_list_arg
        if 'vol_regime' in _entry_feature_cols_base: # Ensure it's removed if passed in
            _entry_feature_cols_base.remove('vol_regime')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df, _entry_feature_cols_base


# --- 3. Model Training & Validation ---

# Modified to accept model specific kwargs
def train_pivot_model(X_train, y_train, X_val, y_val, model_type='lgbm', **kwargs):
    """Trains pivot detection model, accepting kwargs for model parameters."""
    print(f"DEBUG (train_pivot_model): Received kwargs: {kwargs}")
    
    model_params = {'class_weight': 'balanced', 'random_state': 42}
    
    # Filter kwargs to only include those relevant for the specific model type
    if model_type == 'lgbm':
        lgbm_valid_keys = ['num_leaves', 'learning_rate', 'max_depth', 'n_estimators', 'reg_alpha', 'reg_lambda', 'colsample_bytree', 'subsample', 'min_child_samples'] # Add more as tuned by Optuna
        model_params.update({k: v for k, v in kwargs.items() if k in lgbm_valid_keys})
        # Ensure n_estimators is reasonable, Optuna might not always set it.
        if 'n_estimators' not in model_params: model_params['n_estimators'] = 100 # Default
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(stopping_rounds=10, verbose=-1)])
    elif model_type == 'rf':
        rf_valid_keys = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'] # Add more as tuned
        model_params.update({k: v for k, v in kwargs.items() if k in rf_valid_keys})
        if 'n_estimators' not in model_params: model_params['n_estimators'] = 100
        if 'max_depth' not in model_params and 'max_depth' in kwargs : model_params['max_depth'] = kwargs['max_depth'] # Ensure max_depth from Optuna is used if provided
        elif 'max_depth' not in model_params : model_params['max_depth'] = 7 # Default if not from Optuna

        model = RandomForestClassifier(**model_params)
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
    
    pivot_val_metrics = {
        "precision_high": precision_high, "recall_high": recall_high,
        "precision_low": precision_low, "recall_low": recall_low,
        # Add F1 or other metrics if needed
    }
    # print(confusion_matrix(y_val, preds_val)) # Keep for debug if needed
    return model, pivot_val_metrics

def train_entry_model(X_train, y_train, X_val, y_val, model_type='lgbm', **kwargs):
    """Trains entry profitability model, accepting kwargs for model parameters.
       Returns the trained model and its validation metrics."""
    print(f"DEBUG (train_entry_model): Received kwargs: {kwargs}")

    model_params = {'class_weight': 'balanced', 'random_state': 42}

    if model_type == 'lgbm':
        lgbm_valid_keys = ['num_leaves', 'learning_rate', 'max_depth', 'n_estimators', 'reg_alpha', 'reg_lambda', 'colsample_bytree', 'subsample', 'min_child_samples']
        model_params.update({k: v for k, v in kwargs.items() if k in lgbm_valid_keys})
        if 'n_estimators' not in model_params: model_params['n_estimators'] = 100
        
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(stopping_rounds=10, verbose=-1)])
    elif model_type == 'rf':
        rf_valid_keys = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
        model_params.update({k: v for k, v in kwargs.items() if k in rf_valid_keys})
        if 'n_estimators' not in model_params: model_params['n_estimators'] = 100
        if 'max_depth' not in model_params and 'max_depth' in kwargs: model_params['max_depth'] = kwargs['max_depth']
        elif 'max_depth' not in model_params: model_params['max_depth'] = 7

        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
    else:
        raise ValueError("Unsupported model type for entry evaluation.")
    
    # Evaluate (example for binary classification)
    preds_val_entry = model.predict(X_val)
    # y_val is binary (0 for not profitable, 1 for profitable)
    precision_profit = precision_score(y_val, preds_val_entry, pos_label=1, zero_division=0)
    recall_profit = recall_score(y_val, preds_val_entry, pos_label=1, zero_division=0)
    precision_not_profit = precision_score(y_val, preds_val_entry, pos_label=0, zero_division=0)
    recall_not_profit = recall_score(y_val, preds_val_entry, pos_label=0, zero_division=0)

    print(f"Entry Model ({model_type}) Validation (y_val shape: {y_val.shape}, unique: {np.unique(y_val)}):")
    print(f"  Precision (Profitable): {precision_profit:.3f}, Recall (Profitable): {recall_profit:.3f}")
    print(f"  Precision (Not Profitable): {precision_not_profit:.3f}, Recall (Not Profitable): {recall_not_profit:.3f}")

    entry_val_metrics = {
        "precision_profit": precision_profit, "recall_profit": recall_profit,
        "precision_not_profit": precision_not_profit, "recall_not_profit": recall_not_profit
    }
    return model, entry_val_metrics


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
    p_swing_threshold = trial.suggest_float('p_swing_threshold', 0.2, 0.9) # Lowered min further
    profit_threshold = trial.suggest_float('profit_threshold', 0.2, 0.9) # Lowered min (used as Expected R threshold)

    # --- Strategy/Data Processing Parameters to Tune ---
    current_atr_period = trial.suggest_int('atr_period_opt', 10, 24)
    current_pivot_n_left = trial.suggest_int('pivot_n_left_opt', 2, 7)
    current_pivot_n_right = trial.suggest_int('pivot_n_right_opt', 2, 7)
    current_min_atr_distance = trial.suggest_float('min_atr_distance_opt', 0.2, 2.5) # Lowered min
    current_min_bar_gap = trial.suggest_int('min_bar_gap_opt', 2, 15) # Lowered min

    atr_col_name_optuna = f'atr_{current_atr_period}'
    print(f"Optuna Trial - Params: ATR_P={current_atr_period}, Pivot_L={current_pivot_n_left}, Pivot_R={current_pivot_n_right}, MinATR_D={current_min_atr_distance:.2f}, MinBar_G={current_min_bar_gap}")

    # --- Per-Trial Data Processing ---
    df_trial_processed = df_raw.copy() # Start with a fresh copy of the raw data for each trial
    df_trial_processed.reset_index(drop=True, inplace=True) # Ensure 0-based index for concatenated DFs

    # 1. Calculate ATR for the current trial's period
    df_trial_processed = calculate_atr(df_trial_processed, period=current_atr_period)
    if atr_col_name_optuna not in df_trial_processed.columns:
        print(f"Error: ATR column '{atr_col_name_optuna}' not created in trial. Skipping trial.")
        return -100.0 # Penalize heavily

    # 2. Generate candidate pivots (will be used by engineer_pivot_features if force_live_bars_since_pivot_calc=True,
    # but also needed for prune_and_label_pivots if that's still part of the sequence before feature engineering)
    df_trial_processed = generate_candidate_pivots(df_trial_processed, n_left=current_pivot_n_left, n_right=current_pivot_n_right)

    # 3. Prune and label pivots (to get the actual y_target for training)
    df_trial_processed = prune_and_label_pivots(df_trial_processed, atr_col_name=atr_col_name_optuna, 
                                                atr_distance_factor=current_min_atr_distance, 
                                                min_bar_gap=current_min_bar_gap)

    # 4. Simulate Fibonacci entries (for entry model's y_target)
    df_trial_processed = simulate_fib_entries(df_trial_processed, atr_col_name=atr_col_name_optuna)
    
    # Drop rows with NaN in critical columns that might have been introduced or not handled by ATR/simulation
    # This is important before feature engineering
    df_trial_processed.dropna(subset=[atr_col_name_optuna, 'low', 'high', 'close'], inplace=True) # Add other critical columns if necessary
    df_trial_processed.reset_index(drop=True, inplace=True)

    if len(df_trial_processed) < 100: # Check if enough data remains after initial processing
        print(f"Warning: Not enough data ({len(df_trial_processed)} rows) after initial trial processing. Skipping trial.")
        return -99.0

    # 5. Engineer pivot features (returns DataFrame and pivot_feature_names for this trial)
    # Force bars_since_last_pivot to use the candidate-based calculation for training consistency with live
    df_trial_processed, trial_pivot_features = engineer_pivot_features(
        df_trial_processed, 
        atr_col_name=atr_col_name_optuna,
        force_live_bars_since_pivot_calc=True 
    )

    # 6. Engineer entry features (returns DataFrame and entry_feature_names_base for this trial)
    # Pass the static_entry_features_base list which engineer_entry_features will use and potentially extend
    # with atr_col_name_optuna related features.
    df_trial_processed, trial_entry_features_base = engineer_entry_features(
        df_trial_processed, 
        atr_col_name=atr_col_name_optuna, 
        entry_features_base_list_arg=static_entry_features_base
    )

    # df.replace([np.inf, -np.inf], np.nan, inplace=True) # This is now inside engineer_pivot_features & engineer_entry_features
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
    
    initial_train_pivots = df_train[(df_train['pivot_label'].isin([1, 2])) & (df_train['trade_outcome'] != -1)]
    print(f"DEBUG (Optuna): Initial Train Pivots with outcome: {len(initial_train_pivots)}")

    entry_train_candidates = initial_train_pivots[
        initial_train_pivots['P_swing'] >= p_swing_threshold
    ].copy()
    print(f"DEBUG (Optuna): Train Candidates after P_swing ({p_swing_threshold:.2f}) filter: {len(entry_train_candidates)}")


    if len(entry_train_candidates) < 50: # Min candidates for reliable entry model training
        print(f"DEBUG (Optuna): Insufficient train candidates ({len(entry_train_candidates)}) after P_swing. Penalizing trial.")
        return -100.0 # Penalize more if not enough data for entry model

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

    p_profit_val = entry_model.predict_proba(X_entry_eval)[:, 1] # Assuming class 1 is "profitable"
    
    print(f"DEBUG (Optuna): Validation Pivots for Entry Eval: {len(potential_pivots_val)}. P_profit scores sample: {p_profit_val[:5]}")

    final_trades_val = potential_pivots_val[p_profit_val >= profit_threshold]
    print(f"DEBUG (Optuna): Final Validation Trades after P_profit ({profit_threshold:.2f}) filter: {len(final_trades_val)}")


    if len(final_trades_val) == 0:
        return -10.0 # Penalize heavily if no trades are made by the strategy

    # Calculate Net R
    net_r = 0
    for idx, trade in final_trades_val.iterrows():
        outcome = trade['trade_outcome']
        if outcome == 0: net_r -= 1  # SL
        elif outcome == 1: net_r += 1 # TP1
        elif outcome == 2: net_r += 2 # TP2
        elif outcome == 3: net_r += 3 # TP3
    
    # Add a small penalty for very few trades to encourage more robust strategies
    # if a strategy makes positive R but very few trades, it might be less preferred.
    trade_count_penalty = 0
    min_trades_for_no_penalty = 5 # Example value, can be tuned or part of Optuna study
    if len(final_trades_val) < min_trades_for_no_penalty:
        # Penalize by 0.5R for each trade short of min_trades_for_no_penalty,
        # but don't let penalty itself make a positive R negative if it's just a few trades.
        # This penalty is applied to Net R.
        trade_count_penalty = (min_trades_for_no_penalty - len(final_trades_val)) * 0.5 
        # Example: 3 trades, net_r = 2. Penalty = (5-3)*0.5 = 1. Objective = 2-1=1.
        # Example: 1 trade, net_r = 3. Penalty = (5-1)*0.5 = 2. Objective = 3-2=1.
        # Example: 1 trade, net_r = -1. Penalty = 2. Objective = -1-2 = -3.

    final_objective_value = net_r - trade_count_penalty
    return final_objective_value


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
    df_processed.reset_index(drop=True, inplace=True) # Ensure 0-based index

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
    # Force bars_since_last_pivot to use the candidate-based calculation for training consistency with live
    df_processed, final_pivot_features = engineer_pivot_features(
        df_processed, 
        atr_col_name=atr_col_name,
        force_live_bars_since_pivot_calc=True
    )

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
    # Ensure 'predicted_pivot_class' (from pivot_model) is used for pivot direction if available.
    # 'entry_features_base' should be the list of feature names determined when models were finalized.
    potential_pivots_test['norm_dist_entry_pivot'] = (potential_pivots_test['entry_price_sim'] - potential_pivots_test.apply(lambda r: r['low'] if r['predicted_pivot_class'] == 2 else r['high'], axis=1)) / potential_pivots_test[atr_col_name_backtest]
    potential_pivots_test['norm_dist_entry_sl'] = (potential_pivots_test['entry_price_sim'] - potential_pivots_test['sl_price_sim']).abs() / potential_pivots_test[atr_col_name_backtest]

    # Construct the full list of entry features for the backtest
    backtest_full_entry_features = entry_features_base + ['P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl']
    X_entry_test = potential_pivots_test[backtest_full_entry_features].fillna(-1)

    if len(X_entry_test) == 0:
        print("No data for entry model evaluation in test set after feature engineering.")
        return 0, 0, 0, 0 # Return zero for metrics

    # 4. Entry Predictions (Binary: Not Profitable vs. Profitable)
    # Entry model is trained as binary, so predict_proba gives [P_NotProfitable, P_Profitable]
    p_profit_test = entry_model.predict_proba(X_entry_test)[:, 1] # Probability of class 1 (profitable)

    # 5. Filter by P_profit threshold (consistent with Optuna's validation)
    profit_threshold_backtest = best_params.get('profit_threshold', 0.6) # Default from Optuna, should be consistent
    final_trades_test = potential_pivots_test[p_profit_test >= profit_threshold_backtest].copy()
    
    print(f"DEBUG (full_backtest): Test Pivots for Entry Eval: {len(potential_pivots_test)}. P_profit scores sample: {p_profit_test[:5]}")
    print(f"DEBUG (full_backtest): Final Test Trades after P_profit ({profit_threshold_backtest:.2f}) filter: {len(final_trades_test)}")


    if len(final_trades_test) == 0:
        print(f"No trades passed P_profit threshold ({profit_threshold_backtest:.2f}) in test set.")
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

    # --- Enhanced Data Collection for Return ---
    per_trade_details_list = []
    if not final_trades_for_metrics.empty:
        # Ensure relevant columns from simulate_fib_entries are present
        # (entry_bar_idx_sim, exit_bar_idx_sim, trade_direction_sim, exit_price_sim, r_multiple_sim)
        # Also P_swing and P_profit scores need to be aligned with these trades.
        
        # Align P_profit scores (p_profit_test_ml) with final_trades_for_metrics
        # This assumes final_trades_for_metrics is a subset of potential_pivots_ml,
        # and p_profit_test_ml was calculated on potential_pivots_ml.
        # We need to ensure correct indexing if they are not directly aligned.
        # For "Full ML Pipeline", final_trades_for_metrics is potential_pivots_ml[p_profit_test_ml >= profit_threshold]
        # So, we can add p_profit_test_ml as a column to potential_pivots_ml first.
        
        # This section needs careful alignment of P_swing and P_profit scores with the final trades.
        # 'P_swing' is already in df_test, so it's in final_trades_for_metrics.
        # For 'P_profit', it's calculated on X_entry_test_ml (derived from potential_pivots_ml).
        # We need to map these scores back or ensure they are columns in final_trades_for_metrics.
        
        # Let's assume 'P_swing' is available. For 'P_profit', if scenario is "Full ML Pipeline",
        # we can try to add it.
        temp_final_trades = final_trades_for_metrics.copy()
        if scenario_name == "Full ML Pipeline" and 'p_profit_test_ml' in locals() and len(p_profit_test_ml) == len(X_entry_test_ml):
             # This assumes X_entry_test_ml has the same index as potential_pivots_ml
             # And final_trades_for_metrics is a filtered version of potential_pivots_ml
             if X_entry_test_ml.index.equals(potential_pivots_ml.index): # Check index alignment
                 potential_pivots_ml['P_profit_score_calc'] = p_profit_test_ml
                 # Merge this back to temp_final_trades based on index
                 temp_final_trades = temp_final_trades.join(potential_pivots_ml[['P_profit_score_calc']])


        for idx, trade_row in temp_final_trades.iterrows():
            details = {
                "scenario": scenario_name, # Added scenario name
                "trade_id": idx, # Using DataFrame index as trade_id for this backtest
                "timestamp_pivot": trade_row.name if isinstance(trade_row.name, pd.Timestamp) else pd.NaT, # Pivot timestamp
                "symbol": trade_row.get('symbol', 'N/A'), # If 'symbol' column was carried
                "direction": trade_row.get('trade_direction_sim', ("long" if trade_row.get('is_swing_low') == 1 else "short") if pd.notna(trade_row.get('is_swing_low')) else "N/A"),
                "entry_price_sim": trade_row.get('entry_price_sim'),
                "sl_price_sim": trade_row.get('sl_price_sim'),
                "tp1_price_sim": trade_row.get('tp1_price_sim'),
                "tp2_price_sim": trade_row.get('tp2_price_sim'),
                "tp3_price_sim": trade_row.get('tp3_price_sim'),
                "outcome_label": trade_row.get('trade_outcome'),
                "exit_price_sim": trade_row.get('exit_price_sim'),
                "r_multiple": trade_row.get('r_multiple_sim'),
                "duration_bars": trade_row.get('duration_sim'),
                "p_swing_score": trade_row.get('P_swing'),
                "p_profit_score": trade_row.get('P_profit_score_calc', np.nan), # From merged data
                # Add key feature values (example, customize based on pivot_features list)
                # This needs pivot_features list to be available here.
            }
            # Add first few pivot features as an example
            for i, feat_name in enumerate(pivot_features[:3]): # Top 3 pivot features
                 details[f"pivot_feat_{i+1}_{feat_name}"] = trade_row.get(feat_name)
            
            per_trade_details_list.append(details)
    
    per_trade_details_df = pd.DataFrame(per_trade_details_list)

    # Score Distributions (Simplified for now, can be expanded)
    score_distributions = {
        "p_swing_all_considered": df_test['P_swing'].dropna().describe().to_dict() if 'P_swing' in df_test else {},
        "p_swing_trades_taken": final_trades_for_metrics['P_swing'].dropna().describe().to_dict() if 'P_swing' in final_trades_for_metrics and not final_trades_for_metrics.empty else {}
    }
    if scenario_name == "Full ML Pipeline" and 'p_profit_test_ml' in locals():
        # Describe p_profit_test_ml (scores for trades considered by entry model)
        score_distributions["p_profit_all_considered_by_entry_model"] = pd.Series(p_profit_test_ml).dropna().describe().to_dict()
        # Describe p_profit scores for trades actually taken (those that passed profit_threshold)
        if 'P_profit_score_calc' in temp_final_trades and not temp_final_trades.empty: # Use the merged P_profit
             score_distributions["p_profit_trades_taken"] = temp_final_trades['P_profit_score_calc'].dropna().describe().to_dict()


    # Segmented Performance (Long/Short)
    segmented_performance = {}
    if not final_trades_for_metrics.empty and 'trade_direction_sim' in final_trades_for_metrics.columns:
        for direction in ['long', 'short']:
            dir_trades = final_trades_for_metrics[final_trades_for_metrics['trade_direction_sim'] == direction]
            if not dir_trades.empty:
                num_dir_trades = len(dir_trades)
                num_dir_wins = len(dir_trades[dir_trades['trade_outcome'] > 0])
                dir_win_rate = num_dir_wins / num_dir_trades if num_dir_trades > 0 else 0
                
                dir_r_values = dir_trades['r_multiple_sim'].dropna().tolist()
                dir_avg_r = np.mean(dir_r_values) if dir_r_values else 0
                dir_profit_sum_r = sum(r for r in dir_r_values if r > 0)
                dir_loss_sum_r_abs = sum(abs(r) for r in dir_r_values if r < 0)
                dir_profit_factor = dir_profit_sum_r / dir_loss_sum_r_abs if dir_loss_sum_r_abs > 0 else (dir_profit_sum_r if dir_profit_sum_r > 0 else 0)
                
                segmented_performance[direction] = {
                    "trades": num_dir_trades, "win_rate": dir_win_rate, 
                    "avg_r": dir_avg_r, "profit_factor": dir_profit_factor
                }
    
    # Base summary metrics
    summary_metrics_dict = {
        "scenario": scenario_name, "trades": num_trades, "win_rate": win_rate,
        "avg_r": avg_r, "profit_factor": profit_factor, "max_dd_r": max_drawdown_r,
        "trade_frequency": trade_frequency
    }

    # Combine all results
    full_results_dict = {
        "summary_metrics": summary_metrics_dict,
        "per_trade_details": per_trade_details_df, # This is a DataFrame
        "score_distributions": score_distributions, # Dict of Series.describe() dicts
        "equity_curve_r": equity_curve_r, # List
        "segmented_performance": segmented_performance # Dict of dicts
        # "symbol_context": df_test.name if hasattr(df_test, 'name') else 'N/A' # Already in summary_metrics
    }
    return full_results_dict

# --- Backtest Summary Generation ---
def generate_training_backtest_summary(df_processed_full_dataset, # This is the full dataset used for training/val/test split
                                       pivot_model, entry_model, 
                                       best_params_from_optuna, 
                                       pivot_feature_names_list, entry_feature_names_base_list,
                                       app_settings_dict):
    """
    Generates and displays a summary of backtest results for different scenarios.
    Uses a portion of df_processed_full_dataset as a consistent test set for all scenarios.
    """
    log_prefix_summary = "[TrainingSummary]"
    print(f"\n{log_prefix_summary} --- Generating Training Backtest Summary ---")

    all_scenario_results = []
    
    # Define symbols for summary - could come from app_settings_dict or be fixed
    # For now, let's assume the summary is for the universal model's performance on a test split
    # of the data it was trained on, rather than re-processing specific symbols from scratch.
    # This uses the test portion of the universal dataset.
    
    # Determine ATR column name from best_params_from_optuna
    atr_period_summary = best_params_from_optuna.get('atr_period_opt', ATR_PERIOD) # Default if not in optuna params
    atr_col_name_summary = f'atr_{atr_period_summary}'

    # Scenarios to run
    scenarios = ["Rule-Based Baseline", "ML Stage 1 (Pivot Filter)", "Full ML Pipeline"]

    # The df_processed_full_dataset is the one that was split into train/val/test for the main model training.
    # full_backtest already takes the last 15% of this as its test set.
    # We can call run_backtest_scenario with use_full_df_as_test=False, and it will derive the same test set.
    # Or, to be explicit, we can split it here once. Let's rely on run_backtest_scenario's default split for now.
    
    print(f"{log_prefix_summary} Using universal processed dataset for summary. Test split will be derived by run_backtest_scenario.")

    for scenario in scenarios:
        print(f"{log_prefix_summary} Running scenario: {scenario}...")
        # Note: df_processed_full_dataset might not have 'symbol' if it's concatenated.
        # run_backtest_scenario doesn't strictly need symbol name for metrics, but good for context.
        
        # Ensure the atr_col_name_summary exists in df_processed_full_dataset if it's different from default ATR_PERIOD
        # This should be handled if df_processed_full_dataset was created by process_dataframe_with_params
        # using best_params_from_optuna which set the correct atr_period.
        if atr_col_name_summary not in df_processed_full_dataset.columns:
            print(f"{log_prefix_summary} WARNING: ATR column '{atr_col_name_summary}' for scenario '{scenario}' not found in provided df_processed_full_dataset. Skipping scenario.")
            # Fallback: try with default ATR_PERIOD if that column exists? Or just skip.
            # For now, skipping as it implies a data mismatch.
            # A more robust approach would be to re-process a subset of data using the specific ATR period for each symbol.
            # However, the goal here is a summary of the *trained universal model's performance*.
            # So, we use the df_processed_full_dataset that was used for training it.
            continue


        scenario_results = run_backtest_scenario(
            scenario_name=scenario,
            df_processed=df_processed_full_dataset.copy(), # Pass a copy to avoid in-place modifications
            pivot_model=pivot_model,
            entry_model=entry_model,
            best_params=best_params_from_optuna,
            pivot_features=pivot_feature_names_list,
            entry_features_base=entry_feature_names_base_list,
            atr_col_name=atr_col_name_summary, # Use ATR col consistent with model training
            use_full_df_as_test=False # run_backtest_scenario will take its usual test split (last 15%)
        )
        # Add symbol context if possible/relevant for universal model summary (e.g. "Universal")
        scenario_results['symbol_context'] = "UniversalTestSet" 
        all_scenario_results.append(scenario_results)

    if not all_scenario_results:
        print(f"{log_prefix_summary} No results generated for summary.")
        return

    # --- Collect Feature Importances ---
    feature_importances_data = {}
    if pivot_model and hasattr(pivot_model, 'feature_importances_') and pivot_feature_names_list:
        feature_importances_data['pivot_model'] = pd.DataFrame({
            'feature': pivot_feature_names_list,
            'importance': pivot_model.feature_importances_
        }).sort_values(by='importance', ascending=False)
    else:
        print(f"{log_prefix_summary} Pivot model importances not available or feature names missing. Creating empty DataFrame for pivot_model.")
        feature_importances_data['pivot_model'] = pd.DataFrame(columns=['feature', 'importance'])

    if entry_model and hasattr(entry_model, 'feature_importances_') and entry_feature_names_base_list:
        full_entry_feature_names = entry_feature_names_base_list + ['P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl']
        if len(full_entry_feature_names) == len(entry_model.feature_importances_):
            feature_importances_data['entry_model'] = pd.DataFrame({
                'feature': full_entry_feature_names,
                'importance': entry_model.feature_importances_
            }).sort_values(by='importance', ascending=False)
        else:
            print(f"{log_prefix_summary} WARNING: Mismatch in length of entry feature names and importances. Creating empty DataFrame for entry_model.")
            print(f"  Expected {len(full_entry_feature_names)} features (Features: {full_entry_feature_names}), got {len(entry_model.feature_importances_)} importances (Importances: {entry_model.feature_importances_}).")
            feature_importances_data['entry_model'] = pd.DataFrame(columns=['feature', 'importance'])
    else:
        print(f"{log_prefix_summary} Entry model importances not available or base feature names missing. Creating empty DataFrame for entry_model.")
        feature_importances_data['entry_model'] = pd.DataFrame(columns=['feature', 'importance'])

    # --- Optuna Best Parameters ---
    # best_params_from_optuna is already passed in.

    # --- Package all data for display/saving ---
    # all_scenario_results now contains dictionaries with DataFrames and other structures.
    # display_summary_table will need to be significantly reworked to handle this.
    
    # For now, display_summary_table will just handle the main summary metrics DataFrame.
    # The more detailed DataFrames (per-trade, feature importances) will be saved to separate CSVs.
    
    main_summary_metrics_list = [res['summary_metrics'] for res in all_scenario_results if 'summary_metrics' in res]
    main_summary_df = pd.DataFrame(main_summary_metrics_list) if main_summary_metrics_list else pd.DataFrame()
    
    # --- Prepare consolidated data structures ---
    # 1. Consolidated Trade Details
    all_trades_list = []
    if all_scenario_results:
        for res in all_scenario_results:
            if res.get("per_trade_details") is not None and not res["per_trade_details"].empty:
                # Scenario name is already added in run_backtest_scenario
                all_trades_list.append(res["per_trade_details"])
    all_trades_df = pd.concat(all_trades_list, ignore_index=True) if all_trades_list else pd.DataFrame()

    # 2. Consolidated Equity Curves
    all_equity_curves_data_list = []
    if all_scenario_results:
        for res in all_scenario_results:
            scenario_name = res.get("summary_metrics", {}).get("scenario", "unknown_scenario")
            equity_curve_r = res.get("equity_curve_r")
            if equity_curve_r is not None: # Will be a list
                for i, r_val in enumerate(equity_curve_r):
                    all_equity_curves_data_list.append({
                        "scenario": scenario_name,
                        "trade_num": i,
                        "cumulative_r": r_val
                    })
    all_equity_curves_df = pd.DataFrame(all_equity_curves_data_list) if all_equity_curves_data_list else pd.DataFrame()

    # 3. Consolidated Segmented Performance
    all_segmented_performance_data_list = []
    if all_scenario_results:
        for res in all_scenario_results:
            scenario_name = res.get("summary_metrics", {}).get("scenario", "unknown_scenario")
            segmented_perf = res.get("segmented_performance", {})
            if segmented_perf: # Ensure it's not empty
                for segment_name, metrics_dict in segmented_perf.items():
                    for metric_key, metric_value in metrics_dict.items():
                        all_segmented_performance_data_list.append({
                            "scenario": scenario_name,
                            "segment": segment_name,
                            "metric_name": metric_key,
                            "metric_value": metric_value
                        })
    all_segmented_performance_df = pd.DataFrame(all_segmented_performance_data_list) if all_segmented_performance_data_list else pd.DataFrame()
    
    # 4. Consolidated Score Distributions
    all_score_distributions_data_list = []
    if all_scenario_results:
        for res in all_scenario_results:
            scenario_name = res.get("summary_metrics", {}).get("scenario", "unknown_scenario")
            score_dist = res.get("score_distributions", {})
            if score_dist: # Ensure it's not empty
                for score_type, stats_dict in score_dist.items():
                    if isinstance(stats_dict, dict): 
                        for stat_name, stat_value in stats_dict.items():
                            all_score_distributions_data_list.append({
                                "scenario": scenario_name,
                                "score_type": score_type,
                                "stat_name": stat_name,
                                "stat_value": stat_value
                            })
    all_score_distributions_df = pd.DataFrame(all_score_distributions_data_list) if all_score_distributions_data_list else pd.DataFrame()

    output_filename_base = "training_summary_output" 

    # Pass the new consolidated DataFrames to display_summary_table
    display_summary_table(
        main_summary_df=main_summary_df, 
        all_trades_df=all_trades_df,
        all_equity_curves_df=all_equity_curves_df,
        all_segmented_performance_df=all_segmented_performance_df,
        all_score_distributions_df=all_score_distributions_df,
        feature_importances_dict=feature_importances_data, # Renamed for clarity
        optuna_params_dict=best_params_from_optuna,      # Renamed for clarity
        log_prefix=log_prefix_summary,
        output_base_filename=output_filename_base # Renamed for clarity
    )
    print(f"{log_prefix_summary} --- Training Backtest Summary Generation Complete ---")

def display_summary_table(main_summary_df: pd.DataFrame, 
                          all_trades_df: pd.DataFrame,
                          all_equity_curves_df: pd.DataFrame,
                          all_segmented_performance_df: pd.DataFrame,
                          all_score_distributions_df: pd.DataFrame,
                          feature_importances_dict: dict,
                          optuna_params_dict: dict,
                          log_prefix: str = "[SummaryTable]", 
                          output_base_filename: str = "training_summary"):
    """
    Displays the main summary table and saves all detailed components to separate CSV files.
    """
    if main_summary_df.empty:
        print(f"{log_prefix} No main summary results to display.")
        # Still proceed to save other components if they exist
        # return # Original: returned here. Now, continue to save other parts.

    # df_main_summary = pd.DataFrame(main_summary_list) # Already a DataFrame
    
    columns_ordered = ['scenario', 'trades', 'win_rate', 'avg_r', 'profit_factor', 'max_dd_r', 'trade_frequency']
    
    display_columns = [col for col in columns_ordered if col in main_summary_df.columns]
    df_display = main_summary_df[display_columns].copy() if not main_summary_df.empty else pd.DataFrame(columns=columns_ordered)

    # Apply formatting for console display
    formatters = {
        'win_rate': lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A",
        'avg_r': lambda x: f"{x:.2f}R" if pd.notnull(x) else "N/A",
        'profit_factor': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
        'max_dd_r': lambda x: f"{x:.2f}R" if pd.notnull(x) else "N/A",
        'trade_frequency': lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A",
        'trades': lambda x: f"{int(x)}" if pd.notnull(x) and x != '' else "N/A"
    }
    if not df_display.empty:
        for col, func in formatters.items():
            if col in df_display:
                df_display[col] = df_display[col].map(func)

    print(f"\n{log_prefix} === Main Backtest Summary Overview ===")
    table_string = df_display.to_string(index=False)
    print(table_string)
    print(f"{log_prefix} =====================================\n")

    # --- Setup Logger for this function ---
    # Assuming a global logger might be set up elsewhere, or create a specific one.
    # For now, using print for logger.info/logger.error style messages.
    # A proper logging setup would involve:
    # import logging
    # logger = logging.getLogger(__name__) # Or a specific name
    # And then logger.info(), logger.error()
    
    # --- Save Consolidated Files ---

    # File 1: Overview (already have main_summary_df)
    overview_path = f"{output_base_filename}_overview.csv"
    try:
        if main_summary_df is not None and not main_summary_df.empty:
            print(f"{log_prefix} Attempting to write overview summary. Shape: {main_summary_df.shape}. Path: {overview_path}")
            main_summary_df.to_csv(overview_path, index=False)
            print(f"{log_prefix} INFO: Wrote summary: {overview_path}") # Using print as logger.info
        elif main_summary_df is not None: # Empty DataFrame
            print(f"{log_prefix} Overview summary DataFrame is empty. Writing headers only to {overview_path}")
            pd.DataFrame(columns=main_summary_df.columns if main_summary_df.columns.tolist() else columns_ordered).to_csv(overview_path, index=False)
            print(f"{log_prefix} INFO: Wrote empty summary (headers only): {overview_path}")
        else: # main_summary_df is None
            print(f"{log_prefix} Main summary overview DataFrame is None. Skipping save for {overview_path}.")
    except Exception as e:
        print(f"{log_prefix} ERROR: Failed writing {overview_path}: {e}") # Using print as logger.error

    # File 2: Trade Details Log
    trade_details_log_path = f"{output_base_filename}_trade_details_log.csv"
    try:
        if all_trades_df is not None and not all_trades_df.empty:
            print(f"{log_prefix} Attempting to write trade details log. Shape: {all_trades_df.shape}. Path: {trade_details_log_path}")
            all_trades_df.to_csv(trade_details_log_path, index=False)
            print(f"{log_prefix} INFO: Wrote trade details log: {trade_details_log_path}")
        elif all_trades_df is not None: # Empty DataFrame
            print(f"{log_prefix} Trade details log DataFrame is empty. Writing headers only to {trade_details_log_path}")
            pd.DataFrame(columns=all_trades_df.columns if all_trades_df.columns.tolist() else ['scenario', 'trade_id', 'timestamp_pivot']).to_csv(trade_details_log_path, index=False) # Basic default cols
            print(f"{log_prefix} INFO: Wrote empty trade details log (headers only): {trade_details_log_path}")
        else: # all_trades_df is None
            print(f"{log_prefix} Consolidated trade details DataFrame is None. Skipping save for {trade_details_log_path}.")
    except Exception as e:
        print(f"{log_prefix} ERROR: Failed writing {trade_details_log_path}: {e}")

    # File 3: Model Configs and Importances
    model_configs_importances_list = []
    # Optuna parameters
    if optuna_params_dict:
        for key, value in optuna_params_dict.items():
            model_configs_importances_list.append({
                "record_type": "optuna_parameter",
                "parameter_name": key,
                "parameter_value": value,
                "model_context": "global",
                "feature_name": None,
                "importance_value": None
            })
    # Feature importances
    if feature_importances_dict:
        for model_name, importance_df in feature_importances_dict.items():
            if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
                for _, row in importance_df.iterrows():
                    model_configs_importances_list.append({
                        "record_type": "feature_importance",
                        "parameter_name": None,
                        "parameter_value": None,
                        "model_context": model_name,
                        "feature_name": row.get('feature'),
                        "importance_value": row.get('importance')
                    })
            elif isinstance(importance_df, pd.DataFrame) and importance_df.empty:
                 model_configs_importances_list.append({
                        "record_type": "feature_importance", "parameter_name": None, "parameter_value": None,
                        "model_context": model_name, "feature_name": "N/A (Empty Importances)", "importance_value": None
                    })


    model_configs_importances_df = pd.DataFrame(model_configs_importances_list)
    model_configs_path = f"{output_base_filename}_model_configs_importances.csv"
    try:
        if model_configs_importances_df is not None and not model_configs_importances_df.empty:
            print(f"{log_prefix} Attempting to write model configs & importances. Shape: {model_configs_importances_df.shape}. Path: {model_configs_path}")
            model_configs_importances_df.to_csv(model_configs_path, index=False)
            print(f"{log_prefix} INFO: Wrote model configs and importances: {model_configs_path}")
        elif model_configs_importances_df is not None: # Empty DataFrame
            print(f"{log_prefix} Model configs and importances DataFrame is empty. Writing headers only to {model_configs_path}")
            pd.DataFrame(columns=model_configs_importances_df.columns if model_configs_importances_df.columns.tolist() else ['record_type', 'parameter_name', 'model_context']).to_csv(model_configs_path, index=False) # Basic default
            print(f"{log_prefix} INFO: Wrote empty model configs (headers only): {model_configs_path}")
        else: # DataFrame is None
             print(f"{log_prefix} Model configs and importances DataFrame is None. Skipping save for {model_configs_path}.")
    except Exception as e:
        print(f"{log_prefix} ERROR: Failed writing {model_configs_path}: {e}")

    # File 4: Performance Analytics (Equity Curves, Segmented Perf, Score Dists)
    performance_analytics_list = []
    # Equity curves
    if not all_equity_curves_df.empty:
        for _, row in all_equity_curves_df.iterrows():
            performance_analytics_list.append({
                "record_type": "equity_curve", "scenario": row["scenario"], "trade_num": row["trade_num"], 
                "value1_name": "cumulative_r", "value1": row["cumulative_r"],
                "value2_name": None, "value2": None, "category1": None, "category2": None
            })
    # Segmented performance
    if not all_segmented_performance_df.empty:
        for _, row in all_segmented_performance_df.iterrows():
            performance_analytics_list.append({
                "record_type": "segmented_performance", "scenario": row["scenario"], "trade_num": None,
                "value1_name": row["metric_name"], "value1": row["metric_value"],
                "value2_name": None, "value2": None, "category1": "segment", "category2": row["segment"]
            })
    # Score distributions
    if not all_score_distributions_df.empty:
        for _, row in all_score_distributions_df.iterrows():
            performance_analytics_list.append({
                "record_type": "score_distribution", "scenario": row["scenario"], "trade_num": None,
                "value1_name": row["stat_name"], "value1": row["stat_value"],
                "value2_name": None, "value2": None, "category1": "score_type", "category2": row["score_type"]
            })
            
    performance_analytics_df = pd.DataFrame(performance_analytics_list)
    perf_analytics_path = f"{output_base_filename}_performance_analytics.csv"
    try:
        if performance_analytics_df is not None and not performance_analytics_df.empty:
            print(f"{log_prefix} Attempting to write performance analytics. Shape: {performance_analytics_df.shape}. Path: {perf_analytics_path}")
            performance_analytics_df.to_csv(perf_analytics_path, index=False)
            print(f"{log_prefix} INFO: Wrote performance analytics: {perf_analytics_path}")
        elif performance_analytics_df is not None: # Empty DataFrame
            print(f"{log_prefix} Performance analytics DataFrame is empty. Writing headers only to {perf_analytics_path}")
            pd.DataFrame(columns=performance_analytics_df.columns if performance_analytics_df.columns.tolist() else ['record_type', 'scenario', 'value1_name']).to_csv(perf_analytics_path, index=False) # Basic default
            print(f"{log_prefix} INFO: Wrote empty performance analytics (headers only): {perf_analytics_path}")
        else: # DataFrame is None
            print(f"{log_prefix} Performance analytics DataFrame is None. Skipping save for {perf_analytics_path}.")
    except Exception as e:
        print(f"{log_prefix} ERROR: Failed writing {perf_analytics_path}: {e}")


# --- Main Orchestration ---
def get_processed_data_for_symbol(symbol_ticker, kline_interval, start_date, end_date):
    """
    Fetches, preprocesses, and engineers features for a single symbol.
    Returns a processed DataFrame, pivot feature names, and entry feature names.
    """
    print(f"\n--- Initial Data Processing for Symbol: {symbol_ticker} ---")
    # 1. Data Collection using the new function
    print(f"Getting/Downloading historical data for {symbol_ticker}...")
    try:
        # Initialize Binance client if not already done (e.g. if main script part is not run first)
        global app_binance_client, app_trading_configs # Added app_trading_configs
        if app_binance_client is None:
            # Load trading configs which also loads API keys and initializes client
            # This ensures client is ready for get_or_download_historical_data
            load_app_trading_configs() # Loads default "app_trade_config.csv"
            # Check if app_trading_configs was loaded successfully
            if app_trading_configs is None or not app_trading_configs:
                 print(f"CRITICAL: Failed to load app_trading_configs for {symbol_ticker}. Skipping.")
                 return None, None, None
            if not initialize_app_binance_client(env=app_trading_configs.get("app_trading_environment", "mainnet")):
                print(f"CRITICAL: Failed to initialize Binance client for {symbol_ticker} data processing. Skipping.")
                return None, None, None
        
        historical_df = get_or_download_historical_data(
            symbol=symbol_ticker, 
            interval=kline_interval,
            start_date_str=start_date, # Overall start date for the dataset
            end_date_str=end_date,     # Overall end date for the dataset
            data_directory="historical_data" # Standardized directory
            # force_redownload_all can be a parameter to main script if needed
        )
        if historical_df is None or historical_df.empty: # Check for None as well
            print(f"No data obtained for {symbol_ticker} (None or empty DataFrame). Skipping.")
            return None, None, None
            
    except BinanceAPIException as e: # These might still be caught if client init fails within get_or_download
        print(f"Binance API Exception for {symbol_ticker} during data retrieval: {e}. Skipping symbol.")
        return None, None, None
    except BinanceRequestException as e:
        print(f"Binance Request Exception for {symbol_ticker} during data retrieval: {e}. Skipping symbol.")
        return None, None, None
    except Exception as e: 
        print(f"An unexpected error occurred getting/downloading data for {symbol_ticker}: {e}. Skipping symbol.")
        import traceback
        traceback.print_exc()
        return None, None, None

    print(f"Data obtained for {symbol_ticker}: {len(historical_df)} bars")
    if 'timestamp' not in historical_df.columns:
         historical_df.reset_index(inplace=True) # Should not be needed if get_or_download returns consistent format
    if 'timestamp' not in historical_df.columns and 'index' in historical_df.columns and pd.api.types.is_datetime64_any_dtype(historical_df['index']):
        historical_df.rename(columns={'index':'timestamp'}, inplace=True)
    
    historical_df['symbol'] = symbol_ticker # Add symbol identifier

    # 2. Preprocessing & Labeling
    historical_df = calculate_atr(historical_df, period=ATR_PERIOD) # Ensure ATR_PERIOD is used
    if historical_df is None: return None, None, None # Check after each processing step
    atr_col_name_dynamic = f'atr_{ATR_PERIOD}'
    historical_df = generate_candidate_pivots(historical_df)
    if historical_df is None: return None, None, None
    historical_df = prune_and_label_pivots(historical_df, atr_col_name=atr_col_name_dynamic)
    if historical_df is None: return None, None, None
    historical_df = simulate_fib_entries(historical_df, atr_col_name=atr_col_name_dynamic)
    if historical_df is None: return None, None, None
    
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
    if historical_df is None or pivot_feature_names is None: return None, None, None # Check features as well
    historical_df, entry_feature_names_base = engineer_entry_features(historical_df, atr_col_name=atr_col_name_dynamic, entry_features_base_list_arg=None)
    if historical_df is None or entry_feature_names_base is None: return None, None, None

    historical_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    min_required_data_for_features = 30
    if len(historical_df) < min_required_data_for_features:
        print(f"Not enough data after initial processing for {symbol_ticker} ({len(historical_df)} rows). Skipping.")
        return None, None, None

    df_processed = historical_df.iloc[min_required_data_for_features:].copy()
    
    if 'timestamp' not in df_processed.columns:
        print(f"CRITICAL: 'timestamp' column lost for {symbol_ticker} during processing.")
        return None, None, None # Cannot proceed without timestamp

    if pivot_feature_names is None or not pivot_feature_names: # Check features list itself
        print(f"CRITICAL: Pivot feature names list is None or empty for {symbol_ticker}.")
        return None, None, None
    df_processed.dropna(subset=pivot_feature_names, inplace=True) # Ensure core features are present
    df_processed.reset_index(drop=True, inplace=True)

    if len(df_processed) < 100: # Minimum data for meaningful processing/splitting
        print(f"Not enough final processed data for {symbol_ticker} ({len(df_processed)} rows). Skipping.")
        return None, None, None
        
    print(f"Initial data processing complete for {symbol_ticker}: {len(df_processed)} rows")
    return df_processed, pivot_feature_names, entry_feature_names_base


if __name__ == '__main__':
    # Ensure settings are loaded once at the beginning.
    # This will populate app_settings which includes model paths and other configurations.
    # load_app_settings() will also handle initial user prompts if app_settings.json is missing/invalid.
    load_app_settings()

    # Directly call the main application flow orchestrator.
    # start_app_main_flow() will handle:
    # - Checking if models exist.
    # - If not, running the training pipeline (data processing, Optuna, training, saving).
    # - If models exist, loading them.
    # - Presenting the user with the operational menu.
    start_app_main_flow()

    print("\nApplication finished or exited via menu.")
