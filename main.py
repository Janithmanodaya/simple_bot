# Binance Trading Bot - Advance EMA Cross Strategy
# Author: Jules (AI Software Engineer)
# Date: [Current Date] - Will be filled by system or actual date
#
# This script implements the "Advance EMA Cross" trading strategy (ID: 8) for Binance Futures.
# It connects to the Binance API (testnet or mainnet), fetches market data, calculates EMAs,
# identifies trading signals based on EMA crossovers with price validation, manages position sizing
# based on user-defined risk, places orders (entry, stop-loss, take-profit), and dynamically
# adjusts SL/TP based on P&L movements.
# This version includes multithreading for faster scanning of symbols.

import importlib.util
import sys
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import pandas as pd # For kline data handling and analysis
import time # For delays and timing loops
import math # For rounding, floor, ceil operations
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback # For more detailed error logging in threads
import telegram # For sending Telegram messages
import asyncio
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
from telegram.ext import CallbackContext
import requests # For fetching public IP address
import numpy as np # For numerical operations like log and sqrt

# --- Configuration Defaults ---
DEFAULT_RISK_PERCENT = 1.0       # Default account risk percentage per trade (e.g., 1.0 for 1%)
DEFAULT_LEVERAGE = 20            # Default leverage (e.g., 20x)
DEFAULT_MAX_CONCURRENT_POSITIONS = 5 # Default maximum number of concurrent open positions
DEFAULT_MARGIN_TYPE = "ISOLATED" # Default margin type: "ISOLATED" or "CROSS"
DEFAULT_MAX_SCAN_THREADS = 5     # Default threads for scanning symbols, now fixed to 5
DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL = False # Default for allowing higher risk to meet min notional
DEFAULT_PORTFOLIO_RISK_CAP = 5.0 # Default maximum portfolio risk percentage (e.g., 5.0 for 5%)
DEFAULT_ATR_PERIOD = 14          # Default ATR period
DEFAULT_ATR_MULTIPLIER_SL = 2.0  # Default ATR multiplier for Stop Loss
DEFAULT_TP_RR_RATIO = 1.5        # Default Take Profit Risk:Reward Ratio
DEFAULT_MAX_DRAWDOWN_PERCENT = 5.0 # Default max daily drawdown from high equity (e.g., 5.0 for 5%)
DEFAULT_DAILY_STOP_LOSS_PERCENT = 2.0 # Default daily stop loss based on start equity (e.g., 2.0 for 2%)
DEFAULT_TARGET_ANNUALIZED_VOLATILITY = 0.80 # Target annualized volatility for leverage adjustment (80%)
DEFAULT_REALIZED_VOLATILITY_PERIOD = 30   # Period in candles for calculating realized volatility
DEFAULT_MIN_LEVERAGE = 1                  # Minimum leverage to use
DEFAULT_MAX_LEVERAGE = 20                 # Maximum leverage to use (user preference, also check exchange limits)


# --- Global State Variables ---
# Stores details of active trades. Key: symbol (e.g., "BTCUSDT")
# Value: dict with trade info like order IDs, entry/SL/TP prices, quantity, side.
active_trades = {}
active_trades_lock = threading.Lock() # Lock for synchronizing access to active_trades

# Set to keep track of symbols currently being processed by manage_trade_entry to prevent race conditions.
import datetime # Added for last_trading_day
from datetime import datetime as dt # Alias for ease of use if datetime.datetime is needed alongside datetime.date

symbols_currently_processing = set()
symbols_currently_processing_lock = threading.Lock()

# Globals for Cooldown Timer
last_signal_time = {}
last_signal_lock = threading.Lock()

# Globals for Trade Signature Check
recent_trade_signatures = {} # Stores trade_signature: timestamp
recent_trade_signatures_lock = threading.Lock()
recent_trade_signature_cleanup_interval = 60 # seconds, how often to check for cleanup
last_signature_cleanup_time = dt.now() # Initialize last cleanup time

# Daily performance and halt status variables
daily_high_equity = 0.0
day_start_equity = 0.0
last_trading_day = None # Stores datetime.date object
trading_halted_drawdown = False
trading_halted_daily_loss = False
daily_realized_pnl = 0.0
daily_state_lock = threading.Lock() # Lock for synchronizing access to daily state variables

import os # Added for checking file existence
import pandas as pd # Added for CSV operations, though pandas is already imported later, ensure it's available here

# --- Utility and Configuration Functions ---

def load_configuration_from_csv(filepath: str) -> dict | None:
    """Loads configuration from a CSV file."""
    if not os.path.exists(filepath):
        print(f"Info: Configuration file '{filepath}' not found.")
        return None
    try:
        df = pd.read_csv(filepath)
        if 'name' not in df.columns or 'value' not in df.columns:
            print(f"Error: CSV file '{filepath}' must contain 'name' and 'value' columns.")
            return None
        
        # Convert to dictionary, handling potential NaN values from empty cells
        configs = pd.Series(df.value.values, index=df.name).dropna().to_dict()
        
        if not configs:
            print(f"Info: Configuration file '{filepath}' is empty or contains no valid data.")
            return None
        print(f"Configuration loaded from '{filepath}'.")
        return configs
    except pd.errors.EmptyDataError:
        print(f"Info: Configuration file '{filepath}' is empty.")
        return None
    except Exception as e:
        print(f"Error loading configuration from '{filepath}': {e}")
        return None

def save_configuration_to_csv(filepath: str, configs_to_save: dict) -> bool:
    """Saves configuration to a CSV file."""
    try:
        # Ensure only non-sensitive, relevant data is saved
        # API keys and telegram details are explicitly popped before this function is called.
        # Convert dictionary to DataFrame: name, value
        df_to_save = pd.DataFrame(list(configs_to_save.items()), columns=['name', 'value'])
        df_to_save.to_csv(filepath, index=False)
        print(f"Configuration saved to '{filepath}'.")
        return True
    except Exception as e:
        print(f"Error saving configuration to '{filepath}': {e}")
        return False

def validate_configurations(loaded_configs: dict) -> tuple[bool, str, dict]:
    """
    Validates loaded configurations for types and basic range checks.
    Returns: (is_valid, message, validated_configs_dict)
    """
    if not isinstance(loaded_configs, dict):
        return False, "Configuration data is not a dictionary.", {}

    validated_configs = {}
    # Define expected keys, their types, and optional validation functions/ranges
    # Validation functions take the value and return True if valid, or a string error message
    expected_params = {
        "environment": {"type": str, "valid_values": ["testnet", "mainnet"]},
        "mode": {"type": str, "valid_values": ["live", "backtest"]},
        "backtest_days": {"type": int, "optional": True, "condition": lambda x: x > 0},
        "backtest_start_balance_type": {"type": str, "optional": True, "valid_values": ["current", "custom"]},
        "backtest_custom_start_balance": {"type": float, "optional": True, "condition": lambda x: x > 0},
        "risk_percent": {"type": float, "condition": lambda x: 0 < x <= 1.0}, # Stored as 0.01 for 1%
        "leverage": {"type": int, "condition": lambda x: 1 <= x <= 125},
        "max_concurrent_positions": {"type": int, "condition": lambda x: x > 0},
        "margin_type": {"type": str, "valid_values": ["ISOLATED", "CROSS"]},
        "portfolio_risk_cap": {"type": float, "condition": lambda x: 0 < x <= 100.0},
        "atr_period": {"type": int, "condition": lambda x: x > 0},
        "atr_multiplier_sl": {"type": float, "condition": lambda x: x > 0},
        "tp_rr_ratio": {"type": float, "condition": lambda x: x > 0},
        "max_drawdown_percent": {"type": float, "condition": lambda x: 0 <= x <= 100.0},
        "daily_stop_loss_percent": {"type": float, "condition": lambda x: 0 <= x <= 100.0},
        "target_annualized_volatility": {"type": float, "condition": lambda x: 0 < x <= 5.0}, # e.g. 0.80 for 80%
        "realized_volatility_period": {"type": int, "condition": lambda x: x > 0},
        "min_leverage": {"type": int, "condition": lambda x: 1 <= x <= 125},
        "max_leverage": {"type": int, "condition": lambda x: 1 <= x <= 125}, # Further check against min_leverage done in input logic
        "allow_exceed_risk_for_min_notional": {"type": bool}
        # API keys and telegram details are not part of this CSV validation
    }
    
    bool_true_values = ['true', 'yes', '1', 'y', True, 1, 1.0] # Added True, 1, 1.0 for direct bool/num
    bool_false_values = ['false', 'no', '0', 'n', False, 0, 0.0] # Added False, 0, 0.0

    for key, rules in expected_params.items():
        if key not in loaded_configs:
            if not rules.get("optional", False):
                return False, f"Missing required configuration key: '{key}'.", {}
            continue # Skip optional missing keys

        val_str = str(loaded_configs[key]) # Work with string representation for initial parsing for bools
        val_orig = loaded_configs[key] # Keep original for type check if not bool
        
        try:
            converted_val = None
            if rules["type"] == bool:
                if isinstance(val_orig, bool): # Already a boolean
                    converted_val = val_orig
                elif val_str.lower() in [str(btv).lower() for btv in bool_true_values if isinstance(btv, str)]: # check string versions of true values
                    converted_val = True
                elif val_str.lower() in [str(bfv).lower() for bfv in bool_false_values if isinstance(bfv, str)]: # check string versions of false values
                    converted_val = False
                else: # Try direct conversion for numeric bools (1.0, 0.0 etc)
                    try:
                        num_val = float(val_str)
                        if num_val == 1.0: converted_val = True
                        elif num_val == 0.0: converted_val = False
                        else: return False, f"Invalid boolean value for '{key}': {val_str}.", {}
                    except ValueError:
                         return False, f"Invalid boolean value for '{key}': {val_str}.", {}
            elif rules["type"] == int:
                converted_val = int(float(val_str)) # Convert to float first to handle "10.0" then to int
            elif rules["type"] == float:
                converted_val = float(val_str)
            elif rules["type"] == str:
                converted_val = val_str # Already string, or explicitly convert
            else: # Should not happen with defined rules
                return False, f"Unknown expected type for '{key}'.", {}
            
            validated_configs[key] = converted_val

            # Value validation (e.g., enums, ranges)
            if "valid_values" in rules and converted_val not in rules["valid_values"]:
                return False, f"Invalid value for '{key}': '{converted_val}'. Allowed: {rules['valid_values']}.", {}
            if "condition" in rules:
                condition_check = rules["condition"](converted_val)
                if condition_check is False: # Explicitly check for False
                     return False, f"Value for '{key}' ('{converted_val}') does not meet condition: {rules.get('condition_desc', 'Range/value error')}.", {}
                # If condition_check returns a string, it's an error message (not standard here but could be)

        except ValueError:
            return False, f"Invalid type for '{key}'. Expected {rules['type'].__name__}, got '{val_str}'.", {}
        except Exception as e_val: # Catch any other validation error
             return False, f"Validation error for '{key}' with value '{val_str}': {e_val}", {}

    # Specific inter-dependent checks (e.g. max_leverage >= min_leverage)
    if "min_leverage" in validated_configs and "max_leverage" in validated_configs:
        if validated_configs["max_leverage"] < validated_configs["min_leverage"]:
            return False, "max_leverage cannot be less than min_leverage.", {}
            
    if "mode" in validated_configs and validated_configs["mode"] == "backtest":
        if "backtest_days" not in validated_configs:
            return False, "Missing 'backtest_days' for backtest mode.", {}
        if "backtest_start_balance_type" not in validated_configs:
             return False, "Missing 'backtest_start_balance_type' for backtest mode.", {}
        if validated_configs["backtest_start_balance_type"] == "custom" and "backtest_custom_start_balance" not in validated_configs:
            return False, "Missing 'backtest_custom_start_balance' when type is custom.", {}


    return True, "Validation successful.", validated_configs


def get_current_market_price(client, symbol: str) -> float | None:
    """Fetches the current market price for a symbol."""
    try:
        ticker = client.futures_ticker(symbol=symbol)
        return float(ticker['lastPrice'])
    except Exception as e:
        print(f"Error fetching market price for {symbol}: {e}")
        return None

def calculate_unrealized_pnl(trade_details: dict, current_market_price: float) -> float:
    """Calculates unrealized P&L for a single active trade."""
    if not all(k in trade_details for k in ['entry_price', 'quantity', 'side']) or current_market_price is None:
        return 0.0
    
    entry_price = trade_details['entry_price']
    quantity = trade_details['quantity']
    side = trade_details['side']
    
    if side == "LONG":
        pnl = (current_market_price - entry_price) * quantity
    elif side == "SHORT":
        pnl = (entry_price - current_market_price) * quantity
    else:
        pnl = 0.0
    return pnl

def get_current_equity(client, configs: dict, current_balance: float, active_trades_dict: dict, active_trades_lock_ref: threading.Lock) -> float | None:
    """
    Calculates the current total equity (balance + total unrealized P&L).
    Returns None if balance cannot be fetched or a critical error occurs.
    """
    if current_balance is None: # Ensure balance is valid first
        print("Cannot calculate current equity because current_balance is None.")
        # Attempt to fetch balance again as a fallback, though it might indicate a larger issue
        fetched_balance = get_account_balance(client, configs)
        if fetched_balance is None:
            print("Critical: Failed to fetch balance for equity calculation. Returning None for equity.")
            return None
        current_balance = fetched_balance

    total_unrealized_pnl = 0.0
    
    # Iterate over a copy of items for thread safety if active_trades_dict can be modified elsewhere,
    # although lock protection should make direct iteration safe if modifications are also locked.
    with active_trades_lock_ref:
        trades_to_evaluate = list(active_trades_dict.items()) # Create a list of items to iterate over

    for symbol, trade_details in trades_to_evaluate:
        market_price = get_current_market_price(client, symbol)
        if market_price is not None:
            total_unrealized_pnl += calculate_unrealized_pnl(trade_details, market_price)
        else:
            print(f"Warning: Could not fetch market price for {symbol} to calculate its UPNL. Assuming 0 UPNL for this trade in equity calculation.")
            # Optionally, could return None here to indicate equity is uncertain,
            # or proceed with known UPNL. For now, proceeding with 0 for this symbol.

    return current_balance + total_unrealized_pnl

def manage_daily_state(client, configs: dict, active_trades_dict_ref: dict, active_trades_lock_ref: threading.Lock):
    """
    Manages daily state variables like start equity, high equity, P&L, and halt flags.
    Should be called at the start of each trading cycle.
    Returns the current equity.
    """
    global day_start_equity, daily_high_equity, daily_realized_pnl
    global trading_halted_drawdown, trading_halted_daily_loss, last_trading_day
    global daily_state_lock

    today = datetime.date.today()
    current_balance = get_account_balance(client, configs) # Fetch fresh balance

    if current_balance is None:
        print("CRITICAL: Could not fetch account balance in manage_daily_state. Daily state cannot be reliably updated.")
        # Potentially return a special value or raise an exception if this is critical for operation
        # For now, it will try to use the last known equity values if it's not a new day,
        # or might fail to initialize properly on a new day.
        # Let get_current_equity handle the None balance and return None for equity.
        
    # Calculate current equity *before* acquiring the daily_state_lock for new day check,
    # as get_current_equity might take time (API calls) and uses active_trades_lock.
    # We need a consistent view of active_trades for equity calculation.
    calculated_current_equity = get_current_equity(client, configs, current_balance, active_trades_dict_ref, active_trades_lock_ref)

    if calculated_current_equity is None:
        print("CRITICAL: Equity calculation failed in manage_daily_state. Daily limits might not function correctly.")
        # If equity cannot be determined, we cannot reliably manage daily state.
        # Keep existing halt flags, do not reset. This is a degraded state.
        return None # Indicate failure to get equity

    with daily_state_lock:
        if last_trading_day != today:
            print(f"New trading day ({today}). Resetting daily limits and P&L tracking.")
            day_start_equity = calculated_current_equity
            daily_high_equity = calculated_current_equity
            daily_realized_pnl = 0.0
            trading_halted_drawdown = False
            trading_halted_daily_loss = False
            last_trading_day = today
            
            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                send_telegram_message(
                    configs["telegram_bot_token"], 
                    configs["telegram_chat_id"],
                    f"☀️ New Trading Day ({today}) Started.\n"
                    f"Start Equity: {day_start_equity:.2f} USDT.\n"
                    f"Daily limits reset."
                )
        else:
            # Same day, just update daily_high_equity
            daily_high_equity = max(daily_high_equity, calculated_current_equity)
        
        # For logging current state regardless of new day or not
        print(f"Daily State Update: Start Equity: {day_start_equity:.2f}, High Equity: {daily_high_equity:.2f}, Current Equity: {calculated_current_equity:.2f}, Realized PNL: {daily_realized_pnl:.2f}")
        print(f"Halt Status: Drawdown: {trading_halted_drawdown}, Daily Loss: {trading_halted_daily_loss}")

    return calculated_current_equity

def close_all_open_positions(client, configs: dict, active_trades_dict_ref: dict, lock: threading.Lock):
    """
    Attempts to close all open positions managed by the bot and cancels their SL/TP orders.
    Clears the active_trades_dict_ref after attempting closures.
    """
    log_prefix = "[CloseAllPositions]"
    print(f"{log_prefix} Attempting to close all bot-managed open positions...")

    if not active_trades_dict_ref:
        print(f"{log_prefix} No active trades to close.")
        return

    # Iterate over a copy of items because we might modify the dict later (though clearing happens at the end)
    # The primary reason for copy is if active_trades_dict_ref could be modified by another thread during this,
    # but the lock passed should be active_trades_lock, which protects it.
    # However, operations inside loop (API calls) are outside the lock.
    
    # It's safer to get a list of trades to close first, then operate.
    trades_to_close_info = []
    with lock: # Use the passed lock (active_trades_lock)
        for symbol, details in active_trades_dict_ref.items():
            trades_to_close_info.append({
                "symbol": symbol,
                "sl_order_id": details.get('sl_order_id'),
                "tp_order_id": details.get('tp_order_id'),
                "quantity": details.get('quantity'),
                "side": details.get('side'), # "LONG" or "SHORT"
                "symbol_info": details.get('symbol_info')
            })

    closed_successfully_count = 0
    closed_with_errors_count = 0

    for trade_info in trades_to_close_info:
        symbol = trade_info['symbol']
        s_info = trade_info['symbol_info']
        qty_to_close = trade_info['quantity']
        original_side = trade_info['side']
        position_side_to_close = original_side # For Binance, positionSide is LONG or SHORT

        print(f"{log_prefix} Processing closure for {symbol} ({original_side} {qty_to_close})...")

        # 1. Cancel SL order
        if trade_info['sl_order_id']:
            try:
                print(f"{log_prefix} Cancelling SL order {trade_info['sl_order_id']} for {symbol}...")
                client.futures_cancel_order(symbol=symbol, orderId=trade_info['sl_order_id'])
                print(f"{log_prefix} SL order {trade_info['sl_order_id']} for {symbol} cancelled.")
            except BinanceAPIException as e:
                if e.code == -2011: # Order filled or cancelled / Does not exist
                    print(f"{log_prefix} SL order {trade_info['sl_order_id']} for {symbol} already filled/cancelled or does not exist (Code: {e.code}).")
                else:
                    print(f"{log_prefix} API Error cancelling SL order {trade_info['sl_order_id']} for {symbol}: {e}")
            except Exception as e:
                print(f"{log_prefix} Unexpected error cancelling SL order {trade_info['sl_order_id']} for {symbol}: {e}")
        
        # 2. Cancel TP order
        if trade_info['tp_order_id']:
            try:
                print(f"{log_prefix} Cancelling TP order {trade_info['tp_order_id']} for {symbol}...")
                client.futures_cancel_order(symbol=symbol, orderId=trade_info['tp_order_id'])
                print(f"{log_prefix} TP order {trade_info['tp_order_id']} for {symbol} cancelled.")
            except BinanceAPIException as e:
                if e.code == -2011: # Order filled or cancelled / Does not exist
                    print(f"{log_prefix} TP order {trade_info['tp_order_id']} for {symbol} already filled/cancelled or does not exist (Code: {e.code}).")
                else:
                    print(f"{log_prefix} API Error cancelling TP order {trade_info['tp_order_id']} for {symbol}: {e}")
            except Exception as e:
                print(f"{log_prefix} Unexpected error cancelling TP order {trade_info['tp_order_id']} for {symbol}: {e}")

        # 3. Place Market Close Order
        close_side = "SELL" if original_side == "LONG" else "BUY"
        print(f"{log_prefix} Attempting to place MARKET {close_side} order for {qty_to_close} {symbol} (PositionSide: {position_side_to_close})...")
        
        if s_info is None or qty_to_close is None or qty_to_close <= 0:
            print(f"{log_prefix} Insufficient info to close {symbol} (s_info: {s_info is not None}, qty: {qty_to_close}). Skipping market close.")
            closed_with_errors_count +=1
            continue

        close_order_obj, close_error_msg = place_new_order(
            client, 
            s_info, 
            close_side, 
            "MARKET", 
            qty_to_close,
            position_side=position_side_to_close # Important: specify which side of position to close
            # reduce_only=True should implicitly be handled by MARKET close on a positionSide.
            # For SL/TP, is_closing_order=True was used, which sets closePosition=true.
            # For direct market close, just ensuring it reduces the correct positionSide is key.
            # The `place_new_order` doesn't have a direct `closePosition` toggle for market orders
            # unless `is_closing_order` is True (which is for STOP_MARKET/TAKE_PROFIT_MARKET).
            # A simple MARKET order against the positionSide should reduce/close it.
        )

        if close_order_obj and close_order_obj.get('status') == 'FILLED':
            print(f"{log_prefix} Successfully placed MARKET close order for {symbol}. Order ID: {close_order_obj.get('orderId')}")
            closed_successfully_count += 1
            # P&L for this closure will be handled by monitor_active_trades when it sees the position gone,
            # or if a more immediate P&L update is needed, it could be estimated here.
        else:
            error_detail = f"API Error: {close_error_msg}" if close_error_msg else "Order object missing or status not FILLED."
            print(f"{log_prefix} FAILED to place MARKET close order for {symbol}. Details: {error_detail}. Order: {close_order_obj}")
            closed_with_errors_count += 1
            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                send_telegram_message(
                    configs["telegram_bot_token"], 
                    configs["telegram_chat_id"],
                    f"⚠️ {log_prefix} FAILED to market close {symbol} ({original_side} {qty_to_close}). Manual check required. Error: {error_detail}"
                )
    
    print(f"{log_prefix} Closure attempts summary: {closed_successfully_count} closed successfully, {closed_with_errors_count} failed/skipped.")

    # 4. Clear all trades from the bot's management after attempting closure
    with lock: # Use the passed lock (active_trades_lock)
        if active_trades_dict_ref: # Check if not already empty
            print(f"{log_prefix} Clearing all {len(active_trades_dict_ref)} trades from bot's active management list.")
            active_trades_dict_ref.clear()
        else:
            print(f"{log_prefix} Active trades list was already empty before final clear.")
    
    print(f"{log_prefix} Finished closing all positions procedure.")


def get_public_ip():
    """Fetches the current public IP address of the machine."""
    try:
        response = requests.get("https://api.ipify.org?format=json", timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        ip_address = response.json().get("ip")
        if ip_address:
            print(f"Successfully fetched public IP: {ip_address}")
            return ip_address
        else:
            print("Error: Could not parse IP from ipify response.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching public IP: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching public IP: {e}")
        return None

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
        if spec is None: # Check if spec creation failed
            print("Error: Could not prepare to load keys.py. File might be missing or unreadable.")
            sys.exit(1)
        keys_module = importlib.util.module_from_spec(spec)
        if spec.loader is None: # Check if loader is available
             print("Error: Could not get a loader for keys.py.")
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
            raise ValueError("Invalid environment specified for loading API keys.")

        placeholders_binance = ["<your-testnet-api-key>", "<your-testnet-secret>",
                                "<your-mainnet-api-key>", "<your-mainnet-secret>"]
        if not api_key or not api_secret or api_key in placeholders_binance or api_secret in placeholders_binance:
            print(f"Error: Binance API key/secret for {env} not found or not configured in keys.py.")
            print("Please open keys.py and replace placeholder values for Binance.")
            sys.exit(1)
        
        # Load Telegram keys
        telegram_token = getattr(keys_module, "telegram_bot_token", None)
        telegram_chat = getattr(keys_module, "telegram_chat_id", None)
        
        placeholders_telegram = ["<your-telegram-bot-token>", "<your-telegram-chat-id>"] # General placeholders
        # Specific token from user request is "8184556638:AAE4cJMUf0z7yPoXd5si12SrqV_n_2k4eeQ"
        # Specific chat_id from user request is "7144191785"
        # We check if the token/chat_id in keys.py IS STILL a placeholder or the one provided.
        # The values in keys.py were already updated to the user's actual values.
        # So, this check is more about ensuring they are not generic placeholders if the user hadn't provided them.
        # Since they ARE provided and keys.py is updated, this check might seem redundant for *this specific run*,
        # but it's good practice for the function.

        if not telegram_token or not telegram_chat or \
           telegram_token in placeholders_telegram or telegram_chat in placeholders_telegram or \
           telegram_token == "YOUR_TELEGRAM_BOT_TOKEN" or telegram_chat == "YOUR_TELEGRAM_CHAT_ID": # Common placeholders
            print(f"Warning: Telegram bot token or chat ID not found or not configured in keys.py.")
            print("Telegram notifications will be disabled. Please update keys.py with your Telegram details.")
            # Allow bot to run without Telegram if not configured
            telegram_token, telegram_chat = None, None # Disable if not configured

        return api_key, api_secret, telegram_token, telegram_chat
    except FileNotFoundError:
        print("Error: keys.py not found. Please create it and add your API and Telegram credentials.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading keys: {e}")
        sys.exit(1)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hello! I’m your bot.\n"
        "Use /help to see what I can do."
    )

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Here are the commands you can use:\n"
        "/start — Show welcome message\n"
        "/help  — Show this help text\n"
        "/command3 — Run the special Command3 routine"
    )


def build_startup_message(configs, balance, open_positions_text, bot_start_time_str):
    env_name = configs.get('environment', 'N/A').title()
    mode_name = configs.get('mode', 'N/A').title()
    return (
        f"*🚀 Bot Started Successfully ({configs.get('strategy_name', 'Strategy')}) 🚀*\n\n"
        f"*Start Time:* `{bot_start_time_str}`\n"
        f"*Environment:* `{env_name}`\n"
        f"*Mode:* `{mode_name}`\n"
        f"*Initial Balance:* `{balance}`\n"
        f"*Risk Per Trade:* `{configs.get('risk_percent', 0.0) * 100:.2f}%`\n"
        f"*Leverage:* `{configs.get('leverage', 0)}x`\n"
        f"*Max Concurrent Positions:* `{configs.get('max_concurrent_positions', 0)}`\n"
        f"*Monitored Symbols:* `{configs.get('monitored_symbols_count', '?')}`\n"
        f"\n*Initial Open Positions:*\n{open_positions_text}"
    )
def send_telegram_message(bot_token, chat_id, message):
    """Sends a message to a specified Telegram chat using async."""
    if not bot_token or not chat_id:
        print("Telegram bot token or chat ID not configured. Cannot send message.")
        return False

    async def _send():
        try:
            bot = telegram.Bot(token=bot_token)
            await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
            print(f"Telegram message sent successfully to chat ID {chat_id}.")
            return True
        except telegram.error.TelegramError as e:
            print(f"Error sending Telegram message: {e}")
            if "chat not found" in str(e).lower():
                print(f"Ensure the chat ID {chat_id} is correct and the bot has access to it.")
            elif "bot token is invalid" in str(e).lower():
                print("The Telegram Bot Token in keys.py seems to be invalid.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred while sending Telegram message: {e}")
            return False

    return asyncio.run(_send())

def start_command_listener(bot_token, chat_id, last_message):
    async def command3_handler(update, context: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) == str(chat_id):
            await context.bot.send_message(chat_id=chat_id, text=last_message, parse_mode="Markdown")

def start_command_listener(bot_token, chat_id, last_message_content): # Renamed for clarity
    async def actual_bot_runner_async(): # Renamed to indicate it's async
        application = Application.builder().token(bot_token).build()

        async def command3_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if str(update.effective_chat.id) == str(chat_id):
                # Make sure last_message_content is accessible here.
                # It is, due to closure over start_command_listener's scope.
                await context.bot.send_message(chat_id=chat_id, text=last_message_content, parse_mode="Markdown")
        
        application.add_handler(CommandHandler("command3", command3_handler))
        application.add_handler(CommandHandler("start", start_handler)) # Assuming start_handler is defined
        application.add_handler(CommandHandler("help", help_handler))   # Assuming help_handler is defined

        initialized_successfully = False
        try:
            await application.initialize()
            initialized_successfully = True
            print("Starting Telegram bot polling...")
            # run_polling() is blocking and handles its own loop management when run this way.
            # It will run until application.stop() is called or an error occurs.
            await application.run_polling() 
            print("Telegram bot polling stopped.")
        except telegram.error.TelegramError as te:
            print(f"TelegramError in actual_bot_runner_async: {te}")
            traceback.print_exc()
        except Exception as e:
            print(f"General exception in Telegram actual_bot_runner_async: {e}")
            traceback.print_exc()
        finally:
            if initialized_successfully: # Only try to shutdown if initialize was successful
                print("Telegram bot: Attempting to shut down application...")
                try:
                    await application.shutdown()
                    print("Telegram bot: Application shut down successfully.")
                except Exception as e_shutdown:
                    print(f"Telegram bot: Exception during application shutdown: {e_shutdown}")
                    traceback.print_exc()
            else:
                print("Telegram bot: Application was not initialized successfully, skipping shutdown.")

    def thread_starter():
        # `asyncio.run()` creates a new event loop, runs the coroutine, 
        # and handles loop closing and setting/resetting the event loop for the thread.
        print("Telegram thread: Starting actual_bot_runner_async with asyncio.run()...")
        try:
            asyncio.run(actual_bot_runner_async())
            print("Telegram thread: actual_bot_runner_async finished.")
        except Exception as e:
            # This might catch errors from asyncio.run() itself, though errors within
            # actual_bot_runner_async should be handled inside it.
            print(f"Telegram thread: Exception in thread_starter from asyncio.run(): {e}")
            traceback.print_exc()
        finally:
            print("Telegram thread: Exiting thread_starter.")
            # No explicit loop closing needed here, asyncio.run() handles it.
            # asyncio.set_event_loop(None) is also handled by asyncio.run().

    thread = threading.Thread(target=thread_starter, name="TelegramBotThread") # Added name
    thread.daemon = True
    thread.start()


# --- Telegram Notification for Trade Rejection ---
def send_trade_rejection_notification(symbol, signal_type, reason, entry_price, sl_price, tp_price, quantity, symbol_info, configs):
    """
    Sends a Telegram notification about a rejected trade signal.
    """
    if not configs.get("telegram_bot_token") or not configs.get("telegram_chat_id"):
        print(f"Telegram not configured. Cannot send rejection notification for {symbol}.")
        return

    p_prec = symbol_info.get('pricePrecision', 2) if symbol_info else 2
    q_prec = symbol_info.get('quantityPrecision', 0) if symbol_info else 0

    entry_price_str = f"{entry_price:.{p_prec}f}" if entry_price is not None else "N/A"
    sl_price_str = f"{sl_price:.{p_prec}f}" if sl_price is not None else "N/A"
    tp_price_str = f"{tp_price:.{p_prec}f}" if tp_price is not None else "N/A"
    quantity_str = f"{quantity:.{q_prec}f}" if quantity is not None else "N/A"

    message = (
        f"⚠️ TRADE REJECTED ⚠️\n\n"
        f"Symbol: `{symbol}`\n"
        f"Signal: `{signal_type}`\n"
        f"Reason: _{reason}_\n\n"
        f"*Attempted Parameters:*\n"
        f"Entry: `{entry_price_str}`\n"
        f"SL: `{sl_price_str}`\n"
        f"TP: `{tp_price_str}`\n"
        f"Qty: `{quantity_str}`"
    )

    print(f"Sending trade rejection notification for {symbol}: {reason}")
    send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], message)


def get_user_configurations():
    """
    Prompts the user for various trading configurations and returns them as a dictionary.
    Includes input validation for each configuration item.
    Allows loading from or saving to 'configure.csv'.
    """
    print("\n--- Strategy Configuration ---")
    configs = {}
    loaded_configs_from_csv = None
    config_filepath = "configure.csv"
    proceed_to_custom_setup = False

    # Ask user if they want to load from CSV or do custom setup
    while True:
        load_choice = input(f"Load from '{config_filepath}' (L) or Custom Setup (C)? [L]: ").strip().lower()
        if not load_choice or load_choice == 'l':
            loaded_configs_from_csv = load_configuration_from_csv(config_filepath)
            if loaded_configs_from_csv:
                is_valid, validation_msg, validated_configs_csv = validate_configurations(loaded_configs_from_csv)
                if is_valid:
                    print("Configuration loaded successfully from CSV:")
                    for k, v in validated_configs_csv.items(): print(f"  {k}: {v}")
                    
                    while True:
                        change_choice = input("Make changes to these settings? (y/N) [N]: ").strip().lower()
                        if not change_choice or change_choice == 'n':
                            configs = validated_configs_csv # Use loaded and validated configs
                            # Ensure API keys are loaded based on the environment from CSV
                            if "environment" in configs:
                                api_key, api_secret, telegram_token, telegram_chat_id = load_api_keys(configs["environment"])
                                configs["api_key"] = api_key
                                configs["api_secret"] = api_secret
                                configs["telegram_bot_token"] = telegram_token
                                configs["telegram_chat_id"] = telegram_chat_id
                            else:
                                print("Error: 'environment' missing from loaded CSV configuration. Cannot load API keys.")
                                # Fallback to custom setup or ask for environment specifically
                                proceed_to_custom_setup = True
                                break 
                            
                            # Add fixed strategy details
                            configs["strategy_id"] = 8
                            configs["strategy_name"] = "Advance EMA Cross"
                            configs["max_scan_threads"] = 5 # Fixed value
                            print("--- Configuration Complete (Loaded from CSV) ---")
                            return configs
                        elif change_choice == 'y':
                            configs = validated_configs_csv # Start custom setup with these values
                            proceed_to_custom_setup = True
                            break # Break from change_choice loop
                        else:
                            print("Invalid choice. Please enter 'y' or 'n'.")
                    if proceed_to_custom_setup: break # Break from load_choice loop to go to custom setup
                else:
                    print(f"Invalid data in '{config_filepath}': {validation_msg} Proceeding to custom setup.")
                    proceed_to_custom_setup = True
                    break # Break from load_choice loop
            else:
                print(f"'{config_filepath}' not found or empty/corrupted. Proceeding to custom setup.")
                proceed_to_custom_setup = True
                break # Break from load_choice loop
        elif load_choice == 'c':
            proceed_to_custom_setup = True
            break # Break from load_choice loop
        else:
            print("Invalid choice. Please enter 'L' or 'C'.")

    # If proceed_to_custom_setup is True (either by choice or failure to load)
    # `configs` might already hold values from CSV if user chose 'y' to change them.
    # Otherwise, `configs` is empty.

    print("\n--- Custom Configuration Setup ---")
    # Helper to get input, using value from `configs` (loaded from CSV) as default if available
    def get_input_with_default(prompt_message, current_value_key, default_constant_value, type_converter=str):
        default_to_show = configs.get(current_value_key, default_constant_value)
        user_input = input(f"{prompt_message} (default: {default_to_show}): ")
        
        # If user enters nothing, use the default_to_show (which could be from CSV or constant)
        if not user_input:
            # Need to ensure the default_to_show is of the correct type if it came from CSV (already string)
            # or from constant (might need conversion if it's not already a string for display)
            # The type_converter will handle this.
            try:
                return type_converter(default_to_show)
            except ValueError: # If default_to_show can't be converted (e.g. empty string for float)
                 # This case needs careful handling. For now, let's assume default_to_show is valid or re-prompt logic handles it.
                 # Or, use the original default_constant_value if default_to_show fails conversion.
                 return type_converter(default_constant_value)
        try:
            return type_converter(user_input)
        except ValueError:
            # If conversion fails, the validation loop in the main part will catch it and re-prompt.
            # For robustness, could return a specific error marker or re-prompt here.
            # For now, rely on outer validation loop.
            raise # Re-raise to be caught by the calling loop's try-except

    # Environment
    while True:
        # Environment is special, it dictates API key loading.
        # If loaded from CSV, it should already be in `configs`.
        env_default_display = configs.get("environment", "testnet") # Default to testnet if not in CSV
        env_input = input(f"Select environment (1:testnet / 2:mainnet) (current: {env_default_display}): ").strip()
        
        chosen_env = None
        if not env_input and "environment" in configs: # User hit enter, use CSV loaded value
            chosen_env = configs["environment"]
        elif env_input == "1": chosen_env = "testnet"
        elif env_input == "2": chosen_env = "mainnet"
        
        if chosen_env in ["testnet", "mainnet"]:
            configs["environment"] = chosen_env
            break
        print("Invalid environment. Please enter '1' for testnet or '2' for mainnet.")

    # Load API keys based on selected environment
    api_key, api_secret, telegram_token, telegram_chat_id = load_api_keys(configs["environment"])
    configs["api_key"] = api_key
    configs["api_secret"] = api_secret
    configs["telegram_bot_token"] = telegram_token
    configs["telegram_chat_id"] = telegram_chat_id
    
    # Mode
    while True:
        mode_default_display = configs.get("mode", "live")
        mode_input = input(f"Select mode (1:live / 2:backtest) (current: {mode_default_display}): ").strip()
        
        chosen_mode = None
        if not mode_input and "mode" in configs: chosen_mode = configs["mode"]
        elif mode_input == "1": chosen_mode = "live"
        elif mode_input == "2": chosen_mode = "backtest"

        if chosen_mode in ["live", "backtest"]:
            configs["mode"] = chosen_mode
            break
        print("Invalid mode. Please enter '1' for live or '2' for backtest.")

    if configs["mode"] == "backtest":
        while True:
            try:
                days = get_input_with_default(
                    "Enter number of days for backtesting",
                    "backtest_days", 30, int
                )
                if days > 0:
                    configs["backtest_days"] = days
                    break
                print("Number of days must be a positive integer.")
            except ValueError: print("Invalid input. Please enter an integer for the number of days.")
        
        while True:
            bt_balance_type_default_display = configs.get("backtest_start_balance_type", "current")
            balance_choice_input = input(f"For backtest, use (1:current account balance) or (2:set a custom start balance)? (current: {bt_balance_type_default_display}) [1 if new, else current]: ").strip()
            
            chosen_bt_balance_type = None
            # If user hits enter, and there was a value from CSV, use that. Otherwise, default to '1' (current).
            if not balance_choice_input:
                chosen_bt_balance_type = configs.get("backtest_start_balance_type", "current") # 'current' is implied by '1'
                if chosen_bt_balance_type == "current": pass # Already set
                elif chosen_bt_balance_type == "custom": pass # Already set
                else: # If CSV had something else, or nothing, map to '1' or '2' logic
                    chosen_bt_balance_type = "current" # Default to current
            elif balance_choice_input == "1": chosen_bt_balance_type = "current"
            elif balance_choice_input == "2": chosen_bt_balance_type = "custom"

            if chosen_bt_balance_type in ["current", "custom"]:
                configs["backtest_start_balance_type"] = chosen_bt_balance_type
                break
            print("Invalid choice. Please enter '1' for current or '2' for custom.")

        if configs["backtest_start_balance_type"] == "custom":
            while True:
                try:
                    custom_bal = get_input_with_default(
                        "Enter custom start balance for backtest",
                        "backtest_custom_start_balance", 10000.0, float
                    )
                    if custom_bal > 0:
                        configs["backtest_custom_start_balance"] = custom_bal
                        break
                    print("Custom balance must be a positive number.")
                except ValueError: print("Invalid input. Please enter a number for the custom balance.")
    else: # Live mode, ensure backtest keys are not in the final saved CSV if they were from a previous load
        configs.pop("backtest_days", None)
        configs.pop("backtest_start_balance_type", None)
        configs.pop("backtest_custom_start_balance", None)


    # Risk Percent
    while True:
        try:
            # For risk_percent, the stored value is float (e.g., 0.01), display is % (e.g., 1.0)
            # Default constant is 1.0 (meaning 1%)
            # If loaded from CSV, it's already 0.01. Convert to % for display.
            risk_default_display = (configs.get("risk_percent", DEFAULT_RISK_PERCENT / 100.0) * 100.0 
                                    if "risk_percent" in configs else DEFAULT_RISK_PERCENT)

            risk_input_str = input(f"Enter account risk % per trade (e.g., 1 for 1%) (default: {risk_default_display:.2f}%): ")
            
            risk_percent_val = 0
            if not risk_input_str: # User hit enter
                risk_percent_val = float(risk_default_display) # Use the displayed default (already in %)
            else:
                risk_percent_val = float(risk_input_str)

            if 0 < risk_percent_val <= 100:
                configs["risk_percent"] = risk_percent_val / 100.0 # Store as decimal
                break
            print("Risk percentage must be a positive value (e.g., 0.5, 1, up to 100).")
        except ValueError: print("Invalid input for risk percentage. Please enter a number.")

    # Leverage
    while True:
        try:
            leverage = get_input_with_default(
                "Enter leverage (e.g., 10 for 10x)",
                "leverage", DEFAULT_LEVERAGE, int
            )
            if 1 <= leverage <= 125:
                configs["leverage"] = leverage
                break
            print("Leverage must be an integer between 1 and 125.")
        except ValueError: print("Invalid input for leverage. Please enter an integer.")

    # Max Concurrent Positions
    while True:
        try:
            max_pos = get_input_with_default(
                "Enter max concurrent positions",
                "max_concurrent_positions", DEFAULT_MAX_CONCURRENT_POSITIONS, int
            )
            if max_pos > 0:
                configs["max_concurrent_positions"] = max_pos
                break
            print("Max concurrent positions must be a positive integer.")
        except ValueError: print("Invalid input for max positions. Please enter an integer.")

    # Margin Type
    while True:
        margin_default_display = configs.get("margin_type", DEFAULT_MARGIN_TYPE)
        margin_input = input(f"Enter margin type (ISOLATED/CROSS) (default: {margin_default_display}): ").upper().strip()
        
        chosen_margin = None
        if not margin_input and "margin_type" in configs: chosen_margin = configs["margin_type"]
        elif not margin_input : chosen_margin = DEFAULT_MARGIN_TYPE # if no csv val and empty input
        else: chosen_margin = margin_input

        if chosen_margin in ["ISOLATED", "CROSS"]:
            configs["margin_type"] = chosen_margin
            break
        print("Invalid margin type. Please enter 'ISOLATED' or 'CROSS'.")
    
    configs["max_scan_threads"] = 5 # Fixed
    print(f"Maximum symbol scan threads fixed to {configs['max_scan_threads']}.")

    # Portfolio Risk Cap
    while True:
        try:
            portfolio_risk = get_input_with_default(
                f"Enter max portfolio risk % (aggregate open trades, e.g., {DEFAULT_PORTFOLIO_RISK_CAP} for {DEFAULT_PORTFOLIO_RISK_CAP}%)",
                "portfolio_risk_cap", DEFAULT_PORTFOLIO_RISK_CAP, float
            )
            if 0 < portfolio_risk <= 100:
                configs["portfolio_risk_cap"] = portfolio_risk
                break
            print("Portfolio risk percentage must be a positive value (e.g., 3, 5, up to 100).")
        except ValueError: print("Invalid input for portfolio risk percentage. Please enter a number.")

    # ATR Period
    while True:
        try:
            atr_period_val = get_input_with_default(
                "Enter ATR Period for SL/TP",
                "atr_period", DEFAULT_ATR_PERIOD, int
            )
            if atr_period_val > 0:
                configs["atr_period"] = atr_period_val
                break
            print("ATR Period must be a positive integer.")
        except ValueError: print("Invalid input for ATR Period. Please enter an integer.")

    # ATR Multiplier SL
    while True:
        try:
            atr_mult_sl = get_input_with_default(
                "Enter ATR Multiplier for Stop Loss",
                "atr_multiplier_sl", DEFAULT_ATR_MULTIPLIER_SL, float
            )
            if atr_mult_sl > 0:
                configs["atr_multiplier_sl"] = atr_mult_sl
                break
            print("ATR Multiplier for SL must be a positive number.")
        except ValueError: print("Invalid input for ATR Multiplier (SL). Please enter a number.")

    # TP R:R Ratio
    while True:
        try:
            tp_rr = get_input_with_default(
                "Enter Take Profit Risk:Reward Ratio",
                "tp_rr_ratio", DEFAULT_TP_RR_RATIO, float
            )
            if tp_rr > 0:
                configs["tp_rr_ratio"] = tp_rr
                break
            print("Take Profit R:R Ratio must be a positive number.")
        except ValueError: print("Invalid input for TP R:R Ratio. Please enter a number.")
    
    # Max Daily Drawdown %
    while True:
        try:
            max_dd = get_input_with_default(
                f"Enter Max Daily Drawdown % (from high equity, 0 to disable, default: {DEFAULT_MAX_DRAWDOWN_PERCENT}%)",
                "max_drawdown_percent", DEFAULT_MAX_DRAWDOWN_PERCENT, float
            )
            if 0 <= max_dd <= 100:
                configs["max_drawdown_percent"] = max_dd
                break
            print("Max Daily Drawdown % must be between 0 (disabled) and 100.")
        except ValueError: print("Invalid input for Max Daily Drawdown %. Please enter a number.")

    # Daily Stop Loss %
    while True:
        try:
            daily_sl = get_input_with_default(
                f"Enter Daily Stop Loss % (of start equity, 0 to disable, default: {DEFAULT_DAILY_STOP_LOSS_PERCENT}%)",
                "daily_stop_loss_percent", DEFAULT_DAILY_STOP_LOSS_PERCENT, float
            )
            if 0 <= daily_sl <= 100:
                configs["daily_stop_loss_percent"] = daily_sl
                break
            print("Daily Stop Loss % must be between 0 (disabled) and 100.")
        except ValueError: print("Invalid input for Daily Stop Loss %. Please enter a number.")

    # Target Annualized Volatility
    while True:
        try:
            # Display as percentage if it's like 0.80 -> 80%
            target_vol_default_display = (configs.get("target_annualized_volatility", DEFAULT_TARGET_ANNUALIZED_VOLATILITY) * 100.0
                                           if "target_annualized_volatility" in configs else DEFAULT_TARGET_ANNUALIZED_VOLATILITY * 100.0)

            target_vol_input_str = input(f"Enter Target Annualized Volatility % (e.g., {target_vol_default_display:.0f}% for {configs.get('target_annualized_volatility', DEFAULT_TARGET_ANNUALIZED_VOLATILITY):.2f}): ")
            
            target_vol_pct = 0
            if not target_vol_input_str: # User hit enter
                target_vol_pct = float(target_vol_default_display) # Use the displayed default (already in %)
            else:
                target_vol_pct = float(target_vol_input_str)
            
            target_vol_decimal = target_vol_pct / 100.0

            if 0 < target_vol_decimal <= 5.0: # Check decimal value
                configs["target_annualized_volatility"] = target_vol_decimal
                break
            print("Target Annualized Volatility must be a positive number (e.g., input 80 for 80% or 0.8). Sensible range up to 500% (5.0).")
        except ValueError: print("Invalid input for Target Annualized Volatility. Please enter a number.")
            
    # Realized Volatility Period
    while True:
        try:
            vol_period = get_input_with_default(
                "Enter Realized Volatility Period (candles)",
                "realized_volatility_period", DEFAULT_REALIZED_VOLATILITY_PERIOD, int
            )
            if vol_period > 0:
                configs["realized_volatility_period"] = vol_period
                break
            print("Realized Volatility Period must be a positive integer.")
        except ValueError: print("Invalid input for Realized Volatility Period. Please enter an integer.")

    # Minimum Leverage
    while True:
        try:
            min_lev = get_input_with_default(
                "Enter Minimum Leverage for dynamic adjustment",
                "min_leverage", DEFAULT_MIN_LEVERAGE, int
            )
            if 1 <= min_lev <= 125:
                configs["min_leverage"] = min_lev
                break
            print("Minimum Leverage must be an integer between 1 and 125.")
        except ValueError: print("Invalid input for Minimum Leverage. Please enter an integer.")

    # Maximum Leverage
    while True:
        try:
            max_lev = get_input_with_default(
                "Enter Maximum Leverage for dynamic adjustment",
                "max_leverage", DEFAULT_MAX_LEVERAGE, int
            )
            min_lev_for_check = configs.get("min_leverage", DEFAULT_MIN_LEVERAGE) # Should be set by now
            if min_lev_for_check <= max_lev <= 125:
                configs["max_leverage"] = max_lev
                break
            print(f"Maximum Leverage must be an integer between {min_lev_for_check} and 125.")
        except ValueError: print("Invalid input for Maximum Leverage. Please enter an integer.")

    # Allow Exceed Risk for Min Notional
    while True:
        # Determine default display: if loaded from CSV, use that, else use constant
        default_bool_display = ""
        if "allow_exceed_risk_for_min_notional" in configs:
            default_bool_display = 'yes' if configs["allow_exceed_risk_for_min_notional"] else 'no'
        else:
            default_bool_display = 'yes' if DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL else 'no'

        exceed_risk_input = input(f"Allow exceeding individual trade risk % to meet MIN_NOTIONAL? (yes/no) (default: {default_bool_display}): ").lower().strip()
        
        chosen_exceed_risk = None
        if not exceed_risk_input: # User hit enter
            chosen_exceed_risk = configs.get("allow_exceed_risk_for_min_notional", DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL)
        elif exceed_risk_input in ["yes", "y"]: chosen_exceed_risk = True
        elif exceed_risk_input in ["no", "n"]: chosen_exceed_risk = False
        
        if isinstance(chosen_exceed_risk, bool):
            configs["allow_exceed_risk_for_min_notional"] = chosen_exceed_risk
            break
        print("Invalid input. Please enter 'yes' or 'no'.")

    configs["strategy_id"] = 8
    configs["strategy_name"] = "Advance EMA Cross"
    
    # Save configurations to CSV
    # We need to remove API keys before saving to CSV if they were temporarily added
    # However, the current plan is that API keys are NOT part of the CSV.
    # They are loaded from keys.py into the `configs` dict *after* this function returns (or towards the end).
    # So, the `configs` dict at this point should only contain user-configurable parameters.
    
    # The API keys are added to `configs` dict after `environment` is known.
    # So, we need to make a copy for saving that excludes them.
    configs_to_save = configs.copy()
    configs_to_save.pop("api_key", None)
    configs_to_save.pop("api_secret", None)
    configs_to_save.pop("telegram_bot_token", None)
    configs_to_save.pop("telegram_chat_id", None)
    
    if save_configuration_to_csv(config_filepath, configs_to_save):
        print(f"Configurations saved to '{config_filepath}'.")
    else:
        print(f"Failed to save configurations to '{config_filepath}'.")
        
    print("--- Configuration Complete (Custom Setup) ---")
    return configs

# --- Helper function to build symbol_info_map from active_trades ---
def _build_symbol_info_map_from_active_trades(active_trades_dict):
    """
    Constructs a map of {symbol: symbol_info} from the active_trades dictionary.
    """
    s_info_map = {}
    if active_trades_dict:
        for symbol, trade_details in active_trades_dict.items():
            if 'symbol_info' in trade_details:
                s_info_map[symbol] = trade_details['symbol_info']
    return s_info_map

# --- Binance API Interaction Functions (Error handling included) ---

def initialize_binance_client(configs):
    api_key, api_secret, env = configs["api_key"], configs["api_secret"], configs["environment"]
    try:
        client = Client(api_key, api_secret, testnet=(env == "testnet"))
        client.ping() # Verify connection
        server_time = client.get_server_time() # Get server time for the success message
        # print(f"\nSuccessfully connected to Binance {env.title()} API. Server Time: {pd.to_datetime(server_time['serverTime'], unit='ms')} UTC")
        # Return client, environment title, and server time for main() to construct the message
        return client, env, server_time # Modified return
    except BinanceAPIException as e:
        print(f"Binance API Exception (client init): {e}")
        return None, None, None # Modified return
    except Exception as e:
        print(f"Error initializing Binance client: {e}")
        return None, None, None # Modified return

def get_historical_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=500, backtest_days=None):
    """
    Fetches historical klines. If backtest_days is specified, it fetches data for that many days.
    Otherwise, it fetches the most recent 'limit' klines.
    """
    start_time = time.time()
    klines = []
    api_error = None

    try:
        if backtest_days:
            klines_per_day = (24 * 60) // int(interval.replace('m','').replace('h','').replace('d',''))
            if 'h' in interval: klines_per_day = 24 // int(interval.replace('h',''))
            if 'd' in interval: klines_per_day = 1
            total_klines_needed = klines_per_day * backtest_days
            print(f"Fetching klines for {symbol}, interval {interval}, for {backtest_days} days (approx {total_klines_needed} klines)...")
            start_str = f"{backtest_days + 1} days ago UTC"
            klines = client.get_historical_klines(symbol, interval, start_str)
        else: # Live mode
            print(f"Fetching klines for {symbol}, interval {interval}, limit {limit}...")
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except BinanceAPIException as e:
        print(f"API Error fetching klines for {symbol}: {e}")
        api_error = e
        return pd.DataFrame(), api_error # Return empty DataFrame and the error
    except Exception as e: # Catch other potential errors during fetch (e.g., network issues)
        print(f"General error fetching klines for {symbol}: {e}")
        api_error = e # Store general exception as well
        return pd.DataFrame(), api_error # Return empty DataFrame and the error

    duration = time.time() - start_time
    processing_error = None
    try: 
        if not klines:
            # This case means no data returned, but not necessarily an API exception caught above (e.g. symbol exists but has no trades)
            print(f"No kline data for {symbol} (fetch duration: {duration:.2f}s).")
            # If api_error was already set (e.g. from a specific exception during klines = ...), it will be returned.
            # Otherwise, this is just an empty kline list.
            return pd.DataFrame(), api_error # api_error might be None here

        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        
        if df.empty and not api_error : # If after processing, df is empty, and no prior API error
             print(f"Kline data for {symbol} resulted in empty DataFrame after processing (fetch duration: {duration:.2f}s).")
             # Potentially set a generic error or info that processing yielded no usable data
             # For now, returning api_error (which would be None) is fine.

        print(f"Fetched {len(df)} klines for {symbol} (runtime: {duration:.2f}s).")
        if len(df) < 200 and not df.empty: print(f"Warning: Low kline count for {symbol} ({len(df)}), EMA200 may be inaccurate.")
        return df, None # Success, no error object to return for this part
    
    except Exception as e: # Error during DataFrame processing
        print(f"Error processing kline data for {symbol}: {e}")
        processing_error = e
        return pd.DataFrame(), processing_error # Return empty DataFrame and the processing error

def get_account_balance(client, asset="USDT"):
    try:
        balances = client.futures_account_balance()
        for b in balances:
            if b['asset'] == asset:
                print(f"Account Balance ({asset}): {b['balance']}")
                return float(b['balance'])
        print(f"{asset} not found in futures balance.")
        return 0.0  # Return 0.0 if asset not found but call was successful
    except BinanceAPIException as e:
        if e.code == -2015:
            print(f"Critical Error getting balance: {e}. This is likely an API key permission or IP whitelist issue.")
            print("Please check your API key settings on Binance: ensure 'Enable Futures' is checked and your IP is whitelisted if restrictive IP access is enabled.")
            return None # Specific indicator for critical auth/IP error
        else:
            print(f"API Error getting balance: {e}")
            return 0.0 # For other API errors, return 0.0 to indicate balance couldn't be fetched but not necessarily critical auth
    except Exception as e:
        print(f"Unexpected error getting balance: {e}")
        return 0.0 # For non-API unexpected errors

# Modified to accept configs for Telegram alerting on specific errors
def get_account_balance(client, configs, asset="USDT"): # Added configs parameter
    try:
        balances = client.futures_account_balance()
        for b in balances:
            if b['asset'] == asset:
                print(f"Account Balance ({asset}): {b['balance']}")
                return float(b['balance'])
        print(f"{asset} not found in futures balance.")
        return 0.0  # Return 0.0 if asset not found but call was successful
    except BinanceAPIException as e:
        if e.code == -2015:
            print(f"Critical Error getting balance: {e}. This is likely an API key permission or IP whitelist issue.")
            print("Please check your API key settings on Binance: ensure 'Enable Futures' is checked and your IP is whitelisted if restrictive IP access is enabled.")
            
            # Attempt to send Telegram alert
            public_ip = get_public_ip()
            ip_message = f"Bot's current public IP: {public_ip}" if public_ip else "Could not determine bot's public IP."
            
            error_message = (
                f"⚠️ CRITICAL BINANCE API ERROR ⚠️\n\n"
                f"Error Code: -2015 (Likely IP Whitelist Issue)\n"
                f"Message: {e.message}\n\n"
                f"{ip_message}\n\n"
                f"Please check your Binance API key permissions and IP whitelist settings immediately."
            )
            
            telegram_token = configs.get("telegram_bot_token")
            telegram_chat_id = configs.get("telegram_chat_id")

            if telegram_token and telegram_chat_id:
                send_telegram_message(telegram_token, telegram_chat_id, error_message)
            else:
                print("Telegram credentials not found in configs. Cannot send IP error alert via Telegram.")
            return None # Specific indicator for critical auth/IP error
        else:
            print(f"API Error getting balance: {e}")
            return 0.0 # For other API errors, return 0.0 to indicate balance couldn't be fetched but not necessarily critical auth
    except Exception as e:
        print(f"Unexpected error getting balance: {e}")
        return 0.0 # For non-API unexpected errors

def get_open_positions(client, format_for_telegram=False, active_trades_data=None, symbol_info_map=None): # Added active_trades_data and symbol_info_map
    try:
        api_positions = client.futures_position_information()
        positions = [p for p in api_positions if float(p.get('positionAmt', 0)) != 0]

        if format_for_telegram:
            if not positions: return "None"
            pos_texts = []
            if active_trades_data is None: active_trades_data = {}
            if symbol_info_map is None: symbol_info_map = {}

            for p in positions:
                symbol_str = str(p.get('symbol', 'N/A'))
                qty_str = str(p.get('positionAmt', 'N/A'))
                entry_price_str = str(p.get('entryPrice', 'N/A'))
                pnl_str = str(p.get('unRealizedProfit', 'N/A'))
                
                sl_str, tp_str = "N/A (Bot)", "N/A (Bot)"
                price_precision = 2 # Default price precision

                s_info = symbol_info_map.get(symbol_str)
                if s_info:
                    price_precision = int(s_info.get('pricePrecision', 2))

                if symbol_str in active_trades_data:
                    trade_detail = active_trades_data[symbol_str]
                    sl_price = trade_detail.get('current_sl_price')
                    tp_price = trade_detail.get('current_tp_price')
                    
                    if sl_price is not None:
                        sl_str = f"{sl_price:.{price_precision}f}"
                    if tp_price is not None:
                        tp_str = f"{tp_price:.{price_precision}f}"
                
                pos_texts.append(f"- {symbol_str}: Qty={qty_str}, Entry={entry_price_str}, SL={sl_str}, TP={tp_str}, PnL={pnl_str}")
            return "\n".join(pos_texts)

        # Original behavior
        if not positions:
            print("No open positions.")
            return []
        print("Current Open Positions:")
        for p in positions:
            # Try to supplement with SL/TP from active_trades if available for console output too
            sl_console, tp_console = "", ""
            if active_trades_data and p['symbol'] in active_trades_data:
                trade_data = active_trades_data[p['symbol']]
                # Refined logic for s_info_console
                s_info_console = None
                if symbol_info_map and p['symbol'] in symbol_info_map:
                    s_info_console = symbol_info_map[p['symbol']]
                else:
                    # Fallback to fetching if not in provided map (e.g. for positions not in active_trades but we want console info)
                    # This path might be less common if active_trades_data covers all relevant managed trades
                    s_info_console = get_symbol_info(client, p['symbol'])
                
                price_prec_console = 2
                if s_info_console:
                    price_prec_console = int(s_info_console.get('pricePrecision', 2))

                sl_val = trade_data.get('current_sl_price')
                tp_val = trade_data.get('current_tp_price')
                if sl_val is not None: sl_console = f", SL: {sl_val:.{price_prec_console}f}"
                if tp_val is not None: tp_console = f", TP: {tp_val:.{price_prec_console}f}"
            
            print(f"  {p['symbol']}: Amt={p['positionAmt']}, Entry={p['entryPrice']}{sl_console}{tp_console}, PnL={p['unRealizedProfit']}")
        return positions
    except Exception as e:
        print(f"Error getting positions: {e}")
        traceback.print_exc() # Print stack trace for better debugging
        if format_for_telegram:
            return "Error fetching positions"
        return []

def get_open_orders(client, symbol=None):
    try:
        orders = client.futures_get_open_orders(symbol=symbol) if symbol else client.futures_get_open_orders()
        if not orders: print(f"No open orders{(' for '+symbol) if symbol else ''}."); return []
        print(f"Open Orders{(' for '+symbol) if symbol else ''}:")
        for o in orders: print(f"  {o['symbol']}({o['orderId']}): {o['side']} {o['type']} {o['origQty']}@{o['price'] if o['type']!='MARKET' else 'MARKET'}, Stop:{o.get('stopPrice','N/A')}")
        return orders
    except Exception as e: print(f"Error getting orders: {e}"); return []

def set_leverage_on_symbol(client, symbol, leverage):
    try: client.futures_change_leverage(symbol=symbol, leverage=leverage); print(f"Leverage for {symbol} set to {leverage}x."); return True
    except BinanceAPIException as e: print(f"API Error setting leverage for {symbol}: {e}"); return False
    except Exception as e: print(f"Error setting leverage for {symbol}: {e}"); return False

def set_margin_type_on_symbol(client, symbol, margin_type):
    try: client.futures_change_margin_type(symbol=symbol, marginType=margin_type.upper()); print(f"Margin for {symbol} set to {margin_type}."); return True
    except BinanceAPIException as e:
        if e.code == -4046: # (-4046, 'No need to change margin type.')
            print(f"Margin for {symbol} already {margin_type}. No change needed.")
            return True
        print(f"API Error setting margin for {symbol} to {margin_type}: {e}"); return False
    except Exception as e: print(f"Error setting margin for {symbol} to {margin_type}: {e}"); return False

# Modified to accept configs for Telegram alerting
def set_margin_type_on_symbol(client, symbol: str, margin_type: str, configs: dict = None): # Added configs as optional
    try:
        client.futures_change_margin_type(symbol=symbol, marginType=margin_type.upper())
        msg = f"Margin type for {symbol} successfully set to {margin_type}."
        print(msg)
        if configs and configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
            send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], f"ℹ️ {msg}")
        return True
    except BinanceAPIException as e:
        if e.code == -4046:  # (-4046, 'No need to change margin type.')
            print(f"Margin for {symbol} already {margin_type}. No change needed.")
            return True
        print(f"API Error setting margin for {symbol} to {margin_type}: {e}")
        # Optionally send Telegram for failure too, if critical enough
        # if configs and configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
        #     send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], f"⚠️ Failed to set margin type for {symbol} to {margin_type}. Error: {e}")
        return False
    except Exception as e:
        print(f"Error setting margin for {symbol} to {margin_type}: {e}")
        return False

def place_new_order(client, symbol_info, side, order_type, quantity, price=None, stop_price=None, reduce_only=None, position_side=None, is_closing_order=False): # Added position_side and is_closing_order
    symbol, p_prec, q_prec = symbol_info['symbol'], int(symbol_info['pricePrecision']), int(symbol_info['quantityPrecision'])
    params = {"symbol": symbol, "side": side.upper(), "type": order_type.upper(), "quantity": f"{quantity:.{q_prec}f}"}

    if position_side: # Add positionSide if provided
        params["positionSide"] = position_side.upper()

    if order_type.upper() in ["LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if price is None: print(f"Price needed for {order_type} on {symbol}"); return None
        params.update({"price": f"{price:.{p_prec}f}", "timeInForce": "GTC"}) # GTC for limit type orders
    
    if order_type.upper() in ["STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if stop_price is None: print(f"Stop price needed for {order_type} on {symbol}"); return None
        params["stopPrice"] = f"{stop_price:.{p_prec}f}"
        # If it's an SL/TP order, use closePosition=True instead of reduceOnly
        if is_closing_order:
            params["closePosition"] = "true" # API expects string "true"
            if "reduceOnly" in params: # Remove reduceOnly if closePosition is used
                del params["reduceOnly"]
        elif reduce_only is not None: # Fallback to reduce_only if not explicitly a closing order but reduce_only is specified
             params["reduceOnly"] = str(reduce_only).lower()

    # For entry orders (not SL/TP), reduceOnly should not be set unless specifically intended for position reduction without full closure
    # If it's not a closing order and reduce_only is explicitly passed, respect it.
    # Otherwise, for typical entry orders, neither closePosition nor reduceOnly are needed.
    elif reduce_only is not None: # This handles cases where reduce_only is passed for non-SL/TP orders
        params["reduceOnly"] = str(reduce_only).lower()
        
    try:
        order = client.futures_create_order(**params)
        print(f"Order PLACED: {order['symbol']} ID {order['orderId']} {order.get('positionSide','N/A')} {order['side']} {order['type']} {order['origQty']} @ {order.get('price','MARKET')} SP:{order.get('stopPrice','N/A')} CP:{order.get('closePosition','false')} RO:{order.get('reduceOnly','false')} AvgP:{order.get('avgPrice','N/A')} Status:{order['status']}")
        return order, None  # Return order and None for error message on success
    except Exception as e:
        error_msg = f"ORDER FAILED for {symbol} {side} {quantity} {order_type}: {str(e)}"
        print(error_msg)
        return None, str(e) # Return None for order and the error message string on failure

# --- Indicator, Strategy, SL/TP, Sizing ---
def calculate_ema(df, period, column='close'):
    if column not in df or len(df) < period: return None # Basic checks
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Average True Range (ATR) using Wilder's smoothing.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.
        period (int): The period for ATR calculation.

    Returns:
        pd.Series: A pandas Series containing the ATR values.
                   Returns an empty Series if input conditions are not met.
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        print("Error: DataFrame must contain 'high', 'low', 'close' columns for ATR calculation.")
        return pd.Series(dtype='float64')
    if len(df) < period:
        print(f"Error: Data length ({len(df)}) is less than ATR period ({period}). Cannot calculate ATR.")
        return pd.Series(dtype='float64')

    # Calculate True Range (TR)
    df['prev_close'] = df['close'].shift(1)
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['prev_close'])
    df['tr2'] = abs(df['low'] - df['prev_close'])
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)

    # Calculate ATR using Wilder's smoothing
    # The first ATR is the simple average of the TR for the first 'period' values.
    # Subsequent ATRs are calculated as: ATR = (ATR_prev * (period - 1) + TR_current) / period
    # This is equivalent to an EMA with alpha = 1/period.
    # pandas.Series.ewm with alpha=1/period and adjust=False directly implements Wilder's smoothing.
    atr_series = df['tr'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Clean up temporary columns from the input DataFrame
    df.drop(columns=['prev_close', 'tr0', 'tr1', 'tr2', 'tr'], inplace=True)
    
    return atr_series.astype(float) # Ensure the output series is of float type

def calculate_realized_volatility(klines_df: pd.DataFrame, period: int, candles_per_year: int) -> float | None:
    """
    Calculates the annualized realized volatility.

    Args:
        klines_df (pd.DataFrame): DataFrame with 'close' prices.
        period (int): Lookback period for standard deviation of log returns.
        candles_per_year (int): Number of candles in a year for annualization.

    Returns:
        float | None: Annualized volatility, or None if calculation is not possible.
    """
    if 'close' not in klines_df.columns:
        print("Error: DataFrame must contain 'close' column for volatility calculation.")
        return None
    if len(klines_df) < period + 1: # Need period + 1 for one log return at the start of the window
        print(f"Error: Data length ({len(klines_df)}) is insufficient for volatility period ({period}). Needs at least {period + 1} candles.")
        return None

    # Calculate log returns
    # Using a temporary Series to avoid SettingWithCopyWarning if klines_df is a slice
    log_returns = np.log(klines_df['close'] / klines_df['close'].shift(1))
    
    # Calculate rolling standard deviation of log returns
    # The result of rolling().std() will have NaNs for the first `period-1` rows after log_returns start (which is 1 row after klines_df starts)
    stdev_log_returns = log_returns.rolling(window=period).std()
    
    last_stdev = stdev_log_returns.iloc[-1]

    if pd.isna(last_stdev) or last_stdev == 0: # Also check for zero stdev to avoid issues
        print(f"Warning: Standard deviation of log returns is NaN or zero for period {period}. Cannot calculate volatility.")
        return None
        
    # Annualize the standard deviation
    annualized_volatility = last_stdev * np.sqrt(candles_per_year)
    
    return annualized_volatility

def calculate_dynamic_leverage(realized_vol: float | None, target_vol: float, min_lev: int, max_lev: int, default_fallback_lev: int) -> int:
    """
    Calculates dynamic leverage based on realized and target volatility.

    Args:
        realized_vol (float | None): Calculated annualized realized volatility of the asset.
        target_vol (float): User-defined target annualized volatility.
        min_lev (int): Minimum permissible leverage.
        max_lev (int): Maximum permissible leverage.
        default_fallback_lev (int): Leverage to use if realized_vol is invalid.

    Returns:
        int: Calculated and clamped dynamic leverage.
    """
    if realized_vol is None or realized_vol <= 0:
        print(f"Warning: Invalid realized volatility ({realized_vol}). Falling back to default leverage: {default_fallback_lev}x.")
        # Ensure fallback is also clamped, though it should be configured within min/max.
        return max(min_lev, min(default_fallback_lev, max_lev))

    # Ensure target_vol is positive, though input validation should handle this.
    if target_vol <= 0:
        print(f"Warning: Target volatility ({target_vol}) is not positive. Falling back to default leverage: {default_fallback_lev}x.")
        return max(min_lev, min(default_fallback_lev, max_lev))
        
    # Calculate raw leverage
    raw_leverage = target_vol / realized_vol
    
    # Clamp the leverage
    clamped_leverage = max(float(min_lev), min(raw_leverage, float(max_lev)))
    
    # Round to nearest integer for setting leverage
    final_leverage = int(round(clamped_leverage))
    
    # Ensure final_leverage is at least 1 after rounding, and still respects min_lev (e.g. if min_lev was 1 but rounding made it 0)
    final_leverage = max(min_lev, final_leverage) 

    print(f"Dynamic Leverage Calculation: RealizedVol={realized_vol:.4f}, TargetVol={target_vol:.4f}, RawLev={raw_leverage:.2f}, ClampedLev={clamped_leverage:.2f}, FinalIntLev={final_leverage}x")
    return final_leverage

def check_ema_crossover_conditions(df, short_ema_col='EMA100', long_ema_col='EMA200', validation_candles=20, symbol_for_logging=""):
    log_prefix = f"[{threading.current_thread().name}] {symbol_for_logging} check_ema_crossover_conditions:"
    
    required_cols = [short_ema_col, long_ema_col, 'low', 'high']
    if not all(c in df for c in required_cols):
        print(f"{log_prefix} Missing one or more required columns: {required_cols}. Aborting.")
        return None
    if len(df) < validation_candles + 2:
        print(f"{log_prefix} Insufficient data length ({len(df)}) for validation (need {validation_candles + 2}). Aborting.")
        return None
    if df[[short_ema_col, long_ema_col]].iloc[-(validation_candles + 2):].isnull().values.any():
        print(f"{log_prefix} NaN values found in EMAs within the validation lookback period. Aborting.")
        return None

    prev_short, curr_short = df[short_ema_col].iloc[-2], df[short_ema_col].iloc[-1]
    prev_long, curr_long = df[long_ema_col].iloc[-2], df[long_ema_col].iloc[-1]
    
    print(f"{log_prefix} Current Candle ({df.index[-1]}): {short_ema_col}={curr_short:.4f}, {long_ema_col}={curr_long:.4f}")
    print(f"{log_prefix} Previous Candle ({df.index[-2]}): {short_ema_col}={prev_short:.4f}, {long_ema_col}={prev_long:.4f}")

    signal_type = None
    if prev_short <= prev_long and curr_short > curr_long:
        signal_type = "LONG_CROSS"
        print(f"{log_prefix} LONG_CROSS detected.")
    elif prev_short >= prev_long and curr_short < curr_long:
        signal_type = "SHORT_CROSS"
        print(f"{log_prefix} SHORT_CROSS detected.")
    else:
        # print(f"{log_prefix} No crossover event this candle.") # Can be too verbose, enable if needed
        return None

    val_df = df.iloc[-(validation_candles + 1) : -1]
    if len(val_df) < validation_candles:
        print(f"{log_prefix} Potential {signal_type.replace('_CROSS','')} signal, but insufficient validation history ({len(val_df)}/{validation_candles} candles).")
        return "INSUFFICIENT_VALIDATION_HISTORY" 
        
    print(f"{log_prefix} Validating {signal_type.replace('_CROSS','')} signal over {len(val_df)} candles (from {val_df.index[0]} to {val_df.index[-1]}).")

    for i in range(len(val_df)):
        candle_data = val_df.iloc[i]
        ema_short_val = val_df[short_ema_col].iloc[i]
        ema_long_val = val_df[long_ema_col].iloc[i]
        
        touched_short_ema = (candle_data['low'] <= ema_short_val <= candle_data['high'])
        touched_long_ema = (candle_data['low'] <= ema_long_val <= candle_data['high'])

        if touched_short_ema or touched_long_ema:
            reason = ""
            if touched_short_ema: reason += f"Price touched {short_ema_col} ({ema_short_val:.4f}) "
            if touched_long_ema: reason += f"Price touched {long_ema_col} ({ema_long_val:.4f})"
            print(f"{log_prefix} Validation FAILED for {signal_type.replace('_CROSS','')} at {val_df.index[i]} (L:{candle_data['low']}, H:{candle_data['high']}): {reason.strip()}.")
            return "VALIDATION_FAILED"
            
    print(f"{log_prefix} {signal_type.replace('_CROSS','')} signal VALIDATED for {symbol_for_logging}.")
    return signal_type.replace('_CROSS','')

def calculate_swing_high_low(df, window=20, idx=-1):
    if len(df) < window + abs(idx) or idx - window < -len(df): return None, None
    chunk = df.iloc[idx - window : idx]
    if chunk.empty: return (df['high'].iloc[idx-1], df['low'].iloc[idx-1]) if len(df) >= abs(idx)+1 else (None,None)
    return chunk['high'].max(), chunk['low'].min()

def calculate_sl_tp_values(entry_price: float, side: str, atr_value: float, configs: dict, symbol_info: dict = None):
    """
    Calculates Stop Loss (SL) and Take Profit (TP) prices based on ATR.
    SL is defined by ATR * multiplier.
    TP is defined by SL_distance * Risk:Reward ratio.

    Args:
        entry_price (float): The entry price of the trade.
        side (str): "LONG" or "SHORT".
        atr_value (float): The current ATR value for the symbol.
        configs (dict): Dictionary containing 'atr_multiplier_sl' and 'tp_rr_ratio'.
        symbol_info (dict, optional): Symbol information for price precision. Used for logging.


    Returns:
        tuple: (float, float) - (sl_price, tp_price)
               Returns (None, None) if inputs are invalid (e.g., ATR <= 0).
    """
    atr_multiplier_sl = configs.get('atr_multiplier_sl', DEFAULT_ATR_MULTIPLIER_SL)
    tp_rr_ratio = configs.get('tp_rr_ratio', DEFAULT_TP_RR_RATIO)
    
    p_prec = 2 # Default precision for logging if symbol_info not provided
    if symbol_info:
        p_prec = int(symbol_info.get('pricePrecision', 2))

    if atr_value <= 0:
        print(f"Warning: ATR value ({atr_value}) is zero or negative. Cannot calculate ATR-based SL/TP. Returning None, None.")
        return None, None
    if entry_price <= 0:
        print(f"Warning: Entry price ({entry_price}) is zero or negative. Cannot calculate SL/TP. Returning None, None.")
        return None, None


    sl_distance = atr_value * atr_multiplier_sl
    
    # Ensure sl_distance is positive. This is crucial.
    # atr_value is checked > 0, atr_multiplier_sl is validated > 0 during config.
    # However, an explicit check here is a good safeguard.
    if sl_distance <= 0:
        print(f"Critical Error: SL distance calculation resulted in zero or negative value ({sl_distance}). "
              f"ATR: {atr_value}, ATR Multiplier SL: {atr_multiplier_sl}. "
              f"Cannot reliably set SL. Returning None, None.")
        return None, None

    tp_distance = sl_distance * tp_rr_ratio
    # Also ensure tp_distance is positive, which it should be if sl_distance and tp_rr_ratio are positive.
    if tp_distance <= 0:
        print(f"Critical Error: TP distance calculation resulted in zero or negative value ({tp_distance}). "
              f"SL Distance: {sl_distance}, TP R:R Ratio: {tp_rr_ratio}. "
              f"Cannot reliably set TP. Returning None, None.")
        return None, None

    if side == "LONG":
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    elif side == "SHORT":
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance
    else:
        print(f"Error: Invalid side '{side}' in calculate_sl_tp_values. Returning None, None.")
        return None, None

    # Validation: SL/TP should not be equal to entry and should be on the correct side.
    # If SL calculation results in SL >= entry for LONG, or SL <= entry for SHORT, it's invalid.
    # This can happen if sl_distance is extremely small or negative (though guarded above).
    min_tick_size = 1 / (10**p_prec) if p_prec > 0 else 0.01 # Estimate based on precision

    if side == "LONG":
        if sl_price >= entry_price:
            print(f"Warning: Calculated SL ({sl_price:.{p_prec}f}) for LONG is not below entry ({entry_price:.{p_prec}f}). Adjusting SL to entry - min_tick.")
            sl_price = entry_price - min_tick_size
        if tp_price <= entry_price: # TP must be above entry for LONG
            print(f"Warning: Calculated TP ({tp_price:.{p_prec}f}) for LONG is not above entry ({entry_price:.{p_prec}f}). Adjusting TP to entry + min_tick.")
            tp_price = entry_price + min_tick_size
    elif side == "SHORT":
        if sl_price <= entry_price:
            print(f"Warning: Calculated SL ({sl_price:.{p_prec}f}) for SHORT is not above entry ({entry_price:.{p_prec}f}). Adjusting SL to entry + min_tick.")
            sl_price = entry_price + min_tick_size
        if tp_price >= entry_price: # TP must be below entry for SHORT
            print(f"Warning: Calculated TP ({tp_price:.{p_prec}f}) for SHORT is not below entry ({entry_price:.{p_prec}f}). Adjusting TP to entry - min_tick.")
            tp_price = entry_price - min_tick_size
            
    # Ensure SL and TP are not equal after adjustments (highly unlikely but possible if entry_price is extremely small and min_tick_size dominates)
    if abs(sl_price - tp_price) < min_tick_size / 2 : # Check if they are too close
        print(f"Warning: SL and TP are too close or equal after validation. SL: {sl_price:.{p_prec}f}, TP: {tp_price:.{p_prec}f}. Re-adjusting TP.")
        if side == "LONG":
            tp_price = sl_price + min_tick_size # Ensure TP is at least one tick away from SL
        else: # SHORT
            tp_price = sl_price - min_tick_size

    print(f"ATR-based SL: {sl_price:.{p_prec}f}, TP: {tp_price:.{p_prec}f} for {side} from Entry: {entry_price:.{p_prec}f} (ATR: {atr_value:.{p_prec}f}, SL Mult: {atr_multiplier_sl}, TP R:R: {tp_rr_ratio})")
    return sl_price, tp_price

def check_and_adjust_sl_tp_dynamic(cur_price, entry, _, __, cur_sl, cur_tp, side): # initial_sl, initial_tp unused for now
    if entry == 0: return None, None, None
    profit_pct = (cur_price - entry) / entry if side == "LONG" else (entry - cur_price) / entry
    new_sl, new_tp = cur_sl, cur_tp
    adjustment_reason = None

    # Rule 1: If profit >= 0.5%, move SL to +0.2% profit (from entry)
    if profit_pct >= 0.005:
        target_sl = entry * (1 + 0.002 if side == "LONG" else 1 - 0.002)
        # Check if this new SL is an improvement (for LONG, higher; for SHORT, lower)
        if (side == "LONG" and target_sl > new_sl) or \
           (side == "SHORT" and target_sl < new_sl):
            new_sl = target_sl
            adjustment_reason = "SL_PROFIT_LOCK_0.2%"
            print(f"Dynamic SL adjustment for {side} to {new_sl:.4f} ({adjustment_reason})")

    # Rule 2: If loss >= 0.5% (profit_pct <= -0.005), adjust TP to +0.2% profit (from entry)
    # This rule aims to secure a smaller profit if the trade goes into drawdown after initial profit or starts with a loss.
    if profit_pct <= -0.005:
        target_tp = entry * (1 + 0.002 if side == "LONG" else 1 - 0.002)
        # Check if this new TP is an improvement (for LONG, lower but > entry; for SHORT, higher but < entry)
        # And also ensure it's a tighter TP than current, but still profitable
        if (side == "LONG" and target_tp < new_tp and target_tp > entry) or \
           (side == "SHORT" and target_tp > new_tp and target_tp < entry):
            new_tp = target_tp
            adjustment_reason = "TP_DRAWDOWN_REDUCE_0.2%" # Overwrites previous reason if both conditions met, SL prio based on order
            if adjustment_reason and "SL" in adjustment_reason : adjustment_reason += ";TP_DRAWDOWN_REDUCE_0.2%" # Append if SL already adjusted
            else: adjustment_reason = "TP_DRAWDOWN_REDUCE_0.2%"
            print(f"Dynamic TP adjustment for {side} to {new_tp:.4f} ({adjustment_reason})")

    if adjustment_reason:
        return new_sl, new_tp, adjustment_reason
    else:
        return None, None, None # No adjustment made that met criteria

def get_symbol_info(client, symbol):
    try:
        for s in client.futures_exchange_info()['symbols']:
            if s['symbol'] == symbol: return s
        print(f"No info for {symbol}."); return None
    except Exception as e: print(f"Error getting info for {symbol}: {e}"); return None

def calculate_aggregate_open_risk(active_trades_dict, current_balance):
    """
    Calculates the aggregate risk of all open trades as a percentage of the current balance.

    Args:
        active_trades_dict (dict): A dictionary of active trades.
                                   Expected structure: {symbol: {"quantity": float, "entry_price": float, "current_sl_price": float}}
        current_balance (float): The current account balance.

    Returns:
        float: The aggregate risk percentage (e.g., 3.5 for 3.5%), or 0.0 if balance is zero or no trades.
    """
    if not active_trades_dict or current_balance <= 0:
        return 0.0

    total_absolute_risk = 0.0
    for symbol, trade_details in active_trades_dict.items():
        quantity = trade_details.get('quantity')
        entry_price = trade_details.get('entry_price')
        current_sl_price = trade_details.get('current_sl_price')

        if quantity is None or entry_price is None or current_sl_price is None:
            print(f"Warning: Incomplete trade details for {symbol} in calculate_aggregate_open_risk. Skipping this trade for risk calculation.")
            continue
        
        if entry_price == current_sl_price: # Should ideally not happen with valid SL
            print(f"Warning: Entry price and SL price are the same for {symbol} ({entry_price}). Risk for this trade considered zero.")
            individual_risk = 0.0
        else:
            individual_risk = quantity * abs(entry_price - current_sl_price)
        
        total_absolute_risk += individual_risk

    if total_absolute_risk == 0:
        return 0.0
        
    aggregate_risk_percentage = (total_absolute_risk / current_balance) * 100
    return aggregate_risk_percentage

# This is the old definition, which has been replaced by the modified one below.
# def get_account_balance(client, asset="USDT"):
#     try:
#         balances = client.futures_account_balance()
#         for b in balances:
#             if b['asset'] == asset:
#                 print(f"Account Balance ({asset}): {b['balance']}")
#                 return float(b['balance'])
#         print(f"{asset} not found in futures balance.")
#         return 0.0  # Return 0.0 if asset not found but call was successful
#     except BinanceAPIException as e:
#         if e.code == -2015:
#             print(f"Critical Error getting balance: {e}. This is likely an API key permission or IP whitelist issue.")
#             print("Please check your API key settings on Binance: ensure 'Enable Futures' is checked and your IP is whitelisted if restrictive IP access is enabled.")
#             return None # Specific indicator for critical auth/IP error
#         else:
#             print(f"API Error getting balance: {e}")
#             return 0.0 # For other API errors, return 0.0 to indicate balance couldn't be fetched but not necessarily critical auth
#     except Exception as e:
#         print(f"Unexpected error getting balance: {e}")
#         return 0.0 # For non-API unexpected errors

def calculate_position_size(balance, risk_pct, entry, sl, symbol_info, configs=None): # Added configs
    if not symbol_info or balance <= 0 or entry <= 0 or sl <= 0 or abs(entry-sl)<1e-9 : return None
    q_prec = int(symbol_info['quantityPrecision'])
    lot_f = next((f for f in symbol_info['filters'] if f['filterType']=='LOT_SIZE'),None)
    if not lot_f or float(lot_f['stepSize'])==0: print(f"No LOT_SIZE/stepSize for {symbol_info['symbol']}"); return None
    min_qty, step = float(lot_f['minQty']), float(lot_f['stepSize'])
    
    pos_size = (balance * risk_pct) / abs(entry - sl) # Ideal size based on risk_pct
    adj_size = math.floor(pos_size / step) * step
    adj_size = round(adj_size, q_prec)

    if adj_size < min_qty:
        print(f"Initial calculated size {adj_size} based on risk {risk_pct*100:.2f}% is less than min_qty {min_qty}.")
        # If initial size is already too small (even before min_notional check), it implies high risk for min_qty.
        # We can directly check if min_qty itself is too risky.
        risk_for_min_qty = (min_qty * abs(entry - sl)) / balance
        if risk_for_min_qty > risk_pct:
            allow_exceed = configs.get('allow_exceed_risk_for_min_notional', DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL) if configs else DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL
            if allow_exceed:
                print(f"Warning: Using min_qty {min_qty} for {symbol_info['symbol']} results in risk of {risk_for_min_qty*100:.2f}%, exceeding target {risk_pct*100:.2f}%. Allowed by config.")
                adj_size = min_qty
            else:
                # Stricter check: if risk_for_min_qty > risk_pct * 1.5 (original implicit behavior for min_notional adjustment)
                # Or simply, if risk_for_min_qty > risk_pct and not allowed to exceed.
                if risk_for_min_qty > (risk_pct * 1.5): # Maintain a hard cap if not explicitly allowed to exceed any risk
                     print(f"Risk for min_qty {min_qty} ({risk_for_min_qty*100:.2f}%) is too high (>{risk_pct*1.5*100:.2f}%). No trade.")
                     return None
                else: # Risk is > risk_pct but <= risk_pct * 1.5
                     print(f"Warning: Using min_qty {min_qty} for {symbol_info['symbol']} results in risk of {risk_for_min_qty*100:.2f}%. This is > target {risk_pct*100:.2f}% but within 1.5x limit. Proceeding.")
                     adj_size = min_qty

        else: # min_qty is within risk target
            adj_size = min_qty
        
        if adj_size < min_qty: # If after all logic, it's still less (e.g. returned None then somehow adj_size not updated)
            print(f"Final size {adj_size} after min_qty check is still < min_qty {min_qty}. No trade."); return None


    min_not_f = next((f for f in symbol_info['filters'] if f['filterType']=='MIN_NOTIONAL'),None)
    if min_not_f and (adj_size * entry) < float(min_not_f['notional']):
        required_notional_val = float(min_not_f['notional'])
        print(f"Calculated notional for size {adj_size} ({adj_size * entry:.2f}) is below MIN_NOTIONAL ({required_notional_val:.2f}) for {symbol_info['symbol']}.")
        
        # Calculate quantity needed to meet MIN_NOTIONAL
        qty_min_not = math.ceil((required_notional_val / entry) / step) * step
        qty_min_not = round(max(qty_min_not, min_qty), q_prec) # Also ensure it meets min_qty

        risk_for_min_notional_qty = (qty_min_not * abs(entry-sl) / balance)
        
        print(f"Quantity to meet MIN_NOTIONAL: {qty_min_not}. This quantity implies a risk of {risk_for_min_notional_qty*100:.2f}%. Target risk: {risk_pct*100:.2f}%.")

        if risk_for_min_notional_qty > risk_pct:
            allow_exceed = configs.get('allow_exceed_risk_for_min_notional', DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL) if configs else DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL
            if allow_exceed:
                print(f"Warning: Risk for {symbol_info['symbol']} increased to {risk_for_min_notional_qty*100:.2f}% to meet MIN_NOTIONAL (target risk: {risk_pct*100:.2f}%). Allowed by config.")
                adj_size = qty_min_not
            else: # Not allowed to exceed target risk freely
                # Maintain the original stricter check (1.5x risk_pct) if not allowing override
                if risk_for_min_notional_qty > (risk_pct * 1.5):
                    print(f"Risk to meet MIN_NOTIONAL ({risk_for_min_notional_qty*100:.2f}%) is too high (>{risk_pct*1.5*100:.2f}% of target {risk_pct*100:.2f}%). No trade.")
                    return None
                else: # Risk is > risk_pct but <= risk_pct * 1.5
                    print(f"Warning: Risk for {symbol_info['symbol']} increased to {risk_for_min_notional_qty*100:.2f}% to meet MIN_NOTIONAL. This is > target {risk_pct*100:.2f}% but within 1.5x limit. Proceeding.")
                    adj_size = qty_min_not
        else: # Risk for qty_min_not is within target risk_pct
            print(f"Adjusted size to {qty_min_not} to meet MIN_NOTIONAL. Risk ({risk_for_min_notional_qty*100:.2f}%) is within target ({risk_pct*100:.2f}%).")
            adj_size = qty_min_not
            
    if adj_size <= 0: print(f"Final calculated size {adj_size} is zero or negative. No trade."); return None
    # Final check: ensure the chosen adj_size still respects min_qty, as logic for min_notional might overlook it if qty_min_not was small.
    if adj_size < min_qty:
        print(f"Post MIN_NOTIONAL logic, size {adj_size} is < min_qty {min_qty}. This shouldn't happen if max(qty_min_not, min_qty) was used correctly. No trade.")
        return None # Should be captured by max(qty_min_not, min_qty) earlier but as a safeguard.

    print(f"Calculated Position Size: {adj_size} for {symbol_info['symbol']} (Risk: ${(adj_size*abs(entry-sl)):.2f}, Notional: ${(adj_size*entry):.2f})")
    return adj_size

# --- Trade Signature and Cleanup ---
def generate_trade_signature(symbol: str, signal_type: str, entry_price: float, sl_price: float, tp_price: float, quantity: float, precision: int = 4) -> str:
    """Generates a unique signature for a trade to prevent duplicates."""
    # Round prices and quantity to a consistent precision to avoid minor float differences causing new signatures.
    # The precision should be reasonable for the asset.
    return f"{symbol}_{signal_type}_{entry_price:.{precision}f}_{sl_price:.{precision}f}_{tp_price:.{precision}f}_{quantity:.{precision}f}"

def cleanup_recent_trade_signatures():
    """Cleans up old trade signatures to prevent memory bloat."""
    global recent_trade_signatures, recent_trade_signatures_lock, last_signature_cleanup_time, recent_trade_signature_cleanup_interval
    
    now = dt.now()
    # Check if it's time to cleanup
    if (now - last_signature_cleanup_time).total_seconds() < recent_trade_signature_cleanup_interval:
        return

    with recent_trade_signatures_lock:
        last_signature_cleanup_time = now # Update last cleanup time inside lock
        
        signatures_to_delete = []
        # Check each signature's timestamp (stored as value in recent_trade_signatures)
        # Let's define that signatures are kept for, e.g., 2 minutes (120 seconds)
        # This is longer than the cooldown to catch rapid retries that might bypass cooldown if it's very short.
        # Or, make it configurable. For now, 120s.
        max_signature_age_seconds = 120 
        
        for sig, timestamp in recent_trade_signatures.items():
            if (now - timestamp).total_seconds() > max_signature_age_seconds:
                signatures_to_delete.append(sig)
        
        for sig in signatures_to_delete:
            del recent_trade_signatures[sig]
            
        if signatures_to_delete:
            print(f"Cleaned up {len(signatures_to_delete)} old trade signatures. {len(recent_trade_signatures)} remain.")

# --- Pre-Order Sanity Checks ---
def pre_order_sanity_checks(symbol, signal, entry_price, sl_price, tp_price, quantity, 
                            symbol_info, current_balance, risk_percent_config, configs, 
                            specific_leverage_for_trade: int, # Added specific leverage for this trade
                            klines_df_for_debug=None, is_unmanaged_check=False):
    """
    Performs a series of checks before placing an order to ensure parameters are valid
    and risk is managed according to configuration.
    For unmanaged checks, risk percentage validation is skipped.

    Args:
        symbol (str): Trading symbol (e.g., "BTCUSDT").
        signal (str): "LONG" or "SHORT".
        entry_price (float): Proposed entry price.
        sl_price (float): Proposed stop-loss price.
        tp_price (float): Proposed take-profit price.
        quantity (float): Proposed quantity to trade.
        symbol_info (dict): Symbol information from Binance API.
        current_balance (float): Current account balance.
        risk_percent_config (float): User-configured risk percentage (e.g., 0.01 for 1%).
        configs (dict): Overall bot configurations.
        klines_df_for_debug (pd.DataFrame, optional): Kline data, for potential future advanced checks like ATR.

    Returns:
        tuple: (bool, str) - (True, "Checks passed") if all checks are okay.
                             (False, "Reason for failure") if any check fails.
    """
    p_prec = int(symbol_info['pricePrecision'])
    q_prec = int(symbol_info['quantityPrecision'])

    # 1. Price Sanity Checks
    if not all(isinstance(p, (int, float)) and p > 0 and math.isfinite(p) for p in [entry_price, sl_price, tp_price]):
        return False, f"Invalid price(s): Entry={entry_price}, SL={sl_price}, TP={tp_price}. Must be positive finite numbers."

    # 2. SL/TP Validity Checks
    if signal == "LONG":
        if not (sl_price < entry_price):
            return False, f"SL price ({sl_price:.{p_prec}f}) must be below entry price ({entry_price:.{p_prec}f}) for LONG."
        if not (tp_price > entry_price):
            return False, f"TP price ({tp_price:.{p_prec}f}) must be above entry price ({entry_price:.{p_prec}f}) for LONG."
    elif signal == "SHORT":
        if not (sl_price > entry_price):
            return False, f"SL price ({sl_price:.{p_prec}f}) must be above entry price ({entry_price:.{p_prec}f}) for SHORT."
        if not (tp_price < entry_price):
            return False, f"TP price ({tp_price:.{p_prec}f}) must be below entry price ({entry_price:.{p_prec}f}) for SHORT."
    else:
        return False, f"Invalid signal type: {signal}."

    if sl_price == entry_price:
        return False, f"SL price ({sl_price:.{p_prec}f}) cannot be equal to entry price ({entry_price:.{p_prec}f})."
    if tp_price == entry_price:
        return False, f"TP price ({tp_price:.{p_prec}f}) cannot be equal to entry price ({entry_price:.{p_prec}f})."

    # 3. Quantity and Notional Value Checks
    if not (isinstance(quantity, (int, float)) and quantity > 0 and math.isfinite(quantity)):
        return False, f"Invalid quantity ({quantity}). Must be a positive finite number."

    lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
    if lot_size_filter:
        min_qty = float(lot_size_filter['minQty'])
        step_size = float(lot_size_filter['stepSize'])
        if quantity < min_qty:
            return False, f"Quantity ({quantity:.{q_prec}f}) is less than minQty ({min_qty:.{q_prec}f})."
        # Check if quantity adheres to stepSize (modulo check for floats is tricky, use decimal or string formatting)
        # For simplicity, assume calculate_position_size correctly handles stepSize.
        # A more precise check: (Decimal(str(quantity)) - Decimal(str(min_qty))) % Decimal(str(step_size)) == 0
        # Or check if round(quantity / step_size) * step_size == quantity (within tolerance)
        if abs(round(quantity / step_size) * step_size - quantity) > 1e-9 and step_size > 1e-9 : # Tolerance for float math
             # Check if quantity is a multiple of step_size starting from min_qty
            if abs((quantity - min_qty) % step_size) > 1e-9 and abs(quantity % step_size) > 1e-9 : # Check both relative to min_qty and absolute
                return False, f"Quantity ({quantity:.{q_prec}f}) does not meet stepSize ({step_size:.{q_prec}f}) requirement."

    min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
    if min_notional_filter:
        min_notional_val = float(min_notional_filter['notional'])
        current_notional = quantity * entry_price
        if current_notional < min_notional_val:
            return False, f"Notional value ({current_notional:.2f}) is less than MIN_NOTIONAL ({min_notional_val:.2f})."

    # 4. Risk Calculation Verification
    if current_balance <= 0:
        return False, f"Current balance ({current_balance:.2f}) is zero or negative."
        
    risk_amount_abs = quantity * abs(entry_price - sl_price)
    if risk_amount_abs == 0 and entry_price != sl_price : # Should not happen if prices are different and qty > 0
        return False, "Calculated absolute risk amount is zero despite different entry/SL. Check inputs."

    # If entry_price == sl_price, risk_amount_abs would be 0. This case is already caught by "SL price cannot be equal to entry price".
    # So abs(entry_price - sl_price) should always be > 0 here.

    risk_percentage_actual = risk_amount_abs / current_balance
    
    if not is_unmanaged_check: # Only perform detailed risk % checks for normally managed new trades
        allow_exceed_risk_config = configs.get('allow_exceed_risk_for_min_notional', DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL)
        epsilon = 0.00001 # Small value for float comparisons

        if not allow_exceed_risk_config: # "Stricter risk adherence is enabled" path (allow_exceed_risk_for_min_notional is False)
            # If actual risk is greater than configured risk (plus epsilon), reject.
            # Otherwise (actual risk <= configured risk), it's acceptable.
            if risk_percentage_actual > (risk_percent_config + epsilon):
                return False, (f"Actual risk ({risk_percentage_actual*100:.3f}%) exceeds "
                               f"configured risk ({risk_percent_config*100:.3f}%). Stricter adherence enabled, trade rejected.")
        
        else: # allow_exceed_risk_config is True (user explicitly allowed exceeding risk for min_notional)
            # Actual risk can exceed configured_risk, but not beyond 1.5x configured_risk.
            max_permissible_risk = risk_percent_config * 1.5
            if risk_percentage_actual > (max_permissible_risk + epsilon):
                return False, (f"Actual risk ({risk_percentage_actual*100:.3f}%) exceeds "
                               f"the maximum permissible limit ({max_permissible_risk*100:.3f}%, which is 1.5x configured risk) "
                               f"even when 'allow_exceed_risk_for_min_notional' is enabled.")
                               
    else: # For unmanaged checks, just ensure risk is not absurdly high, e.g. > 50% of balance for a single trade SL
        if risk_percentage_actual > 0.5: # Arbitrary high cap for unmanaged safety SL/TP
             return False, (f"Calculated SL for UNMANAGED trade implies extremely high risk ({risk_percentage_actual*100:.2f}% of balance). "
                            f"Entry: {entry_price}, SL: {sl_price}, Qty: {quantity}. Check position and SL logic.")


    # 5. Sufficient Balance (Basic Margin Check)
    # For unmanaged trades, this check is against current balance for an existing position's margin,
    # which is implicitly covered as the position exists.
    # For new trades, it's a pre-check.
    # Use the specific leverage intended for this trade for margin calculation
    if specific_leverage_for_trade <= 0:
        return False, f"Leverage for trade ({specific_leverage_for_trade}) must be positive."
    
    required_margin = (quantity * entry_price) / specific_leverage_for_trade
    if required_margin > current_balance: # This is a simplified check. Binance actual margin req might differ.
        return False, (f"Estimated required margin ({required_margin:.2f} USDT) exceeds "
                       f"current balance ({current_balance:.2f} USDT) for quantity {quantity:.{q_prec}f} at {entry_price:.{p_prec}f} with {specific_leverage_for_trade}x leverage.")

    return True, "Checks passed"


# --- Main Trading Logic ---
def get_all_usdt_perpetual_symbols(client):
    print("\nFetching all USDT perpetual symbols...")
    try:
        syms = [s['symbol'] for s in client.futures_exchange_info()['symbols']
                if s.get('symbol','').endswith('USDT') and s.get('contractType')=='PERPETUAL'
                and s.get('status')=='TRADING' and s.get('quoteAsset')=='USDT' and s.get('marginAsset')=='USDT']
        print(f"Found {len(syms)} USDT perpetuals. Examples: {syms[:5]}")
        return sorted(list(set(syms)))
    except Exception as e: print(f"Error fetching symbols: {e}"); return []

def format_elapsed_time(start_time):
    return f"(Cycle Elapsed: {(time.time() - start_time):.2f}s)"

def process_symbol_task(symbol, client, configs, lock):
    # configs['cycle_start_time_ref'] is vital here
    thread_name = threading.current_thread().name
    cycle_start_ref = configs.get('cycle_start_time_ref', time.time()) # Fallback if not passed
    print(f"[{thread_name}] Processing: {symbol} {format_elapsed_time(cycle_start_ref)}")
    try:
        klines_df, klines_error = get_historical_klines(client, symbol) # Uses default limit 500
        
        if klines_error:
            if isinstance(klines_error, BinanceAPIException) and klines_error.code == -1121:
                # Specific handling for "Invalid symbol"
                msg = f"Skipped: Invalid symbol reported by API (code -1121)."
                print(f"[{thread_name}] {symbol}: {msg} {format_elapsed_time(cycle_start_ref)}")
                # TODO: Consider adding logic here to remove 'symbol' from a shared list of monitored symbols
                # to prevent repeated checks for persistently invalid symbols. This requires careful synchronization.
                return f"{symbol}: {msg}"
            else:
                # General error during kline fetch
                msg = f"Skipped: Error fetching klines ({str(klines_error)})."
                print(f"[{thread_name}] {symbol}: {msg} {format_elapsed_time(cycle_start_ref)}")
                return f"{symbol}: {msg}"

        if klines_df.empty or len(klines_df) < 202:
            msg = f"Skipped: Insufficient klines ({len(klines_df)})."
            print(f"[{thread_name}] {symbol}: {msg} {format_elapsed_time(cycle_start_ref)}")
            return f"{symbol}: {msg}"
        
        print(f"[{thread_name}] {symbol}: Sufficient klines ({len(klines_df)}). Calling manage_trade_entry {format_elapsed_time(cycle_start_ref)}")
        manage_trade_entry(client, configs, symbol, klines_df.copy(), lock)
        return f"{symbol}: Processed"
    except Exception as e:
        error_detail = f"Unhandled error in process_symbol_task: {e}"
        print(f"[{thread_name}] ERROR processing {symbol}: {error_detail} {format_elapsed_time(cycle_start_ref)}")
        traceback.print_exc() # Print full traceback for thread errors
        return f"{symbol}: Error - {error_detail}"

def manage_trade_entry(client, configs, symbol, klines_df, lock): # lock here is active_trades_lock
    global active_trades, symbols_currently_processing, symbols_currently_processing_lock
    global trading_halted_drawdown, trading_halted_daily_loss, daily_state_lock # For checking halt status
    global last_signal_time, last_signal_lock # For Cooldown Timer
    global recent_trade_signatures, recent_trade_signatures_lock # For Trade Signature Check

    log_prefix = f"[{threading.current_thread().name}] {symbol} manage_trade_entry:"

    # --- Cleanup old trade signatures (periodically) ---
    cleanup_recent_trade_signatures()

    # --- Check 1: Overall trading halt status (Daily Drawdown / Daily Loss) ---
    with daily_state_lock:
        if trading_halted_drawdown:
            print(f"{log_prefix} Trade entry BLOCKED for {symbol}. Reason: Max Drawdown trading halt is active.")
            return
        if trading_halted_daily_loss:
            print(f"{log_prefix} Trade entry BLOCKED for {symbol}. Reason: Daily Stop Loss trading halt is active.")
            return
    # --- End trading halt status check ---

    # Attempt to mark this symbol as being processed by this thread.
    with symbols_currently_processing_lock:
        if symbol in symbols_currently_processing:
            print(f"{log_prefix} Symbol is already being processed by another thread. Skipping this instance.")
            return
        symbols_currently_processing.add(symbol)

    try:
        # --- All subsequent logic is now within this try block ---

        # Initial check for sufficient kline data 
        if klines_df.empty or len(klines_df) < 202: 
            print(f"{log_prefix} Insufficient kline data ({len(klines_df)}). Aborting.")
            return

        # --- Generate trading signal based on strategy (EMAs, etc.) ---
        # This part calculates EMAs and determines if a raw signal exists.
        # It does not yet consider cooldowns or other pre-trade checks.
        klines_df_copy = klines_df.copy() # Work on a copy to avoid modifying original DataFrame if passed around
        klines_df_copy['EMA100'] = calculate_ema(klines_df_copy, 100)
        klines_df_copy['EMA200'] = calculate_ema(klines_df_copy, 200)

        if klines_df_copy['EMA100'] is None or klines_df_copy['EMA200'] is None or \
           klines_df_copy['EMA100'].isnull().all() or klines_df_copy['EMA200'].isnull().all() or \
           len(klines_df_copy) < 202: 
            print(f"{log_prefix} EMA calculation failed or insufficient data for signal generation. Aborting.")
            return

        raw_signal = check_ema_crossover_conditions(klines_df_copy, symbol_for_logging=symbol)
        if raw_signal not in ["LONG", "SHORT"]:
            if isinstance(raw_signal, str): # If it's a reason string like "VALIDATION_FAILED"
                 print(f"{log_prefix} No actionable raw signal for {symbol}. Reason: {raw_signal}")
            return # No valid raw signal from strategy
        
        # --- Pre-Trade Checks (Order is important) ---

        # Check 2: Cooldown Timer
        with last_signal_lock:
            cooldown_seconds = configs.get("signal_cooldown_seconds", 30) # Default 30s, make configurable if needed
            if symbol in last_signal_time and (dt.now() - last_signal_time[symbol]).total_seconds() < cooldown_seconds:
                print(f"{log_prefix} New signal ({raw_signal}) processing SKIPPED for {symbol}. Cooldown active ({cooldown_seconds}s). Last signal: {last_signal_time[symbol]}")
                return
            # No need to update last_signal_time[symbol] here yet, do it only if all checks pass and trade is attempted.

        # Check 3: Bot-Managed Active Trade
        with lock: # active_trades_lock
            if symbol in active_trades:
                active_trade_details = active_trades[symbol]
                print(f"{log_prefix} New signal ({raw_signal}) processing SKIPPED. Symbol '{symbol}' already has an active '{active_trade_details.get('side')}' trade @ {active_trade_details.get('entry_price')}.")
                return

        # Check 4: Binance Live Position Check (for this specific symbol)
        try:
            position_info_list = client.futures_position_information(symbol=symbol)
            if position_info_list and isinstance(position_info_list, list):
                # The API returns a list, usually with one item for the symbol if queried directly.
                # If multiple (e.g. for HEDGE mode, though strategy assumes one-way), this needs more complex handling.
                # For now, assume the first entry is the relevant one for one-way mode.
                pos_data = position_info_list[0] if position_info_list else None
                if pos_data and float(pos_data.get('positionAmt', 0.0)) != 0:
                    print(f"{log_prefix} New signal ({raw_signal}) processing SKIPPED. Symbol '{symbol}' already has an open position on Binance. Amount: {pos_data.get('positionAmt')}.")
                    # This implies a desync or manually opened position. Bot should not interfere.
                    return
        except BinanceAPIException as e:
            print(f"{log_prefix} API Error checking live position for {symbol}: {e}. Proceeding cautiously.")
        except Exception as e:
            print(f"{log_prefix} Unexpected error checking live position for {symbol}: {e}. Proceeding cautiously.")

        # Check 5: Open SL/TP Orders on Binance (guards against memory desync / leftover orders)
        try:
            open_orders_on_exchange = client.futures_get_open_orders(symbol=symbol)
            if open_orders_on_exchange:
                print(f"{log_prefix} New signal ({raw_signal}) processing SKIPPED. Symbol '{symbol}' has {len(open_orders_on_exchange)} existing open order(s) on Binance (potential SL/TP or pending entry).")
                for order in open_orders_on_exchange:
                    print(f"  - Order ID: {order['orderId']}, Type: {order['type']}, Side: {order['side']}, Qty: {order['origQty']}, Price: {order.get('price', 'N/A')}, StopPrice: {order.get('stopPrice', 'N/A')}")
                return
        except BinanceAPIException as e:
            print(f"{log_prefix} API Error checking open SL/TP orders for {symbol}: {e}. Proceeding cautiously.")
        except Exception as e:
            print(f"{log_prefix} Unexpected error checking open SL/TP orders for {symbol}: {e}. Proceeding cautiously.")

        # If all preliminary checks passed, this is the validated signal to proceed with.
        signal = raw_signal 
        
        # Max concurrent position check (globally for the bot)
        with lock: # active_trades_lock
            if len(active_trades) >= configs["max_concurrent_positions"]:
                print(f"{log_prefix} Max concurrent positions ({configs['max_concurrent_positions']}) reached globally. Cannot open new {signal} trade for {symbol}.")
                return
        
        print(f"\n{log_prefix} --- Proceeding with New Validated Trade Signal: {signal} for {symbol} ---")
        print(f"{log_prefix} Passed Cooldown, ActiveTrade, LivePosition, OpenOrders, MaxConcurrent checks.")
        
        symbol_info = get_symbol_info(client, symbol)
        if not symbol_info:
            print(f"{log_prefix} Failed to retrieve symbol information for {symbol}. Abort.")
            # Not sending trade rejection here as parameters for it aren't fully formed.
            return

        # entry_p will be determined from klines_df_copy (which has EMAs)
        entry_p = klines_df_copy['close'].iloc[-1] 
        # Ensure sl_p, tp_p, qty_calc are initialized for send_trade_rejection_notification calls
        sl_p_calc, tp_p_calc, qty_calc_val = None, None, None 
        
        print(f"{log_prefix} Proposed entry price (last close from klines_df_copy): {entry_p} for {symbol}")
        
        # --- Dynamic Leverage Calculation and Setting ---
        target_leverage = configs['leverage'] # Start with configured fixed leverage as default/fallback
        
        candles_per_day_approx = 24 * (60 / 15) # Assuming 15min kline interval for default
        candles_per_year_approx = int(candles_per_day_approx * 365)

        realized_vol = calculate_realized_volatility(
            klines_df_copy, # Use klines_df_copy which has necessary columns potentially
            configs.get('realized_volatility_period', DEFAULT_REALIZED_VOLATILITY_PERIOD),
            candles_per_year_approx
        )

        if realized_vol is not None and realized_vol > 0:
            dynamic_lev = calculate_dynamic_leverage(
                realized_vol,
                configs.get('target_annualized_volatility', DEFAULT_TARGET_ANNUALIZED_VOLATILITY),
                configs.get('min_leverage', DEFAULT_MIN_LEVERAGE),
                configs.get('max_leverage', DEFAULT_MAX_LEVERAGE),
                configs['leverage'] # Fallback to user's initially configured fixed leverage
            )
            target_leverage = dynamic_lev
            print(f"{log_prefix} Dynamic leverage calculated: {target_leverage}x (Realized Vol: {realized_vol:.4f})")
            # Alert if dynamic leverage is different from the fixed configuration
            if target_leverage != configs['leverage']:
                if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                    leverage_alert_msg = (
                        f"ℹ️ Dynamic Leverage Update for `{symbol}`\n"
                        f"Set to: `{target_leverage}x` (from fixed config: `{configs['leverage']}x`)\n"
                        f"Basis: Realized Vol: `{realized_vol:.3f}`, Target Vol: `{configs.get('target_annualized_volatility', DEFAULT_TARGET_ANNUALIZED_VOLATILITY):.3f}`"
                    )
                    send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], leverage_alert_msg)
        else:
            print(f"{log_prefix} Could not calculate valid dynamic leverage (Realized Vol: {realized_vol}). Using fixed leverage: {target_leverage}x.")
            # target_leverage already holds configs['leverage']
        
        # Set leverage (either dynamic or fixed if dynamic failed)
        # Pass configs to set_margin_type_on_symbol for potential alerts there
        if not (set_leverage_on_symbol(client, symbol, target_leverage) and \
                set_margin_type_on_symbol(client, symbol, configs['margin_type'], configs)): # Pass configs
            reason = f"Failed to set leverage ({target_leverage}x) or margin type."
            print(f"{log_prefix} {reason} for {symbol}. Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p_calc, tp_p_calc, qty_calc_val, symbol_info, configs)
            return
        # If successful, print the leverage that was actually set.
        print(f"{log_prefix} Leverage for {symbol} set to {target_leverage}x and margin type to {configs['margin_type']}.")
        # --- End Dynamic Leverage ---


        # --- ATR and SL/TP Calculation ---
        # Use klines_df_copy for ATR calculation as well
        klines_df_copy['atr'] = calculate_atr(klines_df_copy, period=configs.get('atr_period', DEFAULT_ATR_PERIOD))
        current_atr_value = klines_df_copy['atr'].iloc[-1]

        if pd.isna(current_atr_value) or current_atr_value <= 0:
            reason = f"Invalid ATR value ({current_atr_value:.4f}) for {symbol}. Cannot proceed with trade."
            print(f"{log_prefix} {reason}")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, None, None, qty_calc_val, symbol_info, configs)
            if 'atr' in klines_df_copy.columns: klines_df_copy.drop(columns=['atr'], inplace=True)
            return
        
        print(f"{log_prefix} Current ATR for {symbol} (period {configs.get('atr_period', DEFAULT_ATR_PERIOD)}): {current_atr_value:.{symbol_info.get('pricePrecision', 2)}f}")

        sl_p_calc, tp_p_calc = calculate_sl_tp_values(entry_p, signal, current_atr_value, configs, symbol_info)
        if sl_p_calc is None or tp_p_calc is None:
            reason = "ATR-based Stop Loss / Take Profit calculation failed."
            print(f"{log_prefix} {reason} for {symbol}. Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p_calc, tp_p_calc, qty_calc_val, symbol_info, configs)
            if 'atr' in klines_df_copy.columns: klines_df_copy.drop(columns=['atr'], inplace=True) # Cleanup
            return
        print(f"{log_prefix} Calculated ATR-based SL: {sl_p_calc:.{symbol_info.get('pricePrecision',2)}f}, TP: {tp_p_calc:.{symbol_info.get('pricePrecision',2)}f} for {symbol}.")
        if 'atr' in klines_df_copy.columns: klines_df_copy.drop(columns=['atr'], inplace=True) # Cleanup ATR column
        # --- End ATR and SL/TP Calculation ---

        acc_bal = get_account_balance(client, configs) 
        if acc_bal is None:
            reason = "Critical error fetching account balance."
            print(f"{log_prefix} {reason} Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p_calc, tp_p_calc, qty_calc_val, symbol_info, configs)
            return
        if acc_bal <= 0:
            reason = f"Zero or negative account balance ({acc_bal})."
            print(f"{log_prefix} {reason} Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p_calc, tp_p_calc, qty_calc_val, symbol_info, configs)
            return
        print(f"{log_prefix} Account balance: {acc_bal} for position sizing.")

        # --- Calculate Position Size (Per-Trade Risk is inherently handled here) ---
        qty_to_order_val = calculate_position_size(acc_bal, configs['risk_percent'], entry_p, sl_p_calc, symbol_info, configs)
        if qty_to_order_val is None or qty_to_order_val <= 0:
            reason = f"Invalid position size calculated (Qty: {qty_to_order_val}). Check logs for per-trade risk details."
            print(f"{log_prefix} {reason} for {symbol}. Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p_calc, tp_p_calc, qty_to_order_val, symbol_info, configs)
            return
        print(f"{log_prefix} Calculated position size (per-trade risk considered): {qty_to_order_val} for {symbol}.")

        # --- Portfolio Risk Check ---
        # Ensure sl_p_calc is valid before using it for new_trade_risk_abs calculation
        if sl_p_calc == entry_p: # Should be caught by sanity checks later, but good to be safe
            reason = "Stop loss price is equal to entry price before portfolio risk check."
            print(f"{log_prefix} {reason} Aborting before portfolio risk check.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p_calc, tp_p_calc, qty_to_order_val, symbol_info, configs)
            return

        new_trade_risk_abs = qty_to_order_val * abs(entry_p - sl_p_calc)
        
        current_portfolio_risk_abs = 0
        current_portfolio_risk_pct = 0.0
        with lock: # active_trades_lock
            # Make a deep copy if calculate_aggregate_open_risk modifies, but it shouldn't
            # For now, direct pass is fine as it only reads.
            current_portfolio_risk_pct = calculate_aggregate_open_risk(active_trades, acc_bal)
            if acc_bal > 0 : # Avoid division by zero if somehow balance is zero but passed initial checks
                 current_portfolio_risk_abs = (current_portfolio_risk_pct / 100) * acc_bal

        potential_total_absolute_risk = current_portfolio_risk_abs + new_trade_risk_abs
        potential_portfolio_risk_pct = (potential_total_absolute_risk / acc_bal) * 100 if acc_bal > 0 else float('inf')
        
        portfolio_risk_cap_config_pct = configs.get("portfolio_risk_cap", DEFAULT_PORTFOLIO_RISK_CAP) # e.g., 5.0 for 5%

        print(f"{log_prefix} Portfolio Risk Check for {symbol}:")
        print(f"  Current Aggregate Portfolio Risk: {current_portfolio_risk_pct:.2f}% ({current_portfolio_risk_abs:.2f} USDT)")
        print(f"  Potential New Trade Risk ({symbol}): {(new_trade_risk_abs / acc_bal * 100):.2f}% ({new_trade_risk_abs:.2f} USDT)")
        print(f"  Projected Total Portfolio Risk: {potential_portfolio_risk_pct:.2f}%")
        print(f"  Portfolio Risk Cap: {portfolio_risk_cap_config_pct:.2f}%")

        if potential_portfolio_risk_pct > portfolio_risk_cap_config_pct:
            reason = (f"Portfolio risk limit exceeded. Current: {current_portfolio_risk_pct:.2f}%, "
                      f"New Trade: {(new_trade_risk_abs / acc_bal * 100):.2f}%, "
                      f"Projected: {potential_portfolio_risk_pct:.2f}%, "
                      f"Cap: {portfolio_risk_cap_config_pct:.2f}%.")
            print(f"{log_prefix} {reason} Trade for {symbol} rejected.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p_calc, tp_p_calc, qty_to_order_val, symbol_info, configs)
            return
        print(f"{log_prefix} Portfolio risk check PASSED for {symbol}.")
        # --- End Portfolio Risk Check ---

        # Use rounded quantity for all further steps including sanity checks and order placement
        qty_to_order_final = round(qty_to_order_val, int(symbol_info['quantityPrecision']))
        if qty_to_order_final == 0.0:
            reason = f"Calculated quantity {qty_to_order_val} rounds to zero."
            print(f"{log_prefix} {reason} for {symbol}. Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p_calc, tp_p_calc, qty_to_order_val, symbol_info, configs) # Log original calc for rejection
            return
        print(f"{log_prefix} Final quantity to order (rounded): {qty_to_order_final} for {symbol}.")

        passed_sanity_checks, sanity_check_reason = pre_order_sanity_checks(
            symbol=symbol, signal=signal, entry_price=entry_p, sl_price=sl_p_calc, tp_price=tp_p_calc,
            quantity=qty_to_order_final, symbol_info=symbol_info, current_balance=acc_bal, 
            risk_percent_config=configs['risk_percent'], configs=configs, 
            specific_leverage_for_trade=target_leverage, # Pass the actual leverage being used
            klines_df_for_debug=klines_df_copy # Pass klines_df_copy for debug
        )

        if not passed_sanity_checks:
            print(f"{log_prefix} Pre-order sanity checks FAILED: {sanity_check_reason}")
            send_trade_rejection_notification(symbol, signal, f"Sanity Check Failed: {sanity_check_reason}", 
                                              entry_p, sl_p_calc, tp_p_calc, qty_to_order_final, symbol_info, configs)
            return 

        print(f"{log_prefix} Pre-order sanity checks PASSED.")

        # --- Check 6: Trade Signature Check (after all parameters are confirmed) ---
        # Precision for signature generation should match symbol's price and quantity precision if possible,
        # or use a reasonable default. Let's use 4 decimal places as a general default for prices in signature.
        # Quantity precision can be derived from symbol_info.
        qty_prec_for_sig = int(symbol_info.get('quantityPrecision', 2)) # Use symbol's qty precision
        price_prec_for_sig = int(symbol_info.get('pricePrecision', 4)) # Use symbol's price precision, default 4

        trade_sig = generate_trade_signature(symbol, signal, entry_p, sl_p_calc, tp_p_calc, qty_to_order_final, precision=price_prec_for_sig)
        # Note: quantity in signature uses same precision as price for simplicity in `generate_trade_signature` default.
        # Consider adjusting `generate_trade_signature` if different precisions for qty are needed.
        # For now, using price_prec_for_sig for qty in signature too. A more robust way:
        # trade_sig = f"{symbol}_{signal}_{entry_p:.{price_prec_for_sig}f}_{sl_p_calc:.{price_prec_for_sig}f}_{tp_p_calc:.{price_prec_for_sig}f}_{qty_to_order_final:.{qty_prec_for_sig}f}"


        with recent_trade_signatures_lock:
            # Check against signatures from last N seconds (e.g., 60 seconds)
            # This is different from cooldown; cooldown is per symbol, signature is per exact trade params.
            # max_signature_age_seconds from cleanup_recent_trade_signatures is 120s.
            # Let's use a shorter window for *blocking* duplicates, e.g., 60s.
            # The cleanup job will remove older ones eventually.
            block_duplicate_signature_within_seconds = 60 
            if trade_sig in recent_trade_signatures and \
               (dt.now() - recent_trade_signatures[trade_sig]).total_seconds() < block_duplicate_signature_within_seconds:
                print(f"{log_prefix} New signal ({signal}) processing SKIPPED for {symbol}. Duplicate trade signature found within {block_duplicate_signature_within_seconds}s: {trade_sig}")
                # Do not update last_signal_time here, as this specific duplicate was caught by signature.
                return
            # If not a recent duplicate, record this signature. It will be cleaned up later.
            # No need to add it here yet, only add *after* successful order placement attempt or confirmation.

        # --- All checks passed, ready to attempt trade ---
        # Update Cooldown Timer's last_signal_time for this symbol *before* placing order.
        # If order placement fails, the cooldown still applies to prevent immediate retries of the same signal.
        with last_signal_lock:
            last_signal_time[symbol] = dt.now()
            print(f"{log_prefix} Updated last_signal_time for {symbol} to {last_signal_time[symbol]} (Cooldown activated).")

        
        # Determine positionSide based on signal
        position_side_for_trade = "LONG" if signal == "LONG" else "SHORT"

        print(f"{log_prefix} Attempting {signal} ({position_side_for_trade}) {qty_to_order_final} {symbol} @MKT (EP:{entry_p:.{symbol_info['pricePrecision']}f}), SL:{sl_p_calc:.{symbol_info['pricePrecision']}f}, TP:{tp_p_calc:.{symbol_info['pricePrecision']}f}")
        
        # Call place_new_order and unpack both order object and potential error message
        entry_order, entry_order_error_msg = place_new_order(client, 
                                                              symbol_info, 
                                                              "BUY" if signal=="LONG" else "SELL", 
                                                              "MARKET", 
                                                              qty_to_order_final,
                                                              position_side=position_side_for_trade)

        if not entry_order or entry_order.get('status') != 'FILLED':
            base_reason = f"Market entry order failed or not filled. Status: {entry_order.get('status') if entry_order else 'N/A'}"
            detailed_reason = f"{base_reason}. API Error: {entry_order_error_msg}" if entry_order_error_msg else base_reason
            print(f"{log_prefix} {detailed_reason} for {symbol}: {entry_order}") # Log includes entry_order object which might be None
            send_trade_rejection_notification(symbol, signal, detailed_reason, entry_p, sl_p_calc, tp_p_calc, qty_to_order_final, symbol_info, configs)
            return
        
        # --- Final Lock for Order Placement and active_trades update ---
        # Note: The entry order has already been attempted (and succeeded if we are here)
        # The original code had a second call to place_new_order within the lock.
        # This is redundant if the first call succeeds and dangerous if it was meant as a retry without proper state handling.
        # Assuming the first successful entry_order is the one to use.
        # The primary purpose of this lock section should be to atomically check max positions and update active_trades.

        with lock: # This is active_trades_lock
            if len(active_trades) >= configs["max_concurrent_positions"]:
                print(f"{log_prefix} Max concurrent positions ({configs['max_concurrent_positions']}) reached just before recording trade. Order for {symbol} was placed but will not be managed by bot. This may require manual intervention for order ID {entry_order.get('orderId') if entry_order else 'UNKNOWN'}.")
                orphan_reason = f"Max positions ({configs['max_concurrent_positions']}) met after order fill. Order ID {entry_order.get('orderId')} for {symbol} is orphaned and needs manual management."
                send_trade_rejection_notification(symbol, signal, orphan_reason, entry_p, sl_p_calc, tp_p_calc, qty_to_order_final, symbol_info, configs)
                # Do NOT add signature here as the trade is orphaned.
                return

            # If we are here, entry_order is successful and there's space for the trade.
            print(f"{log_prefix} Market entry order FILLED. Details: {entry_order}")
            actual_ep = float(entry_order.get('avgPrice', entry_p)) # Use actual fill price
            
            # Add the trade signature to recent_trade_signatures *after* successful fill and *before* SL/TP placement.
            # This ensures that if SL/TP placement fails, the signature is still there to prevent immediate retries
            # of the same entry if the bot restarts or logic loops quickly.
            with recent_trade_signatures_lock:
                recent_trade_signatures[trade_sig] = dt.now()
                print(f"{log_prefix} Trade signature recorded for {symbol}: {trade_sig}")

            final_sl_price = sl_p_calc
            final_tp_price = tp_p_calc

            # Adjust SL/TP based on actual fill price vs proposed entry price
            if entry_p > 0 and abs(actual_ep - entry_p) > 1e-9: # If actual fill is different
                sl_dist_pct = abs(entry_p - sl_p_calc) / entry_p
                tp_dist_pct = abs(entry_p - tp_p_calc) / entry_p
                
                sl_p_adj = actual_ep * (1 - sl_dist_pct if signal == "LONG" else 1 + sl_dist_pct)
                tp_p_adj = actual_ep * (1 + tp_dist_pct if signal == "LONG" else 1 - tp_dist_pct)
                
                price_prec_for_print = int(symbol_info.get('pricePrecision', 2))
                print(f"{log_prefix} SL/TP adjusted for actual fill {actual_ep:.{price_prec_for_print}f} (from proposed {entry_p:.{price_prec_for_print}f}): SL {sl_p_adj:.{price_prec_for_print}f}, TP {tp_p_adj:.{price_prec_for_print}f}")
                final_sl_price, final_tp_price = sl_p_adj, tp_p_adj
            else:
                print(f"{log_prefix} Actual fill price {actual_ep} matches proposed entry price. Using original SL/TP.")


            # Record the trade in active_trades (preliminary, SL/TP order IDs to be updated)
            # current_pd_timestamp is defined once before creating new_trade_data
            new_trade_data_open_timestamp = pd.Timestamp.now(tz='UTC') 
            new_trade_data = {
                "entry_order_id": entry_order['orderId'], 
                "sl_order_id": None, "tp_order_id": None,
                "entry_price": actual_ep,
                "current_sl_price": final_sl_price, "current_tp_price": final_tp_price, 
                "initial_sl_price": final_sl_price, "initial_tp_price": final_tp_price, 
                "quantity": qty_to_order_final, "side": signal, 
                "symbol_info": symbol_info, "open_timestamp": new_trade_data_open_timestamp # Use the defined timestamp
            }
            try:
                active_trades[symbol] = new_trade_data
                print(f"{log_prefix} Trade for {symbol} (Entry ID: {entry_order['orderId']}) preliminarily recorded in active_trades. SL/TP placement follows. Open time: {new_trade_data_open_timestamp}")
            except Exception as e_rec:
                critical_error_msg = f"{log_prefix} CRITICAL ERROR: Order {entry_order['orderId']} for {symbol} placed successfully, BUT FAILED TO RECORD IN active_trades. Reason: {e_rec}. This trade may be orphaned and re-attempted. Manual intervention might be needed."
                print(critical_error_msg)
                traceback.print_exc()
                if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                    send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], f"🆘 {critical_error_msg}")
                # Do not proceed with SL/TP placement if recording failed.
                return 

        # --- SL/TP Placement (occurs outside the immediate active_trades update lock for atomicity of count, but after signature and preliminary recording) ---
        # The safety net (ensure_sl_tp_for_all_open_positions) is crucial to catch this.
        # And immediate SL/TP failure alert is also important.

        sl_ord = place_new_order(client, 
                                 symbol_info, 
                                 "SELL" if signal=="LONG" else "BUY", 
                                 "STOP_MARKET", 
                                 qty_to_order_final, 
                                 stop_price=final_sl_price, 
                                 position_side=position_side_for_trade,
                                 is_closing_order=True)
        if not sl_ord: print(f"{log_prefix} CRITICAL: FAILED TO PLACE SL! Details: {sl_ord}"); 
        else: 
            print(f"{log_prefix} SL order placed. Details: {sl_ord}")
            with lock: # Update active_trades with SL order ID
                if symbol in active_trades and active_trades[symbol]["entry_order_id"] == entry_order['orderId']:
                    active_trades[symbol]['sl_order_id'] = sl_ord.get('orderId')
        
        tp_ord = place_new_order(client, 
                                 symbol_info, 
                                 "SELL" if signal=="LONG" else "BUY", 
                                 "TAKE_PROFIT_MARKET", 
                                 qty_to_order_final, 
                                 stop_price=final_tp_price, 
                                 position_side=position_side_for_trade,
                                 is_closing_order=True)
        if not tp_ord: print(f"{log_prefix} Warning: Failed to place TP. Details: {tp_ord}")
        else: 
            print(f"{log_prefix} TP order placed. Details: {tp_ord}")
            with lock: # Update active_trades with TP order ID
                if symbol in active_trades and active_trades[symbol]["entry_order_id"] == entry_order['orderId']:
                     active_trades[symbol]['tp_order_id'] = tp_ord.get('orderId')

        # Check for SL/TP placement failures and send alert
        if not sl_ord or not tp_ord:
            failure_reason_list = []
            if not sl_ord: failure_reason_list.append("SL placement failed")
            if not tp_ord: failure_reason_list.append("TP placement failed")
            combined_reason_str = " and ".join(failure_reason_list)
            error_message_str = f"❌ {log_prefix} CRITICAL SL/TP PLACEMENT FAILURE for {symbol} immediately after entry: {combined_reason_str}. Entry ID: {entry_order['orderId']}. SL: {sl_ord}, TP: {tp_ord}"
            print(error_message_str)
            # Telegram alert logic (already exists)
            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                telegram_alert_msg = (
                    f"⚠️ IMMEDIATE SL/TP PLACEMENT FAILURE ⚠️\n\n"
                    f"Symbol: `{symbol}`\n"
                    f"Reason: _{combined_reason_str}_\n"
                    f"Entry Order ID: `{entry_order['orderId']}`\n"
                    f"Avg Entry Price: `{actual_ep}`\n"
                    f"Intended SL Price: `{final_sl_price}`\n" # Use final prices
                    f"Intended TP Price: `{final_tp_price}`\n\n" # Use final prices
                    f"Bot will still record this trade in `active_trades` but SL/TP might be missing or incorrect. "
                    f"The safety net should attempt to fix this in the next cycle. Monitor closely."
                )
                send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], telegram_alert_msg)

        with lock:
            if symbol in active_trades:
                existing_trade_data = active_trades[symbol]
                preserved_open_timestamp = existing_trade_data.get('open_timestamp')

                updated_trade_data = {
                    "entry_order_id": entry_order['orderId'],
                    "sl_order_id": sl_ord.get('orderId') if sl_ord else existing_trade_data.get('sl_order_id'),
                    "tp_order_id": tp_ord.get('orderId') if tp_ord else existing_trade_data.get('tp_order_id'),
                    "entry_price": actual_ep,
                    "current_sl_price": final_sl_price,
                    "current_tp_price": final_tp_price,
                    "initial_sl_price": final_sl_price,
                    "initial_tp_price": final_tp_price,
                    "quantity": qty_to_order_final,
                    "side": signal,
                    "symbol_info": symbol_info,
                    "open_timestamp": preserved_open_timestamp 
                }
                active_trades[symbol] = updated_trade_data
                print(f"{log_prefix} Trade for {symbol} finalized in active_trades. Open time: {preserved_open_timestamp}")
            else:
                # This path should ideally not be hit if preliminary recording was successful.
                # If it is, log a warning and fall back to current timestamp for safety, though it indicates an issue.
                current_pd_timestamp = pd.Timestamp.now(tz='UTC')
                print(f"{log_prefix} WARNING: Symbol {symbol} was not in active_trades for final update. Recording with current time {current_pd_timestamp}.")
                updated_trade_data = {
                    "entry_order_id": entry_order['orderId'],
                    "sl_order_id": sl_ord.get('orderId') if sl_ord else None,
                    "tp_order_id": tp_ord.get('orderId') if tp_ord else None,
                    "entry_price": actual_ep,
                    "current_sl_price": final_sl_price,
                    "current_tp_price": final_tp_price,
                    "initial_sl_price": final_sl_price,
                    "initial_tp_price": final_tp_price,
                    "quantity": qty_to_order_final,
                    "side": signal,
                    "symbol_info": symbol_info,
                    "open_timestamp": current_pd_timestamp 
                }
                active_trades[symbol] = updated_trade_data
                
        if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
            current_balance_for_msg = get_account_balance(client, configs)
            if current_balance_for_msg is None: current_balance_for_msg = "N/A (Error)"
            else: current_balance_for_msg = f"{current_balance_for_msg:.2f}"

            with active_trades_lock: # Ensure this is the correct lock variable 'lock'
                s_info_map_for_new_trade_msg = _build_symbol_info_map_from_active_trades(active_trades)
            open_positions_str = get_open_positions(client, format_for_telegram=True, active_trades_data=active_trades.copy(), symbol_info_map=s_info_map_for_new_trade_msg)

            qty_prec_msg = int(symbol_info.get('quantityPrecision', 0))
            price_prec_msg = int(symbol_info.get('pricePrecision', 2))

            new_trade_message = (
                f"🚀 NEW TRADE PLACED 🚀\n\n"
                f"Symbol: {symbol}\n"
                f"Side: {signal}\n"
                f"Quantity: {qty_to_order_final:.{qty_prec_msg}f}\n"
                f"Entry Price: {actual_ep:.{price_prec_msg}f}\n"
                f"SL: {final_sl_price:.{price_prec_msg}f}\n"
                f"TP: {final_tp_price:.{price_prec_msg}f}\n\n"
                f"💰 Account Balance: {current_balance_for_msg} USDT\n"
                f"📊 Current Open Positions:\n{open_positions_str}"
            )
            send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], new_trade_message)
            print(f"{log_prefix} New trade Telegram notification sent for {symbol}.")

        get_open_positions(client); get_open_orders(client, symbol) 
    
    finally:
        # This block ensures the symbol is removed from symbols_currently_processing
        # if this thread was the one that added it.
        with symbols_currently_processing_lock:
            symbols_currently_processing.discard(symbol)
            # print(f"{log_prefix} Released processing rights for symbol.") # Less verbose

def monitor_active_trades(client, configs): # Needs lock for active_trades access
    global active_trades, active_trades_lock
    if not active_trades: return
    print(f"\nMonitoring {len(active_trades)} active bot trades... {format_elapsed_time(configs.get('cycle_start_time_ref', time.time()))}")
    
    symbols_to_remove = []
    # Iterate over a copy of keys if modifying dict during iteration (though del happens later)
    # For reading, direct iteration is fine if modifications are locked.
    # Let's lock for the whole iteration to be safe with reads/potential dynamic adjustments.
    with active_trades_lock:
        trades_to_check = list(active_trades.items()) # Create a list to iterate over

    for symbol, trade_details in trades_to_check:
        # Re-acquire lock if we need to modify trade_details inside active_trades
        # Or, pass lock to check_and_adjust_sl_tp_dynamic if it modifies orders then active_trades
        # For now, let's assume check_and_adjust_sl_tp_dynamic returns new SL/TP and modification happens here under lock

        print(f"Checking {symbol} (Side: {trade_details['side']}, Entry: {trade_details['entry_price']:.4f})...")
        pos_exists, pos_qty = False, 0.0
        try:
            pos_info_list = client.futures_position_information(symbol=symbol)
            if pos_info_list and isinstance(pos_info_list, list):
                pos_data = next((p for p in pos_info_list if p['symbol'] == symbol), None)
                if pos_data: 
                    pos_qty = float(pos_data.get('positionAmt', 0.0))
                    if abs(pos_qty) > 1e-9 : pos_exists = True
        except Exception as e: print(f"Error getting position for {symbol} in monitor: {e}"); continue

        if not pos_exists:
            print(f"Position for {symbol} closed/zero. Attempting OCO cancellation and removing from active list.")
            
            # --- OCO Logic Implementation ---
            # If position is closed, try to cancel the corresponding SL and TP orders.
            # One of them might have triggered the closure, the other is now orphaned.
            sl_order_id_to_cancel = trade_details.get('sl_order_id')
            tp_order_id_to_cancel = trade_details.get('tp_order_id')

            if sl_order_id_to_cancel:
                try:
                    print(f"Attempting to cancel SL order {sl_order_id_to_cancel} for closed position {symbol} (OCO)...")
                    client.futures_cancel_order(symbol=symbol, orderId=sl_order_id_to_cancel)
                    print(f"Successfully cancelled SL order {sl_order_id_to_cancel} for {symbol} as part of OCO.")
                except BinanceAPIException as e:
                    if e.code == -2011: # Order filled or cancelled / Does not exist
                        print(f"SL order {sl_order_id_to_cancel} for {symbol} already filled or does not exist (OCO check). Code: {e.code}")
                    else:
                        print(f"API Error cancelling SL order {sl_order_id_to_cancel} for {symbol} (OCO): {e}")
                except Exception as e:
                    print(f"Unexpected error cancelling SL order {sl_order_id_to_cancel} for {symbol} (OCO): {e}")
            
            if tp_order_id_to_cancel:
                try:
                    print(f"Attempting to cancel TP order {tp_order_id_to_cancel} for closed position {symbol} (OCO)...")
                    client.futures_cancel_order(symbol=symbol, orderId=tp_order_id_to_cancel)
                    print(f"Successfully cancelled TP order {tp_order_id_to_cancel} for {symbol} as part of OCO.")
                except BinanceAPIException as e:
                    if e.code == -2011: # Order filled or cancelled / Does not exist
                        print(f"TP order {tp_order_id_to_cancel} for {symbol} already filled or does not exist (OCO check). Code: {e.code}")
                    else:
                        print(f"API Error cancelling TP order {tp_order_id_to_cancel} for {symbol} (OCO): {e}")
                except Exception as e:
                    print(f"Unexpected error cancelling TP order {tp_order_id_to_cancel} for {symbol} (OCO): {e}")
            
            # --- Realized P&L Calculation on Closure ---
            # This part is tricky as we need to know the exit price.
            # We assume if a position is gone, it was closed by SL, TP, or manually.
            # The OCO cancellation logic below attempts to cancel remaining SL/TP.
            # If one was filled, its cancellation will fail with error -2011.
            realized_pnl_for_trade = 0.0
            exit_price_assumed = None
            closure_reason = "Unknown (external or liquidation)"

            # Try to determine if SL or TP was hit by checking cancellation status
            sl_order_id_to_cancel = trade_details.get('sl_order_id')
            tp_order_id_to_cancel = trade_details.get('tp_order_id')
            sl_filled, tp_filled = False, False

            if sl_order_id_to_cancel:
                try:
                    client.futures_cancel_order(symbol=symbol, orderId=sl_order_id_to_cancel)
                    print(f"Successfully cancelled SL order {sl_order_id_to_cancel} for {symbol} as part of OCO (position closed).")
                except BinanceAPIException as e:
                    if e.code == -2011: # Order filled or cancelled / Does not exist
                        sl_filled = True # Assume it was filled
                        print(f"SL order {sl_order_id_to_cancel} for {symbol} likely FILLED (or already cancelled). Code: {e.code}")
                    else:
                        print(f"API Error cancelling SL order {sl_order_id_to_cancel} for {symbol} (OCO): {e}")
                except Exception as e:
                    print(f"Unexpected error cancelling SL order {sl_order_id_to_cancel} for {symbol} (OCO): {e}")
            
            if tp_order_id_to_cancel:
                try:
                    client.futures_cancel_order(symbol=symbol, orderId=tp_order_id_to_cancel)
                    print(f"Successfully cancelled TP order {tp_order_id_to_cancel} for {symbol} as part of OCO (position closed).")
                except BinanceAPIException as e:
                    if e.code == -2011: # Order filled or cancelled / Does not exist
                        tp_filled = True # Assume it was filled
                        print(f"TP order {tp_order_id_to_cancel} for {symbol} likely FILLED (or already cancelled). Code: {e.code}")
                    else:
                        print(f"API Error cancelling TP order {tp_order_id_to_cancel} for {symbol} (OCO): {e}")
                except Exception as e:
                    print(f"Unexpected error cancelling TP order {tp_order_id_to_cancel} for {symbol} (OCO): {e}")

            entry_p = trade_details['entry_price']
            qty = trade_details['quantity']
            trade_side = trade_details['side']

            if sl_filled:
                exit_price_assumed = trade_details['current_sl_price']
                closure_reason = "Stop-Loss Hit"
            elif tp_filled:
                exit_price_assumed = trade_details['current_tp_price']
                closure_reason = "Take-Profit Hit"
            # If neither SL nor TP seems to have filled but position is gone, it's harder to determine P&L accurately here.
            # For daily P&L tracking, this estimation is a compromise.
            
            if exit_price_assumed is not None:
                if trade_side == "LONG":
                    realized_pnl_for_trade = (exit_price_assumed - entry_p) * qty
                elif trade_side == "SHORT":
                    realized_pnl_for_trade = (entry_p - exit_price_assumed) * qty
                
                print(f"Position {symbol} closed. Reason: {closure_reason}. Entry: {entry_p}, Exit: {exit_price_assumed}, Qty: {qty}, PNL: {realized_pnl_for_trade:.2f}")
                
                with daily_state_lock:
                    global daily_realized_pnl
                    daily_realized_pnl += realized_pnl_for_trade
                    print(f"Updated daily realized PNL: {daily_realized_pnl:.2f}")
            else:
                print(f"Position {symbol} closed, but exact exit (SL/TP hit) could not be determined from order cancellations. PNL not added to daily total from this event.")

            symbols_to_remove.append(symbol)
            continue

        # Dynamic SL/TP adjustment logic
        try: cur_price = float(client.futures_ticker(symbol=symbol)['lastPrice'])
        except Exception as e: print(f"Error getting ticker for {symbol} in monitor: {e}"); continue
        
        new_sl, new_tp = check_and_adjust_sl_tp_dynamic(cur_price, trade_details['entry_price'], 
                                                        trade_details['initial_sl_price'], trade_details['initial_tp_price'],
                                                        trade_details['current_sl_price'], trade_details['current_tp_price'],
                                                        trade_details['side'])
        s_info = trade_details['symbol_info']
        qty = trade_details['quantity']
        original_trade_side = trade_details['side'] # This is "LONG" or "SHORT" for the main position
        # positionSide for SL/TP orders should match the main position's side.
        position_side_for_sl_tp = original_trade_side 
        updated_orders = False

        if new_sl is not None and abs(new_sl - trade_details['current_sl_price']) > 1e-9:
            print(f"Adjusting SL for {symbol} ({position_side_for_sl_tp}) to {new_sl:.4f}")
            if trade_details.get('sl_order_id'): # Cancel old
                try: client.futures_cancel_order(symbol=symbol, orderId=trade_details['sl_order_id'])
                except Exception as e: print(f"Warn: Old SL {trade_details['sl_order_id']} for {symbol} cancel fail: {e}")
            
            # Place new SL order
            sl_ord_new = place_new_order(client, 
                                         s_info, 
                                         "SELL" if original_trade_side == "LONG" else "BUY", # Order side is opposite to position
                                         "STOP_MARKET", 
                                         qty, 
                                         stop_price=new_sl, 
                                         position_side=position_side_for_sl_tp,
                                         is_closing_order=True)
            if sl_ord_new: 
                with active_trades_lock: 
                    if symbol in active_trades: 
                         active_trades[symbol]['current_sl_price'] = new_sl
                         active_trades[symbol]['sl_order_id'] = sl_ord_new.get('orderId')
                         updated_orders = True
            else: print(f"CRITICAL: FAILED TO PLACE NEW DYNAMIC SL FOR {symbol}!")
        
        if new_tp is not None and abs(new_tp - trade_details['current_tp_price']) > 1e-9:
            print(f"Adjusting TP for {symbol} ({position_side_for_sl_tp}) to {new_tp:.4f}")
            if trade_details.get('tp_order_id'): # Cancel old
                try: client.futures_cancel_order(symbol=symbol, orderId=trade_details['tp_order_id'])
                except Exception as e: print(f"Warn: Old TP {trade_details['tp_order_id']} for {symbol} cancel fail: {e}")

            # Place new TP order
            tp_ord_new = place_new_order(client, 
                                         s_info, 
                                         "SELL" if original_trade_side == "LONG" else "BUY", # Order side is opposite to position
                                         "TAKE_PROFIT_MARKET", 
                                         qty, 
                                         stop_price=new_tp, 
                                         position_side=position_side_for_sl_tp,
                                         is_closing_order=True)
            if tp_ord_new: 
                with active_trades_lock: 
                     if symbol in active_trades:
                        active_trades[symbol]['current_tp_price'] = new_tp
                        active_trades[symbol]['tp_order_id'] = tp_ord_new.get('orderId')
                        updated_orders = True
            else: print(f"Warning: Failed to place new dynamic TP for {symbol}.")
        
        if updated_orders: get_open_orders(client, symbol) # Show updated orders

    if symbols_to_remove:
        # First, gather all details for notifications outside the main active_trades_lock if possible,
        # to minimize lock duration, though PNL calculation might need some details from active_trades.
        # The current structure calculates PNL and determines reason inside the loop for each trade.
        # Let's refine it slightly: Store the calculated PNL and reason with the details to be removed.
        
        processed_closures_for_notification = []

        for sym_to_remove_candidate_info in symbols_to_remove: # This list now contains dicts with {symbol, pnl, reason}
            symbol_name = sym_to_remove_candidate_info["symbol"]
            realized_pnl = sym_to_remove_candidate_info["pnl"]
            determined_closure_reason = sym_to_remove_candidate_info["reason"]

            with active_trades_lock: # Lock for reading and then deleting
                if symbol_name in active_trades:
                    trade_details_for_notification = active_trades[symbol_name].copy() # Get a copy of details
                    trade_details_for_notification['realized_pnl'] = realized_pnl
                    trade_details_for_notification['closure_reason'] = determined_closure_reason
                    
                    processed_closures_for_notification.append(trade_details_for_notification)
                    # Actual deletion will happen after notifications or in a separate locked block
                else:
                    print(f"Warning: Symbol {symbol_name} intended for removal was not found in active_trades. Might have been removed by another process.")
        
        # Now send notifications and delete from active_trades
        for closed_trade_details in processed_closures_for_notification:
            sym_to_remove = closed_trade_details['symbol'] # Get symbol from the copied details

            # Send Telegram notification for closed trade
            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                        current_balance_for_msg = get_account_balance(client, configs)
                        if current_balance_for_msg is None: current_balance_for_msg = "N/A (Error)"
                        else: current_balance_for_msg = f"{current_balance_for_msg:.2f}"
                        
                        # Fetch open positions *after* this one is conceptually closed
                        # For an accurate list of *remaining* positions, it's tricky here as the removal happens after this loop.
                        # For simplicity, get_open_positions will reflect state *before* this specific sym_to_remove is visibly gone from API if check is fast.
                        # A more accurate "remaining" might require fetching positions, then filtering out sym_to_remove.
                        # Let's use current open positions, user will understand one less is about to be listed by bot internally.
                        # Build symbol_info_map from the current state of active_trades (before this one is removed)
                        s_info_map_for_closed_trade_msg = _build_symbol_info_map_from_active_trades(active_trades)
                        open_positions_str = get_open_positions(client, format_for_telegram=True, active_trades_data=active_trades.copy(), symbol_info_map=s_info_map_for_closed_trade_msg)
                        
                        # Determine precision from symbol_info stored in closed_trade_details
                        price_precision = closed_trade_details['symbol_info']['pricePrecision']
                        qty_precision = closed_trade_details['symbol_info']['quantityPrecision']

                        closed_trade_message = (
                            f"✅ TRADE CLOSED ✅\n\n"
                            f"Symbol: {closed_trade_details['symbol_info']['symbol']}\n"
                            f"Side: {closed_trade_details['side']}\n"
                            f"Quantity: {closed_trade_details['quantity']:.{qty_precision}f}\n"
                            f"Entry Price: {closed_trade_details['entry_price']:.{price_precision}f}\n"
                            f"Reason: {closed_trade_details.get('closure_reason', 'SL/TP hit or external closure detected')}\n" # Use determined reason
                            f"Realized PNL for trade: {closed_trade_details.get('realized_pnl', 'N/A'):.2f} USDT\n\n" # Add PNL
                            f"💰 Account Balance: {current_balance_for_msg} USDT\n"
                            f"📊 Current Open Positions:\n{open_positions_str}"
                        )
                        send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], closed_trade_message)
                        print(f"Trade closure Telegram notification sent for {sym_to_remove}.")
            
        # After notifications, remove the trades from the main active_trades dictionary
        if processed_closures_for_notification:
            symbols_to_delete_from_active = [item['symbol'] for item in processed_closures_for_notification]
            if symbols_to_delete_from_active:
                with active_trades_lock:
                    for sym_to_del in symbols_to_delete_from_active:
                        if sym_to_del in active_trades:
                            del active_trades[sym_to_del]
                            print(f"Removed {sym_to_del} from bot's active trades.")
                        # else: (already warned if not found during processing_closures_for_notification population)
                            

def trading_loop(client, configs, monitored_symbols):
    print("\n--- Starting Trading Loop ---")
    if not monitored_symbols: print("No symbols to monitor. Exiting."); return
    print(f"Monitoring {len(monitored_symbols)} symbols. Examples: {monitored_symbols[:5]}")

    # Declare globals used and modified within this loop
    global daily_high_equity, day_start_equity, last_trading_day
    global trading_halted_drawdown, trading_halted_daily_loss, daily_realized_pnl
    # daily_state_lock and active_trades_lock are used with 'with' or passed, not assigned directly in this scope.
    # active_trades is modified via helpers or within locks.

    current_cycle_number = 0
    executor = ThreadPoolExecutor(max_workers=configs.get('max_scan_threads', DEFAULT_MAX_SCAN_THREADS))
    
    try:
        while True:
            current_cycle_number += 1
            cycle_start_time = time.time()
            configs['cycle_start_time_ref'] = cycle_start_time # For threads to use consistent base
            iter_ts = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')
            print(f"\n--- Starting Scan Cycle #{current_cycle_number}: {iter_ts} {format_elapsed_time(cycle_start_time)} ---")

            # --- Manage Daily State and Get Current Equity ---
            # Note: current_balance is fetched inside manage_daily_state and used for equity calculation.
            # active_trades (global) and active_trades_lock (global) are passed directly.
            current_equity = manage_daily_state(client, configs, active_trades, active_trades_lock)

            if current_equity is None:
                print("CRITICAL: Failed to determine current equity. Skipping risk checks for this cycle. Bot may not operate safely.")
                # Decide on behavior: sleep and retry, or continue cautiously.
                # For now, let it proceed to monitoring, but new trades will be blocked by manage_trade_entry if balance is None there.
                # This path implies a severe issue like persistent API failure for balance/prices.
            else:
                # --- Max Drawdown Check (on current equity) ---
                # This check happens regardless of trading_halted_daily_loss, as drawdown is based on equity movement.
                max_dd_config = configs.get('max_drawdown_percent', 0.0)
                if max_dd_config > 0: # Only if drawdown limit is enabled
                    with daily_state_lock: # Protect access to daily_high_equity and trading_halted_drawdown
                        if not trading_halted_drawdown and daily_high_equity > 0: # Avoid division by zero
                            drawdown_pct = (daily_high_equity - current_equity) / daily_high_equity * 100
                            print(f"Max Drawdown Check: Current Equity: {current_equity:.2f}, Daily High Equity: {daily_high_equity:.2f}, Drawdown: {drawdown_pct:.2f}% (Limit: {max_dd_config:.2f}%)")
                            if drawdown_pct >= max_dd_config:
                                trading_halted_drawdown = True # Set halt flag under lock
                                # Unlock before potentially long operation like close_all_open_positions and Telegram msg
                                print(f"!!! MAX DRAWDOWN LIMIT HIT ({drawdown_pct:.2f}% >= {max_dd_config:.2f}%) !!!")
                                if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                                    send_telegram_message(
                                        configs["telegram_bot_token"], configs["telegram_chat_id"],
                                        f"🚨 MAX DRAWDOWN LIMIT HIT! 🚨\n"
                                        f"Drawdown: {drawdown_pct:.2f}% (Limit: {max_dd_config:.2f}%)\n"
                                        f"Daily High Equity: {daily_high_equity:.2f}\n"
                                        f"Current Equity: {current_equity:.2f}\n"
                                        f"Closing all positions and halting trading for the day."
                                    )
                                # Call close_all_open_positions (passes active_trades and active_trades_lock)
                                close_all_open_positions(client, configs, active_trades, active_trades_lock)
                        elif trading_halted_drawdown:
                             print("Trading remains halted due to Max Drawdown limit previously hit.")


                # --- Daily Stop Loss Check (on realized P&L) ---
                # This check also happens regardless of trading_halted_drawdown, as it's a separate limit.
                # However, if trading_halted_drawdown is true, new trades are already blocked.
                daily_sl_config = configs.get('daily_stop_loss_percent', 0.0)
                if daily_sl_config > 0: # Only if daily SL is enabled
                    with daily_state_lock: # Protect daily_realized_pnl, day_start_equity, trading_halted_daily_loss
                        if not trading_halted_daily_loss and day_start_equity > 0: # Avoid division by zero
                            # daily_realized_pnl is negative for a loss
                            current_loss_pct = (daily_realized_pnl / day_start_equity) * 100 
                            print(f"Daily Stop Loss Check: Realized PNL: {daily_realized_pnl:.2f}, Start Equity: {day_start_equity:.2f}, Loss Pct: {current_loss_pct:.2f}% (Limit: -{daily_sl_config:.2f}%)")
                            if current_loss_pct <= -daily_sl_config:
                                trading_halted_daily_loss = True # Set halt flag under lock
                                print(f"!!! DAILY STOP LOSS LIMIT HIT ({current_loss_pct:.2f}% <= -{daily_sl_config:.2f}%) !!!")
                                if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                                     send_telegram_message(
                                        configs["telegram_bot_token"], configs["telegram_chat_id"],
                                        f"📛 DAILY STOP LOSS LIMIT HIT! 📛\n"
                                        f"Realized Daily PNL: {daily_realized_pnl:.2f} ({current_loss_pct:.2f}% of start equity)\n"
                                        f"Limit: -{daily_sl_config:.2f}%\n"
                                        f"Halting new trades for the day. Existing positions will be managed."
                                    )
                        elif trading_halted_daily_loss:
                            print("New trades remain halted due to Daily Stop Loss limit previously hit.")
            
            # --- Main Trading Operations ---
            try:
                # Fetch current balance for this cycle if not done or if needed separately
                # current_cycle_balance = get_account_balance(client, configs) # This is already done in manage_daily_state
                # For clarity, let's assume current_equity from manage_daily_state is what we need for display/checks here.
                # If manage_daily_state returned None for current_equity, some operations might be skipped.
                
                # Display general status
                if current_equity is not None: # Only if equity was determined
                    get_open_positions(client) # Shows current positions from API
                    # The print below was for current_cycle_balance, let's use current_equity
                    print(f"Account status: Current Equity: {current_equity:.2f} USDT {format_elapsed_time(cycle_start_time)}")
                else:
                    print(f"Account status: Equity could not be determined this cycle. {format_elapsed_time(cycle_start_time)}")


                # --- Check for Halts or Max Concurrent Positions BEFORE attempting new trades ---
                halt_dd_flag, halt_dsl_flag = False, False
                with daily_state_lock: # Read halt flags safely
                    halt_dd_flag = trading_halted_drawdown
                    halt_dsl_flag = trading_halted_daily_loss
                
                num_active_trades = 0
                with active_trades_lock:
                    num_active_trades = len(active_trades)

                max_pos_limit = configs["max_concurrent_positions"]

                proceed_with_new_trades = True
                if halt_dd_flag:
                    print(f"Trading HALTED for the day (Max Drawdown). No new trades will be initiated. {format_elapsed_time(cycle_start_time)}")
                    proceed_with_new_trades = False
                elif halt_dsl_flag:
                    print(f"New trades HALTED for the day (Daily Stop Loss). No new trades will be initiated. {format_elapsed_time(cycle_start_time)}")
                    proceed_with_new_trades = False
                
                if proceed_with_new_trades and num_active_trades >= max_pos_limit:
                    print(f"Max concurrent positions ({max_pos_limit}) reached. Pausing new trade scanning. {format_elapsed_time(cycle_start_time)}")
                    proceed_with_new_trades = False

                # --- Symbol Processing for New Trades (if not halted/paused) ---
                if proceed_with_new_trades:
                    print(f"Space available for new trades ({num_active_trades}/{max_pos_limit}). Proceeding with symbol scan. {format_elapsed_time(cycle_start_time)}")
                    futures = []
                    print(f"Submitting {len(monitored_symbols)} symbol tasks to {configs.get('max_scan_threads')} threads... {format_elapsed_time(cycle_start_time)}")
                    for symbol in monitored_symbols:
                        # manage_trade_entry will internally check halt flags again before processing
                        futures.append(executor.submit(process_symbol_task, symbol, client, configs, active_trades_lock))
                    
                    processed_count = 0
                    for future in as_completed(futures):
                        try: result = future.result(); # print(f"Task result: {result}") # Optional: log task result
                        except Exception as e_future: print(f"Task error: {e_future}")
                        processed_count += 1
                        if processed_count % (len(monitored_symbols)//5 or 1) == 0 or processed_count == len(monitored_symbols): # Log progress periodically
                             print(f"Symbol tasks progress: {processed_count}/{len(monitored_symbols)} completed. {format_elapsed_time(cycle_start_time)}")
                    print(f"All symbol tasks completed for new trade scanning. {format_elapsed_time(cycle_start_time)}")
                
                # --- Operations for when new trades are paused/halted but monitoring continues ---
                else: # Not proceeding with new trades (halted, or max positions)
                    if halt_dd_flag: # Max drawdown halt implies all positions should have been closed
                        print(f"Max Drawdown Halt is active. All positions should be closed. Waiting for next trading day. {format_elapsed_time(cycle_start_time)}")
                        # No monitoring needed if all positions are intended to be closed.
                        # The ensure_sl_tp_for_all_open_positions might run if there were closure errors, which is fine.
                    elif halt_dsl_flag:
                        print(f"Daily Stop Loss Halt for new trades is active. Monitoring existing positions. {format_elapsed_time(cycle_start_time)}")
                    elif num_active_trades >= max_pos_limit:
                         print(f"Max concurrent positions limit reached. Monitoring existing positions. {format_elapsed_time(cycle_start_time)}")
                    # If none of the above, it implies proceed_with_new_trades was true, so this 'else' shouldn't be hit without a reason.
                    # This block primarily serves to articulate actions when new trades are *not* sought.

                # --- Monitor Existing Trades (Always run, unless specific conditions prevent it) ---
                # If max drawdown occurred and positions were closed, active_trades should be empty.
                # If daily SL hit, existing trades are still managed.
                if not halt_dd_flag: # If not in max drawdown hard stop (where positions are closed)
                    monitor_active_trades(client, configs) # Monitor after scan or if scan was skipped
                    print(f"Active trades monitoring complete. {format_elapsed_time(cycle_start_time)}")
                else:
                    print(f"Skipping active trade monitoring as Max Drawdown halt is active and positions should be closed. {format_elapsed_time(cycle_start_time)}")


                # --- SL/TP Safety Net Check for All Open Positions (Always run) ---
                # This is important even if halted, to manage any stragglers or manually opened positions,
                # or if close_all_open_positions had issues.
                symbol_info_cache_for_safety_net = {} 
                print(f"Running SL/TP safety net check for all open positions... {format_elapsed_time(cycle_start_time)}")
                ensure_sl_tp_for_all_open_positions(client, configs, active_trades, symbol_info_cache_for_safety_net)
                print(f"SL/TP safety net check complete. {format_elapsed_time(cycle_start_time)}")
                # --- End SL/TP Safety Net Check ---
                
                # --- Loop Delay ---
                # Determine appropriate sleep time
                # If any halt is active, or max positions, might use a shorter delay to check for new day or slot availability sooner.
                current_loop_delay_seconds = configs['loop_delay_minutes'] * 25 # Default delay
                if halt_dd_flag or halt_dsl_flag or (num_active_trades >= max_pos_limit and proceed_with_new_trades == False) :
                    # Using a shorter delay if halted or at max positions, to check for day reset or position closure sooner
                    current_loop_delay_seconds = 25 
                    print(f"Using shorter loop delay ({current_loop_delay_seconds}s) due to trading halt or max positions.")
                
                print(f"Waiting for {current_loop_delay_seconds / 25:.1f} m...")
                time.sleep(current_loop_delay_seconds)

            except Exception as loop_err: # Catch errors within the try block of the loop
                print(f"ERROR in trading loop cycle: {loop_err} {format_elapsed_time(cycle_start_time)}")
                traceback.print_exc()

            cycle_dur_s = time.time() - cycle_start_time
            print(f"\n--- Scan Cycle #{current_cycle_number} Completed (Runtime: {cycle_dur_s:.2f}s / {(cycle_dur_s/60):.2f}min). ---")

            # Send status update every 100 cycles (This part remains largely the same, just ensure it's outside the 'else' of the pause logic)
            if current_cycle_number > 0 and current_cycle_number % 100 == 0:
                print(f"Scan cycle {current_cycle_number} reached. Sending status update to Telegram...")
                if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                    current_balance_for_update = get_account_balance(client, configs)
                    if current_balance_for_update is None: 
                        current_balance_for_update = "Error fetching" 

                    open_pos_text_update = "None"
                    if client: 
                        if configs["mode"] == "live": 
                            with active_trades_lock: 
                                s_info_map_for_status_update = _build_symbol_info_map_from_active_trades(active_trades)
                                current_active_trades_copy = active_trades.copy()
                            open_pos_text_update = get_open_positions(client, format_for_telegram=True, active_trades_data=current_active_trades_copy, symbol_info_map=s_info_map_for_status_update)
                        else:
                            open_pos_text_update = "None (Backtest Mode)" 
                    else:
                        open_pos_text_update = "N/A (Client not initialized)" 

                    retrieved_bot_start_time_str = configs.get('bot_start_time_str', 'N/A')
                    
                    status_update_msg = build_startup_message(configs, current_balance_for_update, open_pos_text_update, retrieved_bot_start_time_str)
                    send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], status_update_msg)
                    configs["last_startup_message"] = status_update_msg 
                    print("Telegram status update sent.")
                else:
                    print("Telegram not configured, skipping status update message.")
            
            # This `time.sleep` was part of the original `else` block for normal operation.
            # If the loop was paused, it `continue`d before reaching here.
            # If it was a normal cycle, this sleep is now handled within the `else` block above.
            # So, this specific `print` and `time.sleep` might be redundant or needs to be placed carefully
            # if there's a path that reaches here without either sleeping for pause or sleeping for normal delay.
            # Let's assume the `time.sleep` is correctly placed within the `else` for normal operation,
            # and the `continue` handles the pause. The `print(f"Waiting for {configs['loop_delay_minutes']} m...")`
            # should also be within that `else`.

            # Corrected structure: The loop_delay_minutes sleep should only happen if NOT paused.
            # The current diff places it correctly inside the `else` block for normal operation.
            # The status update logic can remain common after the main try/except of the cycle.
    finally:
        print("Shutting down thread pool executor...")
        executor.shutdown(wait=True) # Ensure all threads finish before exiting loop/program
        print("Thread pool executor shut down.")


# --- Main Execution ---
def main():
    print("Initializing Binance Trading Bot - Advance EMA Cross Strategy (ID: 8)")
    bot_start_time_utc = pd.Timestamp.now(tz='UTC')
    bot_start_time_str = bot_start_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    configs = get_user_configurations()
    print("\nLoaded Configurations:")
    for k, v in configs.items(): 
        if k not in ["api_key", "api_secret", "telegram_bot_token", "telegram_chat_id"]: # Hide sensitive keys
             print(f"  {k.replace('_',' ').title()}: {v}")

    # Call initialize_binance_client once and unpack its results
    # The first item in the returned tuple is the actual client object.
    client_connection_details = initialize_binance_client(configs)
    client_obj = client_connection_details[0]
    env_for_msg = client_connection_details[1]
    server_time_obj = client_connection_details[2]
    
    if not client_obj: # Check if the actual client object is None
        # Error messages are printed by initialize_binance_client already
        print("Exiting: Binance client init failed.") 
        sys.exit(1)

    # Now use client_obj for operations
    print("\nFetching initial account balance...")
    initial_balance = get_account_balance(client_obj, configs) # Use the actual client object, pass configs

    if configs['mode'] == 'live' and initial_balance is None:
        # IP-related error or other critical issue detected by get_account_balance returning None.
        # Telegram alert with IP details is sent from within get_account_balance for -2015.
        print("CRITICAL: Initial API connection failed (e.g., IP whitelist issue or invalid API key).")
        while initial_balance is None:
            print("Retrying initial connection in 60 seconds... Check Telegram for IP details if this is a whitelist issue.")
            time.sleep(60)
            initial_balance = get_account_balance(client_obj, configs)
            if initial_balance is not None:
                print(f"Initial API connection successful! Current balance: {initial_balance:.2f} USDT")
                # Re-send startup message now that connection is established
                if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                    open_pos_text_retry = get_open_positions(client_obj, format_for_telegram=True)
                    # configs['bot_start_time_str'] should already be set
                    startup_msg_retry = build_startup_message(configs, initial_balance, open_pos_text_retry, configs.get('bot_start_time_str', 'N/A'))
                    send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], startup_msg_retry)
                    configs["last_startup_message"] = startup_msg_retry 
                    print("Startup message resent to Telegram after successful reconnection.")
            else:
                # This else block is within the while, meaning get_account_balance still returned None
                print("Retry failed. Will try again after 60 seconds.")
                # The loop will continue, and get_account_balance will send another Telegram if it's -2015.
    
    elif initial_balance == 0.0 and configs.get("environment") == "mainnet": # Use elif to avoid this message if initial_balance is None initially
        print("Warning: Initial account balance is 0.0 USDT on Mainnet. Ensure funds are available for trading.")
        # If it was critical, the 'initial_balance is None' check above would have caught it for live mode.
        # If it's 0.0 not due to a critical error, it's just a warning.
        
    # Print the success message using the unpacked env and server_time
    if server_time_obj and isinstance(server_time_obj, dict) and 'serverTime' in server_time_obj and env_for_msg:
        print(f"\nSuccessfully connected to Binance {env_for_msg.title()} API. Server Time: {pd.to_datetime(server_time_obj['serverTime'], unit='ms')} UTC")
    elif env_for_msg: # If client initialized but server time fetch failed or env_for_msg is None (should be rare)
        print(f"\nSuccessfully connected to Binance {env_for_msg.title()} API. (Server time not fully available)")
    else: # Fallback if client_obj is somehow valid but other details are not
        print(f"\nSuccessfully connected to Binance API. (Connection details partially unavailable)")

    configs.setdefault("api_delay_short", 1) 
    configs.setdefault("api_delay_symbol_processing", 0.1) # Can be very short with threads
    configs.setdefault("loop_delay_minutes", 5)

    monitored_symbols = get_all_usdt_perpetual_symbols(client_obj) # Use client_obj
    if not monitored_symbols: print("Exiting: No symbols to monitor."); sys.exit(1)
    
    # --- Initial Telegram Notification ---
    # --- Telegram Startup Notification ---
    if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
        open_pos_text = "None"
        # active_trades is initially empty, so s_info_map will be empty.
        # This is fine as there are no bot-managed SL/TP to show at this very start.
        # If there were pre-existing positions from API, they'd show "N/A (Bot)" for SL/TP.
        s_info_map_initial = _build_symbol_info_map_from_active_trades(active_trades) # active_trades is global
        
        if client_obj:
            if configs["mode"] == "live":
                open_pos_text = get_open_positions(client_obj, format_for_telegram=True, active_trades_data=active_trades.copy(), symbol_info_map=s_info_map_initial)
            else:
                open_pos_text = "None (Backtest Mode)"
        else:
            open_pos_text = "N/A (Client not initialized)"

        startup_msg = build_startup_message(configs, initial_balance, open_pos_text, bot_start_time_str)
        send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], startup_msg)
        configs["last_startup_message"] = startup_msg

        # Start /command3 listener
        start_command_listener(configs["telegram_bot_token"], configs["telegram_chat_id"], startup_msg)

    else:
        print("Telegram notifications disabled (token or chat_id not configured in keys.py or load failed).")
    configs["monitored_symbols_count"] = len(monitored_symbols)
    configs['bot_start_time_str'] = bot_start_time_str # Make bot_start_time_str available in configs

    print(f"Found {len(monitored_symbols)} USDT perpetuals. Proceeding to monitor all for {'live trading' if configs['mode'] == 'live' else 'backtesting'}.")
    # confirm = input(f"Found {len(monitored_symbols)} USDT perpetuals. Monitor all for {'live trading' if configs['mode'] == 'live' else 'backtesting'}? (yes/no) [yes]: ").lower().strip()
    # if confirm == 'no': print("Exiting by user choice."); sys.exit(0)

    if configs["mode"] == "live":
        try:
            trading_loop(client_obj, configs, monitored_symbols) # Use client_obj
        except KeyboardInterrupt: 
            print("\nBot stopped by user (Ctrl+C).")
            # Send a notification for graceful shutdown by user
            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], "ℹ️ Bot stopped by user (Ctrl+C).")
        except Exception as e: 
            print(f"\nCRITICAL UNEXPECTED ERROR IN LIVE TRADING: {e}")
            traceback.print_exc()
            # Send a notification for critical error
            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                error_message_for_telegram = f"🆘 CRITICAL BOT ERROR 🆘\nBot encountered an unhandled exception and may have stopped.\nError: {str(e)[:1000]}\nCheck logs immediately!"
                send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], error_message_for_telegram)
        finally:
            print("\n--- Live Trading Bot Shutting Down ---")

            # Cancel orders if any
            if client_obj and active_trades:
                print(f"Cancelling {len(active_trades)} bot-managed active SL/TP orders...")
                with active_trades_lock:
                    for symbol, trade_details in list(active_trades.items()):
                        for oid_key in ['sl_order_id', 'tp_order_id']:
                            oid = trade_details.get(oid_key)
                            if oid:
                                try:
                                    print(f"Cancelling {oid_key} {oid} for {symbol}...")
                                    client_obj.futures_cancel_order(symbol=symbol, orderId=oid)
                                except Exception as e_c:
                                    print(f"Failed to cancel {oid_key} {oid} for {symbol}: {e_c}")

            print("Live Bot shutdown sequence complete.")

            # ✅ Send Telegram Shutdown Message
            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                shutdown_time_str = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')
                shutdown_msg = (
                    f"*⚠️ Bot Stopped*\n\n"
                    f"*Stop Time:* `{shutdown_time_str}`\n"
                    f"*Strategy:* `{configs.get('strategy_name', 'Unknown')}`\n"
                    f"*Environment:* `{configs.get('environment', 'Unknown')}`"
                )
                send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], shutdown_msg)

            if client_obj and active_trades: # Use client_obj
                print(f"Cancelling {len(active_trades)} bot-managed active SL/TP orders...")
                with active_trades_lock:
                    for symbol, trade_details in list(active_trades.items()):
                        for oid_key in ['sl_order_id', 'tp_order_id']:
                            oid = trade_details.get(oid_key)
                            if oid:
                                try:
                                    print(f"Cancelling {oid_key} {oid} for {symbol}...")
                                    client_obj.futures_cancel_order(symbol=symbol, orderId=oid) # Use client_obj
                                except Exception as e_c: print(f"Failed to cancel {oid_key} {oid} for {symbol}: {e_c}")
            print("Live Bot shutdown sequence complete.")
    elif configs["mode"] == "backtest":
        try:
            # Pass client_obj to backtesting_loop; it might also be used for initializing backtest env
            backtesting_loop(client_obj, configs, monitored_symbols) # Use client_obj
        except KeyboardInterrupt: print("\nBacktest stopped by user (Ctrl+C).")
        except Exception as e: print(f"\nCRITICAL UNEXPECTED ERROR IN BACKTESTING: {e}"); traceback.print_exc()
        finally:
            print("\n--- Backtesting Complete ---")
            # No orders to cancel in backtest mode unless simulating exchange interactions
            # For now, active_trades will just be cleared or analyzed.
        
            active_trades.clear() # Clear trades for a clean slate if re-run or for reporting
            print("Backtest shutdown sequence complete.")


# --- SL/TP Safety Net Function ---
def ensure_sl_tp_for_all_open_positions(client, configs, active_trades_ref, symbol_info_cache):
    """
    Checks all open positions on Binance.
    For positions managed by the bot (in active_trades_ref), it verifies SL/TP orders exist.
    If not, it attempts to re-place them based on stored values.
    For positions NOT managed by the bot, it attempts to calculate and set new SL/TP orders
    based on current strategy logic.
    Sends Telegram alerts for failures or significant actions.
    """
    log_prefix = "[SL/TP SafetyNet]"
    print(f"\n{log_prefix} Starting check for all open positions to ensure SL/TP...")

    try:
        all_positions = client.futures_position_information()
        open_positions = [p for p in all_positions if float(p.get('positionAmt', 0)) != 0]

        if not open_positions:
            print(f"{log_prefix} No open positions found.")
            return

        print(f"{log_prefix} Found {len(open_positions)} open position(s). Processing...")
        current_balance_for_check = get_account_balance(client, configs) # For sanity checks

        for pos in open_positions:
            symbol = pos['symbol']
            entry_price = float(pos['entryPrice'])
            position_qty_raw = float(pos['positionAmt']) # Can be positive or negative
            leverage = int(pos.get('leverage', configs.get('leverage'))) # Get actual leverage from position

            print(f"\n{log_prefix} Checking position: {symbol}, Entry: {entry_price}, Qty: {position_qty_raw}, Leverage: {leverage}x")

            # Determine side and absolute quantity
            side = "LONG" if position_qty_raw > 0 else "SHORT"
            abs_position_qty = abs(position_qty_raw)

            with active_trades_lock: # Ensure thread-safe access to active_trades_ref
                is_managed_trade = symbol in active_trades_ref

            if is_managed_trade:
                with active_trades_lock: # Re-acquire lock if reading mutable details
                    trade_details = active_trades_ref.get(symbol) # Use .get for safety, though 'in' check was done
                
                if not trade_details: # Should not happen if is_managed_trade is true and lock is proper
                    print(f"{log_prefix} CRITICAL: {symbol} was in active_trades_ref but now missing. Concurrency issue?")
                    continue

                print(f"{log_prefix} {symbol} is MANAGED by the bot. Verifying SL/TP orders.")
                
                s_info_managed = trade_details['symbol_info'] # Already available
                target_sl_price = trade_details['current_sl_price']
                target_tp_price = trade_details['current_tp_price']
                expected_qty_for_sl_tp = trade_details['quantity'] # Bot's tracked quantity

                # It's possible the position quantity on exchange differs slightly from bot's tracked quantity
                # due to partial fills on SL/TP, or manual intervention.
                # For re-placing SL/TP, use the *actual current position quantity* on the exchange.
                qty_for_new_sl_tp_orders = abs_position_qty


                open_orders_for_symbol = client.futures_get_open_orders(symbol=symbol)
                
                sl_order_active = False
                if trade_details.get('sl_order_id'):
                    for o in open_orders_for_symbol:
                        if o['orderId'] == trade_details['sl_order_id'] and \
                           o['type'] == 'STOP_MARKET' and \
                           abs(float(o['origQty']) - expected_qty_for_sl_tp) < 1e-9 : # Check if qty matches expected
                            sl_order_active = True
                            print(f"{log_prefix} SL order {trade_details['sl_order_id']} for {symbol} is ACTIVE.")
                            break
                
                tp_order_active = False
                if trade_details.get('tp_order_id'):
                    for o in open_orders_for_symbol:
                        if o['orderId'] == trade_details['tp_order_id'] and \
                           o['type'] == 'TAKE_PROFIT_MARKET' and \
                           abs(float(o['origQty']) - expected_qty_for_sl_tp) < 1e-9:
                            tp_order_active = True
                            print(f"{log_prefix} TP order {trade_details['tp_order_id']} for {symbol} is ACTIVE.")
                            break

                if not sl_order_active:
                    print(f"{log_prefix} SL order for managed trade {symbol} ({side}) is MISSING or incorrect. Attempting to re-place.")
                    sl_order_obj, sl_error_msg = place_new_order(client, 
                                                                 s_info_managed, 
                                                                 "SELL" if side == "LONG" else "BUY", 
                                                                 "STOP_MARKET", 
                                                                 qty_for_new_sl_tp_orders, 
                                                                 stop_price=target_sl_price, 
                                                                 position_side=side, # side here is the position's side (LONG/SHORT)
                                                                 is_closing_order=True)
                    if sl_order_obj and sl_order_obj.get('orderId'):
                        print(f"{log_prefix} Successfully re-placed SL order for {symbol}. New ID: {sl_order_obj['orderId']}")
                        with active_trades_lock: 
                             if symbol in active_trades_ref: 
                                active_trades_ref[symbol]['sl_order_id'] = sl_order_obj['orderId']
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"),
                                              f"✅ {log_prefix} Re-placed MISSING SL for managed {symbol} ({side}) @ {target_sl_price:.{s_info_managed['pricePrecision']}f}")
                    else:
                        err_detail = f"API Error: {sl_error_msg}" if sl_error_msg else "Order object missing or no orderId."
                        err_msg_sl = f"⚠️ {log_prefix} FAILED to re-place SL for managed {symbol} ({side}). Target SL: {target_sl_price:.{s_info_managed['pricePrecision']}f}. Details: {err_detail}"
                        print(err_msg_sl)
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), err_msg_sl)

                if not tp_order_active:
                    print(f"{log_prefix} TP order for managed trade {symbol} ({side}) is MISSING or incorrect. Attempting to re-place.")
                    tp_order_obj, tp_error_msg = place_new_order(client, 
                                                                 s_info_managed,
                                                                 "SELL" if side == "LONG" else "BUY",
                                                                 "TAKE_PROFIT_MARKET", 
                                                                 qty_for_new_sl_tp_orders,
                                                                 stop_price=target_tp_price, 
                                                                 position_side=side, # side here is the position's side (LONG/SHORT)
                                                                 is_closing_order=True)
                    if tp_order_obj and tp_order_obj.get('orderId'):
                        print(f"{log_prefix} Successfully re-placed TP order for {symbol}. New ID: {tp_order_obj['orderId']}")
                        with active_trades_lock: 
                            if symbol in active_trades_ref:
                                active_trades_ref[symbol]['tp_order_id'] = tp_order_obj['orderId']
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"),
                                              f"✅ {log_prefix} Re-placed MISSING TP for managed {symbol} ({side}) @ {target_tp_price:.{s_info_managed['pricePrecision']}f}")
                    else:
                        err_detail = f"API Error: {tp_error_msg}" if tp_error_msg else "Order object missing or no orderId."
                        err_msg_tp = f"⚠️ {log_prefix} FAILED to re-place TP for managed {symbol} ({side}). Target TP: {target_tp_price:.{s_info_managed['pricePrecision']}f}. Details: {err_detail}"
                        print(err_msg_tp)
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), err_msg_tp)
            else: # UNMANAGED TRADE
                print(f"{log_prefix} {symbol} is UNMANAGED by the bot. Attempting to calculate and set SL/TP ({side} position).")
                
                # 1. Fetch Symbol Info (use cache)
                s_info_unmanaged = symbol_info_cache.get(symbol)
                if not s_info_unmanaged:
                    s_info_unmanaged = get_symbol_info(client, symbol)
                    if s_info_unmanaged:
                        symbol_info_cache[symbol] = s_info_unmanaged
                    else:
                        msg = f"⚠️ {log_prefix} Cannot get symbol_info for UNMANAGED {symbol}. Cannot set SL/TP."
                        print(msg)
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), msg)
                        continue # Skip to next position
                
                p_prec_unmanaged = int(s_info_unmanaged['pricePrecision'])
                # q_prec_unmanaged = int(s_info_unmanaged['quantityPrecision']) # Not strictly needed for placing SL/TP with existing qty

                # 2. Fetch Klines
                # get_historical_klines returns a tuple (df, error_object)
                klines_df_unmanaged, klines_error = get_historical_klines(client, symbol, limit=250) # Sufficient for EMA100/200
                
                if klines_error:
                    error_message = f"Error fetching klines for UNMANAGED {symbol}: {klines_error}"
                    msg = f"⚠️ {log_prefix} {error_message}. Cannot calculate SL/TP."
                    print(msg)
                    send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), msg)
                    continue

                if klines_df_unmanaged.empty or len(klines_df_unmanaged) < 202: # Need enough for EMA200 + some history
                    msg = f"⚠️ {log_prefix} Insufficient kline data for UNMANAGED {symbol} (got {len(klines_df_unmanaged)}). Cannot calculate SL/TP."
                    print(msg)
                    send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), msg)
                    continue

                # 3. Calculate EMAs
                klines_df_unmanaged['EMA100'] = calculate_ema(klines_df_unmanaged, 100)
                klines_df_unmanaged['EMA200'] = calculate_ema(klines_df_unmanaged, 200) # Though not directly used by calc_sl_tp_values, good practice
                
                if klines_df_unmanaged['EMA100'].isnull().all():
                    msg = f"⚠️ {log_prefix} Failed to calculate EMA100 for UNMANAGED {symbol}. Cannot determine SL/TP."
                    print(msg)
                    send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), msg)
                    continue
                
                ema100_val_unmanaged = klines_df_unmanaged['EMA100'].iloc[-1]

                # 4. Calculate SL/TP
                # Calculate ATR for unmanaged trade
                klines_df_unmanaged['atr'] = calculate_atr(klines_df_unmanaged, period=configs.get('atr_period', DEFAULT_ATR_PERIOD))
                raw_atr_value = klines_df_unmanaged['atr'].iloc[-1]
                
                current_atr_unmanaged = None
                try:
                    current_atr_unmanaged = float(raw_atr_value)
                except (ValueError, TypeError) as e:
                    msg = (f"⚠️ {log_prefix} ATR value '{raw_atr_value}' for UNMANAGED {symbol} is not a valid number. Error: {e}. "
                           f"Cannot calculate SL/TP.")
                    print(msg)
                    send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), msg)
                    if 'atr' in klines_df_unmanaged.columns: klines_df_unmanaged.drop(columns=['atr'], inplace=True)
                    continue # Skip to next position

                if pd.isna(current_atr_unmanaged) or current_atr_unmanaged <= 0:
                    msg = (f"⚠️ {log_prefix} Invalid or non-positive ATR value ({current_atr_unmanaged}) for UNMANAGED {symbol}. "
                           f"Cannot calculate SL/TP.")
                    print(msg)
                    send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), msg)
                    if 'atr' in klines_df_unmanaged.columns: klines_df_unmanaged.drop(columns=['atr'], inplace=True)
                    continue

                print(f"{log_prefix} ATR for UNMANAGED {symbol}: {current_atr_unmanaged:.{p_prec_unmanaged}f}")

                # Correctly call calculate_sl_tp_values with ATR, configs, and symbol_info
                calc_sl_price, calc_tp_price = calculate_sl_tp_values(
                    entry_price, 
                    side, 
                    current_atr_unmanaged,  # Pass calculated ATR value
                    configs,                # Pass the main configs dictionary
                    s_info_unmanaged        # Pass the symbol_info for this unmanaged symbol
                )
                if 'atr' in klines_df_unmanaged.columns: klines_df_unmanaged.drop(columns=['atr'], inplace=True) # Clean up ATR column

                if calc_sl_price is None or calc_tp_price is None:
                    msg = (f"⚠️ {log_prefix} Failed to calculate SL/TP for UNMANAGED {symbol} using ATR. "
                           f"Entry: {entry_price:.{p_prec_unmanaged}f}, Side: {side}, ATR: {current_atr_unmanaged:.{p_prec_unmanaged}f}.")
                    print(msg)
                    send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), msg)
                    continue
                
                print(f"{log_prefix} For UNMANAGED {symbol}, calculated SL: {calc_sl_price:.{p_prec_unmanaged}f}, TP: {calc_tp_price:.{p_prec_unmanaged}f}")

                # 5. Perform Sanity Checks on calculated SL/TP
                # Use current_balance_for_check fetched at the start of the function.
                # Risk percent for unmanaged trades: use default or a specific "safety_net_risk_percent" from configs if available.
                # For now, use the main risk_percent. This might be too aggressive for unmanaged.
                # Consider adding a specific, potentially smaller, risk % for unmanaged trades in future.
                # The sanity check's risk validation part might be less relevant here if we're just applying SL/TP
                # to an existing position whose size is fixed. The key is valid SL/TP prices.
                
                # For unmanaged trades, the quantity is fixed by the existing position.
                # The risk % check in pre_order_sanity_checks might not be directly applicable in the same way.
                # We are not calculating quantity based on risk, but applying SL/TP to an existing quantity.
                # However, other checks (price validity, SL/TP sides) are still important.
                # Let's call it, but be mindful of its risk interpretation for this context.
                # We might need a modified sanity check or interpret its results carefully.

                # leverage variable is already calculated as int:
                # leverage = int(pos.get('leverage', configs.get('leverage')))
                sanity_passed, sanity_reason = pre_order_sanity_checks(
                    symbol=symbol, 
                    signal=side, 
                    entry_price=entry_price, 
                    sl_price=calc_sl_price, 
                    tp_price=calc_tp_price, 
                    quantity=abs_position_qty,
                    symbol_info=s_info_unmanaged, 
                    current_balance=(current_balance_for_check if current_balance_for_check is not None else 10000),
                    risk_percent_config=configs.get('risk_percent'), 
                    configs=configs, 
                    specific_leverage_for_trade=leverage, # Pass the integer leverage here
                    klines_df_for_debug=klines_df_unmanaged, # Pass klines_df here
                    is_unmanaged_check=True
                )

                if not sanity_passed:
                    msg = (f"⚠️ {log_prefix} Sanity check FAILED for calculated SL/TP for UNMANAGED {symbol}. "
                           f"Entry: {entry_price:.{p_prec_unmanaged}f}, Side: {side}, Qty: {abs_position_qty}. "
                           f"Attempted SL: {calc_sl_price:.{p_prec_unmanaged}f}, TP: {calc_tp_price:.{p_prec_unmanaged}f}. Reason: {sanity_reason}")
                    print(msg)
                    send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), msg)
                    continue

                # 6. Place SL & TP orders
                sl_placed_unmanaged, tp_placed_unmanaged = False, False
                
                # Cancel existing SL/TP orders for this symbol before placing new ones for unmanaged position
                # This is to avoid conflicts if there are manually placed or very old bot orders.
                print(f"{log_prefix} Cancelling any existing SL/TP orders for unmanaged {symbol} before placing new ones.")
                open_orders_unmanaged = client.futures_get_open_orders(symbol=symbol)
                for order in open_orders_unmanaged:
                    # Check if the order is a conditional order (SL/TP) and is reduceOnly
                    # The `reduceOnly` field from Binance API is boolean.
                    if order['type'] in ['STOP_MARKET', 'TAKE_PROFIT_MARKET'] and order.get('reduceOnly') is True:
                        try:
                            client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                            print(f"{log_prefix} Cancelled existing conditional order {order['orderId']} for {symbol}.")
                        except Exception as e_cancel:
                            print(f"{log_prefix} Failed to cancel existing order {order['orderId']} for {symbol}: {e_cancel}")
                
                sl_order_obj, sl_error_msg = place_new_order(client, s_info_unmanaged,
                                                             "SELL" if side == "LONG" else "BUY",
                                                             "STOP_MARKET", abs_position_qty,
                                                             stop_price=calc_sl_price,
                                                             position_side=side, # Pass the determined side of the unmanaged position
                                                             is_closing_order=True) # Use is_closing_order instead of reduce_only
                if sl_order_obj and sl_order_obj.get('orderId'):
                    sl_placed_unmanaged = True
                    print(f"{log_prefix} Successfully placed SL for UNMANAGED {symbol} ({side}) @ {calc_sl_price:.{p_prec_unmanaged}f}")
                else:
                    err_detail = f"API Error: {sl_error_msg}" if sl_error_msg else "Order object missing or no orderId."
                    msg = (f"⚠️ {log_prefix} FAILED to place SL for UNMANAGED {symbol} ({side}). "
                           f"Entry: {entry_price:.{p_prec_unmanaged}f}, Target SL: {calc_sl_price:.{p_prec_unmanaged}f}. Details: {err_detail}")
                    print(msg)
                    send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), msg)

                tp_order_obj, tp_error_msg = place_new_order(client, s_info_unmanaged,
                                                             "SELL" if side == "LONG" else "BUY",
                                                             "TAKE_PROFIT_MARKET", abs_position_qty,
                                                             stop_price=calc_tp_price,
                                                             position_side=side, # Pass the determined side of the unmanaged position
                                                             is_closing_order=True) # Use is_closing_order instead of reduce_only
                if tp_order_obj and tp_order_obj.get('orderId'):
                    tp_placed_unmanaged = True
                    print(f"{log_prefix} Successfully placed TP for UNMANAGED {symbol} ({side}) @ {calc_tp_price:.{p_prec_unmanaged}f}")
                else:
                    err_detail = f"API Error: {tp_error_msg}" if tp_error_msg else "Order object missing or no orderId."
                    msg = (f"⚠️ {log_prefix} FAILED to place TP for UNMANAGED {symbol} ({side}). "
                           f"Entry: {entry_price:.{p_prec_unmanaged}f}, Target TP: {calc_tp_price:.{p_prec_unmanaged}f}. Details: {err_detail}")
                    print(msg)
                    send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), msg)
                
                if sl_placed_unmanaged or tp_placed_unmanaged:
                    final_msg = f"✅ {log_prefix} For UNMANAGED {symbol} (Entry: {entry_price:.{p_prec_unmanaged}f}, Qty: {abs_position_qty}): "
                    if sl_placed_unmanaged: final_msg += f"SL set @ {calc_sl_price:.{p_prec_unmanaged}f}. "
                    if tp_placed_unmanaged: final_msg += f"TP set @ {calc_tp_price:.{p_prec_unmanaged}f}."
                    send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), final_msg.strip())


    except BinanceAPIException as e:
        print(f"{log_prefix} Binance API Exception while fetching/processing positions: {e}")
        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"),
                              f"⚠️ {log_prefix} Binance API Error: {e}")
    except Exception as e:
        print(f"{log_prefix} Unexpected error during SL/TP check: {e}")
        traceback.print_exc()
        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"),
                              f"🆘 {log_prefix} Unexpected Error: {e}")

    print(f"{log_prefix} Finished SL/TP check for all open positions.")


# --- Backtesting Specific Functions ---

# Global state for backtesting simulation
backtest_current_time = None
backtest_simulated_balance = None # Will be initialized from actual balance or a preset
backtest_simulated_orders = {} # symbol -> list of order dicts
backtest_simulated_positions = {} # symbol -> position dict
backtest_trade_log = [] # Log of all simulated trade actions

# Backtesting specific daily state and halt flags
day_start_equity_bt = 0.0
daily_high_equity_bt = 0.0
daily_realized_pnl_bt = 0.0
last_trading_day_bt = None # Stores datetime.date object
trading_halted_drawdown_bt = False
trading_halted_daily_loss_bt = False
# No separate lock needed for backtest globals as it's single-threaded per simulation run

def get_current_equity_backtest(current_sim_balance: float, sim_positions: dict, candle_data_map: dict) -> float:
    """Calculates current equity for backtesting (sim_balance + unrealized P&L)."""
    total_unrealized_pnl = 0.0
    for symbol, pos_details in sim_positions.items():
        current_candle = candle_data_map.get(symbol)
        if current_candle is not None:
            market_price = current_candle['close'] # Use current candle's close for UPNL
            # Calculate UPNL for this position (re-using live logic structure)
            entry_price = pos_details['entryPrice']
            position_amount = pos_details['positionAmt'] # This is signed (positive for long, negative for short)
            
            # Universal formula for P&L: (market_price - entry_price) * position_amount
            # For LONG: (market_price - entry_price) * (+qty)
            # For SHORT: (market_price - entry_price) * (-qty) which is (entry_price - market_price) * (+qty)
            pnl = (market_price - entry_price) * position_amount
            total_unrealized_pnl += pnl
        else:
            # If no candle data for an open position's symbol, UPNL for it is unknown.
            # This shouldn't happen if candle_data_map is updated correctly each step for all symbols with positions.
            print(f"[SIM_ EQUITY_WARNING] No current candle data for symbol {symbol} with an open position. UPNL for it assumed 0.")
            
    return current_sim_balance + total_unrealized_pnl

def close_all_open_positions_backtest(configs: dict, current_candle_data_map: dict):
    """
    Simulates closing all open positions in backtesting at current candle's close.
    Updates daily_realized_pnl_bt and logs closures.
    Clears simulated positions, orders, and the bot's active_trades.
    """
    global backtest_simulated_positions, backtest_simulated_orders, backtest_trade_log
    global daily_realized_pnl_bt, active_trades, backtest_simulated_balance

    log_prefix_bt_close = f"[SIM-{backtest_current_time} CloseAllBT]"
    print(f"{log_prefix_bt_close} Max Drawdown triggered. Simulating closure of all open positions.")

    if not backtest_simulated_positions:
        print(f"{log_prefix_bt_close} No simulated positions to close.")
    else:
        for symbol, pos_details in list(backtest_simulated_positions.items()):
            current_candle = current_candle_data_map.get(symbol)
            if current_candle is None:
                print(f"{log_prefix_bt_close} CRITICAL: No candle data for {symbol} to simulate closure. Position may remain open in simulation state.")
                continue

            close_price = current_candle['close']
            entry_price = pos_details['entryPrice']
            quantity_abs = abs(pos_details['positionAmt']) # Absolute quantity
            side = pos_details['side']
            
            pnl_trade = 0
            if side == "LONG":
                pnl_trade = (close_price - entry_price) * quantity_abs
            elif side == "SHORT":
                pnl_trade = (entry_price - close_price) * quantity_abs
            
            print(f"{log_prefix_bt_close} Closing {symbol} ({side} {quantity_abs} @ {entry_price:.4f}) at {close_price:.4f}. PNL: {pnl_trade:.2f}")
            
            backtest_simulated_balance += pnl_trade # Update balance first
            daily_realized_pnl_bt += pnl_trade # Add to daily P&L

            backtest_trade_log.append({
                "time": backtest_current_time, "symbol": symbol, "type": "MARKET_CLOSE_DRAWDOWN_HALT", 
                "side": "SELL" if side == "LONG" else "BUY", # Action taken
                "qty": quantity_abs, "price": close_price, "pnl": pnl_trade, 
                "balance": backtest_simulated_balance,
                "reason": "Max Drawdown Halt"
            })
            del backtest_simulated_positions[symbol]

    # Clear all pending SL/TP orders for all symbols
    if backtest_simulated_orders:
        print(f"{log_prefix_bt_close} Clearing all ({len(backtest_simulated_orders)}) pending simulated SL/TP orders.")
        backtest_simulated_orders.clear()
    
    # Clear the bot's active_trades tracking
    if active_trades:
        print(f"{log_prefix_bt_close} Clearing bot's active_trades list ({len(active_trades)} items).")
        active_trades.clear()
    
    print(f"{log_prefix_bt_close} All positions closed/cleared due to Max Drawdown. Daily Realized PNL: {daily_realized_pnl_bt:.2f}")


def initialize_backtest_environment(client, configs):
    global backtest_simulated_balance, active_trades, backtest_trade_log, backtest_simulated_orders, backtest_simulated_positions
    # Also initialize backtest daily state variables here
    global day_start_equity_bt, daily_high_equity_bt, daily_realized_pnl_bt
    global last_trading_day_bt, trading_halted_drawdown_bt, trading_halted_daily_loss_bt

    active_trades.clear()
    backtest_trade_log = []
    backtest_simulated_orders = {}
    backtest_simulated_positions = {}
    
    # Reset backtest-specific daily state variables
    day_start_equity_bt = 0.0
    daily_high_equity_bt = 0.0
    daily_realized_pnl_bt = 0.0
    last_trading_day_bt = None
    trading_halted_drawdown_bt = False
    trading_halted_daily_loss_bt = False
    print("Backtest daily state variables reset.")

    start_balance_type = configs.get("backtest_start_balance_type", "current") # Default to current if not set
    
    if start_balance_type == "custom":
        backtest_simulated_balance = configs.get("backtest_custom_start_balance", 10000) # Default custom to 10k if somehow not set
        print(f"Backtest initialized with CUSTOM starting balance: {backtest_simulated_balance:.2f} USDT")
    else: # 'current' or default
        # Pass configs, though for backtesting the -2015 error is less likely / relevant for IP alert
        live_balance = get_account_balance(client, configs) 
        # Handle if live_balance is None (critical error)
        if live_balance is None:
            print("Warning: Could not fetch live balance for backtest initialization due to an API error. Defaulting to 10000 USDT.")
            backtest_simulated_balance = 10000
        else:
            backtest_simulated_balance = live_balance if live_balance > 0 else 10000 # Default to 10k if no live balance
        print(f"Backtest initialized with starting balance: {backtest_simulated_balance:.2f} USDT")
    
    # Initialize day_start_equity_bt and daily_high_equity_bt with the starting balance
    day_start_equity_bt = backtest_simulated_balance
    daily_high_equity_bt = backtest_simulated_balance


def get_simulated_account_balance(asset="USDT"): # Overrides live version for backtest
    global backtest_simulated_balance
    return backtest_simulated_balance

def get_simulated_open_positions(symbol=None): # Overrides live version for backtest
    global backtest_simulated_positions
    if symbol:
        return [backtest_simulated_positions[symbol]] if symbol in backtest_simulated_positions else []
    return list(backtest_simulated_positions.values())

def get_simulated_position_info(symbol):
    global backtest_simulated_positions
    return backtest_simulated_positions.get(symbol)


def place_simulated_order(symbol_info, side, order_type, quantity, price=None, stop_price=None, reduce_only=None, current_candle_price=None):
    global backtest_simulated_orders, backtest_simulated_balance, backtest_simulated_positions, backtest_trade_log, backtest_current_time

    symbol = symbol_info['symbol']
    order_id = f"sim_{symbol}_{int(time.time()*1000)}_{len(backtest_trade_log)}" # Unique enough for sim
    
    print(f"[SIM-{backtest_current_time}] Attempting to place {side} {order_type} for {quantity} {symbol} @ {price or 'MARKET'}")

    # Basic order structure
    sim_order = {
        "symbol": symbol, "orderId": order_id, "side": side, "type": order_type,
        "origQty": quantity, "executedQty": 0.0, "status": "NEW", "price": price,
        "stopPrice": stop_price, "reduceOnly": reduce_only or False,
        "timestamp": backtest_current_time, "avgPrice": 0.0
    }

    # For MARKET orders in simulation, we assume they fill at the current candle's close/open or a slight slippage
    # For LIMIT/STOP orders, they will be checked each candle to see if they trigger
    fill_price = current_candle_price # Assume market orders fill at current candle's close for simplicity
    
    if order_type == "MARKET":
        if reduce_only: # Closing part or all of an existing position
            current_pos = backtest_simulated_positions.get(symbol)
            if not current_pos or current_pos['positionAmt'] == 0:
                print(f"[SIM-{backtest_current_time}] ReduceOnly MARKET order for {symbol} but no position exists."); return None
            
            # Ensure side is opposite to current position
            if (side == "BUY" and current_pos['side'] == "SHORT") or \
               (side == "SELL" and current_pos['side'] == "LONG"):
                
                actual_reduce_qty = min(quantity, abs(current_pos['positionAmt']))
                pnl = (fill_price - current_pos['entryPrice']) * actual_reduce_qty if current_pos['side'] == "LONG" else \
                      (current_pos['entryPrice'] - fill_price) * actual_reduce_qty
                
                global daily_realized_pnl_bt # Ensure it's accessible
                backtest_simulated_balance += pnl # Add PnL to balance
                daily_realized_pnl_bt += pnl # Update daily realized P&L for backtest
                current_pos['positionAmt'] += actual_reduce_qty if side == "BUY" else -actual_reduce_qty # Reduce position
                
                sim_order.update({"status": "FILLED", "executedQty": actual_reduce_qty, "avgPrice": fill_price})
                backtest_trade_log.append({
                    "time": backtest_current_time, "symbol": symbol, "type": "MARKET_CLOSE", "side": side,
                    "qty": actual_reduce_qty, "price": fill_price, "pnl": pnl, "balance": backtest_simulated_balance,
                    "daily_realized_pnl_bt_after_trade": daily_realized_pnl_bt, # Log current daily PNL
                    "order_id": order_id
                })
                print(f"[SIM-{backtest_current_time}] {side} {actual_reduce_qty} {symbol} CLOSED @ {fill_price:.4f}. PnL: {pnl:.2f}. New Bal: {backtest_simulated_balance:.2f}. Daily Realized PNL_BT: {daily_realized_pnl_bt:.2f}")

                if abs(current_pos['positionAmt']) < 1e-9: # Position fully closed
                    del backtest_simulated_positions[symbol]
                    print(f"[SIM-{backtest_current_time}] Position for {symbol} fully closed.")
                else: # Position partially closed
                    print(f"[SIM-{backtest_current_time}] Position for {symbol} partially closed. Remaining: {current_pos['positionAmt']}")
                return sim_order
            else:
                print(f"[SIM-{backtest_current_time}] ReduceOnly MARKET order side mismatch for {symbol}."); return None

        else: # Opening a new position or increasing existing
            # Cost of trade (simplified, not including fees or precise margin calc for this example)
            # For simplicity, assume sufficient balance. Real backtester needs margin checks.
            sim_order.update({"status": "FILLED", "executedQty": quantity, "avgPrice": fill_price})
            
            current_pos = backtest_simulated_positions.get(symbol)
            if current_pos: # Adding to existing position (averaging down/up)
                # Complex logic for avg entry price, etc. - simplified here
                print(f"[SIM-{backtest_current_time}] WARNING: Adding to existing position for {symbol} - simplified logic.")
                # This needs proper handling of combined positions. For now, let's assume new trades are for new positions
                # or that the strategy is designed not to add to existing ones unless explicitly handled.
                # Let's overwrite for simplicity in this phase, or better, reject if position exists and not reduceOnly
                print(f"[SIM-{backtest_current_time}] Market order to open {symbol}, but position already exists and not reduceOnly. Order rejected for simplicity.")
                return None

            backtest_simulated_positions[symbol] = {
                "symbol": symbol, "entryPrice": fill_price, "positionAmt": quantity if side == "BUY" else -quantity,
                "side": "LONG" if side == "BUY" else "SHORT", "leverage": symbol_info.get('leverage', DEFAULT_LEVERAGE), # Assuming leverage is set elsewhere
                "marginType": symbol_info.get('marginType', DEFAULT_MARGIN_TYPE), "unRealizedProfit": 0.0,
                "initial_margin": (quantity * fill_price) / symbol_info.get('leverage', DEFAULT_LEVERAGE) # Simplified
            }
            # backtest_simulated_balance -= backtest_simulated_positions[symbol]['initial_margin'] # Simplified margin deduction
            
            backtest_trade_log.append({
                "time": backtest_current_time, "symbol": symbol, "type": "MARKET_OPEN", "side": side,
                "qty": quantity, "price": fill_price, "balance": backtest_simulated_balance, # Balance before margin deduction for this log
                "order_id": order_id
            })
            print(f"[SIM-{backtest_current_time}] {side} {quantity} {symbol} OPENED @ {fill_price:.4f}. New Pos: {backtest_simulated_positions[symbol]['positionAmt']}")
            return sim_order
            
    elif order_type in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]: # These are conditional orders
        if not stop_price: print(f"[SIM-{backtest_current_time}] Stop price needed for {order_type}."); return None
        # Store it to be checked each candle
        if symbol not in backtest_simulated_orders: backtest_simulated_orders[symbol] = []
        backtest_simulated_orders[symbol].append(sim_order)
        print(f"[SIM-{backtest_current_time}] {order_type} for {symbol} qty {quantity} at stop {stop_price} PLACED (pending trigger).")
        return sim_order
    else:
        print(f"[SIM-{backtest_current_time}] Order type {order_type} not fully supported in simplified simulation yet.")
        return None


def cancel_simulated_order(symbol, order_id):
    global backtest_simulated_orders, backtest_trade_log, backtest_current_time
    if symbol in backtest_simulated_orders:
        orders_for_symbol = backtest_simulated_orders[symbol]
        for i, order in enumerate(orders_for_symbol):
            if order['orderId'] == order_id:
                removed_order = orders_for_symbol.pop(i)
                print(f"[SIM-{backtest_current_time}] Cancelled simulated order {order_id} for {symbol}: {removed_order['type']} {removed_order['side']} Qty {removed_order['origQty']} Stop {removed_order.get('stopPrice')}")
                backtest_trade_log.append({
                    "time": backtest_current_time, "symbol": symbol, "type": "CANCEL_ORDER",
                    "order_id": order_id, "details": removed_order
                })
                if not backtest_simulated_orders[symbol]: # Clean up if list is empty
                    del backtest_simulated_orders[symbol]
                return True
    print(f"[SIM-{backtest_current_time}] Could not find order {order_id} for {symbol} to cancel.")
    return False

def process_pending_simulated_orders(symbol, current_candle_high, current_candle_low, current_candle_close):
    global backtest_simulated_orders, backtest_simulated_positions, backtest_trade_log, backtest_current_time
    
    if symbol not in backtest_simulated_orders: return

    triggered_orders_indices = []
    for i, order in enumerate(backtest_simulated_orders[symbol]):
        if order['status'] != "NEW": continue # Already processed or cancelled

        triggered = False
        trigger_price = None

        # Check STOP_MARKET / TAKE_PROFIT_MARKET
        if order['type'] in ["STOP_MARKET", "TAKE_PROFIT_MARKET"]:
            sp = order['stopPrice']
            # For LONG SL (SELL STOP_MARKET) or SHORT TP (BUY TAKE_PROFIT_MARKET)
            if (order['side'] == "SELL" and current_candle_low <= sp) or \
               (order['side'] == "BUY" and current_candle_high >= sp):
                triggered = True
                # Simplification: assume trigger at stop_price. Realistically, it could be worse (slippage).
                # For backtesting, it's common to assume it triggers at stop_price if breached.
                trigger_price = sp 
        
        if triggered:
            print(f"[SIM-{backtest_current_time}] TRIGGERED: {order['type']} {order['side']} for {symbol} Qty {order['origQty']} at Stop {order['stopPrice']} (Candle H:{current_candle_high}, L:{current_candle_low})")
            order['status'] = "TRIGGERED_PENDING_FILL" # Mark as triggered
            
            # Now simulate the fill (which is like a MARKET order)
            # This is a recursive call, essentially. Let's simplify by directly creating the market effect.
            # Assuming it's always reduceOnly for SL/TP from the main strategy logic
            
            current_pos = backtest_simulated_positions.get(symbol)
            if not current_pos:
                print(f"[SIM-{backtest_current_time}] ERROR: Triggered SL/TP for {symbol} but no position found. Order: {order['orderId']}")
                triggered_orders_indices.append(i) # Remove it as it can't be processed
                continue

            # Ensure the order side makes sense for closing the position
            # e.g. if pos is LONG, SL/TP must be SELL. If pos is SHORT, SL/TP must be BUY.
            if (current_pos['side'] == "LONG" and order['side'] == "SELL") or \
               (current_pos['side'] == "SHORT" and order['side'] == "BUY"):
                
                actual_filled_qty = min(order['origQty'], abs(current_pos['positionAmt']))
                pnl = (trigger_price - current_pos['entryPrice']) * actual_filled_qty if current_pos['side'] == "LONG" else \
                      (current_pos['entryPrice'] - trigger_price) * actual_filled_qty
                
                global backtest_simulated_balance, daily_realized_pnl_bt # Added daily_realized_pnl_bt
                backtest_simulated_balance += pnl
                daily_realized_pnl_bt += pnl # Update daily realized P&L for backtest
                current_pos['positionAmt'] += actual_filled_qty if order['side'] == "BUY" else -actual_filled_qty
                
                order.update({"status": "FILLED", "executedQty": actual_filled_qty, "avgPrice": trigger_price})
                backtest_trade_log.append({
                    "time": backtest_current_time, "symbol": symbol, "type": f"{order['type']}_FILL", "side": order['side'],
                    "qty": actual_filled_qty, "price": trigger_price, "pnl": pnl, "balance": backtest_simulated_balance,
                    "daily_realized_pnl_bt_after_trade": daily_realized_pnl_bt, # Log current daily PNL
                    "triggered_order_id": order['orderId']
                })
                print(f"[SIM-{backtest_current_time}] {order['side']} {actual_filled_qty} {symbol} (from {order['type']}) FILLED @ {trigger_price:.4f}. PnL: {pnl:.2f}. New Bal: {backtest_simulated_balance:.2f}. Daily Realized PNL_BT: {daily_realized_pnl_bt:.2f}")

                if abs(current_pos['positionAmt']) < 1e-9: # Position fully closed
                    del backtest_simulated_positions[symbol]
                    print(f"[SIM-{backtest_current_time}] Position for {symbol} (from {order['type']}) fully closed.")
                else: # Position partially closed
                    print(f"[SIM-{backtest_current_time}] Position for {symbol} (from {order['type']}) partially closed. Remaining: {current_pos['positionAmt']}")
                
                triggered_orders_indices.append(i) # Mark for removal from pending list
            else:
                print(f"[SIM-{backtest_current_time}] ERROR: Triggered SL/TP for {symbol} has mismatched side for current position. Order: {order['orderId']}, Pos Side: {current_pos['side']}")
                triggered_orders_indices.append(i) # Remove it

    # Remove orders that were filled or errored
    for index in sorted(triggered_orders_indices, reverse=True):
        del backtest_simulated_orders[symbol][index]
    if symbol in backtest_simulated_orders and not backtest_simulated_orders[symbol]:
        del backtest_simulated_orders[symbol]


def manage_trade_entry_backtest(client_dummy, configs, symbol, klines_df_current_slice, symbol_info_map, current_active_trades_bt):
    # This function is a wrapper around the core strategy logic for backtesting.
    # It uses simulated order placement and state management.
    # `current_active_trades_bt` is passed in, representing the state of active_trades for the backtest.
    global backtest_current_time, backtest_simulated_positions # backtest_simulated_positions is the ground truth for positions
    # Assuming these flags are managed by the backtesting_loop and passed or globally accessible for backtest context
    global trading_halted_drawdown_bt, trading_halted_daily_loss_bt 

    log_prefix_bt = f"[SIM-{backtest_current_time}] {symbol} manage_trade_entry_backtest:"

    # --- Check overall trading halt status for backtest ---
    # These flags (trading_halted_drawdown_bt, trading_halted_daily_loss_bt)
    # will need to be set appropriately within the backtesting_loop.
    if trading_halted_drawdown_bt:
        print(f"{log_prefix_bt} Trade entry BLOCKED. Reason: Max Drawdown trading halt is active (Backtest).")
        # Optionally log this to backtest_trade_log if desired
        # backtest_trade_log.append({"time": backtest_current_time, "symbol": symbol, "type": "REJECTED_HALT_DRAWDOWN"})
        return
    if trading_halted_daily_loss_bt:
        print(f"{log_prefix_bt} Trade entry BLOCKED. Reason: Daily Stop Loss trading halt is active (Backtest).")
        # backtest_trade_log.append({"time": backtest_current_time, "symbol": symbol, "type": "REJECTED_HALT_DAILY_LOSS"})
        return
    # --- End trading halt status check for backtest ---

    # The lock (`active_trades_lock`) is not used here as backtesting is typically single-threaded per candle loop.
    
    current_candle_close = klines_df_current_slice['close'].iloc[-1]
    current_candle_high = klines_df_current_slice['high'].iloc[-1]
    current_candle_low = klines_df_current_slice['low'].iloc[-1]

    # --- Calculations (same as live, moved before active trade check for signal determination) ---
    # Ensure EMAs are calculated on a copy to avoid SettingWithCopyWarning if klines_df_current_slice is a view
    klines_df_for_calc = klines_df_current_slice.copy()
    klines_df_for_calc['EMA100'] = calculate_ema(klines_df_for_calc, 100)
    klines_df_for_calc['EMA200'] = calculate_ema(klines_df_for_calc, 200)

    if klines_df_for_calc['EMA100'] is None or klines_df_for_calc['EMA200'] is None or \
       klines_df_for_calc['EMA100'].isnull().all() or klines_df_for_calc['EMA200'].isnull().all() or \
       len(klines_df_for_calc) < 202:
        # print(f"[SIM-{backtest_current_time}] EMA calculation failed or insufficient data for {symbol}. Length: {len(klines_df_current_slice)}")
        return

    signal_status = check_ema_crossover_conditions(klines_df_for_calc) 
    
    if signal_status not in ["LONG", "SHORT"]: # Not a valid, validated signal
        # Log reasons for no signal if needed from check_ema_crossover_conditions internal prints
        return # No actionable signal

    # --- NEW TRADE ENTRY CONDITION for Backtest: Check if symbol is already in active_trades ---
    # `current_active_trades_bt` is the dictionary representing active_trades passed into this function
    if symbol in current_active_trades_bt:
        # print(f"[SIM-{backtest_current_time}] Signal {signal_status} for {symbol} IGNORED. Symbol already has an active trade in backtest.")
        return # Exit if symbol already has an active trade.

    # Max concurrent positions check using the passed current_active_trades_bt
    if len(current_active_trades_bt) >= configs["max_concurrent_positions"]:
         # print(f"[SIM-{backtest_current_time}] Max concurrent positions ({configs['max_concurrent_positions']}) reached. Cannot open for new symbol {symbol}.")
         return
    
    # If here, signal is valid, symbol not in current_active_trades_bt, and not exceeding max positions.
    signal = signal_status 
    print(f"\n[SIM-{backtest_current_time}] --- New Validated Trade Signal for {symbol}: {signal} (No existing active trade in backtest) ---")
    
    symbol_info = symbol_info_map.get(symbol) 
    if not symbol_info: 
        print(f"[SIM-{backtest_current_time}] No symbol info for {symbol}. Abort signal processing."); return

    entry_p_signal = klines_df_for_calc['close'].iloc[-1] 

    # --- Dynamic Leverage Calculation for Backtest ---
    leverage_for_trade_bt = configs['leverage'] # Default to fixed leverage
    # Assuming 15-minute klines for backtest as well for candles_per_year
    candles_per_day_approx_bt = 24 * (60 / 15) 
    candles_per_year_approx_bt = int(candles_per_day_approx_bt * 365)

    realized_vol_bt = calculate_realized_volatility(
        klines_df_for_calc, # This is the slice for the current backtest candle
        configs.get('realized_volatility_period', DEFAULT_REALIZED_VOLATILITY_PERIOD),
        candles_per_year_approx_bt
    )

    if realized_vol_bt is not None and realized_vol_bt > 0:
        dynamic_lev_bt = calculate_dynamic_leverage(
            realized_vol_bt,
            configs.get('target_annualized_volatility', DEFAULT_TARGET_ANNUALIZED_VOLATILITY),
            configs.get('min_leverage', DEFAULT_MIN_LEVERAGE),
            configs.get('max_leverage', DEFAULT_MAX_LEVERAGE),
            configs['leverage'] 
        )
        leverage_for_trade_bt = dynamic_lev_bt
        print(f"[SIM-{backtest_current_time}] Dynamic leverage for {symbol} (Backtest): {leverage_for_trade_bt}x (Realized Vol: {realized_vol_bt:.4f})")
    else:
        print(f"[SIM-{backtest_current_time}] Could not calculate dynamic leverage for {symbol} (Backtest Realized Vol: {realized_vol_bt}). Using fixed leverage: {leverage_for_trade_bt}x.")
    # --- End Dynamic Leverage Calculation for Backtest ---

    # --- ATR and SL/TP Calculation for Backtest ---
    klines_df_for_calc['atr'] = calculate_atr(klines_df_for_calc, period=configs.get('atr_period', DEFAULT_ATR_PERIOD))
    current_atr_value_bt = klines_df_for_calc['atr'].iloc[-1]

    # Initial evaluated_trade_details for logging, may be updated with rejection reasons
    evaluated_trade_details = {
        "time": backtest_current_time, "symbol": symbol, "type": "SIGNAL_EVALUATED",
        "signal_side": signal, "calc_entry": entry_p_signal, 
        "atr_value": current_atr_value_bt if pd.notna(current_atr_value_bt) else None,
        "realized_vol_bt": realized_vol_bt if pd.notna(realized_vol_bt) else None,
        "leverage_used_bt": leverage_for_trade_bt
        # SL, TP, Qty, RR will be added after calculation
    }

    if pd.isna(current_atr_value_bt) or current_atr_value_bt <= 0:
        reason_atr_bt = f"Invalid ATR value ({current_atr_value_bt}) for {symbol} in backtest."
        print(f"[SIM-{backtest_current_time}] {reason_atr_bt} Cannot proceed.")
        evaluated_trade_details.update({
            "status": "REJECTED_INVALID_ATR", "reason": reason_atr_bt,
            "calc_sl": None, "calc_tp": None, "calc_qty": None, "initial_rr_ratio": None
        })
        backtest_trade_log.append(evaluated_trade_details)
        if 'atr' in klines_df_for_calc.columns: klines_df_for_calc.drop(columns=['atr'], inplace=True)
        return
    
    print(f"[SIM-{backtest_current_time}] Current ATR for {symbol} (Backtest): {current_atr_value_bt:.{symbol_info.get('pricePrecision', 2)}f}")
    
    sl_p, tp_p = calculate_sl_tp_values(entry_p_signal, signal, current_atr_value_bt, configs, symbol_info)
    if 'atr' in klines_df_for_calc.columns: klines_df_for_calc.drop(columns=['atr'], inplace=True) # Cleanup ATR column

    evaluated_trade_details.update({"calc_sl": sl_p, "calc_tp": tp_p})

    if sl_p is None or tp_p is None:
        reason_sltp_bt = "ATR-based SL/TP calculation failed in backtest."
        print(f"[SIM-{backtest_current_time}] {reason_sltp_bt} for {symbol}. Abort.")
        evaluated_trade_details.update({
            "status": "REJECTED_SLTP_CALC_FAIL_BT", "reason": reason_sltp_bt,
            "calc_qty": None, "initial_rr_ratio": None
        })
        backtest_trade_log.append(evaluated_trade_details)
        return
    # --- End ATR and SL/TP Calculation for Backtest ---
    
    current_sim_balance = get_simulated_account_balance()
    qty = calculate_position_size(current_sim_balance, configs['risk_percent'], entry_p_signal, sl_p, symbol_info, configs) 
    
    evaluated_trade_details.update({"calc_qty": qty if qty is not None else 0.0})

    # Calculate Risk:Reward Ratio
    rr_ratio = None
    if sl_p is not None and tp_p is not None and entry_p_signal != sl_p and abs(entry_p_signal - sl_p) > 1e-9:
        reward_abs = abs(tp_p - entry_p_signal)
        risk_abs = abs(entry_p_signal - sl_p)
        rr_ratio = reward_abs / risk_abs
    evaluated_trade_details.update({"initial_rr_ratio": rr_ratio})
    
    if current_sim_balance <= 0:
        print(f"[SIM-{backtest_current_time}] Zero/unavailable simulated balance. Abort."); 
        evaluated_trade_details.update({"status": "REJECTED", "reason": "ZERO_BALANCE"})
        backtest_trade_log.append(evaluated_trade_details)
        return

    if qty is None or qty <= 0: 
        print(f"[SIM-{backtest_current_time}] Invalid position size for {symbol}. Abort."); 
        evaluated_trade_details.update({"status": "REJECTED", "reason": "INVALID_POS_SIZE"}) # calc_qty already updated
        backtest_trade_log.append(evaluated_trade_details)
        return
        
    if round(qty, int(symbol_info['quantityPrecision'])) == 0.0:
        print(f"[SIM-{backtest_current_time}] Qty for {symbol} rounds to 0. Abort."); 
        evaluated_trade_details.update({"status": "REJECTED", "reason": "QTY_ROUNDS_TO_ZERO"})
        backtest_trade_log.append(evaluated_trade_details)
        return

    # --- Portfolio Risk Check for Backtest ---
    # Ensure sl_p is valid (not None, not equal to entry) before proceeding
    if sl_p is None or sl_p == entry_p_signal: 
        reason_portfolio_bt = f"Invalid SL price ({sl_p}) before portfolio risk check (Backtest)."
        print(f"[SIM-{backtest_current_time}] {reason_portfolio_bt} for {symbol}. Abort.")
        evaluated_trade_details.update({"status": "REJECTED_INVALID_SL_FOR_PORTFOLIO_CHECK", "reason": reason_portfolio_bt})
        backtest_trade_log.append(evaluated_trade_details)
        return

    new_trade_risk_abs_bt = qty * abs(entry_p_signal - sl_p)
    current_portfolio_risk_abs_bt = 0
    current_portfolio_risk_pct_bt = 0.0

    # `current_active_trades_bt` is the backtester's equivalent of live `active_trades`
    # No lock needed in backtest as it's single-threaded per candle
    current_portfolio_risk_pct_bt = calculate_aggregate_open_risk(current_active_trades_bt, current_sim_balance)
    if current_sim_balance > 0:
        current_portfolio_risk_abs_bt = (current_portfolio_risk_pct_bt / 100) * current_sim_balance

    potential_total_absolute_risk_bt = current_portfolio_risk_abs_bt + new_trade_risk_abs_bt
    potential_portfolio_risk_pct_bt = (potential_total_absolute_risk_bt / current_sim_balance) * 100 if current_sim_balance > 0 else float('inf')

    portfolio_risk_cap_config_pct_bt = configs.get("portfolio_risk_cap", DEFAULT_PORTFOLIO_RISK_CAP)

    print(f"[SIM-{backtest_current_time}] Portfolio Risk Check (Backtest) for {symbol}:")
    print(f"  Current Aggregate Portfolio Risk: {current_portfolio_risk_pct_bt:.2f}% ({current_portfolio_risk_abs_bt:.2f} USDT)")
    print(f"  Potential New Trade Risk ({symbol}): {(new_trade_risk_abs_bt / current_sim_balance * 100 if current_sim_balance > 0 else 0):.2f}% ({new_trade_risk_abs_bt:.2f} USDT)")
    print(f"  Projected Total Portfolio Risk: {potential_portfolio_risk_pct_bt:.2f}%")
    print(f"  Portfolio Risk Cap: {portfolio_risk_cap_config_pct_bt:.2f}%")

    if potential_portfolio_risk_pct_bt > portfolio_risk_cap_config_pct_bt:
        reason_portfolio_bt = (f"Portfolio risk limit exceeded (Backtest). Current: {current_portfolio_risk_pct_bt:.2f}%, "
                               f"New: {(new_trade_risk_abs_bt / current_sim_balance * 100 if current_sim_balance > 0 else 0):.2f}%, "
                               f"Projected: {potential_portfolio_risk_pct_bt:.2f}%, "
                               f"Cap: {portfolio_risk_cap_config_pct_bt:.2f}%.")
        print(f"[SIM-{backtest_current_time}] {reason_portfolio_bt} Trade for {symbol} rejected.")
        evaluated_trade_details.update({"status": "REJECTED_PORTFOLIO_RISK", "reason": reason_portfolio_bt})
        backtest_trade_log.append(evaluated_trade_details) # Log the initial evaluation then the rejection reason
        return
    print(f"[SIM-{backtest_current_time}] Portfolio risk check PASSED (Backtest) for {symbol}.")
    # --- End Portfolio Risk Check for Backtest ---

    # --- Perform Pre-Order Sanity Checks for Backtest ---
    # Note: `current_sim_balance` is used for `current_balance` parameter.
    # `klines_df_current_slice` is passed for `klines_df_for_debug`.
    passed_sanity_checks_bt, sanity_check_reason_bt = pre_order_sanity_checks(
        symbol=symbol,
        signal=signal,
        entry_price=entry_p_signal, # entry_p_signal is used as the proposed entry for backtest
        sl_price=sl_p,
        tp_price=tp_p,
        quantity=qty,
        symbol_info=symbol_info,
        current_balance=current_sim_balance,
        risk_percent_config=configs['risk_percent'],
        configs=configs,
        specific_leverage_for_trade=leverage_for_trade_bt, # Pass the dynamic/fallback leverage for backtest
        klines_df_for_debug=klines_df_for_calc # Use the df with EMAs
    )

    if not passed_sanity_checks_bt:
        print(f"[SIM-{backtest_current_time}] Pre-order sanity checks FAILED for {symbol}: {sanity_check_reason_bt}")
        evaluated_trade_details.update({"status": "REJECTED_SANITY_CHECK", "reason": sanity_check_reason_bt})
        backtest_trade_log.append(evaluated_trade_details)
        return # Abort trade entry for backtest

    print(f"[SIM-{backtest_current_time}] Pre-order sanity checks PASSED for {symbol}.")
    # --- End of Pre-Order Sanity Checks for Backtest ---

    # If all checks passed, log the initial signal evaluation details
    # (status will implicitly be "Passed all checks up to order placement")
    # or we can add an explicit status like "CHECKS_PASSED_PROCEEDING_TO_ORDER"
    evaluated_trade_details.update({"status": "CHECKS_PASSED_ATTEMPTING_ORDER"})
    backtest_trade_log.append(evaluated_trade_details)

    print(f"[SIM-{backtest_current_time}] Attempting {signal} {qty} {symbol} @SIM_MARKET (EP_signal:{entry_p_signal:.4f}), SL:{sl_p:.4f}, TP:{tp_p:.4f}, RR:{(f'{rr_ratio:.2f}' if isinstance(rr_ratio, float) else 'N/A')})")
    
    # In backtest, MARKET order fills at current_candle_close (or open of next, configurable)
    # For this simulation, let's assume it fills at `entry_p_signal` which is current_candle_close
    entry_order = place_simulated_order(symbol_info, "BUY" if signal=="LONG" else "SELL", "MARKET", qty, 
                                        current_candle_price=entry_p_signal)

    if entry_order and entry_order.get('status') == 'FILLED':
        actual_ep = float(entry_order.get('avgPrice', entry_p_signal)) # Should be entry_p_signal here
        
        # SL/TP might need slight re-calc if actual_ep differs from entry_p_signal due to simulated slippage
        # For now, assume actual_ep IS entry_p_signal for simplicity if no slippage model.
        # If slippage was modeled in place_simulated_order, then recalculate:
        if abs(actual_ep - entry_p_signal) > 1e-9 : # If there was slippage
            sl_dist_pct = abs(entry_p_signal - sl_p) / entry_p_signal
            tp_dist_pct = abs(entry_p_signal - tp_p) / entry_p_signal
            sl_p = actual_ep * (1 - sl_dist_pct if signal == "LONG" else 1 + sl_dist_pct)
            tp_p = actual_ep * (1 + tp_dist_pct if signal == "LONG" else 1 - tp_dist_pct)
            print(f"[SIM-{backtest_current_time}] SL/TP adjusted for actual fill {actual_ep:.4f}: SL {sl_p:.4f}, TP {tp_p:.4f}")

        sl_ord = place_simulated_order(symbol_info, "SELL" if signal=="LONG" else "BUY", "STOP_MARKET", qty, stop_price=sl_p, reduce_only=True)
        if not sl_ord: print(f"[SIM-{backtest_current_time}] CRITICAL: FAILED TO PLACE SIMULATED SL FOR {symbol}!")
        
        tp_ord = place_simulated_order(symbol_info, "SELL" if signal=="LONG" else "BUY", "TAKE_PROFIT_MARKET", qty, stop_price=tp_p, reduce_only=True)
        if not tp_ord: print(f"[SIM-{backtest_current_time}] Warning: Failed to place simulated TP for {symbol}.")

        # Update active_trades for the backtester's internal tracking (mirroring live bot)
        # This active_trades is the global one, ensure it's managed correctly for backtesting context
        # IMPORTANT: current_active_trades_bt is the one passed in, which is the global active_trades.
        current_active_trades_bt[symbol] = {
            "entry_order_id": entry_order['orderId'], "sl_order_id": sl_ord.get('orderId') if sl_ord else None,
            "tp_order_id": tp_ord.get('orderId') if tp_ord else None, "entry_price": actual_ep,
            "current_sl_price": sl_p, "current_tp_price": tp_p, "initial_sl_price": sl_p, "initial_tp_price": tp_p,
            "quantity": qty, "side": signal, "symbol_info": symbol_info, 
            "open_timestamp": backtest_current_time # Use backtest_current_time
        }
        print(f"[SIM-{backtest_current_time}] Trade for {symbol} recorded in active_trades at {current_active_trades_bt[symbol]['open_timestamp']}.")
        
        # Update last trade time for cooldown in backtest - REMOVED
        # symbol_last_trade_time[symbol] = active_trades[symbol]['open_timestamp'] 
        # print(f"[SIM-{backtest_current_time}] Updated last trade time for {symbol} to {symbol_last_trade_time[symbol]} for backtest cooldown.")
    else:
        print(f"[SIM-{backtest_current_time}] Simulated Market order for {symbol} failed or not filled: {entry_order}")
    klines_df_current_slice['EMA200'] = calculate_ema(klines_df_current_slice, 200)
    if klines_df_current_slice['EMA100'] is None or klines_df_current_slice['EMA200'] is None or \
       klines_df_current_slice['EMA100'].isnull().all() or klines_df_current_slice['EMA200'].isnull().all() or \
       len(klines_df_current_slice) < 202:
        # print(f"[SIM-{backtest_current_time}] EMA calculation failed or insufficient data for {symbol}. Length: {len(klines_df_current_slice)}")
        return

    signal_status = check_ema_crossover_conditions(klines_df_current_slice) 
    
    if signal_status not in ["LONG", "SHORT"]: # Not a valid, validated signal
        if signal_status == "VALIDATION_FAILED":
            backtest_trade_log.append({
                "time": backtest_current_time, "symbol": symbol, "type": "SIGNAL_VALIDATION_FAILED",
                "reason": "Price touched EMAs during validation period",
                "details": f"EMA100: {klines_df_current_slice['EMA100'].iloc[-1]:.4f}, EMA200: {klines_df_current_slice['EMA200'].iloc[-1]:.4f}"
            })
        elif signal_status == "INSUFFICIENT_VALIDATION_HISTORY":
             backtest_trade_log.append({
                "time": backtest_current_time, "symbol": symbol, "type": "SIGNAL_VALIDATION_FAILED",
                "reason": "Insufficient validation history",
                "details": f"EMA100: {klines_df_current_slice['EMA100'].iloc[-1]:.4f}, EMA200: {klines_df_current_slice['EMA200'].iloc[-1]:.4f}"
            })
        # If signal_status is None, it means no crossover, so no log needed here, just return.
        return

    signal = signal_status # Now signal is confirmed "LONG" or "SHORT"
    print(f"\n[SIM-{backtest_current_time}] --- New Validated Trade Signal for {symbol}: {signal} ---")
    
    symbol_info = symbol_info_map.get(symbol) 
    if not symbol_info: 
        print(f"[SIM-{backtest_current_time}] No symbol info for {symbol}. Abort signal processing."); return

    entry_p_signal = klines_df_current_slice['close'].iloc[-1] 
    ema100_val = klines_df_current_slice['EMA100'].iloc[-1]
    sl_p, tp_p = calculate_sl_tp_values(entry_p_signal, signal, ema100_val, klines_df_current_slice)
    
    current_sim_balance = get_simulated_account_balance()
    # Pass configs to calculate_position_size for backtesting as well
    qty = calculate_position_size(current_sim_balance, configs['risk_percent'], entry_p_signal, sl_p, symbol_info, configs) 
    
    # Calculate Risk:Reward Ratio
    rr_ratio = None
    if sl_p is not None and tp_p is not None and entry_p_signal != sl_p:
        reward_abs = abs(tp_p - entry_p_signal)
        risk_abs = abs(entry_p_signal - sl_p)
        if risk_abs > 1e-9 : rr_ratio = reward_abs / risk_abs

    evaluated_trade_details = {
        "time": backtest_current_time, "symbol": symbol, "type": "SIGNAL_EVALUATED",
        "signal_side": signal, "calc_entry": entry_p_signal, "calc_sl": sl_p, "calc_tp": tp_p,
        "calc_qty": qty if qty is not None else 0.0, "initial_rr_ratio": rr_ratio
    }

    if sl_p is None or tp_p is None:
        print(f"[SIM-{backtest_current_time}] SL/TP calc failed for {symbol}. Abort."); 
        evaluated_trade_details.update({"status": "REJECTED", "reason": "SL_TP_CALC_FAILED"})
        backtest_trade_log.append(evaluated_trade_details)
        return
    
    if current_sim_balance <= 0:
        print(f"[SIM-{backtest_current_time}] Zero/unavailable simulated balance. Abort."); 
        evaluated_trade_details.update({"status": "REJECTED", "reason": "ZERO_BALANCE"})
        backtest_trade_log.append(evaluated_trade_details)
        return

    if qty is None or qty <= 0: 
        print(f"[SIM-{backtest_current_time}] Invalid position size for {symbol}. Abort."); 
        evaluated_trade_details.update({"status": "REJECTED", "reason": "INVALID_POS_SIZE", "calc_qty": qty if qty is not None else 0.0})
        backtest_trade_log.append(evaluated_trade_details)
        return
        
    if round(qty, int(symbol_info['quantityPrecision'])) == 0.0:
        print(f"[SIM-{backtest_current_time}] Qty for {symbol} rounds to 0. Abort."); 
        evaluated_trade_details.update({"status": "REJECTED", "reason": "QTY_ROUNDS_TO_ZERO"})
        backtest_trade_log.append(evaluated_trade_details)
        return

    # Log evaluated signal before attempting to place
    backtest_trade_log.append(evaluated_trade_details)

    # --- Perform Pre-Order Sanity Checks for Backtest ---
    # Note: `current_sim_balance` is used for `current_balance` parameter.
    # `klines_df_current_slice` is passed for `klines_df_for_debug`.
    passed_sanity_checks_bt, sanity_check_reason_bt = pre_order_sanity_checks(
        symbol=symbol,
        signal=signal,
        entry_price=entry_p_signal, # entry_p_signal is used as the proposed entry for backtest
        sl_price=sl_p,
        tp_price=tp_p,
        quantity=qty,
        symbol_info=symbol_info,
        current_balance=current_sim_balance,
        risk_percent_config=configs['risk_percent'],
        configs=configs,
        klines_df_for_debug=klines_df_current_slice 
    )

    if not passed_sanity_checks_bt:
        print(f"[SIM-{backtest_current_time}] Pre-order sanity checks FAILED for {symbol}: {sanity_check_reason_bt}")
        # Log this failure specifically
        backtest_trade_log.append({
            "time": backtest_current_time, "symbol": symbol, "type": "SANITY_CHECK_FAILED",
            "reason": sanity_check_reason_bt, "signal_side": signal, 
            "calc_entry": entry_p_signal, "calc_sl": sl_p, "calc_tp": tp_p, "calc_qty": qty
        })
        # Update the status of the previously logged "SIGNAL_EVALUATED" to "REJECTED_SANITY_CHECK"
        # Find the last "SIGNAL_EVALUATED" for this symbol and time (should be the one just added)
        for log_item in reversed(backtest_trade_log):
            if log_item.get("type") == "SIGNAL_EVALUATED" and \
               log_item.get("symbol") == symbol and \
               log_item.get("time") == backtest_current_time:
                log_item["status"] = "REJECTED_SANITY_CHECK"
                log_item["reject_reason"] = sanity_check_reason_bt
                break
        return # Abort trade entry for backtest

    print(f"[SIM-{backtest_current_time}] Pre-order sanity checks PASSED for {symbol}.")
    # --- End of Pre-Order Sanity Checks for Backtest ---

    print(f"[SIM-{backtest_current_time}] Attempting {signal} {qty} {symbol} @SIM_MARKET (EP_signal:{entry_p_signal:.4f}), SL:{sl_p:.4f}, TP:{tp_p:.4f}, RR:{(f'{rr_ratio:.2f}' if isinstance(rr_ratio, float) else 'N/A')})")
    
    # In backtest, MARKET order fills at current_candle_close (or open of next, configurable)
    # For this simulation, let's assume it fills at `entry_p_signal` which is current_candle_close
    entry_order = place_simulated_order(symbol_info, "BUY" if signal=="LONG" else "SELL", "MARKET", qty, 
                                        current_candle_price=entry_p_signal)

    if entry_order and entry_order.get('status') == 'FILLED':
        actual_ep = float(entry_order.get('avgPrice', entry_p_signal)) # Should be entry_p_signal here
        
        # SL/TP might need slight re-calc if actual_ep differs from entry_p_signal due to simulated slippage
        # For now, assume actual_ep IS entry_p_signal for simplicity if no slippage model.
        # If slippage was modeled in place_simulated_order, then recalculate:
        if abs(actual_ep - entry_p_signal) > 1e-9 : # If there was slippage
            sl_dist_pct = abs(entry_p_signal - sl_p) / entry_p_signal
            tp_dist_pct = abs(entry_p_signal - tp_p) / entry_p_signal
            sl_p = actual_ep * (1 - sl_dist_pct if signal == "LONG" else 1 + sl_dist_pct)
            tp_p = actual_ep * (1 + tp_dist_pct if signal == "LONG" else 1 - tp_dist_pct)
            print(f"[SIM-{backtest_current_time}] SL/TP adjusted for actual fill {actual_ep:.4f}: SL {sl_p:.4f}, TP {tp_p:.4f}")

        sl_ord = place_simulated_order(symbol_info, "SELL" if signal=="LONG" else "BUY", "STOP_MARKET", qty, stop_price=sl_p, reduce_only=True)
        if not sl_ord: print(f"[SIM-{backtest_current_time}] CRITICAL: FAILED TO PLACE SIMULATED SL FOR {symbol}!")
        
        tp_ord = place_simulated_order(symbol_info, "SELL" if signal=="LONG" else "BUY", "TAKE_PROFIT_MARKET", qty, stop_price=tp_p, reduce_only=True)
        if not tp_ord: print(f"[SIM-{backtest_current_time}] Warning: Failed to place simulated TP for {symbol}.")

        # Update active_trades for the backtester's internal tracking (mirroring live bot)
        # This active_trades is the global one, ensure it's managed correctly for backtesting context
        active_trades[symbol] = {
            "entry_order_id": entry_order['orderId'], "sl_order_id": sl_ord.get('orderId') if sl_ord else None,
            "tp_order_id": tp_ord.get('orderId') if tp_ord else None, "entry_price": actual_ep,
            "current_sl_price": sl_p, "current_tp_price": tp_p, "initial_sl_price": sl_p, "initial_tp_price": tp_p,
            "quantity": qty, "side": signal, "symbol_info": symbol_info, 
            "open_timestamp": backtest_current_time # Use backtest_current_time
        }
        print(f"[SIM-{backtest_current_time}] Trade for {symbol} recorded in active_trades at {active_trades[symbol]['open_timestamp']}.")
        
        # Update last trade time for cooldown in backtest - REMOVED
        # symbol_last_trade_time[symbol] = active_trades[symbol]['open_timestamp'] 
        # print(f"[SIM-{backtest_current_time}] Updated last trade time for {symbol} to {symbol_last_trade_time[symbol]} for backtest cooldown.")
    else:
        print(f"[SIM-{backtest_current_time}] Simulated Market order for {symbol} failed or not filled: {entry_order}")


def monitor_active_trades_backtest(configs):
    global active_trades, active_trades_lock, backtest_simulated_positions, backtest_current_time, backtest_trade_log
    
    if not active_trades: return
    # print(f"\n[SIM-{backtest_current_time}] Monitoring {len(active_trades)} active bot trades...")
    
    symbols_to_remove_from_active_trades = []

    for symbol, trade_details in list(active_trades.items()): # Iterate copy for safe modification
        # print(f"[SIM-{backtest_current_time}] Checking {symbol} (Side: {trade_details['side']}, Entry: {trade_details['entry_price']:.4f})...")
        
        sim_pos = backtest_simulated_positions.get(symbol)
        if not sim_pos or sim_pos['positionAmt'] == 0: # Position closed (likely by SL/TP trigger in process_pending_simulated_orders)
            print(f"[SIM-{backtest_current_time}] Position for {symbol} seems closed. Removing from active_trades.")
            symbols_to_remove_from_active_trades.append(symbol)
            # No need to cancel SL/TP orders here, as they would have been consumed or become irrelevant
            # if they were simulated. If `process_pending_simulated_orders` handles their removal, this is fine.
            continue

        # Dynamic SL/TP adjustment logic (same as live, but uses simulated current price)
        # The "current price" for a candle in backtest can be Open, High, Low, or Close.
        # Typically, for monitoring, you'd use the Close of the current candle.
        # This requires the klines_df for the symbol up to backtest_current_time.
        # This function is called AFTER all symbols have had their pending orders processed for the current candle.
        # So, `sim_pos['entryPrice']` etc. are post-any-fills for this candle.
        
        # We need the current kline data for the symbol to get the current price.
        # This implies monitor_active_trades_backtest needs access to the symbol's latest kline slice.
        # This is a bit tricky design-wise. Let's assume it's passed or accessible.
        # For now, let's assume `current_candle_data_map` is available, mapping symbol to its current candle.
        # This will be populated in the main backtesting_loop.
        current_candle_data = current_candle_data_map.get(symbol)
        if not current_candle_data:
            # print(f"[SIM-{backtest_current_time}] No current candle data for {symbol} in monitor_active_trades_backtest. Skipping dynamic SL/TP.")
            continue
        cur_price = current_candle_data['close'] # Use close of the current candle for adjustment checks

        adj_sl, adj_tp, adj_reason = check_and_adjust_sl_tp_dynamic(
            cur_price, trade_details['entry_price'], 
            trade_details['initial_sl_price'], trade_details['initial_tp_price'],
            trade_details['current_sl_price'], trade_details['current_tp_price'],
            trade_details['side']
        )
        
        s_info = trade_details['symbol_info']
        qty = trade_details['quantity']
        side = trade_details['side']
        made_adjustment_in_log = False

        if adj_reason: # If any adjustment was made
            log_entry_dyn_adj = {
                "time": backtest_current_time, "symbol": symbol, "type": "DYNAMIC_ADJUSTMENT",
                "reason": adj_reason,
                "old_sl": trade_details['current_sl_price'], "new_sl": adj_sl if adj_sl is not None else trade_details['current_sl_price'],
                "old_tp": trade_details['current_tp_price'], "new_tp": adj_tp if adj_tp is not None else trade_details['current_tp_price'],
                "current_price_at_adj": cur_price
            }
            # Note: adj_sl and adj_tp from check_and_adjust_sl_tp_dynamic will be the new values IF they changed,
            # otherwise they are the same as current_sl_price/current_tp_price passed in.
            # We only proceed to cancel/replace if the specific SL or TP value has actually changed.

            if adj_sl is not None and abs(adj_sl - trade_details['current_sl_price']) > 1e-9:
                print(f"[SIM-{backtest_current_time}] Adjusting SL for {symbol} from {trade_details['current_sl_price']:.4f} to {adj_sl:.4f} (Reason: {adj_reason})")
                if trade_details.get('sl_order_id'): 
                    cancel_simulated_order(symbol, trade_details['sl_order_id'])
                
                sl_ord_new = place_simulated_order(s_info, "SELL" if side=="LONG" else "BUY", "STOP_MARKET", qty, stop_price=adj_sl, reduce_only=True)
                if sl_ord_new: 
                    active_trades[symbol]['current_sl_price'] = adj_sl
                    active_trades[symbol]['sl_order_id'] = sl_ord_new.get('orderId')
                    made_adjustment_in_log = True
                else: 
                    print(f"[SIM-{backtest_current_time}] CRITICAL: FAILED TO PLACE NEW SIMULATED SL FOR {symbol}!")
                    log_entry_dyn_adj["sl_update_status"] = "FAILED_TO_PLACE_NEW_SL"
            
            if adj_tp is not None and abs(adj_tp - trade_details['current_tp_price']) > 1e-9:
                print(f"[SIM-{backtest_current_time}] Adjusting TP for {symbol} from {trade_details['current_tp_price']:.4f} to {adj_tp:.4f} (Reason: {adj_reason})")
                if trade_details.get('tp_order_id'):
                    cancel_simulated_order(symbol, trade_details['tp_order_id'])

                tp_ord_new = place_simulated_order(s_info, "SELL" if side=="LONG" else "BUY", "TAKE_PROFIT_MARKET", qty, stop_price=adj_tp, reduce_only=True)
                if tp_ord_new:
                    active_trades[symbol]['current_tp_price'] = adj_tp
                    active_trades[symbol]['tp_order_id'] = tp_ord_new.get('orderId')
                    made_adjustment_in_log = True
                else: 
                    print(f"[SIM-{backtest_current_time}] Warning: Failed to place new simulated TP for {symbol}.")
                    log_entry_dyn_adj["tp_update_status"] = "FAILED_TO_PLACE_NEW_TP"

            if made_adjustment_in_log: # Only log if an actual order was changed
                 backtest_trade_log.append(log_entry_dyn_adj)

    for sym in symbols_to_remove_from_active_trades:
        if sym in active_trades:
            # Before deleting, ensure any associated pending orders are also cleared if not already handled
            trade_detail_to_remove = active_trades[sym]
            if trade_detail_to_remove.get('sl_order_id'):
                cancel_simulated_order(sym, trade_detail_to_remove['sl_order_id'])
            if trade_detail_to_remove.get('tp_order_id'):
                cancel_simulated_order(sym, trade_detail_to_remove['tp_order_id'])
            del active_trades[sym]
            print(f"[SIM-{backtest_current_time}] Removed {sym} from bot's active_trades list.")


# Global map for current candle data per symbol, used by monitor_active_trades_backtest
current_candle_data_map = {}

def fetch_initial_backtest_data_for_symbol(symbol, client, backtest_days_config, existing_symbol_info=None):
    """
    Helper function to fetch symbol info and historical klines for a single symbol for backtesting.
    """
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Fetching initial data for {symbol}...")
    s_info = existing_symbol_info
    if not s_info:
        s_info = get_symbol_info(client, symbol) # Live call, okay for setup

    if not s_info:
        print(f"[{thread_name}] Could not get symbol info for {symbol}. Skipping.")
        return symbol, None, None

    # Fetch data for each symbol
    # The `get_historical_klines` fetches based on "days ago UTC"
    kl_df = get_historical_klines(client, symbol, backtest_days=backtest_days_config)
    if kl_df.empty:
        print(f"[{thread_name}] No kline data for {symbol} for backtest period. Skipping.")
        return symbol, s_info, None
    
    print(f"[{thread_name}] Successfully fetched {len(kl_df)} klines for {symbol}.")
    return symbol, s_info, kl_df

def backtesting_loop(client, configs, monitored_symbols):
    global backtest_current_time, active_trades, backtest_simulated_balance, backtest_trade_log, current_candle_data_map
    global day_start_equity_bt, daily_high_equity_bt, daily_realized_pnl_bt, last_trading_day_bt # Added
    global trading_halted_drawdown_bt, trading_halted_daily_loss_bt # Added
    global backtest_simulated_positions, backtest_simulated_orders # Added for clarity, though modified via helpers mostly

    print("\n--- Starting Backtesting Loop ---")
    if not monitored_symbols: print("No symbols to monitor for backtest. Exiting."); return

    initialize_backtest_environment(client, configs) # Sets up balance, clears old state
    
    backtest_days_config = configs['backtest_days']
    print(f"Fetching historical data for {len(monitored_symbols)} symbols for {backtest_days_config} days...")
    
    all_symbol_historical_data_raw = {} # Store raw fetched data before alignment
    symbol_info_map = {}
    master_klines_df = pd.DataFrame()
    reference_symbol_candidates = ['BTCUSDT', 'ETHUSDT'] + [s for s in monitored_symbols if s not in ['BTCUSDT', 'ETHUSDT']] # Prioritize common symbols

    # 1. Determine the overall timespan for the backtest using a reference symbol
    print("Step 1: Establishing master timeline for backtest...")
    reference_symbol_found = None
    for ref_sym in reference_symbol_candidates:
        if ref_sym not in monitored_symbols: # If BTC/ETH not in user's list, skip
            # However, if monitored_symbols is short, it might be the first one.
            if not monitored_symbols or ref_sym != monitored_symbols[0]: # ensure we check user's symbols
                 continue
            # If BTC/ETH not explicitly in monitored_symbols, but monitored_symbols is not empty,
            # ensure we use a symbol from monitored_symbols for reference.
            # The loop structure of reference_symbol_candidates handles this by trying BTC/ETH first if they are in monitored_symbols,
            # then others from monitored_symbols.
            # The condition `if ref_sym not in monitored_symbols:` might be too strict if we want to use, e.g. BTCUSDT
            # from `reference_symbol_candidates` even if it's not in the `monitored_symbols` list from the user to establish time.
            # For now, let's stick to symbols the user wants to monitor.
            if ref_sym not in monitored_symbols : continue


        print(f"Attempting to use {ref_sym} to establish backtest time range...")
        # Fetch data for the reference symbol directly first
        _, _, temp_df = fetch_initial_backtest_data_for_symbol(ref_sym, client, backtest_days_config)
        if temp_df is not None and not temp_df.empty:
            master_klines_df = temp_df
            reference_symbol_found = ref_sym
            print(f"Successfully fetched master kline data using {reference_symbol_found} ({len(master_klines_df)} candles).")
            break
        else:
            print(f"Failed to fetch kline data for {ref_sym} as reference. Trying next symbol.")

    if master_klines_df.empty or reference_symbol_found is None:
        print(f"Could not establish master kline data from any candidate reference symbols. Aborting backtest."); return
    
    # Filter master_klines_df to the precise "backtest_days" from its end
    end_date_of_data = master_klines_df.index[-1]
    start_date_cutoff = end_date_of_data - pd.Timedelta(days=backtest_days_config)
    master_klines_df = master_klines_df[master_klines_df.index >= start_date_cutoff]
    print(f"Master timeline for backtest established: {len(master_klines_df)} candles from {master_klines_df.index[0]} to {master_klines_df.index[-1]}")

    # 2. Fetch data for all other symbols concurrently
    print(f"\nStep 2: Concurrently fetching data for {len(monitored_symbols)} symbols...")
    all_symbol_historical_data = {} 
    
    # Add reference symbol's data to maps if it was successfully fetched
    if reference_symbol_found:
        # s_info_ref = get_symbol_info(client, reference_symbol_found) # Re-fetch or store from fetch_initial
        # For simplicity, assume fetch_initial_backtest_data_for_symbol would give us info if we called it again
        # Or better, store it from the initial call. Let's assume it's stored if needed.
        # For now, symbol_info_map will be populated from the concurrent fetches.
        # The master_klines_df is already populated with the kline data for the reference symbol.
        # We'll add its klines to all_symbol_historical_data_raw to be processed like others.
        all_symbol_historical_data_raw[reference_symbol_found] = master_klines_df.copy()
        # And ensure its symbol_info is also fetched and stored
        # This will be handled if reference_symbol_found is part of monitored_symbols in the loop below.
        # To be safe, explicitly get its info if not already done by fetch_initial...
        # However, fetch_initial_backtest_data_for_symbol returns symbol_info. We should use that.
        # Let's refine the ref symbol handling slightly:
        # We need its symbol_info. Let's call fetch_initial_backtest_data_for_symbol and store all its results.
        
        # Re-evaluate: The reference symbol's data IS master_klines_df. Its info needs to be in symbol_info_map.
        # The concurrent fetching below will include the reference symbol if it's in monitored_symbols.

    symbols_to_fetch_concurrently = [s for s in monitored_symbols] # All symbols including reference if it's in the list

    with ThreadPoolExecutor(max_workers=configs.get('max_scan_threads', DEFAULT_MAX_SCAN_THREADS)) as executor:
        futures = {executor.submit(fetch_initial_backtest_data_for_symbol, sym, client, backtest_days_config): sym for sym in symbols_to_fetch_concurrently}
        
        for future in as_completed(futures):
            original_symbol = futures[future]
            try:
                res_symbol, res_s_info, res_kl_df = future.result()
                if res_symbol != original_symbol: # Should not happen if symbol is returned correctly
                     print(f"Warning: Mismatch in returned symbol {res_symbol} and original {original_symbol}")
                
                if res_s_info and res_kl_df is not None and not res_kl_df.empty:
                    symbol_info_map[res_symbol] = res_s_info
                    all_symbol_historical_data_raw[res_symbol] = res_kl_df
                    print(f"Successfully processed data for {res_symbol} from concurrent fetch.")
                else:
                    print(f"Failed to fetch or no data for {res_symbol} during concurrent fetch.")
            except Exception as exc:
                print(f"Symbol {original_symbol} generated an exception during fetch: {exc}")
                traceback.print_exc()

    # 3. Align all fetched data to the master timeline
    print("\nStep 3: Aligning all symbol data to master timeline...")
    processed_symbols_count = 0
    for symbol, kl_df_raw in all_symbol_historical_data_raw.items():
        if symbol not in symbol_info_map: # Should have info if data is present
            print(f"Warning: Symbol {symbol} has kline data but no symbol_info. Skipping alignment.")
            continue

        # Trim to the exact range of master_klines_df
        kl_df_aligned = kl_df_raw[kl_df_raw.index >= master_klines_df.index[0]]
        kl_df_aligned = kl_df_aligned[kl_df_aligned.index <= master_klines_df.index[-1]]
        
        # Reindex to master timeline, forward-filling missing data.
        # Important: Ensure that NaNs resulting from reindexing (e.g., for symbols listed later than master's start)
        # are handled appropriately by downstream logic (e.g., strategy conditions, EMA calculations).
        # The strategy already checks for NaN EMAs and sufficient data length.
        kl_df_aligned = kl_df_aligned.reindex(master_klines_df.index, method='ffill')
        
        # A symbol might have no data for the early part of the master timeline.
        # `ffill` will fill these. We need to ensure that calculations like EMA
        # are not skewed by artificially filled old data.
        # The strategy's checks for `len(df) < period` and `df.isnull().values.any()` for EMAs should handle this.
        # However, it's good to also check if a symbol has *any* non-NaN data after alignment.
        if kl_df_aligned.empty or kl_df_aligned['close'].isnull().all():
            print(f"No valid data for {symbol} after aligning to master timeline. Skipping this symbol.")
            # Remove from symbol_info_map as well if we skip it entirely
            if symbol in symbol_info_map: del symbol_info_map[symbol]
            continue

        all_symbol_historical_data[symbol] = kl_df_aligned
        print(f"Data for {symbol} aligned and prepared: {len(kl_df_aligned)} candles.")
        processed_symbols_count +=1

    if not all_symbol_historical_data:
        print("No historical data loaded and aligned for any symbol. Aborting backtest."); return
    
    print(f"\nSuccessfully prepared and aligned data for {processed_symbols_count} symbols out of {len(monitored_symbols)} monitored.")
    
    # The number of "candles" or simulation steps is determined by the length of master_klines_df (which should be non-empty)
    # We need at least 202 candles of history for EMA calculations at any point.
    # So, the actual simulation will start from the 202nd candle of the fetched data.
    simulation_start_index = 201 # Start processing from the 202nd candle (index 201)
    if len(master_klines_df) <= simulation_start_index:
        print(f"Not enough historical data ({len(master_klines_df)} candles) to start simulation (needs > {simulation_start_index}).")
        return

    print(f"\n--- Starting Simulation ({len(master_klines_df) - simulation_start_index} steps) ---")
    initial_balance_for_report = backtest_simulated_balance
    
    # 2. Iterate candle by candle through the master timeline
    for i in range(simulation_start_index, len(master_klines_df)):
        backtest_current_time = master_klines_df.index[i]
        current_candle_data_map.clear() # Clear for the new candle
        
        # --- Backtest Daily State Management ---
        # Global declarations for these variables are now at the top of the function.
        
        current_date_bt = backtest_current_time.date()
        # First, populate current_candle_data_map for equity calculation for *this specific candle*
        for symbol_for_candle_map in all_symbol_historical_data.keys():
            if symbol_for_candle_map in all_symbol_historical_data:
                 symbol_klines_for_map = all_symbol_historical_data[symbol_for_candle_map].iloc[:i+1]
                 if not symbol_klines_for_map.empty:
                      current_candle_data_map[symbol_for_candle_map] = symbol_klines_for_map.iloc[-1]

        current_equity_bt = get_current_equity_backtest(backtest_simulated_balance, backtest_simulated_positions, current_candle_data_map)

        if last_trading_day_bt != current_date_bt:
            print(f"[SIM-{backtest_current_time}] New trading day ({current_date_bt}). Resetting daily limits for backtest.")
            day_start_equity_bt = current_equity_bt
            daily_high_equity_bt = current_equity_bt
            daily_realized_pnl_bt = 0.0 # Realized PNL resets daily
            trading_halted_drawdown_bt = False
            trading_halted_daily_loss_bt = False
            last_trading_day_bt = current_date_bt
            backtest_trade_log.append({
                "time": backtest_current_time, "type": "NEW_DAY_STATE_RESET_BT",
                "day_start_equity_bt": day_start_equity_bt, "daily_high_equity_bt": daily_high_equity_bt
            })
        else:
            daily_high_equity_bt = max(daily_high_equity_bt, current_equity_bt)
        
        if i % 100 == 0 or last_trading_day_bt != current_date_bt : # Log progress and key daily stats
            print(f"\n[SIM] Candle {i - simulation_start_index + 1}/{len(master_klines_df) - simulation_start_index} | Time: {backtest_current_time}")
            print(f"  Sim Balance: {backtest_simulated_balance:.2f}, Current Equity_BT: {current_equity_bt:.2f}")
            print(f"  Day Start Equity_BT: {day_start_equity_bt:.2f}, Daily High Equity_BT: {daily_high_equity_bt:.2f}, Daily Realized PNL_BT: {daily_realized_pnl_bt:.2f}")
            print(f"  Halt Status_BT: Drawdown: {trading_halted_drawdown_bt}, Daily Loss: {trading_halted_daily_loss_bt}")

        # --- Backtest Max Drawdown Check ---
        max_dd_config_bt = configs.get('max_drawdown_percent', 0.0)
        if max_dd_config_bt > 0 and not trading_halted_drawdown_bt and daily_high_equity_bt > 0:
            drawdown_pct_bt = (daily_high_equity_bt - current_equity_bt) / daily_high_equity_bt * 100
            if drawdown_pct_bt >= max_dd_config_bt:
                trading_halted_drawdown_bt = True
                print(f"[SIM-{backtest_current_time}] !!! MAX DRAWDOWN LIMIT HIT (Backtest) {drawdown_pct_bt:.2f}% >= {max_dd_config_bt:.2f}% !!!")
                backtest_trade_log.append({
                    "time": backtest_current_time, "type": "HALT_MAX_DRAWDOWN_BT",
                    "daily_high_equity_bt": daily_high_equity_bt, "current_equity_bt": current_equity_bt,
                    "drawdown_pct_bt": drawdown_pct_bt, "limit_pct": max_dd_config_bt
                })
                close_all_open_positions_backtest(configs, current_candle_data_map) # Pass map for close prices

        # --- Backtest Daily Stop Loss Check (on realized P&L) ---
        # This check is based on daily_realized_pnl_bt which is updated when trades close.
        daily_sl_config_bt = configs.get('daily_stop_loss_percent', 0.0)
        if daily_sl_config_bt > 0 and not trading_halted_daily_loss_bt and not trading_halted_drawdown_bt and day_start_equity_bt > 0:
            current_loss_pct_bt = (daily_realized_pnl_bt / day_start_equity_bt) * 100
            if current_loss_pct_bt <= -daily_sl_config_bt:
                trading_halted_daily_loss_bt = True
                print(f"[SIM-{backtest_current_time}] !!! DAILY STOP LOSS LIMIT HIT (Backtest) {current_loss_pct_bt:.2f}% <= -{daily_sl_config_bt:.2f}% !!!")
                backtest_trade_log.append({
                    "time": backtest_current_time, "type": "HALT_DAILY_STOP_LOSS_BT",
                    "daily_realized_pnl_bt": daily_realized_pnl_bt, "day_start_equity_bt": day_start_equity_bt,
                    "loss_pct_bt": current_loss_pct_bt, "limit_pct": -daily_sl_config_bt,
                    "reason": "Halting new trades for the day."
                })
        # --- End Backtest Daily State and Halt Checks ---

        # For each symbol:
        # A. Process pending orders (SL/TP triggers) based on current candle's H/L
        # B. Update position P&L (if any open positions)
        # C. Check for new trade entry signals
        # D. Perform dynamic SL/TP adjustments for active trades
        
        # Step A & B: Process pending orders and update P&L for existing positions
        for symbol in list(backtest_simulated_positions.keys()): # Iterate over copy as it might be modified
            if symbol not in all_symbol_historical_data: continue
            
            symbol_klines_so_far = all_symbol_historical_data[symbol].iloc[:i+1] # Data up to current candle
            if symbol_klines_so_far.empty: continue

            current_candle = symbol_klines_so_far.iloc[-1]
            current_candle_data_map[symbol] = current_candle # Store for monitor_active_trades_backtest
            
            # Update P&L for open positions (can be done here or in monitor_active_trades)
            pos = backtest_simulated_positions.get(symbol)
            if pos:
                pnl_unrealized = (current_candle['close'] - pos['entryPrice']) * pos['positionAmt'] if pos['side'] == "LONG" else \
                                 (pos['entryPrice'] - current_candle['close']) * abs(pos['positionAmt'])
                pos['unRealizedProfit'] = pnl_unrealized
                # print(f"[SIM-{backtest_current_time}] {symbol} Pos PnL: {pnl_unrealized:.2f}")


            process_pending_simulated_orders(symbol, current_candle['high'], current_candle['low'], current_candle['close'])

        # Step C: Check for new trade entries
        # This part needs to be careful about threading if we adapt the live `process_symbol_task`
        # For backtesting, sequential processing per symbol is simpler and deterministic.
        for symbol in monitored_symbols:
            if symbol not in all_symbol_historical_data: continue
            
            # Prepare data slice for this symbol: needs at least 202 candles up to current time
            # The slice should end at index `i` (current candle)
            start_idx_for_slice = max(0, i - (201 + 50)) # Ensure enough data for EMAs + some buffer, e.g. 202 for EMAs, +50 for lookbacks if any
                                                       # Original `get_historical_klines` live uses limit=500.
                                                       # The strat needs 202 for EMA200 and validation candles.
            if i < simulation_start_index: continue # Should not happen due to loop range but good check

            klines_slice_for_symbol = all_symbol_historical_data[symbol].iloc[max(0, i - 500 +1) : i+1].copy() # Use a rolling window of 500 candles like live
            if len(klines_slice_for_symbol) < 202: # Minimum needed by strategy
                # print(f"[SIM-{backtest_current_time}] Insufficient data history for {symbol} at this point ({len(klines_slice_for_symbol)}). Skipping entry check.")
                continue
            
            # Call the modified trade entry logic
                # The `client` object passed here is a dummy or the real one but its API calls for orders/balance are mocked/redirected
        # Pass active_trades (the global one, used by backtester) to manage_trade_entry_backtest
        manage_trade_entry_backtest(client, configs, symbol, klines_slice_for_symbol, symbol_info_map, active_trades)

        # Step D: Perform dynamic SL/TP adjustments (after all entries and SL/TP hits for the current candle)
        monitor_active_trades_backtest(configs) # This function uses the global active_trades


    # 3. Print summary report
    print("\n--- Backtest Simulation Finished ---")
    print(f"Initial Balance: {initial_balance_for_report:.2f} USDT")
    print(f"Final Balance: {backtest_simulated_balance:.2f} USDT")
    profit = backtest_simulated_balance - initial_balance_for_report
    profit_pct = (profit / initial_balance_for_report) * 100 if initial_balance_for_report > 0 else 0
    print(f"Total Profit: {profit:.2f} USDT ({profit_pct:.2f}%)")
    
    print(f"\nTotal Trades Logged: {len(backtest_trade_log)}")
    # Further analysis of backtest_trade_log can be done here (e.g., win rate, max drawdown, etc.)
    # For example, count winning/losing trades:
    wins = sum(1 for t in backtest_trade_log if t.get('pnl', 0) > 0 and ("CLOSE" in t.get("type","") or "FILL" in t.get("type","")))
    losses = sum(1 for t in backtest_trade_log if t.get('pnl', 0) < 0 and ("CLOSE" in t.get("type","") or "FILL" in t.get("type","")))
    num_closing_trades = wins + losses
    win_rate = (wins / num_closing_trades * 100) if num_closing_trades > 0 else 0
    print(f"Closing Trades: {num_closing_trades} (Wins: {wins}, Losses: {losses})")
    print(f"Win Rate: {win_rate:.2f}%")

    # Save trade log to a file (optional)
    try:
        df_tradelog = pd.DataFrame(backtest_trade_log)
        log_filename = f"backtest_tradelog_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_tradelog.to_csv(log_filename, index=False)
        print(f"Trade log saved to {log_filename}")
    except Exception as e_log:
        print(f"Error saving trade log: {e_log}")


if __name__ == "__main__":
    main()
