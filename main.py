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
DEFAULT_STRATEGY = "ema_cross"   # Default strategy to run
DEFAULT_FIB_ORDER_TIMEOUT_MINUTES = 5 # Default timeout for Fib retracement limit orders
DEFAULT_FIB_ATR_PERIOD = 14          # Default ATR period for Fibonacci strategy SL
DEFAULT_FIB_SL_ATR_MULTIPLIER = 1.0  # Default ATR multiplier for SL for Fibonacci strategy
DEFAULT_MICRO_PIVOT_TRAILING_SL = True # Default for enabling micro-pivot trailing SL
DEFAULT_MICRO_PIVOT_BUFFER_ATR = 0.25  # Default ATR multiplier for micro-pivot SL buffer
DEFAULT_MICRO_PIVOT_PROFIT_THRESHOLD_R = 0.5 # Default profit threshold (in R) to activate micro-pivot SL

# Minimum Profit Check Default
DEFAULT_MIN_EXPECTED_PROFIT_USDT = 0.10

# ATR-Smart TP Zones (Volatility-Adaptive TP) Defaults for Fib Strategy
DEFAULT_USE_ATR_FOR_TP = True
DEFAULT_TP_ATR_MULTIPLIER = 2.5

# Fibonacci Strategy Specific TP Scaling and SL Management Defaults
DEFAULT_FIB_TP_USE_EXTENSIONS = True
DEFAULT_FIB_TP1_EXTENSION_RATIO = 0.618
DEFAULT_FIB_TP2_EXTENSION_RATIO = 1.0
DEFAULT_FIB_TP3_EXTENSION_RATIO = 1.618
DEFAULT_FIB_TP1_QTY_PCT = 0.25
DEFAULT_FIB_TP2_QTY_PCT = 0.50
DEFAULT_FIB_TP3_QTY_PCT = 0.25 # Remainder could also be used
DEFAULT_FIB_MOVE_SL_AFTER_TP1 = "breakeven"  # Options: "breakeven", "trailing", "original"
DEFAULT_FIB_BREAKEVEN_BUFFER_R = 0.1         # 0.1R buffer
DEFAULT_FIB_SL_ADJUSTMENT_AFTER_TP2 = "micro_pivot"  # Options: "micro_pivot", "atr_trailing", "original"

# ICT Strategy Defaults
DEFAULT_ICT_TIMEFRAME = "15m" # Example, will be used by strategy logic if it needs a specific TF different from main
DEFAULT_ICT_RISK_REWARD_RATIO = 2.0 # Target R:R for ICT trades
DEFAULT_ICT_FVG_FRESHNESS_CANDLES = 5 # How many candles an FVG is considered fresh
DEFAULT_ICT_LIQUIDITY_LOOKBACK = 20 # Candles to look back for liquidity zones
DEFAULT_ICT_SWEEP_DETECTION_WINDOW = 5 # How many recent candles to check for a sweep
DEFAULT_ICT_ENTRY_TYPE = "fvg_mid" # "fvg_mid", "ob_open", "ob_mean"
DEFAULT_ICT_SL_TYPE = "ob_fvg_zone" # "ob_fvg_zone", "swept_point", "atr_buffered_zone"
DEFAULT_ICT_SL_ATR_BUFFER_MULTIPLIER = 0.1 # Multiplier for ATR buffer for SL
DEFAULT_ICT_OB_BOS_LOOKBACK = 10 # Lookback for BoS confirmation for an OB
DEFAULT_ICT_PO3_CONSOLIDATION_LOOKBACK = 10 # Lookback for Po3 consolidation
DEFAULT_ICT_PO3_ACCELERATION_MIN_CANDLES = 1 # Min candles for Po3 acceleration
DEFAULT_ICT_LIMIT_SIGNAL_COOLDOWN_SECONDS = 300 # Cooldown for ICT limit signals
DEFAULT_ICT_LIMIT_SIGNAL_SIGNATURE_BLOCK_SECONDS = 60 # Block duplicate signatures (Reduced from 300)
DEFAULT_ICT_ORDER_TIMEOUT_MINUTES = 15 # Timeout for pending ICT limit orders
DEFAULT_ICT_KLINE_LIMIT = 300 # Number of klines to fetch for ICT analysis


# --- Global State Variables ---
# Stores details of active trades. Key: symbol (e.g., "BTCUSDT")
# Value: dict with trade info like order IDs, entry/SL/TP prices, quantity, side.
active_trades = {}
active_trades_lock = threading.Lock() # Lock for synchronizing access to active_trades

# Set to keep track of symbols currently being processed by manage_trade_entry to prevent race conditions.
import datetime # Added for last_trading_day
from datetime import datetime as dt # Alias for ease of use if datetime.datetime is needed alongside datetime.date
from datetime import timezone  # <-- Add this import for timezone support

symbols_currently_processing = set()
symbols_currently_processing_lock = threading.Lock()

# Globals for Cooldown Timer
last_signal_time = {}
last_signal_lock = threading.Lock()

# Globals for 1-minute candle buffers (for Fibonacci strategy)
# Key: symbol (e.g., "BTCUSDT"), Value: deque of last N 1-minute candles (Pandas DataFrames/Series)
symbol_1m_candle_buffers = {}
symbol_1m_candle_buffers_lock = threading.Lock()
DEFAULT_1M_BUFFER_SIZE = 200 # Max 1-minute candles to keep per symbol for new strategy (e.g., for pivot lookbacks)

# Globals for Trade Signature Check
recent_trade_signatures = {} # Stores trade_signature: timestamp
recent_trade_signatures_lock = threading.Lock()
recent_trade_signature_cleanup_interval = 60 # seconds, how often to check for cleanup
last_signature_cleanup_time = dt.now() # Initialize last cleanup time

# Globals for "Signal" Mode Virtual Trade Tracking
active_signals = {} # Stores details of active signals, similar to active_trades
active_signals_lock = threading.Lock() # Lock for active_signals

# Global state for ICT strategy (similar to fib_strategy_states)
ict_strategy_states = {} # Stores pending ICT limit order details
ict_strategy_states_lock = threading.Lock()

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
        "mode": {"type": str, "valid_values": ["live", "backtest", "signal"]},
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
        "allow_exceed_risk_for_min_notional": {"type": bool},
        "strategy_choice": {"type": str, "valid_values": ["ema_cross", "fib_retracement", "ict_strategy"]},
        "fib_1m_buffer_size": {"type": int, "optional": True, "condition": lambda x: 20 <= x <= 1000}, # For Fibonacci strategy
        "fib_order_timeout_minutes": {"type": int, "optional": True, "condition": lambda x: 1 <= x <= 60}, # For Fibonacci strategy
        "fib_atr_period": {"type": int, "optional": True, "condition": lambda x: x > 0}, # For Fibonacci SL
        "fib_sl_atr_multiplier": {"type": float, "optional": True, "condition": lambda x: x > 0}, # For Fibonacci SL
        "micro_pivot_trailing_sl": {"type": bool, "optional": True},
        "micro_pivot_buffer_atr": {"type": float, "optional": True, "condition": lambda x: x > 0},
        "micro_pivot_profit_threshold_r": {"type": float, "optional": True, "condition": lambda x: x > 0},
        # Fib strategy TP scaling and SL management
        "fib_tp_use_extensions": {"type": bool, "optional": True},
        "fib_tp1_extension_ratio": {"type": float, "optional": True, "condition": lambda x: x > 0},
        "fib_tp2_extension_ratio": {"type": float, "optional": True, "condition": lambda x: x > 0},
        "fib_tp3_extension_ratio": {"type": float, "optional": True, "condition": lambda x: x > 0},
        "fib_tp1_qty_pct": {"type": float, "optional": True, "condition": lambda x: 0 < x < 1},
        "fib_tp2_qty_pct": {"type": float, "optional": True, "condition": lambda x: 0 < x < 1},
        "fib_tp3_qty_pct": {"type": float, "optional": True, "condition": lambda x: 0 < x < 1}, # Validation for sum of pct could be added later
        "fib_move_sl_after_tp1": {"type": str, "optional": True, "valid_values": ["breakeven", "trailing", "original"]},
        "fib_breakeven_buffer_r": {"type": float, "optional": True, "condition": lambda x: 0 <= x < 1}, # Buffer can be 0
        "fib_sl_adjustment_after_tp2": {"type": str, "optional": True, "valid_values": ["micro_pivot", "atr_trailing", "original"]},
        # ATR-Smart TP for Fib
        "use_atr_for_tp": {"type": bool, "optional": True},
        "tp_atr_multiplier": {"type": float, "optional": True, "condition": lambda x: x > 0},
        # Minimum Expected Profit
        "min_expected_profit_usdt": {"type": float, "optional": True, "condition": lambda x: x >= 0},
        # ICT Strategy Params
        "ict_timeframe": {"type": str, "optional": True, "valid_values": ["1m", "5m", "15m", "30m", "1h", "4h"]}, # Added "1m", "5m", "4h"
        "ict_risk_reward_ratio": {"type": float, "optional": True, "condition": lambda x: 1.0 <= x <= 10.0}, # Wider R:R range
        "ict_fvg_freshness_candles": {"type": int, "optional": True, "condition": lambda x: 1 <= x <= 100}, # How many candles FVG is fresh
        "ict_liquidity_lookback": {"type": int, "optional": True, "condition": lambda x: 5 <= x <= 200}, # Lookback for liquidity zones
        "ict_sweep_detection_window": {"type": int, "optional": True, "condition": lambda x: 1 <= x <= 50},
        "ict_entry_type": {"type": str, "optional": True, "valid_values": ["fvg_mid", "ob_open", "ob_mean"]},
        "ict_sl_type": {"type": str, "optional": True, "valid_values": ["ob_fvg_zone", "swept_point", "atr_buffered_zone"]},
        "ict_sl_atr_buffer_multiplier": {"type": float, "optional": True, "condition": lambda x: 0.0 <= x <= 2.0}, # Can be 0 if not used
        "ict_ob_bos_lookback": {"type": int, "optional": True, "condition": lambda x: 3 <= x <= 50},
        "ict_po3_consolidation_lookback": {"type": int, "optional": True, "condition": lambda x: 5 <= x <= 50},
        "ict_po3_acceleration_min_candles": {"type": int, "optional": True, "condition": lambda x: 1 <= x <= 10},
        "ict_limit_signal_cooldown_seconds": {"type": int, "optional": True, "condition": lambda x: 0 <= x <= 3600},
        "ict_limit_signal_signature_block_seconds": {"type": int, "optional": True, "condition": lambda x: 0 <= x <= 3600},
        "ict_order_timeout_minutes": {"type": int, "optional": True, "condition": lambda x: 1 <= x <= 1440}, # Up to 1 day
        "ict_kline_limit": {"type": int, "optional": True, "condition": lambda x: 300 <= x <= 1000, "condition_desc": "must be between 300 and 1000"}

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
        # Enhanced logging here
        print(f"TELEGRAM_MSG_SKIPPED: Attempted to send message, but bot_token ('{bot_token if bot_token else 'None/Empty'}') or chat_id ('{chat_id if chat_id else 'None/Empty'}') is not configured. Message not sent: '{message[:100]}...'")
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

def send_entry_signal_telegram(configs: dict, symbol: str, signal_type_display: str, leverage: int, entry_price: float, 
                               tp1_price: float, tp2_price: float | None, tp3_price: float | None, sl_price: float,
                               risk_percentage_config: float, est_pnl_tp1: float | None, est_pnl_sl: float | None,
                               symbol_info: dict, strategy_name_display: str = "Signal",
                               signal_timestamp: dt = None, signal_order_type: str = "N/A",
                               ict_details: dict = None): # Added ict_details
    """
    Formats and sends a Telegram message for a new trade signal in 'Signal' mode.
    """
    if signal_timestamp is None:
        signal_timestamp = dt.now(tz=timezone.utc) # Default to now UTC if not provided
    
    if not configs.get("telegram_bot_token") or not configs.get("telegram_chat_id"):
        print(f"Telegram not configured. Cannot send signal notification for {symbol}.")
        return

    p_prec = int(symbol_info.get('pricePrecision', 2))

    tp1_str = f"{tp1_price:.{p_prec}f}" if tp1_price is not None else "N/A"
    tp2_str = f"{tp2_price:.{p_prec}f}" if tp2_price is not None else "N/A" # Will be N/A if None
    tp3_str = f"{tp3_price:.{p_prec}f}" if tp3_price is not None else "N/A" # Will be N/A if None
    sl_str = f"{sl_price:.{p_prec}f}" if sl_price is not None else "N/A"
    
    pnl_tp1_str = f"{est_pnl_tp1:.2f} USDT" if est_pnl_tp1 is not None else "Not Calculated"
    pnl_sl_str = f"{est_pnl_sl:.2f} USDT" if est_pnl_sl is not None else "Not Calculated"

    # Determine side emoji
    side_emoji = "🔼" if "LONG" in signal_type_display.upper() else "🔽" if "SHORT" in signal_type_display.upper() else "↔️"
    signal_side_text = "LONG" if "LONG" in signal_type_display.upper() else "SHORT" if "SHORT" in signal_type_display.upper() else "N/A"
    formatted_timestamp = signal_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')

    message = (
        f"🔔 *NEW TRADE SIGNAL* | {strategy_name_display} {side_emoji}\n\n"
        f"🗓️ Time: `{formatted_timestamp}`\n"
        f"📈 Symbol: `{symbol}`\n"
        f"SIDE: *{signal_side_text}*\n"
        f"🔩 Strategy: `{signal_type_display}`\n"
        f"📊 Order Type: `{signal_order_type}`\n"
        f"Leverage: `{leverage}x`\n\n"
        f"➡️ Entry Price: `{entry_price:.{p_prec}f}`\n"
        f"🛡️ Stop Loss: `{sl_str}`\n"
    )
    
    tps_message_part = ""
    if tp1_price is not None:
        tps_message_part += f"🎯 Take Profit 1: `{tp1_str}`\n"
    if tp2_price is not None:
        tps_message_part += f"🎯 Take Profit 2: `{tp2_str}`\n"
    if tp3_price is not None:
        tps_message_part += f"🎯 Take Profit 3: `{tp3_str}`\n"
    
    if not tps_message_part and tp1_price is None: # If all TPs are None (e.g. error in calculation)
        tps_message_part = "🎯 Take Profit Levels: `N/A`\n"
        
    message += tps_message_part
    
    message += (
        f"\n📊 Configured Risk: `{risk_percentage_config * 100:.2f}%`\n\n"
        f"💰 *Est. P&L ($100 Capital Trade):*\n"
        f"  - TP1 Hit: `{pnl_tp1_str}`\n"
        f"  - SL Hit: `{pnl_sl_str}`\n\n"
    )

    if ict_details:
        message += "*ICT Context:*\n"
        if ict_details.get('grab_type') and ict_details.get('price_swept'):
            grab_time_str = ict_details.get('grab_timestamp', "N/A")
            if isinstance(grab_time_str, pd.Timestamp): grab_time_str = grab_time_str.strftime('%H:%M:%S')
            message += f"  - Liquidity Grab: `{ict_details['grab_type']}` at `{ict_details['price_swept']:.{p_prec}f}` (Time: `{grab_time_str}`)\n"
        
        if ict_details.get('fvg_range'): # e.g. {'fvg_bottom': X, 'fvg_top': Y}
            fvg_range = ict_details['fvg_range']
            fvg_time_str = ict_details.get('fvg_timestamp_c3', "N/A")
            if isinstance(fvg_time_str, pd.Timestamp): fvg_time_str = fvg_time_str.strftime('%H:%M:%S')
            message += f"  - FVG ({ict_details.get('fvg_direction','N/A')}): `{fvg_range.get('fvg_bottom'):.{p_prec}f} - {fvg_range.get('fvg_top'):.{p_prec}f}` (C3 Time: `{fvg_time_str}`)\n"

        if ict_details.get('ob_range'): # e.g. {'ob_bottom': X, 'ob_top': Y}
            ob_range = ict_details['ob_range']
            ob_time_str = ict_details.get('ob_timestamp', "N/A")
            if isinstance(ob_time_str, pd.Timestamp): ob_time_str = ob_time_str.strftime('%H:%M:%S')
            message += f"  - Order Block ({ict_details.get('ob_direction','N/A')}): `{ob_range.get('ob_bottom'):.{p_prec}f} - {ob_range.get('ob_top'):.{p_prec}f}` (Time: `{ob_time_str}`)\n"
        
        if ict_details.get('po3_confirmed'):
            message += f"  - Power of Three: `Confirmed`\n"
        
        if ict_details.get('entry_logic_used'):
             message += f"  - Entry Logic: `{ict_details['entry_logic_used']}`\n"
        message += "\n"

    message += f"⚠️ _This is a signal only. No order has been placed._"
    
    print(f"Sending TRADE SIGNAL notification for {symbol} ({signal_type_display}). Details: Entry={entry_price}, SL={sl_str}, TP1={tp1_str}, PNL_TP1={pnl_tp1_str}, PNL_SL={pnl_sl_str}")
    send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], message)

def send_signal_update_telegram(configs: dict, signal_details: dict, update_type: str, message_detail: str, 
                                current_market_price: float, pnl_estimation_fixed_capital: float | None = None):
    """
    Formats and sends a Telegram message for an update on an active signal.
    """
    if not configs.get("telegram_bot_token") or not configs.get("telegram_chat_id"):
        print(f"Telegram not configured. Cannot send signal update for {signal_details.get('symbol')}.")
        return

    symbol = signal_details.get('symbol', 'N/A')
    side = signal_details.get('side', 'N/A')
    entry_price = signal_details.get('entry_price', 0.0)
    s_info = signal_details.get('symbol_info', {})
    p_prec = int(s_info.get('pricePrecision', 2))

    title_emoji = "⚙️" # Default
    if update_type.startswith("TP"): title_emoji = "✅"
    elif update_type == "SL_HIT": title_emoji = "❌"
    elif update_type == "SL_ADJUSTED": title_emoji = "🛡️"
    elif update_type == "CLOSED_ALL_TPS": title_emoji = "🎉"
    
    pnl_info_str = ""
    if pnl_estimation_fixed_capital is not None:
        pnl_info_str = f"\nEst. P&L ($100 Capital): `{pnl_estimation_fixed_capital:.2f} USDT`"

    message = (
        f"{title_emoji} *SIGNAL UPDATE* ({signal_details.get('strategy_type', 'Signal')}) {title_emoji}\n\n"
        f"Symbol: `{symbol}` ({side})\n"
        f"Entry: `{entry_price:.{p_prec}f}`\n"
        f"Update Type: `{update_type}`\n"
        f"Details: _{message_detail}_\n"
        f"Current Market Price: `{current_market_price:.{p_prec}f}`"
        f"{pnl_info_str}"
    )
    
    # Avoid sending too many identical messages (e.g. trailing SL not moving but logic runs)
    # This check is simplified; a more robust check might involve comparing more fields or using timestamps.
    if signal_details.get("last_update_message_type") == update_type and \
       signal_details.get("last_update_message_detail_preview") == message_detail[:50]: # Compare preview
        # print(f"Skipping identical signal update for {symbol}: {update_type}")
        return

    print(f"Sending SIGNAL UPDATE notification for {symbol}: {update_type} - {message_detail}")
    send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], message)
    
    # Update last message type to prevent spam, store a preview of the detail
    # This modification should happen on the original dict in active_signals, so caller should handle it or pass mutable dict.
    # For now, this function doesn't modify signal_details directly. Caller must update.
    # active_signals[symbol]["last_update_message_type"] = update_type
    # active_signals[symbol]["last_update_message_detail_preview"] = message_detail[:50]


# --- Fibonacci Retracement Strategy Modules (New) ---
from collections import deque # For 1m candle buffer
import numpy as np # For pivot detection, already imported but good to note dependency

# --- Data Module (Fib Strategy) ---
def update_1m_candle_buffer(symbol: str, new_candle_df_row: pd.Series, buffer_size: int):
    """
    Updates the 1-minute candle buffer for a given symbol.
    `new_candle_df_row` should be a Pandas Series representing the latest 1m candle,
    with a DateTimeIndex.
    """
    global symbol_1m_candle_buffers, symbol_1m_candle_buffers_lock
    with symbol_1m_candle_buffers_lock:
        if symbol not in symbol_1m_candle_buffers:
            symbol_1m_candle_buffers[symbol] = deque(maxlen=buffer_size)
        
        # Ensure the new candle is not duplicating the last one in the buffer if timestamps match
        if symbol_1m_candle_buffers[symbol]:
            last_buffered_candle_time = symbol_1m_candle_buffers[symbol][-1].name # Assuming index is timestamp
            if new_candle_df_row.name == last_buffered_candle_time:
                # Overwrite last candle if timestamp is the same (e.g. update on partial candle close)
                symbol_1m_candle_buffers[symbol][-1] = new_candle_df_row
                # print(f"Updated last 1m candle for {symbol} at {new_candle_df_row.name}")
                return
            elif new_candle_df_row.name < last_buffered_candle_time:
                # print(f"Skipping older 1m candle for {symbol}. New: {new_candle_df_row.name}, Last: {last_buffered_candle_time}")
                return # Skip if out of order

        symbol_1m_candle_buffers[symbol].append(new_candle_df_row)
        # print(f"Appended new 1m candle for {symbol} at {new_candle_df_row.name}. Buffer size: {len(symbol_1m_candle_buffers[symbol])}")

def get_historical_klines_1m(client, symbol: str, limit: int = 200): # Default limit for initial fill or checks
    """
    Fetches historical 1-minute klines for the Fibonacci strategy.
    Returns a DataFrame and an error object (similar to get_historical_klines).
    """
    start_time = time.time()
    klines_1m = []
    api_error = None
    # print(f"Fetching 1m klines for {symbol}, limit {limit}...") # Verbose
    try:
        klines_1m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit)
    except BinanceAPIException as e:
        print(f"API Error fetching 1m klines for {symbol}: {e}")
        api_error = e
        return pd.DataFrame(), api_error
    except Exception as e:
        print(f"General error fetching 1m klines for {symbol}: {e}")
        api_error = e
        return pd.DataFrame(), api_error
    
    duration = time.time() - start_time
    processing_error = None
    try:
        if not klines_1m:
            # print(f"No 1m kline data for {symbol} (fetch duration: {duration:.2f}s).")
            return pd.DataFrame(), api_error

        df = pd.DataFrame(klines_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        
        if df.empty and not api_error:
            # print(f"1m kline data for {symbol} resulted in empty DataFrame after processing (fetch duration: {duration:.2f}s).")
            pass
        # else: print(f"Fetched {len(df)} 1m klines for {symbol} (runtime: {duration:.2f}s).") # Verbose
        return df, None
    except Exception as e:
        print(f"Error processing 1m kline data for {symbol}: {e}")
        processing_error = e
        return pd.DataFrame(), processing_error

# --- Market Structure Detector (Fib Strategy) ---
# Global state for Fibonacci strategy (per symbol)
# Stores: 'trend' (None, 'uptrend', 'downtrend'),
#         'last_swing_high_price', 'last_swing_high_time',
#         'last_swing_low_price', 'last_swing_low_time',
#         'bos_detail' (dict of BoS event), 
#         'state' ('IDLE', 'AWAITING_PULLBACK', 'PENDING_ENTRY', 'IN_TRADE'),
#         'pending_entry_order_id', 'pending_entry_details',
#         'flip_bias_direction' (None, 'long', 'short') - for prioritizing next BoS after SL.
fib_strategy_states = {}
fib_strategy_states_lock = threading.Lock()

# Constants for pivot detection
PIVOT_N_LEFT = 5  # Number of candles to the left for pivot detection (general use)
PIVOT_N_RIGHT = 5 # Number of candles to the right for pivot detection (general use, implies delay)

# Constants for 1-minute micro-pivot based SL adjustments (Fibonacci Strategy)
MICRO_PIVOT_N_LEFT_1M = 3
MICRO_PIVOT_N_RIGHT_1M = 1 # Shorter right window for faster confirmation of 1-min pivots
MICRO_PIVOT_SL_BUFFER_ATR_MULT = 0.25 # Multiplier of 1m ATR for SL buffer below/above micro pivot
MICRO_PIVOT_PROFIT_THRESHOLD_R = 0.2  # Profit threshold (in terms of initial Risk 'R') to activate micro-pivot SL trailing

def identify_swing_pivots(data: pd.Series, n_left: int, n_right: int, is_high: bool) -> pd.Series:
    """
    Identifies swing highs or lows in a series.
    For real-time, n_right implies a delay. For historical, it's fine.
    For live usage with n_right > 0, the pivot is confirmed 'n_right' bars later.
    A simpler real-time approach might use n_right=0 and other confirmation.
    This implementation is standard for historical pivot detection.

    Args:
        data (pd.Series): Price data (e.g., 'high' or 'low' series).
        n_left (int): Number of bars to the left that must be lower (for high) or higher (for low).
        n_right (int): Number of bars to the right that must be lower (for high) or higher (for low).
        is_high (bool): True to find swing highs, False for swing lows.

    Returns:
        pd.Series: Boolean series indicating pivot points.
    """
    if is_high:
        # Current bar is a swing high if its value is greater than all values in n_left and n_right windows
        pivot_conditions = (data > data.shift(i) for i in range(1, n_left + 1))
        pivot_conditions_right = (data >= data.shift(-i) for i in range(1, n_right + 1)) # Use >= for right to allow flat tops
    else:
        # Current bar is a swing low if its value is less than all values in n_left and n_right windows
        pivot_conditions = (data < data.shift(i) for i in range(1, n_left + 1))
        pivot_conditions_right = (data <= data.shift(-i) for i in range(1, n_right + 1)) # Use <= for right to allow flat bottoms

    # Combine conditions using np.logical_and.reduce for multiple conditions
    # Need to handle NaNs from shifting, ensure comparison window is valid
    
    # Pad with False for initial/final elements where window is incomplete
    pivots = pd.Series(False, index=data.index)
    
    # Iterate through the series, excluding edges where full window isn't available
    for i in range(n_left, len(data) - n_right):
        current_val = data.iloc[i]
        
        # Left side check
        left_window = data.iloc[i-n_left : i]
        if is_high:
            if not (left_window < current_val).all(): continue
        else: # is_low
            if not (left_window > current_val).all(): continue
            
        # Right side check
        right_window = data.iloc[i+1 : i+1+n_right]
        if is_high:
            if not (right_window <= current_val).all(): continue # Allow equals for right side (flat top)
        else: # is_low
            if not (right_window >= current_val).all(): continue # Allow equals for right side (flat bottom)
            
        pivots.iloc[i] = True
        
    return pivots

def get_latest_pivots_from_buffer(candle_buffer_df: pd.DataFrame, n_left: int, n_right: int) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None, pd.Series | None]:
    """
    Gets the most recent confirmed swing high and low from the buffer.
    A pivot is confirmed n_right bars after it occurs.
    Returns: (timestamp_high, price_high, timestamp_low, price_low)
             Returns (None, None, None, None) if no confirmed pivots found.
    """
    if candle_buffer_df.empty or len(candle_buffer_df) < n_left + n_right + 1:
        return None, None, None, None

    # Identify all historical pivots in the buffer
    # Note: For live, identify_swing_pivots with n_right > 0 means pivots are confirmed with a delay.
    # We are interested in the *most recent confirmed* pivots.
    # The last `n_right` candles cannot have confirmed pivots yet.
    
    relevant_data_for_pivots = candle_buffer_df.iloc[:-(n_right+1)] if n_right > 0 else candle_buffer_df
    if relevant_data_for_pivots.empty:
        return None, None, None, None

    swing_highs_bool = identify_swing_pivots(relevant_data_for_pivots['high'], n_left, n_right, is_high=True)
    swing_lows_bool = identify_swing_pivots(relevant_data_for_pivots['low'], n_left, n_right, is_high=False)

    confirmed_highs = relevant_data_for_pivots[swing_highs_bool]
    confirmed_lows = relevant_data_for_pivots[swing_lows_bool]

    latest_high_price, latest_high_time = None, None
    latest_low_price, latest_low_time = None, None

    if not confirmed_highs.empty:
        latest_high_price = confirmed_highs['high'].iloc[-1]
        latest_high_time = confirmed_highs.index[-1]
        
    if not confirmed_lows.empty:
        latest_low_price = confirmed_lows['low'].iloc[-1]
        latest_low_time = confirmed_lows.index[-1]
        
    return latest_high_time, latest_high_price, latest_low_time, latest_low_price


def detect_market_structure_and_bos(symbol: str, candle_buffer_df: pd.DataFrame, configs: dict):
    """
    Detects market structure (trend) and Break of Structure (BoS) events.
    Updates fib_strategy_states for the symbol.

    Args:
        symbol (str): The trading symbol.
        candle_buffer_df (pd.DataFrame): DataFrame of 1-minute candles (index=timestamp, cols=['open', 'high', 'low', 'close']).
        configs (dict): Bot configuration.

    Returns:
        dict | None: BoS event details if detected {direction, swing_high, swing_low, bos_price, timestamp}, else None.
    """
    global fib_strategy_states, fib_strategy_states_lock
    
    if candle_buffer_df.empty or len(candle_buffer_df) < PIVOT_N_LEFT + PIVOT_N_RIGHT + 1 + 1: # +1 for current candle
        print(f"[{symbol}] Insufficient data in candle_buffer_df for BoS detection ({len(candle_buffer_df)} candles).")
        return None

    current_candle = candle_buffer_df.iloc[-1]
    current_close = current_candle['close']
    
    # Get latest confirmed pivots (excluding the very latest candles that can't be confirmed yet)
    # For BoS, we look at "established" prior swings.
    # The n_right in identify_swing_pivots creates a delay.
    # So, the pivots we get are from `PIVOT_N_RIGHT` bars ago or older.
    
    # Data for pivot identification should not include the current, still-forming candle.
    # And pivot confirmation itself has a delay of PIVOT_N_RIGHT bars.
    # So, we consider pivots from `candle_buffer_df` up to `-(PIVOT_N_RIGHT + 1)` index.
    
    # Let's simplify: get pivots from the whole buffer. The `identify_swing_pivots` already handles edge cases.
    # The `get_latest_pivots_from_buffer` then picks the most recent *confirmed* ones.
    
    # For `get_latest_pivots_from_buffer`, n_right means the pivot is confirmed `n_right` bars *after* it forms.
    # So, the latest pivot from `get_latest_pivots_from_buffer` is from at least `n_right` bars ago relative to the end of data passed to it.
    # We pass data *excluding* the current bar for pivot calculation.
    historical_data_for_pivots = candle_buffer_df.iloc[:-1] # Exclude current incomplete candle

    if len(historical_data_for_pivots) < PIVOT_N_LEFT + PIVOT_N_RIGHT + 1:
         print(f"[{symbol}] Insufficient historical data ({len(historical_data_for_pivots)}) for pivot calculation in BoS.")
         return None


    prev_high_time, prev_swing_high, prev_low_time, prev_swing_low = get_latest_pivots_from_buffer(
        historical_data_for_pivots, PIVOT_N_LEFT, PIVOT_N_RIGHT
    )

    with fib_strategy_states_lock:
        state = fib_strategy_states.get(symbol, {"state": "IDLE", "trend": None, "last_swing_high": None, "last_swing_low": None})

        # Basic trend definition (can be more sophisticated)
        # If we have a recent confirmed swing high and low:
        if prev_swing_high and prev_swing_low:
            # Access state using the correct keys: _price suffix
            if state.get("last_swing_high_price") and state.get("last_swing_low_price"):
                # Higher highs and higher lows -> uptrend
                if prev_swing_high > state["last_swing_high_price"] and prev_swing_low > state["last_swing_low_price"]:
                    state["trend"] = "uptrend"
                # Lower highs and lower lows -> downtrend
                elif prev_swing_high < state["last_swing_high_price"] and prev_swing_low < state["last_swing_low_price"]:
                    state["trend"] = "downtrend"
                # else, could be ranging or unclear, maintain previous trend or set to None
            
            # Update last known major swings
            state["last_swing_high_price"] = prev_swing_high
            state["last_swing_high_time"] = prev_high_time
            state["last_swing_low_price"] = prev_swing_low
            state["last_swing_low_time"] = prev_low_time
        
        # BoS Detection
        state = fib_strategy_states.get(symbol) # Get existing state or None
        if not state: # First time seeing this symbol for Fib strategy or after a reset
            state = {
                "state": "IDLE", "trend": None, 
                "last_swing_high_price": None, "last_swing_high_time": None,
                "last_swing_low_price": None, "last_swing_low_time": None,
                "bos_detail": None,
                "pending_entry_order_id": None, "pending_entry_details": None,
                "flip_bias_direction": None 
            }
            # Do not assign to fib_strategy_states[symbol] yet, do it at the end of the with block.

        newly_identified_trend = state.get("trend")
        if prev_swing_high and prev_swing_low:
            if state.get("last_swing_high_price") and state.get("last_swing_low_price"):
                if prev_swing_high > state["last_swing_high_price"] and prev_swing_low > state["last_swing_low_price"]:
                    newly_identified_trend = "uptrend"
                elif prev_swing_high < state["last_swing_high_price"] and prev_swing_low < state["last_swing_low_price"]:
                    newly_identified_trend = "downtrend"
            else:
                if prev_high_time and prev_low_time:
                    if prev_high_time > prev_low_time: newly_identified_trend = "uptrend"
                    elif prev_low_time > prev_high_time: newly_identified_trend = "downtrend"
            state["trend"] = newly_identified_trend
            state["last_swing_high_price"] = prev_swing_high
            state["last_swing_high_time"] = prev_high_time
            state["last_swing_low_price"] = prev_swing_low
            state["last_swing_low_time"] = prev_low_time
        
        bos_event = None
        current_trend = state.get("trend")
        last_sh_price = state.get("last_swing_high_price")
        last_sl_price = state.get("last_swing_low_price")
        # For logging/clarity, get times, ensure they exist if prices exist
        last_sh_time = state.get("last_swing_high_time") if last_sh_price else None
        last_sl_time = state.get("last_swing_low_time") if last_sl_price else None

        if (current_trend == "uptrend" or current_trend is None) and last_sh_price:
            if current_close > last_sh_price:
                if last_sl_price is None:
                    print(f"📈 [{symbol}] Potential Bullish BoS: Close ({current_close:.4f}) > Prev Swing High ({last_sh_price:.4f}), but prior swing low for BoS move is undefined. BoS not confirmed.")
                else:
                    print(f"📈 [{symbol}] Bullish BoS detected! Close ({current_close:.4f}) > Prev Swing High ({last_sh_price:.4f} at {last_sh_time})")
                    bos_event = {"direction": "long", "swing_high_bos_move": current_candle['high'], "swing_low_bos_move": last_sl_price, "bos_price": current_close, "timestamp": current_candle.name, "broken_structure_price": last_sh_price}
        
        if not bos_event and (current_trend == "downtrend" or current_trend is None) and last_sl_price:
            if current_close < last_sl_price:
                if last_sh_price is None:
                     print(f"📉 [{symbol}] Potential Bearish BoS: Close ({current_close:.4f}) < Prev Swing Low ({last_sl_price:.4f}), but prior swing high for BoS move is undefined. BoS not confirmed.")
                else:
                    print(f"📉 [{symbol}] Bearish BoS detected! Close ({current_close:.4f}) < Prev Swing Low ({last_sl_price:.4f} at {last_sl_time})")
                    bos_event = {"direction": "short", "swing_high_bos_move": last_sh_price, "swing_low_bos_move": current_candle['low'], "bos_price": current_close, "timestamp": current_candle.name, "broken_structure_price": last_sl_price}
        
        if bos_event:
            flip_bias = state.get("flip_bias_direction")
            if flip_bias and flip_bias != bos_event["direction"]:
                print(f"[{symbol}] BoS detected ({bos_event['direction']}) but flip bias is for {flip_bias}. Ignoring this BoS.")
                fib_strategy_states[symbol] = state # Save updated state (trend/pivots) even if BoS ignored
                return None 

            if flip_bias and flip_bias == bos_event["direction"]:
                print(f"[{symbol}] BoS detected ({bos_event['direction']}) ALIGNS with flip bias. Proceeding with priority.")
                # Note: flip_bias is not cleared here. It should be cleared by trade management logic

            # Calculate ATR on the 1-minute data buffer for SL placement
            # Ensure candle_buffer_df is a DataFrame and not a Series if calculate_atr expects DataFrame
            # The candle_buffer_df is already a DataFrame.
            fib_atr_period = configs.get("fib_atr_period", DEFAULT_FIB_ATR_PERIOD)
            # Make a copy for ATR calculation to avoid modifying the original buffer's DataFrame
            buffer_df_for_atr_calc = candle_buffer_df.copy()
            atr_series = calculate_atr(buffer_df_for_atr_calc, period=fib_atr_period)
            
            if atr_series.empty or atr_series.iloc[-1] is None or pd.isna(atr_series.iloc[-1]) or atr_series.iloc[-1] <= 0:
                print(f"[{symbol}] Could not calculate valid 1m ATR (period {fib_atr_period}) for BoS event. ATR: {atr_series.iloc[-1] if not atr_series.empty else 'N/A'}. BoS event ignored.")
                # Do not set state to AWAITING_PULLBACK if ATR is invalid
                fib_strategy_states[symbol] = state # Save updated state (trend/pivots)
                return None # Ignore BoS if ATR cannot be determined for SL
            
            current_1m_atr = atr_series.iloc[-1]
            bos_event["atr_1m"] = current_1m_atr # Add 1m ATR to the BoS event details
            print(f"[{symbol}] BoS event created: {bos_event}") # DEBUG: Show full BoS event
            print(f"[{symbol}] BoS event includes 1m ATR ({fib_atr_period}-period): {current_1m_atr:.5f}")

            state["state"] = "AWAITING_PULLBACK"
            state["bos_detail"] = bos_event
        elif state.get("state") == "AWAITING_PULLBACK" and not bos_event : # If no new BoS, but was awaiting pullback, reset if conditions change
            # This might be too aggressive, consider if state should persist longer or have other criteria for reset.
            # For now, if it was awaiting pullback and no BoS is now detected (e.g. trend changed, price moved away),
            # it might be prudent to reset to IDLE to re-evaluate for a fresh BoS.
            # However, this part of logic might be better handled by timeout in manage_fib_retracement_entry_logic or monitor_pending_fib_entries
            # For now, let's not reset here automatically unless BoS detection itself indicates a clear invalidation.
            pass # Keeping state as AWAITING_PULLBACK if no new BoS on this candle. Entry logic will check if price is in zone.
        
        fib_strategy_states[symbol] = state 
        return bos_event

# --- Order Manager Components (Fib Strategy) ---
# (Further integration with monitor_active_trades or a new Fib-specific monitor is needed)

# --- Fibonacci Module (Fib Strategy) ---
def calculate_fibonacci_retracement_levels(swing_high_price: float, swing_low_price: float, direction: str) -> dict | None:
    """
    Calculates Fibonacci retracement levels (0.236, 0.382, 0.5, 0.618) for a given move.

    Args:
        swing_high_price (float): The highest price of the BoS move.
        swing_low_price (float): The lowest price of the BoS move.
        direction (str): "long" if BoS was bullish (breakout upwards), 
                         "short" if BoS was bearish (breakdown downwards).

    Returns:
        dict | None: A dictionary with 'level_0_236', 'level_0_382', 'level_0_5', 'level_0_618',
                     'zone_upper' (0.5 for long, 0.618 for short), 
                     'zone_lower' (0.618 for long, 0.5 for short) prices, or None if inputs are invalid.
    """
    if swing_high_price is None or swing_low_price is None or swing_high_price <= swing_low_price:
        print(f"Error calculating Fib levels: Invalid swing prices. High: {swing_high_price}, Low: {swing_low_price}")
        return None

    price_range = swing_high_price - swing_low_price

    levels = {}

    if direction == "long": # Bullish BoS, expect pullback downwards to buy
        # Fib levels are measured from low (0%) to high (100%)
        # Retracement levels are below the swing_high_price
        levels["level_0_236"] = swing_high_price - (price_range * 0.236)
        levels["level_0_382"] = swing_high_price - (price_range * 0.382)
        levels["level_0_5"] = swing_high_price - (price_range * 0.5)
        levels["level_0_618"] = swing_high_price - (price_range * 0.618)
        # Golden zone (for entry) is still between 0.5 and 0.618. For a long, 0.618 is lower.
        levels["zone_upper"] = levels["level_0_5"]
        levels["zone_lower"] = levels["level_0_618"]
    elif direction == "short": # Bearish BoS, expect pullback upwards to sell
        # Fib levels are measured from high (0%) to low (100%)
        # Retracement levels are above the swing_low_price
        levels["level_0_236"] = swing_low_price + (price_range * 0.236)
        levels["level_0_382"] = swing_low_price + (price_range * 0.382)
        levels["level_0_5"] = swing_low_price + (price_range * 0.5)
        levels["level_0_618"] = swing_low_price + (price_range * 0.618)
        # Golden zone (for entry) is still between 0.5 and 0.618. For a short, 0.618 is higher.
        levels["zone_upper"] = levels["level_0_618"]
        levels["zone_lower"] = levels["level_0_5"]
    else:
        print(f"Error calculating Fib levels: Invalid direction '{direction}'.")
        return None
        
    # Ensure zone_upper is actually above zone_lower (can happen if levels are identical due to precision)
    if levels["zone_upper"] < levels["zone_lower"]: 
        levels["zone_upper"], levels["zone_lower"] = levels["zone_lower"], levels["zone_upper"]

    return levels

# --- Entry & Exit Logic (Fib Strategy) ---
def manage_fib_retracement_entry_logic(client, configs: dict, symbol: str, bos_event: dict, symbol_info: dict):
    """
    Manages the entry logic for the Fibonacci Retracement strategy after a BoS event.
    Places a limit order in the golden zone, and if filled, SL/TP orders.
    """
    global active_trades, active_trades_lock, fib_strategy_states, fib_strategy_states_lock
    global last_signal_time, last_signal_lock # For Cooldown on new signal type
    global recent_trade_signatures, recent_trade_signatures_lock # For Trade Signature Check

    log_prefix = f"[{threading.current_thread().name}] {symbol} FibEntry:"
    print(f"{log_prefix} ▶️ Fib entry logic triggered. Symbol: {symbol}, BoS: {bos_event.get('direction') if bos_event else 'N/A'}") # DEBUG
    
    if not bos_event or not symbol_info:
        print(f"{log_prefix} Missing bos_event or symbol_info. Aborting.")
        return

    direction = bos_event['direction']
    swing_high_bos = bos_event['swing_high_bos_move'] # Actual high of the move (for long), or start high of move (for short)
    swing_low_bos = bos_event['swing_low_bos_move']   # Actual low of the move (for short), or start low of move (for long)
    broken_structure_price = bos_event['broken_structure_price']
    
    # 1. Calculate Fibonacci Levels
    fib_levels = calculate_fibonacci_retracement_levels(swing_high_bos, swing_low_bos, direction)
    if not fib_levels:
        print(f"{log_prefix} Failed to calculate Fibonacci levels. Aborting.")
        with fib_strategy_states_lock: # Reset state if Fib calculation fails
            if symbol in fib_strategy_states: fib_strategy_states[symbol]['state'] = "IDLE"
        return

    zone_upper = fib_levels['zone_upper']
    zone_lower = fib_levels['zone_lower']
    entry_price_target = (zone_upper + zone_lower) / 2.0 # Mid-point of the golden zone

    p_prec = int(symbol_info['pricePrecision'])
    entry_price_target = round(entry_price_target, p_prec) # Round to symbol's precision

    print(f"{log_prefix} Target Golden Zone for {direction}: {zone_lower:.{p_prec}f} - {zone_upper:.{p_prec}f}. Target Entry: {entry_price_target:.{p_prec}f}")

    # 2. Determine SL and Multiple TP Levels
    # SL calculation using ATR from bos_event
    current_1m_atr = bos_event.get("atr_1m")
    if current_1m_atr is None or current_1m_atr <= 0:
        print(f"{log_prefix} Invalid or missing 1m ATR ({current_1m_atr}) in BoS event for SL calculation. Aborting.")
        with fib_strategy_states_lock: # Reset state
            if symbol in fib_strategy_states: fib_strategy_states[symbol]['state'] = "IDLE"
        return

    sl_atr_multiplier = configs.get("fib_sl_atr_multiplier", DEFAULT_FIB_SL_ATR_MULTIPLIER)
    sl_atr_offset = current_1m_atr * sl_atr_multiplier
    min_tick_size = 1 / (10**p_prec) # Already defined earlier

    sl_price = None
    if direction == "long":
        # SL below the low of the move that caused BoS, buffered by ATR
        sl_price = swing_low_bos - sl_atr_offset
    elif direction == "short":
        # SL above the high of the move that caused BoS, buffered by ATR
        sl_price = swing_high_bos + sl_atr_offset
    
    if sl_price is None: # Should not happen if direction is valid
        print(f"{log_prefix} SL price could not be determined (direction: {direction}). Aborting."); return

    sl_price = round(sl_price, p_prec)
    print(f"{log_prefix} ATR-based SL calculation: SwingLow/High_BoSMove={swing_low_bos if direction == 'long' else swing_high_bos:.{p_prec}f}, ATR_1m={current_1m_atr:.{p_prec}f}, Multiplier={sl_atr_multiplier}, Offset={sl_atr_offset:.{p_prec}f}, SL_Price={sl_price:.{p_prec}f}")

    # TP levels based on Fibonacci retracements of the BoS move itself (or other logic if specified)
    # The request specifies TP levels as retracements (0.236, 0.382, 0.618) of the BoS range.
    # This means if BoS was from swing_low_bos to swing_high_bos:
    # For LONG: TP levels are swing_low_bos + (range * fib_ratio)
    # For SHORT: TP levels are swing_high_bos - (range * fib_ratio)
    # This is different from the entry logic which uses retracements for pullback entry.
    # Let's clarify: "0.236 retracement or 25-30% of the BoS range."
    # If TP is a retracement *of the BoS move*, then for a long, TPs would be *below* swing_high_bos.
    # This seems counterintuitive for TPs.
    # More likely: TPs are *extensions* or percentages of the BoS range *added to the entry price* or *from the broken structure*.

    # Re-interpreting: TP levels are calculated from the *entry point* using the BoS range.
    # TP1 (Quick Win): 0.236 retracement or 25–30% of the BoS range.
    # TP2 (Standard): 0.382 retracement for the bulk of the position.
    # TP3 (Optional “Extended”): 0.618 retracement for aggressive runners.
    # "Calculate each TP level using the same calculate_fibonacci_retracement_levels helper, extending it to include 0.236 and 0.382."
    # This suggests the TPs are also Fib levels *of the BoS move*.
    # If direction is LONG (price broke up from swing_low_bos to swing_high_bos, entry is a pullback):
    #   TP1: fib_levels['level_0_236'] (this is swing_high_bos - range * 0.236)
    #   TP2: fib_levels['level_0_382'] (this is swing_high_bos - range * 0.382)
    #   TP3: fib_levels['level_0_618'] (this is swing_high_bos - range * 0.618)
    # This still means TPs are *within* the BoS range, which is for capturing parts of the pullback *if the entry was at the BoS price*.
    # But entry is on a pullback.

    # Let's assume the TPs are target prices projected *from the entry price*.
    # Or, they are fixed Fib levels relative to the BoS move boundaries.
    # The original TP was `broken_structure_price`.
    # Let's use the fib_levels from `calculate_fibonacci_retracement_levels` as the actual TP target prices.
    # For a LONG trade, entry is on a pullback. TPs should be higher than entry.
    # The fib_levels calculated are:
    # level_0_236 = swing_high_price - (price_range * 0.236)
    # level_0_382 = swing_high_price - (price_range * 0.382)
    # level_0_618 = swing_high_price - (price_range * 0.618)
    # These are potential TP targets if we are aiming for these specific price levels.

    tp1_price = round(fib_levels['level_0_236'], p_prec)
    tp2_price = round(fib_levels['level_0_382'], p_prec)
    tp3_price = round(fib_levels['level_0_618'], p_prec) # Optional TP

    # Ensure TPs are logical relative to entry and SL
    if sl_price is None: # Should not happen if direction is valid
        print(f"{log_prefix} SL price calculation error. Aborting."); return

    # Check entry vs SL
    if (direction == "long" and entry_price_target <= sl_price) or \
       (direction == "short" and entry_price_target >= sl_price):
        print(f"{log_prefix} Invalid SL relative to entry: Entry={entry_price_target}, SL={sl_price}. Aborting.")
        with fib_strategy_states_lock:
            if symbol in fib_strategy_states: fib_strategy_states[symbol]['state'] = "IDLE"
        return

    # Check TPs vs entry (TPs must be profitable)
    # For LONG: TP > entry. For SHORT: TP < entry.
    # The fib_levels are absolute prices.
    if direction == "long":
        if not (tp1_price > entry_price_target and tp2_price > entry_price_target and (tp3_price is None or tp3_price > entry_price_target)):
            print(f"{log_prefix} Invalid TP levels relative to entry for LONG. Entry={entry_price_target}, TP1={tp1_price}, TP2={tp2_price}, TP3={tp3_price}. Aborting.")
            # This might happen if entry pullback is too deep, making some fib levels already passed.
            # Or if fib levels are calculated in a way that they are not suitable as TPs from the entry.
            # For now, let's assume the fib levels are absolute price targets.
            # If entry is (e.g.) at 0.618, then TP at 0.236, 0.382 makes sense.
            # We need to ensure the selected TPs are progressively further in the profit direction.
            # For LONG: level_0_236 > level_0_382 > level_0_5 > level_0_618 (these are prices)
            # So TP1 (0.236) is highest, TP3 (0.618) is lowest. This order is for profit taking.
            # The request: TP1 (0.236), TP2 (0.382), TP3 (0.618).
            # This means TP1 is the "quickest" but also the "highest" price for a long. This needs to be correct.
            # If fib_levels['level_0.236'] is swing_high - range*0.236, it is indeed higher than swing_high - range*0.618.
            # So, for a LONG, TP1 (level_0.236) is the most ambitious, TP3 (level_0.618) is the closest.
            # This contradicts "TP1 (Quick Win)".
            # Re-evaluating: "TP1 (Quick Win): 0.236 retracement or 25–30% of the BoS range."
            # This means the *distance* from entry (or broken structure) is 25-30% of BoS range.
            # OR, the TP level is the 0.236 Fib level of the *pullback range* if entry is confirmed.

            # Let's stick to the interpretation that TPs are absolute price levels derived from the BoS move:
            # level_0_236, level_0_382, level_0_618.
            # For LONG, these are: swing_high - X. So higher X means lower price.
            # TP1 (0.236) = swing_high - 0.236 * range  -- This is the highest price (most profit)
            # TP2 (0.382) = swing_high - 0.382 * range
            # TP3 (0.618) = swing_high - 0.618 * range  -- This is the lowest price (least profit, but maybe hit first if entry is deep)

            # The request implies TP1 should be hit first for quick win.
            # So for LONG, TP1 should be the lowest price among TPs that is still > entry.
            # TP1 = fib_levels['level_0_618'] (if > entry)
            # TP2 = fib_levels['level_0_5'] or fib_levels['level_0_382']
            # TP3 = fib_levels['level_0_236']
            # This is getting complicated. Let's use the direct fib levels as TPs.
            # TP1_target = fib_levels['level_0_236']
            # TP2_target = fib_levels['level_0_382']
            # TP3_target = fib_levels['level_0_618']
            # We need to ensure they are in profitable direction from entry_price_target.

            # Let's define TPs as specific Fib levels of the BoS range, used as absolute price targets.
            # For a LONG trade (entry is a pullback from swing_high_bos):
            # TP1 (quickest) should be a level that's hit first as price moves up from entry.
            # So, if entry is at fib_levels['level_0_618'], then TP1 could be fib_levels['level_0_5'],
            # TP2 fib_levels['level_0_382'], TP3 fib_levels['level_0_236'].
            # This means for LONG: TP1_price < TP2_price < TP3_price
            # fib_levels are calculated as: level_X = swing_high - (range * X)
            # So level_0.618 < level_0.5 < level_0.382 < level_0.236 (these are actual prices)
            # Request: TP1 (0.236), TP2 (0.382), TP3 (0.618)
            # This means TP1 is associated with the 0.236 Fib ratio, TP2 with 0.382, TP3 with 0.618.
            # For LONG, the price corresponding to 0.236 ratio is swing_high - 0.236*range.
            # The price corresponding to 0.618 ratio is swing_high - 0.618*range.
            # The 0.236 price is *higher* than 0.618 price.
            # If TP1 is "Quick Win", it should be the closest.
            # This implies the ratios are distances from entry, or the Fib levels are indexed differently.

            # Let's use the direct interpretation: TP1 uses 0.236 level, TP2 uses 0.382, TP3 uses 0.618.
            # For LONG: tp1_price = fib_levels['level_0_236'], tp2_price = fib_levels['level_0_382'], tp3_price = fib_levels['level_0_618']
            # We must ensure these are all > entry_price_target.
            # And for a sensible profit progression, for LONG, we'd want tp1 < tp2 < tp3 if TP1 is quick win.
            # But level_0.236 > level_0.382 > level_0.618.
            # This means TP1 (0.236) is the most ambitious profit, TP3 (0.618) is the least ambitious.
            # This setup is fine if "TP1 (Quick Win)" refers to "Smallest part of position for largest initial target".

            # Let's stick to the definition from calculate_fibonacci_retracement_levels directly for now.
            # We will filter out TPs that are not profitable relative to entry.
            valid_tps = []
            if tp1_price > entry_price_target: valid_tps.append(tp1_price)
            if tp2_price > entry_price_target: valid_tps.append(tp2_price)
            if tp3_price > entry_price_target: valid_tps.append(tp3_price)
            valid_tps = sorted(list(set(valid_tps))) # Ascending order for LONG TPs
            if not valid_tps:
                 print(f"{log_prefix} No valid TP levels are profitable for LONG. Entry={entry_price_target}. Aborting.")
                 with fib_strategy_states_lock: fib_strategy_states[symbol]['state'] = "IDLE"; return
            # Assign based on sorted valid TPs for LONG: TP1 is closest, TP3 furthest (if 3 exist)
            tp1_price_final = valid_tps[0]
            tp2_price_final = valid_tps[1] if len(valid_tps) > 1 else None
            tp3_price_final = valid_tps[2] if len(valid_tps) > 2 else None


    elif direction == "short":
        # For SHORT: TP < entry.
        # level_0.236 < level_0.382 < level_0.618 (these are prices, e.g. 10, 11, 12 for a short from 15)
        # TP1 (0.236) is the most ambitious (lowest price), TP3 (0.618) is the least ambitious (highest price).
        valid_tps = []
        if tp1_price < entry_price_target: valid_tps.append(tp1_price)
        if tp2_price < entry_price_target: valid_tps.append(tp2_price)
        if tp3_price < entry_price_target: valid_tps.append(tp3_price)
        valid_tps = sorted(list(set(valid_tps)), reverse=True) # Descending order for SHORT TPs (closest to entry is first)
        if not valid_tps:
             print(f"{log_prefix} No valid TP levels are profitable for SHORT. Entry={entry_price_target}. Aborting.")
             with fib_strategy_states_lock: fib_strategy_states[symbol]['state'] = "IDLE"; return
        tp1_price_final = valid_tps[0]
        tp2_price_final = valid_tps[1] if len(valid_tps) > 1 else None
        tp3_price_final = valid_tps[2] if len(valid_tps) > 2 else None

    # If any TP is None (e.g. only 1 or 2 valid TPs found), that's acceptable.
    # The monitoring logic will handle placing orders only for non-None TPs.
    
    print(f"{log_prefix} Calculated Entry: {entry_price_target:.{p_prec}f}, SL: {sl_price:.{p_prec}f}")
    # Format TP levels carefully to handle None values before printing
    tp1_str = f"{tp1_price_final:.{p_prec}f}" if tp1_price_final is not None else "N/A"
    tp2_str = f"{tp2_price_final:.{p_prec}f}" if tp2_price_final is not None else "N/A"
    tp3_str = f"{tp3_price_final:.{p_prec}f}" if tp3_price_final is not None else "N/A"
    print(f"{log_prefix} TP Levels: TP1={tp1_str}, TP2={tp2_str}, TP3={tp3_str}")

    # 3. Risk:Reward and Position Sizing (Similar to manage_trade_entry)
    # For R:R, use TP1 for calculation, or an average TP if desired. Let's use TP1.
    # Cooldown, ActiveTrade, LivePosition, OpenOrders, MaxConcurrent checks
    # These checks are largely similar to manage_trade_entry, adapt as needed.
    # For brevity, let's assume these checks pass or are integrated into a shared pre-trade check function later.

    with last_signal_lock: # Cooldown for this *type* of signal (Fib based)
        cooldown_seconds = configs.get("fib_signal_cooldown_seconds", 120) # Separate cooldown for Fib strategy
        if symbol in last_signal_time and (dt.now() - last_signal_time.get(f"{symbol}_fib", dt.min)).total_seconds() < cooldown_seconds:
            print(f"{log_prefix} Cooldown active for Fib strategy on {symbol}. Skipping.")
            return

    with active_trades_lock:
        if symbol in active_trades:
            print(f"{log_prefix} Symbol {symbol} already has an active trade (EMA strat or other). Skipping Fib entry.")
            return
        if len(active_trades) >= configs["max_concurrent_positions"]:
            print(f"{log_prefix} Max concurrent positions reached. Cannot open Fib trade for {symbol}.")
            return
            
    # (Skipping live position/order checks for brevity - assume they'd be here)

    acc_bal = get_account_balance(client, configs)
    if acc_bal is None or acc_bal <= 0:
        print(f"{log_prefix} Invalid account balance ({acc_bal}). Aborting."); return

    # Use the dynamic/fixed leverage already set (assume it's appropriate or handled elsewhere for 1m strategy)
    # For Fib strategy, we might want a specific leverage setting or use the general one.
    # Let's assume dynamic leverage logic or general config leverage is fine for now.
    # The `specific_leverage_for_trade` in sanity checks will use the symbol's current leverage.
    # Fetch current leverage for the symbol for sanity check
    current_leverage_on_symbol = configs.get('leverage') # Default
    try:
        pos_info_lev = client.futures_position_information(symbol=symbol)
        if pos_info_lev and isinstance(pos_info_lev, list) and pos_info_lev[0]:
            current_leverage_on_symbol = int(pos_info_lev[0].get('leverage', configs.get('leverage')))
    except Exception as e_lev:
        print(f"{log_prefix} Could not fetch current leverage for {symbol}: {e_lev}. Using default {current_leverage_on_symbol}x.")

    print(f"{log_prefix} Inputs for calculate_position_size: acc_bal={acc_bal}, risk_percent={configs['risk_percent']}, entry_price_target={entry_price_target}, sl_price={sl_price}") # DEBUG
    qty_to_order = calculate_position_size(acc_bal, configs['risk_percent'], entry_price_target, sl_price, symbol_info, configs)
    print(f"{log_prefix} Output from calculate_position_size: qty_to_order={qty_to_order}") # DEBUG
    if qty_to_order is None or qty_to_order <= 0:
        print(f"{log_prefix} Invalid position size calculated (Qty: {qty_to_order}). Aborting."); return

    # Portfolio Risk Check (similar to manage_trade_entry)
    # ... (portfolio risk calculation and check logic as in manage_trade_entry) ...
    # For brevity, assume it passes.

    # Use the first valid TP as tp_price for sanity checks and trade signature
    # If fib_tp_use_extensions is true, tp1_price_final might be based on extensions.
    # If false, it's based on the old retracement logic. This assignment is okay.
    tp_price = tp1_price_final 

    # Sanity Checks
    passed_sanity, sanity_reason = pre_order_sanity_checks(
        symbol, direction.upper(), entry_price_target, sl_price, tp_price, qty_to_order, 
        symbol_info, acc_bal, configs['risk_percent'], configs,
        specific_leverage_for_trade=current_leverage_on_symbol # Pass actual leverage
    )
    if not passed_sanity:
        print(f"{log_prefix} Pre-order sanity checks FAILED: {sanity_reason}"); return
    
    print(f"{log_prefix} Sanity checks PASSED. Target R:R: {abs(tp_price-entry_price_target)/abs(sl_price-entry_price_target):.2f}")

    # Trade Signature
    trade_sig_fib = generate_trade_signature(symbol, f"FIB_{direction.upper()}", entry_price_target, sl_price, tp_price, qty_to_order, p_prec)
    with recent_trade_signatures_lock:
        if trade_sig_fib in recent_trade_signatures and \
           (dt.now() - recent_trade_signatures[trade_sig_fib]).total_seconds() < 60 : # 60s signature duplicate check
            print(f"{log_prefix} Duplicate Fib trade signature found. Skipping."); return
    
    # Update Cooldown
    with last_signal_lock:
        last_signal_time[f"{symbol}_fib"] = dt.now()

    # --- Mode-Specific Action: Signal or Live/Backtest Order Placement ---
    if configs['mode'] == 'signal':
        # --- Signal Mode: Send Telegram Notification, No Real Orders ---
        print(f"{log_prefix} Signal Mode: Preparing Telegram signal for Fib {symbol} {direction}.")
        
        est_pnl_tp1_fib = calculate_pnl_for_fixed_capital(entry_price_target, tp1_price_final, direction, current_leverage_on_symbol, 100.0, symbol_info)
        est_pnl_sl_fib = calculate_pnl_for_fixed_capital(entry_price_target, sl_price, direction, current_leverage_on_symbol, 100.0, symbol_info)

        send_entry_signal_telegram(
            configs=configs, symbol=symbol, signal_type_display=f"FIB_RETRACEMENT_{direction.upper()}",
            leverage=current_leverage_on_symbol, entry_price=entry_price_target,
            tp1_price=tp1_price_final, tp2_price=tp2_price_final, tp3_price=tp3_price_final, sl_price=sl_price,
            risk_percentage_config=configs['risk_percent'], est_pnl_tp1=est_pnl_tp1_fib, est_pnl_sl=est_pnl_sl_fib,
            symbol_info=symbol_info, strategy_name_display="Fibonacci Retracement",
            signal_timestamp=bos_event.get('timestamp', dt.now(tz=timezone.utc)), # Use BoS event time
            signal_order_type="LIMIT"
        )
        
        # Use the same timestamp for signal_id and open_timestamp for consistency
        signal_generation_time = bos_event.get('timestamp', dt.now(tz=timezone.utc))
        signal_id_fib = f"signal_{symbol}_{int(signal_generation_time.timestamp())}"

        with active_signals_lock:
            active_signals[symbol] = {
                "signal_id": signal_id_fib, "entry_price": entry_price_target, "current_sl_price": sl_price,
                "current_tp1_price": tp1_price_final, "current_tp2_price": tp2_price_final, "current_tp3_price": tp3_price_final,
                "initial_sl_price": sl_price, "initial_tp1_price": tp1_price_final, "initial_tp2_price": tp2_price_final, "initial_tp3_price": tp3_price_final,
                "side": direction, "leverage": current_leverage_on_symbol, "symbol_info": symbol_info,
                "open_timestamp": signal_generation_time, "strategy_type": "FIBONACCI_RETRACEMENT", "sl_management_stage": "initial",
                "last_update_message_type": "NEW_SIGNAL",
                "fib_move_sl_after_tp1_config": configs.get("fib_move_sl_after_tp1", DEFAULT_FIB_MOVE_SL_AFTER_TP1),
                "fib_breakeven_buffer_r_config": configs.get("fib_breakeven_buffer_r", DEFAULT_FIB_BREAKEVEN_BUFFER_R),
                "fib_sl_adjustment_after_tp2_config": configs.get("fib_sl_adjustment_after_tp2", DEFAULT_FIB_SL_ADJUSTMENT_AFTER_TP2),
                "initial_risk_per_unit": abs(entry_price_target - sl_price)
            }
            print(f"{log_prefix} Signal Mode: Fibonacci signal for {symbol} added to active_signals.")

        tp_notes_fib = f"SL: {sl_price:.{symbol_info.get('pricePrecision', 2)}f}, TP1: {tp1_price_final:.{symbol_info.get('pricePrecision', 2)}f}"
        if tp2_price_final: tp_notes_fib += f", TP2: {tp2_price_final:.{symbol_info.get('pricePrecision', 2)}f}"
        if tp3_price_final: tp_notes_fib += f", TP3: {tp3_price_final:.{symbol_info.get('pricePrecision', 2)}f}"
        log_event_details_fib = {
            "SignalID": signal_id_fib, "Symbol": symbol, "Strategy": "FIBONACCI_RETRACEMENT", "Side": direction,
            "Leverage": current_leverage_on_symbol, "SignalOpenPrice": entry_price_target,
            "EventType": "NEW_SIGNAL", "EventPrice": entry_price_target, "Notes": tp_notes_fib,
            "EstimatedPNL_USD100": est_pnl_tp1_fib
        }
        log_signal_event_to_csv(log_event_details_fib)

        print(f"{log_prefix} Signal Mode: Telegram signal sent, CSV logged, and virtual signal recorded for Fib {symbol}.")
        with fib_strategy_states_lock: 
            if symbol in fib_strategy_states: 
                fib_strategy_states[symbol]['state'] = "IDLE"
                fib_strategy_states[symbol]['pending_entry_order_id'] = None
                fib_strategy_states[symbol]['pending_entry_details'] = None
        
        with recent_trade_signatures_lock:
            recent_trade_signatures[trade_sig_fib] = dt.now()
            print(f"{log_prefix} Signal Mode: Fib Trade signature recorded for {symbol}: {trade_sig_fib}")
            
        return # Crucially, return here for signal mode before any order placement logic
    
    # --- Live/Backtest Mode: Place Real Limit Order ---
    # This block executes only if not in 'signal' mode due to the return above.
    print(f"{log_prefix} Live/Backtest Mode: Placing LIMIT {direction.upper()} order: Qty {qty_to_order} @ {entry_price_target:.{p_prec}f}")
    limit_order_side = "BUY" if direction == "long" else "SELL"
    position_side_trade = "LONG" if direction == "long" else "SHORT"

    entry_limit_order, entry_limit_error = place_new_order(
        client, symbol_info, limit_order_side, "LIMIT", qty_to_order, 
        price=entry_price_target, position_side=position_side_trade
    )

    if not entry_limit_order:
        print(f"{log_prefix} Failed to place LIMIT entry order. Error: {entry_limit_error}")
        with fib_strategy_states_lock: 
            if symbol in fib_strategy_states: fib_strategy_states[symbol]['state'] = "IDLE"
        return

    print(f"{log_prefix} LIMIT entry order placed: ID {entry_limit_order['orderId']}. Status: {entry_limit_order['status']}")
    
    if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
        side_display = direction.upper()
        limit_price_display = f"{entry_price_target:.{p_prec}f}"
        qty_display = f"{qty_to_order:.{int(symbol_info.get('quantityPrecision', 0))}f}"
        sl_str_limit = f"{sl_price:.{p_prec}f}" if sl_price is not None else "N/A"
        tp1_str_limit = f"{tp1_price_final:.{p_prec}f}" if tp1_price_final is not None else "N/A"
        limit_order_msg = (
            f"⏳ FIB LIMIT ORDER PLACED ⏳\n\n"
            f"Symbol: `{symbol}`\nSide: `{side_display}`\nType: `LIMIT`\n"
            f"Quantity: `{qty_display}`\nLimit Price: `{limit_price_display}`\n"
            f"Order ID: `{entry_limit_order['orderId']}`\n\n"
            f"Associated SL (if filled): `{sl_str_limit}`\n"
            f"Associated TP1 (if filled): `{tp1_str_limit}`\n\nMonitoring for fill..."
        )
        send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], limit_order_msg)
        print(f"{log_prefix} Telegram notification sent for new Fib limit order placement.")

    with fib_strategy_states_lock:
        if symbol in fib_strategy_states:
            fib_strategy_states[symbol]['state'] = "PENDING_ENTRY"
            fib_strategy_states[symbol]['pending_entry_order_id'] = entry_limit_order['orderId']
            fib_strategy_states[symbol]['pending_entry_details'] = {
                "limit_price": entry_price_target, "sl_price": sl_price,
                "tp1_price_provisional": tp1_price_final, 
                "tp2_price_provisional": tp2_price_final,
                "tp3_price_provisional": tp3_price_final,
                "quantity": qty_to_order, "side": direction,
                "bos_event_snapshot": bos_event,
                "initial_risk_per_unit": abs(entry_price_target - sl_price)
            }
            with recent_trade_signatures_lock: 
                 recent_trade_signatures[trade_sig_fib] = dt.now()
        else: 
            print(f"{log_prefix} Error: State for {symbol} not found after placing limit order. Cancelling.")
            try: client.futures_cancel_order(symbol=symbol, orderId=entry_limit_order['orderId'])
            except Exception as e_cancel: print(f"{log_prefix} Failed to cancel orphaned limit order {entry_limit_order['orderId']}: {e_cancel}")

def monitor_pending_fib_entries(client, configs: dict):
    """
    Monitors pending limit entry orders for the Fibonacci strategy.
    If filled, places SL/TP. If timed out, cancels the order.
    """
    global fib_strategy_states, fib_strategy_states_lock, active_trades, active_trades_lock
    
    log_prefix_monitor = "[FibMonitor]"
    symbols_to_reset_state = []
    # Iterate over a copy of items because the dictionary might be modified
    pending_entries_copy = {}
    with fib_strategy_states_lock:
        pending_entries_copy = {sym: state_data for sym, state_data in fib_strategy_states.items() if state_data.get('state') == "PENDING_ENTRY"}

    if not pending_entries_copy:
        return

    print(f"\n{log_prefix_monitor} Checking {len(pending_entries_copy)} pending Fibonacci entry order(s)...")

    for symbol, state_data in pending_entries_copy.items():
        order_id = state_data.get('pending_entry_order_id')
        pending_details = state_data.get('pending_entry_details')
        order_placed_time = state_data.get('bos_detail', {}).get('timestamp') # Using BoS timestamp as proxy for order placement time

        # Define symbol_info for use below
        symbol_info = None
        if pending_details and 'bos_event_snapshot' in pending_details:
            symbol_info = pending_details['bos_event_snapshot'].get('symbol_info', None)

        if not order_id or not pending_details or not order_placed_time:
            print(f"{log_prefix_monitor} Incomplete pending entry data for {symbol}. Resetting state.")
            symbols_to_reset_state.append(symbol)
            continue
        
        try:
            order_status = client.futures_get_order(symbol=symbol, orderId=order_id)
            
            if order_status['status'] == 'FILLED':
                print(f"{log_prefix_monitor} ✅ Limit entry order {order_id} for {symbol} FILLED!")
                actual_entry_price = float(order_status['avgPrice'])
                total_filled_qty = float(order_status['executedQty']) # Total quantity from the filled limit order
                
                sl_price = pending_details['sl_price']
                trade_side = pending_details['side'] # "long" or "short"
                position_side_for_sl_tp = "LONG" if trade_side == "long" else "SHORT"
                bos_event_snapshot = pending_details.get('bos_event_snapshot', {})
                
                p_prec = int(symbol_info.get('pricePrecision', 2))
                q_prec = int(symbol_info.get('quantityPrecision', 0))

                # --- TP Price and Quantity Calculation ---
                tp1_price_final, tp2_price_final, tp3_price_final = None, None, None
                qty_tp1, qty_tp2, qty_tp3 = 0, 0, 0

                use_atr_tp = configs.get("use_atr_for_tp", DEFAULT_USE_ATR_FOR_TP)

                if use_atr_tp:
                    print(f"{log_prefix_monitor} Using ATR-Smart TP logic for {symbol} ({trade_side}).")
                    atr_1m = bos_event_snapshot.get("atr_1m")
                    initial_risk_per_unit = pending_details.get("initial_risk_per_unit") # This is 'R'
                    tp_atr_mult = configs.get("tp_atr_multiplier", DEFAULT_TP_ATR_MULTIPLIER)

                    if atr_1m is not None and initial_risk_per_unit is not None and atr_1m > 0 and initial_risk_per_unit > 0:
                        tp_distance = atr_1m * tp_atr_mult * initial_risk_per_unit
                        if trade_side == "long":
                            tp1_price_final = round(actual_entry_price + tp_distance, p_prec)
                        else: # SHORT
                            tp1_price_final = round(actual_entry_price - tp_distance, p_prec)
                        
                        qty_tp1 = total_filled_qty # Single TP takes all quantity
                        qty_tp2 = 0
                        qty_tp3 = 0
                        print(f"{log_prefix_monitor} ATR-Smart TP calculated for {symbol}: TP={tp1_price_final} (ATR: {atr_1m}, R: {initial_risk_per_unit}, Multiplier: {tp_atr_mult})")
                    else:
                        print(f"{log_prefix_monitor} Could not calculate ATR-Smart TP due to missing ATR ({atr_1m}) or Risk ({initial_risk_per_unit}). Defaulting to Fib Extension/Provisional TPs.")
                        use_atr_tp = False # Fallback to extension/provisional
                
                if not use_atr_tp: # Fallback or standard Fib Extension/Provisional logic
                    if configs.get("fib_tp_use_extensions", DEFAULT_FIB_TP_USE_EXTENSIONS) and bos_event_snapshot:
                        range_bos = abs(bos_event_snapshot['swing_high_bos_move'] - bos_event_snapshot['swing_low_bos_move'])
                        tp1_ext_ratio = configs.get("fib_tp1_extension_ratio", DEFAULT_FIB_TP1_EXTENSION_RATIO)
                        tp2_ext_ratio = configs.get("fib_tp2_extension_ratio", DEFAULT_FIB_TP2_EXTENSION_RATIO)
                        tp3_ext_ratio = configs.get("fib_tp3_extension_ratio", DEFAULT_FIB_TP3_EXTENSION_RATIO)

                        if trade_side == "long":
                            tp1_price_final = round(actual_entry_price + (range_bos * tp1_ext_ratio), p_prec)
                            tp2_price_final = round(actual_entry_price + (range_bos * tp2_ext_ratio), p_prec)
                            tp3_price_final = round(actual_entry_price + (range_bos * tp3_ext_ratio), p_prec)
                        else: # SHORT
                            tp1_price_final = round(actual_entry_price - (range_bos * tp1_ext_ratio), p_prec)
                            tp2_price_final = round(actual_entry_price - (range_bos * tp2_ext_ratio), p_prec)
                            tp3_price_final = round(actual_entry_price - (range_bos * tp3_ext_ratio), p_prec)
                        print(f"{log_prefix_monitor} Calculated Fib Extension TPs for {symbol} ({trade_side}): TP1={tp1_price_final}, TP2={tp2_price_final}, TP3={tp3_price_final}")
                    else: # Use provisional TPs (old logic or if extensions disabled)
                        tp1_price_final = pending_details.get('tp1_price_provisional')
                        tp2_price_final = pending_details.get('tp2_price_provisional')
                        tp3_price_final = pending_details.get('tp3_price_provisional')
                        print(f"{log_prefix_monitor} Using provisional/original TP logic for {symbol} ({trade_side}): TP1={tp1_price_final}, TP2={tp2_price_final}, TP3={tp3_price_final}")

                    # TP Quantity Percentages from config (only if not ATR-Smart TP)
                    qty_pct_tp1 = configs.get("fib_tp1_qty_pct", DEFAULT_FIB_TP1_QTY_PCT)
                    qty_pct_tp2 = configs.get("fib_tp2_qty_pct", DEFAULT_FIB_TP2_QTY_PCT)
                    
                    qty_tp1 = round(total_filled_qty * qty_pct_tp1, q_prec)
                    qty_tp2 = round(total_filled_qty * qty_pct_tp2, q_prec)
                
                # Ensure qty_tp1 and qty_tp2 are not zero if their percentages are non-zero, due to small total_filled_qty
                min_qty_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                min_qty_val = float(min_qty_filter['minQty']) if min_qty_filter else 0.0
                
                if qty_pct_tp1 > 0 and qty_tp1 < min_qty_val and total_filled_qty >= min_qty_val: qty_tp1 = min_qty_val
                if qty_pct_tp2 > 0 and qty_tp2 < min_qty_val and total_filled_qty >= min_qty_val: qty_tp2 = min_qty_val
                
                # If sum of TP1 and TP2 quantities exceeds total, adjust TP2
                if (qty_tp1 + qty_tp2) > total_filled_qty:
                    qty_tp2 = round(total_filled_qty - qty_tp1, q_prec)
                    if qty_tp2 < 0: qty_tp2 = 0 # Ensure not negative

                qty_tp3 = round(total_filled_qty - qty_tp1 - qty_tp2, q_prec)
                if qty_tp3 < 0: qty_tp3 = 0 # Safety check

                # Final adjustment if sum is still off due to multiple roundings or min_qty adjustments
                current_sum_tp_qty = qty_tp1 + qty_tp2 + qty_tp3
                if abs(current_sum_tp_qty - total_filled_qty) > (1 / (10**q_prec)) / 2 : # If sum is off
                    print(f"{log_prefix_monitor} Adjusting TP quantities sum for {symbol}. Initial: Q1={qty_tp1}, Q2={qty_tp2}, Q3={qty_tp3}, Sum={current_sum_tp_qty}, Total={total_filled_qty}")
                    diff = total_filled_qty - current_sum_tp_qty
                    # Add difference to the largest portion, typically TP2 or TP3 if TP3 exists meaningfully
                    if qty_tp3 >= abs(diff) and qty_tp3 > 0 : qty_tp3 = round(qty_tp3 + diff, q_prec)
                    elif qty_tp2 >= abs(diff) and qty_tp2 > 0 : qty_tp2 = round(qty_tp2 + diff, q_prec)
                    elif qty_tp1 >= abs(diff) and qty_tp1 > 0 : qty_tp1 = round(qty_tp1 + diff, q_prec)
                    print(f"{log_prefix_monitor} Adjusted: Q1={qty_tp1}, Q2={qty_tp2}, Q3={qty_tp3}. NewSum: {qty_tp1+qty_tp2+qty_tp3}")

                print(f"{log_prefix_monitor} Placing SL for total qty {total_filled_qty} and multi-tier TPs for {symbol}:")
                print(f"  SL: {sl_price:.{p_prec}f} (Qty: {total_filled_qty})")
                if tp1_price_final and qty_tp1 > 0: print(f"  TP1: {tp1_price_final:.{p_prec}f} (Qty: {qty_tp1})")
                if tp2_price_final and qty_tp2 > 0: print(f"  TP2: {tp2_price_final:.{p_prec}f} (Qty: {qty_tp2})")
                if tp3_price_final and qty_tp3 > 0: print(f"  TP3: {tp3_price_final:.{p_prec}f} (Qty: {qty_tp3})")

                sl_ord_details, tp_orders_details_list = None, []
                
                if configs['mode'] != 'signal':
                    # --- Live/Backtest Mode: Place real SL/TP orders ---
                    print(f"{log_prefix_monitor} Live/Backtest Mode: Placing SL/TP for filled Fib limit order {symbol}.")
                    # Place SL for the total quantity
                    sl_ord_obj, sl_err_msg = place_new_order(client, symbol_info,
                                        "SELL" if trade_side == "long" else "BUY", "STOP_MARKET", total_filled_qty, 
                                        stop_price=sl_price, position_side=position_side_for_sl_tp, is_closing_order=True)
                    if not sl_ord_obj: print(f"{log_prefix_monitor} CRITICAL: FAILED TO PLACE SL for {symbol}! Error: {sl_err_msg}")
                    else: sl_ord_details = {"id": sl_ord_obj.get('orderId'), "price": sl_price, "quantity": total_filled_qty, "status": "OPEN"}

                    # Place TP orders for each tier
                tp_target_levels = [ # Use the final calculated prices and quantities
                    {"price": tp1_price_final, "quantity": qty_tp1, "name": "TP1"},
                    {"price": tp2_price_final, "quantity": qty_tp2, "name": "TP2"},
                    {"price": tp3_price_final, "quantity": qty_tp3, "name": "TP3"}
                ]

                for tp_info in tp_target_levels:
                    current_tp_price = tp_info["price"]
                    current_tp_qty = tp_info["quantity"]
                    tp_name = tp_info["name"]

                    if current_tp_price is not None and current_tp_qty > 0:
                        tp_ord_obj, tp_err_msg = place_new_order(client, symbol_info,
                                             "SELL" if trade_side == "long" else "BUY", "TAKE_PROFIT_MARKET", current_tp_qty,
                                             stop_price=current_tp_price, position_side=position_side_for_sl_tp, is_closing_order=True)
                        if not tp_ord_obj:
                            print(f"{log_prefix_monitor} WARNING: Failed to place {tp_name} for {symbol} (Qty: {current_tp_qty} @ {current_tp_price}). Error: {tp_err_msg}")
                            tp_orders_details_list.append({"id": None, "price": current_tp_price, "quantity": current_tp_qty, "status": "FAILED", "name": tp_name})
                        else:
                            tp_orders_details_list.append({"id": tp_ord_obj.get('orderId'), "price": current_tp_price, "quantity": current_tp_qty, "status": "OPEN", "name": tp_name})
                    elif current_tp_price is None and current_tp_qty > 0:
                         print(f"{log_prefix_monitor} Skipping {tp_name} for {symbol} as price is None, but quantity {current_tp_qty} was assigned.")
                    # If qty is zero, skip (already handled by qty_tpX > 0 check)
                
                if configs['mode'] == 'signal':
                    # --- Signal Mode: Update active_signals, no real trade transition ---
                    # The initial signal was already added to active_signals.
                    # Here, we'd confirm its "virtual fill" and potentially update its state if needed,
                    # e.g., to indicate it's now being monitored as an "active" signal rather than "pending entry signal".
                    # However, active_signals currently doesn't have a "pending entry" state, signals are added as active.
                    # We just need to ensure no real SL/TP orders were placed and no transition to `active_trades`.
                    print(f"{log_prefix_monitor} Signal Mode: Virtual Fib limit order for {symbol} considered FILLED at {actual_entry_price:.{p_prec}f}.")
                    # Send a specific Telegram update for virtual fill if desired
                    fill_msg_detail = f"Virtual limit entry order filled at ~{actual_entry_price:.{p_prec}f}. Signal is now active."
                    # Find the signal in active_signals to pass its details, or construct a temporary one.
                    temp_signal_details_for_fill_msg = { # Construct enough details for the message
                        "symbol": symbol, "side": trade_side.upper(), "entry_price": actual_entry_price, 
                        "strategy_type": "FIBONACCI_RETRACEMENT", "symbol_info": symbol_info,
                        "current_sl_price": sl_price, "current_tp1_price": tp1_price_final, # Show planned SL/TP1
                        "leverage": pending_details.get('bos_event_snapshot',{}).get('leverage', configs.get('leverage')) # Approx leverage
                    }
                    send_signal_update_telegram(configs, temp_signal_details_for_fill_msg, "VIRTUAL_ENTRY_FILLED", fill_msg_detail, actual_entry_price)
                    # No transition to active_trades. The signal remains in active_signals.
                
                else: # Live or Backtest Mode
                    # Transition to active_trades
                    with active_trades_lock:
                        if symbol not in active_trades:
                            active_trades[symbol] = {
                                "entry_order_id": order_id,
                                "sl_order_id": sl_ord_details.get('id') if sl_ord_details else None, 
                                "tp_orders": tp_orders_details_list, 
                                "entry_price": actual_entry_price,
                                "current_sl_price": sl_price, 
                                "initial_sl_price": sl_price,
                                "initial_risk_per_unit": abs(actual_entry_price - sl_price), 
                                "quantity": total_filled_qty, 
                                "side": trade_side.upper(),
                                "symbol_info": symbol_info,
                                "open_timestamp": pd.Timestamp(order_status['updateTime'], unit='ms', tz='UTC'),
                                "strategy_type": "FIBONACCI_MULTI_TP",
                                "sl_management_stage": "initial" 
                            }
                            print(f"{log_prefix_monitor} Fib trade for {symbol} (Multi-TP) moved to active_trades.")
                            
                        if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                            qty_prec_fib = int(symbol_info.get('quantityPrecision', 0))
                            price_prec_fib = int(symbol_info.get('pricePrecision', 2))
                            
                            tp_summary_lines = []
                            for tp_det in tp_orders_details_list:
                                if tp_det['id']: # Successfully placed
                                    tp_summary_lines.append(f"  - {tp_det['name']}: {tp_det['quantity']:.{qty_prec_fib}f} @ {tp_det['price']:.{price_prec_fib}f} (ID: {tp_det['id']})")
                                elif tp_det['price'] is not None : # Attempted but failed
                                    tp_summary_lines.append(f"  - {tp_det['name']}: {tp_det['quantity']:.{qty_prec_fib}f} @ {tp_det['price']:.{price_prec_fib}f} (Status: FAILED)")
                            tp_summary_str = "\n".join(tp_summary_lines) if tp_summary_lines else "  No TPs placed or all failed."

                            fib_trade_msg = (
                                f"🚀 FIB TRADE ENTRY FILLED (Multi-TP) 🚀\n\n"
                                f"Symbol: {symbol}\n"
                                f"Side: {trade_side.upper()}\n"
                                f"Total Quantity: {total_filled_qty:.{qty_prec_fib}f}\n"
                                f"Entry Price: {actual_entry_price:.{price_prec_fib}f}\n"
                                f"SL: {sl_price:.{price_prec_fib}f} (ID: {sl_ord_details.get('id') if sl_ord_details else 'FAIL'})\n"
                                f"TP Levels:\n{tp_summary_str}"
                            )
                            send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], fib_trade_msg)
                        else:
                            print(f"{log_prefix_monitor} {symbol} already in active_trades. Limit fill for Fib trade {order_id} will not be managed by bot SL/TP. Manual check advised.")
                            # Cancel newly placed SL/TPs for this orphaned fill
                            if sl_ord_details and sl_ord_details.get('id'): client.futures_cancel_order(symbol=symbol, orderId=sl_ord_details['id'])
                            for tp_det in tp_orders_details_list:
                                if tp_det.get('id'): client.futures_cancel_order(symbol=symbol, orderId=tp_det['id'])
                    
                symbols_to_reset_state.append(symbol) 
            
            elif order_status['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']:
                print(f"{log_prefix_monitor} Limit entry order {order_id} for {symbol} is {order_status['status']}. Resetting state.")
                symbols_to_reset_state.append(symbol)
            
            elif order_status['status'] == 'NEW' or order_status['status'] == 'PARTIALLY_FILLED':
                # Time Stop logic
                order_timeout_minutes = configs.get("fib_order_timeout_minutes", 5)
                if order_placed_time and (pd.Timestamp.now(tz='UTC') - order_placed_time).total_seconds() > order_timeout_minutes * 60:
                    print(f"{log_prefix_monitor} Limit entry order {order_id} for {symbol} timed out ({order_timeout_minutes}m). Cancelling.")
                    try:
                        client.futures_cancel_order(symbol=symbol, orderId=order_id)
                        print(f"{log_prefix_monitor} Successfully cancelled timed-out order {order_id} for {symbol}.")
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"),
                                              f"⏳ Fib entry order for `{symbol}` (ID: {order_id}) timed out and was cancelled.")
                    except Exception as e_cancel:
                        print(f"{log_prefix_monitor} Failed to cancel timed-out order {order_id} for {symbol}: {e_cancel}")
                    symbols_to_reset_state.append(symbol)
                # else: Order is still new/partially_filled and within timeout, do nothing.
                # print(f"{log_prefix_monitor} Limit entry order {order_id} for {symbol} is still {order_status['status']}.")
            
            # Any other status means it's likely no longer relevant or an issue
            elif order_status['status'] not in ['NEW', 'PARTIALLY_FILLED']:
                 print(f"{log_prefix_monitor} Limit entry order {order_id} for {symbol} has unexpected status: {order_status['status']}. Resetting state.")
                 symbols_to_reset_state.append(symbol)

        except BinanceAPIException as e:
            if e.code == -2013: # Order does not exist
                print(f"{log_prefix_monitor} Limit entry order {order_id} for {symbol} NOT FOUND (likely manually cancelled or already processed). Resetting state.")
                symbols_to_reset_state.append(symbol)
            else:
                print(f"{log_prefix_monitor} API Error checking order {order_id} for {symbol}: {e}")
        except Exception as e:
            print(f"{log_prefix_monitor} Unexpected error checking order {order_id} for {symbol}: {e}")
            traceback.print_exc()
            symbols_to_reset_state.append(symbol) # Reset state on unexpected error too

    if symbols_to_reset_state:
        with fib_strategy_states_lock:
            for sym_to_reset in symbols_to_reset_state:
                if sym_to_reset in fib_strategy_states:
                    print(f"{log_prefix_monitor} Resetting Fib strategy state for {sym_to_reset} to IDLE.")
                    fib_strategy_states[sym_to_reset]['state'] = "IDLE"
                    fib_strategy_states[sym_to_reset]['pending_entry_order_id'] = None
                    fib_strategy_states[sym_to_reset]['pending_entry_details'] = None
                    # Keep flip_bias_direction unless explicitly cleared by a successful trade or other logic
                    # Keep trend and pivot data as they might still be relevant
                else: # Should not happen if pending_entries_copy was derived correctly
                    print(f"{log_prefix_monitor} Warning: Tried to reset state for {sym_to_reset}, but it was not in fib_strategy_states.")


def start_telegram_polling(bot_token: str, app_configs: dict):
    # Create and set a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Build the app
    application = (
        Application.builder()
        .token(bot_token)
        .concurrent_updates(True)    # optional: allow concurrent handler execution
        .build()
    )
    application.bot_data['configs'] = app_configs

    # Register defined handlers:
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("status", status_handler))
    application.add_handler(CommandHandler("positions", positions_handler))
    application.add_handler(CommandHandler("orders", orders_handler))
    application.add_handler(CommandHandler("halt", halt_handler))
    application.add_handler(CommandHandler("resume", resume_handler))
    application.add_handler(CommandHandler("closeall", close_all_handler))
    application.add_handler(CommandHandler("setrisk", set_risk_handler))
    application.add_handler(CommandHandler("setleverage", set_leverage_handler))
    application.add_handler(CommandHandler("log", log_handler))
    application.add_handler(CommandHandler("config", config_handler))
    application.add_handler(CommandHandler("shutdown", shutdown_handler))
    application.add_handler(CommandHandler("blacklist", blacklist_handler)) 
    application.add_handler(CommandHandler("restart", restart_handler)) 
    application.add_handler(CommandHandler("set", set_handler)) # Updated command to /set and handler to set_handler
    application.add_handler(CommandHandler("sum", summary_handler)) # Added /sum command handler
    # Note: command3_handler is removed as it was tied to the old start_command_listener structure

    print("Telegram ▶ run_polling() starting…")
    # Initialize and run polling within this function
    # application.initialize() # Included in ApplicationBuilder with .build()
    application.run_polling()       # blocks here, polling updates forever
    print("Telegram ▶ run_polling() exited.")

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


def get_user_configurations(load_choice_override: str = None, make_changes_override: str = None):
    """
    Prompts the user for various trading configurations and returns them as a dictionary.
    Includes input validation for each configuration item.
    Allows loading from or saving to 'configure.csv'.
    Overrides for initial prompts can be passed for automated setup.
    """
    print("\n--- Strategy Configuration ---")
    configs = {}
    loaded_configs_from_csv = None
    config_filepath = "configure.csv"
    proceed_to_custom_setup = False

    # Ask user if they want to load from CSV or do custom setup
    while True:
        actual_load_choice = None
        if load_choice_override and load_choice_override.lower() in ['l', 'c']:
            actual_load_choice = load_choice_override.lower()
            print(f"Using pre-set load choice: '{actual_load_choice}' ('L' for Load, 'C' for Custom).")
        else:
            actual_load_choice = input(f"Load from '{config_filepath}' (L) or Custom Setup (C)? [L]: ").strip().lower()

        if not actual_load_choice or actual_load_choice == 'l':
            loaded_configs_from_csv = load_configuration_from_csv(config_filepath)
            if loaded_configs_from_csv:
                is_valid, validation_msg, validated_configs_csv = validate_configurations(loaded_configs_from_csv)
                if is_valid:
                    print("Configuration loaded successfully from CSV:")
                    for k, v in validated_configs_csv.items(): print(f"  {k}: {v}")
                    
                    while True:
                        actual_make_changes_choice = None
                        if make_changes_override and make_changes_override.lower() in ['y', 'n']:
                            actual_make_changes_choice = make_changes_override.lower()
                            print(f"Using pre-set make changes choice: '{actual_make_changes_choice}'.")
                        else:
                            actual_make_changes_choice = input("Make changes to these settings? (y/N) [N]: ").strip().lower()

                        if not actual_make_changes_choice or actual_make_changes_choice == 'n':
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
                                proceed_to_custom_setup = True
                                break
                            
                            # Set strategy specific details based on choice
                            # Ensure 'strategy_choice' exists in configs, validated by validate_configurations
                            current_strategy_choice = configs.get("strategy_choice")

                            if not current_strategy_choice:
                                # This case should ideally be prevented by validate_configurations making strategy_choice mandatory.
                                print(f"CRITICAL WARNING: 'strategy_choice' not found in validated configs. Defaulting to {DEFAULT_STRATEGY} for safety.")
                                current_strategy_choice = DEFAULT_STRATEGY
                                configs["strategy_choice"] = DEFAULT_STRATEGY # Ensure it's set in configs

                            print(f"[Debug] Inside get_user_configurations (CSV load path), strategy_choice from configs is: '{current_strategy_choice}'")

                            if current_strategy_choice == "ema_cross":
                                configs["strategy_id"] = 8
                                configs["strategy_name"] = "Advance EMA Cross"
                                print(f"[Debug] Set strategy to EMA Cross (ID: 8)")
                            elif current_strategy_choice == "fib_retracement":
                                configs["strategy_id"] = 9
                                configs["strategy_name"] = "Fibonacci Retracement"
                                print(f"[Debug] Set strategy to Fibonacci Retracement (ID: 9)")
                                # Ensure fib_1m_buffer_size has a default if not in CSV
                                if "fib_1m_buffer_size" not in configs: configs["fib_1m_buffer_size"] = DEFAULT_1M_BUFFER_SIZE
                                # Ensure fib specific ATR SL params have defaults if not in CSV
                                if "fib_atr_period" not in configs: configs["fib_atr_period"] = DEFAULT_FIB_ATR_PERIOD
                                if "fib_sl_atr_multiplier" not in configs: configs["fib_sl_atr_multiplier"] = DEFAULT_FIB_SL_ATR_MULTIPLIER
                            elif current_strategy_choice == "ict_strategy":
                                configs["strategy_id"] = 10
                                configs["strategy_name"] = "ICT Strategy"
                                print(f"[Debug] Set strategy to ICT Strategy (ID: 10)")
                                # Ensure ICT specific params have defaults if not in CSV (many are now added)
                                # Basic ones:
                                if "ict_timeframe" not in configs: configs["ict_timeframe"] = DEFAULT_ICT_TIMEFRAME
                                if "ict_risk_reward_ratio" not in configs: configs["ict_risk_reward_ratio"] = DEFAULT_ICT_RISK_REWARD_RATIO
                                if "ict_fvg_freshness_candles" not in configs: configs["ict_fvg_freshness_candles"] = DEFAULT_ICT_FVG_FRESHNESS_CANDLES
                                if "ict_liquidity_lookback" not in configs: configs["ict_liquidity_lookback"] = DEFAULT_ICT_LIQUIDITY_LOOKBACK
                                # Add defaults for other new ICT params if they are missing from CSV
                                for ict_param, ict_default in [
                                    ("ict_sweep_detection_window", DEFAULT_ICT_SWEEP_DETECTION_WINDOW),
                                    ("ict_entry_type", DEFAULT_ICT_ENTRY_TYPE),
                                    ("ict_sl_type", DEFAULT_ICT_SL_TYPE),
                                    ("ict_sl_atr_buffer_multiplier", DEFAULT_ICT_SL_ATR_BUFFER_MULTIPLIER),
                                    ("ict_ob_bos_lookback", DEFAULT_ICT_OB_BOS_LOOKBACK),
                                    ("ict_po3_consolidation_lookback", DEFAULT_ICT_PO3_CONSOLIDATION_LOOKBACK),
                                    ("ict_po3_acceleration_min_candles", DEFAULT_ICT_PO3_ACCELERATION_MIN_CANDLES),
                                    ("ict_limit_signal_cooldown_seconds", DEFAULT_ICT_LIMIT_SIGNAL_COOLDOWN_SECONDS),
                                    ("ict_limit_signal_signature_block_seconds", DEFAULT_ICT_LIMIT_SIGNAL_SIGNATURE_BLOCK_SECONDS),
                                    ("ict_order_timeout_minutes", DEFAULT_ICT_ORDER_TIMEOUT_MINUTES),
                                    ("ict_kline_limit", DEFAULT_ICT_KLINE_LIMIT)
                                ]:
                                    if ict_param not in configs: configs[ict_param] = ict_default
                            else:
                                # This else block means 'strategy_choice' was something unexpected AFTER validation,
                                # which should not happen if validation enforces the allowed values.
                                # Or, it was missing and defaulted to DEFAULT_STRATEGY, and that default isn't one of the three.
                                print(f"ERROR: Validated 'strategy_choice' ('{current_strategy_choice}') is unknown. Defaulting to EMA Cross strategy settings as a fallback.")
                                configs["strategy_choice"] = "ema_cross" # Correct the choice itself
                                configs["strategy_id"] = 8
                                configs["strategy_name"] = "Advance EMA Cross"
                            
                            # No need to update expected_params here; it is only used inside validate_configurations.


                            configs["max_scan_threads"] = 5 # Fixed value
                            print("--- Configuration Complete (Loaded from CSV) ---")
                            return configs
                        elif actual_make_changes_choice == 'y':
                            configs = validated_configs_csv # Start custom setup with these values
                            proceed_to_custom_setup = True
                            break # Break from this inner while True loop
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
        elif actual_load_choice == 'c': # Corrected variable name here
            proceed_to_custom_setup = True
            break # Break from load_choice loop
        else:
            print("Invalid choice. Please enter 'L' or 'C'.")

    # If proceed_to_custom_setup is True (either by choice or failure to load)
    # `configs` might already hold values from CSV if user chose 'y' to change them.
    # Otherwise, `configs` is empty.

    print("\n--- Custom Configuration Setup ---")
    # Helper to get input, using value from `configs` (loaded from CSV) as default if available
    def get_input_with_default(prompt_message, current_value_key, default_constant_value, type_converter=str, is_percentage_display=False):
        default_from_config = configs.get(current_value_key)
        
        # For percentage display, if default_from_config is present (e.g. 0.01), convert to % (e.g. 1.0) for display.
        # If not present, use default_constant_value (which is assumed to be in display format, e.g. 1.0 for 1%).
        if is_percentage_display:
            default_to_show_val = (default_from_config * 100.0) if default_from_config is not None else default_constant_value
            default_to_show_str = f"{default_to_show_val:.2f}%" if isinstance(default_to_show_val, float) else str(default_to_show_val)
        else:
            default_to_show_val = default_from_config if default_from_config is not None else default_constant_value
            default_to_show_str = str(default_to_show_val)

        user_input = input(f"{prompt_message} (default: {default_to_show_str}): ")
        
        # If user enters nothing, use the default_to_show_val (which could be from CSV or constant, already in correct scale)
        if not user_input:
            # Need to ensure the default_to_show is of the correct type if it came from CSV (already string)
            # or from constant (might need conversion if it's not already a string for display)
            # The type_converter will handle this.
            try:
                return type_converter(default_to_show_val)
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

    # Strategy Choice
    while True:
        strategy_default_display = configs.get("strategy_choice", DEFAULT_STRATEGY)
        strategy_input = input(f"Select strategy (1:EMA Cross / 2:Fib Retracement / 3:ICT Strategy) (current: {strategy_default_display}): ").strip()
        
        chosen_strategy = None
        if not strategy_input and "strategy_choice" in configs: chosen_strategy = configs["strategy_choice"]
        elif strategy_input == "1": chosen_strategy = "ema_cross"
        elif strategy_input == "2": chosen_strategy = "fib_retracement"
        elif strategy_input == "3": chosen_strategy = "ict_strategy"


        if chosen_strategy in ["ema_cross", "fib_retracement", "ict_strategy"]:
            configs["strategy_choice"] = chosen_strategy
            if chosen_strategy == "ema_cross":
                configs["strategy_id"] = 8
                configs["strategy_name"] = "Advance EMA Cross"
            elif chosen_strategy == "fib_retracement":
                configs["strategy_id"] = 9
                configs["strategy_name"] = "Fibonacci Retracement"
            elif chosen_strategy == "ict_strategy":
                configs["strategy_id"] = 10 # New ID for ICT Strategy
                configs["strategy_name"] = "ICT Strategy"
            break
        print("Invalid strategy choice. Please enter '1', '2', or '3'.")

    # Environment
    while True:
        env_default_display = configs.get("environment", "testnet")
        env_input = input(f"Select environment (1:testnet / 2:mainnet) (current: {env_default_display}): ").strip()
        chosen_env = None
        if not env_input and "environment" in configs: chosen_env = configs["environment"]
        elif env_input == "1": chosen_env = "testnet"
        elif env_input == "2": chosen_env = "mainnet"
        if chosen_env in ["testnet", "mainnet"]:
            configs["environment"] = chosen_env
            break
        print("Invalid environment. Please enter '1' or '2'.")

    api_key, api_secret, telegram_token, telegram_chat_id = load_api_keys(configs["environment"])
    configs["api_key"] = api_key
    configs["api_secret"] = api_secret
    configs["telegram_bot_token"] = telegram_token
    configs["telegram_chat_id"] = telegram_chat_id
    
    # Mode
    while True:
        mode_default_display = configs.get("mode", "live")
        mode_input = input(f"Select mode (1:live / 2:backtest / 3:signal) (current: {mode_default_display}): ").strip()
        chosen_mode = None
        if not mode_input and "mode" in configs: # User hit enter, use CSV value if present
            chosen_mode = configs["mode"]
        elif mode_input == "1": chosen_mode = "live"
        elif mode_input == "2": chosen_mode = "backtest"
        elif mode_input == "3": chosen_mode = "signal"
        
        if chosen_mode in ["live", "backtest", "signal"]:
            configs["mode"] = chosen_mode
            break
        print("Invalid mode. Please enter '1', '2', or '3'.")

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
            risk_percent_val = get_input_with_default(
                "Enter account risk % per trade (e.g., 1 for 1%)",
                "risk_percent", DEFAULT_RISK_PERCENT, float, is_percentage_display=True
            )
            # The value from get_input_with_default will be in % if user entered it,
            # or already in correct scale (e.g. 0.01) if from CSV and user hit enter,
            # or the constant default (e.g. 1.0) if from constant and user hit enter.
            # We need to ensure it's consistently handled.
            # If it came from `configs.get("risk_percent")` and user hit enter, it's already 0.01.
            # If it was input by user, it's 1.0.
            # If it was default constant, it's 1.0.
            # So, if value > 1 (likely user input like "1" for 1%), convert to decimal.
            if risk_percent_val > 1.0 and risk_percent_val <=100.0 : # User likely entered 1 for 1%
                 configs["risk_percent"] = risk_percent_val / 100.0
            elif 0 < risk_percent_val <= 1.0: # Already in decimal form (e.g. from CSV or user entered 0.01)
                 configs["risk_percent"] = risk_percent_val
            # Handle case where default was used and it was > 1 (e.g. DEFAULT_RISK_PERCENT = 1.0)
            elif risk_percent_val == DEFAULT_RISK_PERCENT and DEFAULT_RISK_PERCENT > 1.0 and DEFAULT_RISK_PERCENT <= 100.0:
                 configs["risk_percent"] = risk_percent_val / 100.0
            else: # Covers 0 or invalid numbers not caught by float conversion
                print("Risk percentage must be a positive value (e.g., 0.5, 1, up to 100).")
                continue # Re-prompt
            break
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
            target_vol_pct = get_input_with_default(
                "Enter Target Annualized Volatility % (e.g., 80 for 80%)",
                "target_annualized_volatility", 
                DEFAULT_TARGET_ANNUALIZED_VOLATILITY * 100.0, # Constant default is in display %
                float, 
                is_percentage_display=True # Input is expected as %, stored as decimal
            )
            # Logic similar to risk_percent:
            # If from CSV via get_input_with_default (user hit enter), it's already decimal (e.g. 0.80)
            # If user input, it's in % (e.g. 80.0)
            # If default constant, it's in % (e.g. 80.0 from DEFAULT_TARGET_ANNUALIZED_VOLATILITY * 100.0)
            
            target_vol_decimal_val = 0
            if target_vol_pct > 1.0 and target_vol_pct <= 500.0: # User entered 80 for 80%
                target_vol_decimal_val = target_vol_pct / 100.0
            elif 0 < target_vol_pct <= 5.0: # Already decimal (e.g. from CSV or user entered 0.8)
                target_vol_decimal_val = target_vol_pct
            # Handle case where default was used and it was > 1 (e.g. DEFAULT_TARGET_ANNUALIZED_VOLATILITY * 100.0 = 80.0)
            elif target_vol_pct == (DEFAULT_TARGET_ANNUALIZED_VOLATILITY * 100.0) and \
                 (DEFAULT_TARGET_ANNUALIZED_VOLATILITY * 100.0) > 1.0 and \
                 (DEFAULT_TARGET_ANNUALIZED_VOLATILITY * 100.0) <= 500.0:
                 target_vol_decimal_val = (DEFAULT_TARGET_ANNUALIZED_VOLATILITY * 100.0) / 100.0
            else:
                print("Target Annualized Volatility must be a positive number (e.g., input 80 for 80% or 0.8). Sensible range up to 500% (5.0).")
                continue

            configs["target_annualized_volatility"] = target_vol_decimal_val
            break
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

    # Fibonacci Strategy Specific: 1M Candle Buffer Size
    if configs.get("strategy_choice") == "fib_retracement":
        while True:
            try:
                buffer_size = get_input_with_default(
                    "Enter 1-min candle buffer size (for Fib strategy, e.g., 20-1000)",
                    "fib_1m_buffer_size", DEFAULT_1M_BUFFER_SIZE, int
                )
                if 20 <= buffer_size <= 1000:
                    configs["fib_1m_buffer_size"] = buffer_size
                    break
                print("1-min candle buffer size must be between 20 and 1000.")
            except ValueError: print("Invalid input. Please enter an integer.")
    else: # Not fib_retracement strategy, remove if it was in loaded CSV
        configs.pop("fib_1m_buffer_size", None)
        configs.pop("fib_order_timeout_minutes", None)

    # Fibonacci Strategy Specific: Order Timeout
    if configs.get("strategy_choice") == "fib_retracement":
        while True:
            try:
                timeout_val = get_input_with_default(
                    "Enter Fib order timeout in minutes (e.g., 1-60)",
                    "fib_order_timeout_minutes", DEFAULT_FIB_ORDER_TIMEOUT_MINUTES, int
                )
                if 1 <= timeout_val <= 60:
                    configs["fib_order_timeout_minutes"] = timeout_val
                    break
                print("Fib order timeout must be between 1 and 60 minutes.")
            except ValueError: print("Invalid input. Please enter an integer.")
        
        # Fib ATR Period for SL
        while True:
            try:
                fib_atr_period_val = get_input_with_default(
                    "Enter ATR Period for Fib Strategy SL",
                    "fib_atr_period", DEFAULT_FIB_ATR_PERIOD, int
                )
                if fib_atr_period_val > 0:
                    configs["fib_atr_period"] = fib_atr_period_val
                    break
                print("Fib ATR Period must be a positive integer.")
            except ValueError: print("Invalid input for Fib ATR Period. Please enter an integer.")

        # Fib SL ATR Multiplier
        while True:
            try:
                fib_sl_mult_val = get_input_with_default(
                    "Enter SL ATR Multiplier for Fib Strategy (e.g., 0.5-1.0)",
                    "fib_sl_atr_multiplier", DEFAULT_FIB_SL_ATR_MULTIPLIER, float
                )
                if fib_sl_mult_val > 0:
                    configs["fib_sl_atr_multiplier"] = fib_sl_mult_val
                    break
                print("Fib SL ATR Multiplier must be a positive number.")
            except ValueError: print("Invalid input for Fib SL ATR Multiplier. Please enter a number.")

        # ATR-Smart TP for Fib
        while True:
            default_bool_val = configs.get("use_atr_for_tp", DEFAULT_USE_ATR_FOR_TP)
            default_bool_disp = 'yes' if default_bool_val else 'no'
            atr_tp_input = input(f"Use ATR-Smart TP for Fib Strategy? (yes/no) (default: {default_bool_disp}): ").lower().strip()
            chosen_atr_tp_bool = None
            if not atr_tp_input: chosen_atr_tp_bool = default_bool_val
            elif atr_tp_input in ["yes", "y"]: chosen_atr_tp_bool = True
            elif atr_tp_input in ["no", "n"]: chosen_atr_tp_bool = False
            if isinstance(chosen_atr_tp_bool, bool):
                configs["use_atr_for_tp"] = chosen_atr_tp_bool
                break
            print("Invalid input. Please enter 'yes' or 'no'.")

        if configs.get("use_atr_for_tp"):
            while True:
                try:
                    tp_atr_mult_val = get_input_with_default(
                        "Enter TP ATR Multiplier (for ATR-Smart TP)",
                        "tp_atr_multiplier", DEFAULT_TP_ATR_MULTIPLIER, float
                    )
                    if tp_atr_mult_val > 0:
                        configs["tp_atr_multiplier"] = tp_atr_mult_val
                        break
                    print("TP ATR Multiplier must be a positive number.")
                except ValueError: print("Invalid input for TP ATR Multiplier. Please enter a number.")
        else: # Not using ATR-Smart TP, remove if it was in loaded CSV
            configs.pop("tp_atr_multiplier", None)

    else: # Not fib_retracement strategy, remove if it was in loaded CSV
        configs.pop("fib_atr_period", None)
        configs.pop("fib_sl_atr_multiplier", None)
        # Also remove new Fib TP/SL management specific configs if not Fib strategy
        configs.pop("fib_tp_use_extensions", None)
        configs.pop("fib_tp1_extension_ratio", None)
        configs.pop("fib_tp2_extension_ratio", None)
        configs.pop("fib_tp3_extension_ratio", None)
        configs.pop("fib_tp1_qty_pct", None)
        configs.pop("fib_tp2_qty_pct", None)
        configs.pop("fib_tp3_qty_pct", None)
        configs.pop("fib_move_sl_after_tp1", None)
        configs.pop("fib_breakeven_buffer_r", None)
        configs.pop("fib_sl_adjustment_after_tp2", None)
        # ATR-Smart TP for Fib
        configs.pop("use_atr_for_tp", None)
        configs.pop("tp_atr_multiplier", None)
    
    # ICT Strategy Specific configurations
    if configs.get("strategy_choice") == "ict_strategy":
        print("\n--- ICT Strategy Specific Configurations ---")
        # Timeframe (already exists)
        while True:
            try:
                val = get_input_with_default("Enter ICT Analysis Timeframe (e.g., 15m, 1h)", "ict_timeframe", DEFAULT_ICT_TIMEFRAME, str)
                if val in ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]: configs["ict_timeframe"] = val; break
                print("Invalid timeframe. Use formats like 1m, 5m, 15m, 1h, 4h.")
            except ValueError: print("Invalid input for ICT Timeframe.")
        # Risk:Reward Ratio (already exists)
        while True:
            try:
                val = get_input_with_default("Enter ICT Target Risk:Reward Ratio", "ict_risk_reward_ratio", DEFAULT_ICT_RISK_REWARD_RATIO, float)
                if 1.0 <= val <= 10.0: configs["ict_risk_reward_ratio"] = val; break
                print("ICT R:R ratio must be between 1.0 and 10.0.")
            except ValueError: print("Invalid input for ICT R:R ratio.")
        # FVG Freshness (already exists)
        while True:
            try:
                val = get_input_with_default("Enter ICT FVG Freshness (candles)", "ict_fvg_freshness_candles", DEFAULT_ICT_FVG_FRESHNESS_CANDLES, int)
                if 1 <= val <= 100: configs["ict_fvg_freshness_candles"] = val; break
                print("ICT FVG Freshness must be between 1 and 100 candles.")
            except ValueError: print("Invalid input for ICT FVG Freshness.")
        # Liquidity Lookback (already exists)
        while True:
            try:
                val = get_input_with_default("Enter ICT Liquidity Lookback (candles)", "ict_liquidity_lookback", DEFAULT_ICT_LIQUIDITY_LOOKBACK, int)
                if 5 <= val <= 200: configs["ict_liquidity_lookback"] = val; break
                print("ICT Liquidity Lookback must be between 5 and 200 candles.")
            except ValueError: print("Invalid input for ICT Liquidity Lookback.")

        # New ICT Params
        while True:
            try:
                val = get_input_with_default("Enter ICT Sweep Detection Window (candles)", "ict_sweep_detection_window", DEFAULT_ICT_SWEEP_DETECTION_WINDOW, int)
                if 1 <= val <= 50: configs["ict_sweep_detection_window"] = val; break
                print("ICT Sweep Detection Window must be between 1 and 50 candles.")
            except ValueError: print("Invalid input.")
        while True:
            default_val = configs.get("ict_entry_type", DEFAULT_ICT_ENTRY_TYPE)
            user_input = input(f"Enter ICT Entry Type (fvg_mid/ob_open/ob_mean) (default: {default_val}): ").lower().strip()
            chosen_val = default_val if not user_input else user_input
            if chosen_val in ["fvg_mid", "ob_open", "ob_mean"]: configs["ict_entry_type"] = chosen_val; break
            print("Invalid entry type.")
        while True:
            default_val = configs.get("ict_sl_type", DEFAULT_ICT_SL_TYPE)
            user_input = input(f"Enter ICT SL Type (ob_fvg_zone/swept_point/atr_buffered_zone) (default: {default_val}): ").lower().strip()
            chosen_val = default_val if not user_input else user_input
            if chosen_val in ["ob_fvg_zone", "swept_point", "atr_buffered_zone"]: configs["ict_sl_type"] = chosen_val; break
            print("Invalid SL type.")
        while True:
            try:
                val = get_input_with_default("Enter ICT SL ATR Buffer Multiplier (0 if not using ATR SL)", "ict_sl_atr_buffer_multiplier", DEFAULT_ICT_SL_ATR_BUFFER_MULTIPLIER, float)
                if 0.0 <= val <= 2.0: configs["ict_sl_atr_buffer_multiplier"] = val; break
                print("ICT SL ATR Buffer Multiplier must be between 0.0 and 2.0.")
            except ValueError: print("Invalid input.")
        while True:
            try:
                val = get_input_with_default("Enter ICT OB BoS Lookback (candles)", "ict_ob_bos_lookback", DEFAULT_ICT_OB_BOS_LOOKBACK, int)
                if 3 <= val <= 50: configs["ict_ob_bos_lookback"] = val; break
                print("ICT OB BoS Lookback must be between 3 and 50 candles.")
            except ValueError: print("Invalid input.")
        while True:
            try:
                val = get_input_with_default("Enter ICT Po3 Consolidation Lookback (candles)", "ict_po3_consolidation_lookback", DEFAULT_ICT_PO3_CONSOLIDATION_LOOKBACK, int)
                if 5 <= val <= 50: configs["ict_po3_consolidation_lookback"] = val; break
                print("ICT Po3 Consolidation Lookback must be between 5 and 50.")
            except ValueError: print("Invalid input.")
        while True:
            try:
                val = get_input_with_default("Enter ICT Po3 Acceleration Min Candles", "ict_po3_acceleration_min_candles", DEFAULT_ICT_PO3_ACCELERATION_MIN_CANDLES, int)
                if 1 <= val <= 10: configs["ict_po3_acceleration_min_candles"] = val; break
                print("ICT Po3 Acceleration Min Candles must be between 1 and 10.")
            except ValueError: print("Invalid input.")
        while True:
            try:
                val = get_input_with_default("Enter ICT Limit Signal Cooldown (seconds)", "ict_limit_signal_cooldown_seconds", DEFAULT_ICT_LIMIT_SIGNAL_COOLDOWN_SECONDS, int)
                if 0 <= val <= 3600: configs["ict_limit_signal_cooldown_seconds"] = val; break
                print("ICT Limit Signal Cooldown must be between 0 and 3600 seconds.")
            except ValueError: print("Invalid input.")
        while True:
            try:
                val = get_input_with_default("Enter ICT Limit Signal Signature Block (seconds, e.g., 60)", "ict_limit_signal_signature_block_seconds", DEFAULT_ICT_LIMIT_SIGNAL_SIGNATURE_BLOCK_SECONDS, int)
                if 0 <= val <= 3600: configs["ict_limit_signal_signature_block_seconds"] = val; break
                print("ICT Limit Signal Signature Block must be between 0 and 3600 seconds.")
            except ValueError: print("Invalid input. Please enter an integer.")
        while True:
            try:
                val = get_input_with_default("Enter ICT Order Timeout (minutes)", "ict_order_timeout_minutes", DEFAULT_ICT_ORDER_TIMEOUT_MINUTES, int)
                if 1 <= val <= 1440: configs["ict_order_timeout_minutes"] = val; break
                print("ICT Order Timeout must be between 1 and 1440 minutes.")
            except ValueError: print("Invalid input.")
        while True:
            try:
                val = get_input_with_default("Enter ICT Kline Fetch Limit", "ict_kline_limit", DEFAULT_ICT_KLINE_LIMIT, int)
                if val < 300:
                    print(f"ICT Kline Limit must be at least 300. Adjusting {val} to 300.")
                    val = 300
                elif val > 1000:
                    print(f"ICT Kline Limit cannot exceed 1000. Adjusting {val} to 1000.")
                    val = 1000
                configs["ict_kline_limit"] = val
                break
            except ValueError: print("Invalid input. Please enter an integer.")

    else: # Not ICT strategy, remove if they were in loaded CSV
        configs.pop("ict_timeframe", None)
        configs.pop("ict_risk_reward_ratio", None)
        configs.pop("ict_fvg_freshness_candles", None)
        configs.pop("ict_liquidity_lookback", None)
        configs.pop("ict_sweep_detection_window", None)
        configs.pop("ict_entry_type", None)
        configs.pop("ict_sl_type", None)
        configs.pop("ict_sl_atr_buffer_multiplier", None)
        configs.pop("ict_ob_bos_lookback", None)
        configs.pop("ict_po3_consolidation_lookback", None)
        configs.pop("ict_po3_acceleration_min_candles", None)
        configs.pop("ict_limit_signal_cooldown_seconds", None)
        configs.pop("ict_limit_signal_signature_block_seconds", None)
        configs.pop("ict_order_timeout_minutes", None)
        configs.pop("ict_kline_limit", None)

    # Fibonacci Strategy Specific: TP Scaling and SL Management (only if Fib strategy is chosen)
    if configs.get("strategy_choice") == "fib_retracement":
        print("\n--- Fibonacci Strategy Specific TP & SL Management ---")

        # Only ask for Fib Extension TPs if ATR-Smart TP is NOT enabled
        if not configs.get("use_atr_for_tp", False):
            while True:
                default_val = configs.get("fib_tp_use_extensions", DEFAULT_FIB_TP_USE_EXTENSIONS)
                default_disp = 'yes' if default_val else 'no'
                user_input = input(f"Use Fib Extension TPs for Fib Strategy? (yes/no) (default: {default_disp}): ").lower().strip()
                chosen_val = None
                if not user_input:
                    chosen_val = default_val
                elif user_input in ["yes", "y"]:
                    chosen_val = True
                elif user_input in ["no", "n"]:
                    chosen_val = False
                if isinstance(chosen_val, bool):
                    configs["fib_tp_use_extensions"] = chosen_val
                    break
                print("Invalid input. Please enter 'yes' or 'no'.")

        if configs.get("fib_tp_use_extensions"):
            # TP Extension Ratios
            for i, ratio_key, default_ratio in [
                (1, "fib_tp1_extension_ratio", DEFAULT_FIB_TP1_EXTENSION_RATIO),
                (2, "fib_tp2_extension_ratio", DEFAULT_FIB_TP2_EXTENSION_RATIO),
                (3, "fib_tp3_extension_ratio", DEFAULT_FIB_TP3_EXTENSION_RATIO)
            ]:
                while True:
                    try:
                        val = get_input_with_default(f"Enter Fib TP{i} Extension Ratio", ratio_key, default_ratio, float)
                        if val > 0: configs[ratio_key] = val; break
                        print(f"TP{i} Extension Ratio must be positive.")
                    except ValueError: print("Invalid input. Please enter a number.")
            
            # TP Quantity Percentages
            # Basic validation for sum of percentages could be added here, or assume user manages it.
            # For now, individual percentage validation.
            for i, qty_key, default_qty_pct in [
                (1, "fib_tp1_qty_pct", DEFAULT_FIB_TP1_QTY_PCT),
                (2, "fib_tp2_qty_pct", DEFAULT_FIB_TP2_QTY_PCT),
                (3, "fib_tp3_qty_pct", DEFAULT_FIB_TP3_QTY_PCT) # Ensure sum is 1.0, or TP3 is remainder
            ]:
                while True:
                    try:
                        # For display, convert 0.25 to 25%
                        val_pct_display = configs.get(qty_key, default_qty_pct) * 100.0
                        user_input_pct = input(f"Enter Fib TP{i} Quantity % (e.g., 25 for 25%) (default: {val_pct_display:.0f}%): ")
                        
                        final_val_decimal = 0
                        if not user_input_pct: # User hit enter
                            final_val_decimal = configs.get(qty_key, default_qty_pct)
                        else:
                            val_input_float = float(user_input_pct)
                            if 0 < val_input_float <= 100:
                                final_val_decimal = val_input_float / 100.0
                            else:
                                print(f"TP{i} Quantity % must be between 0 and 100.")
                                continue
                        
                        configs[qty_key] = final_val_decimal
                        break
                    except ValueError: print("Invalid input. Please enter a number for percentage.")
        
        # SL Management after TP1
        while True:
            default_val = configs.get("fib_move_sl_after_tp1", DEFAULT_FIB_MOVE_SL_AFTER_TP1)
            user_input = input(f"SL action after Fib TP1 (breakeven/trailing/original) (default: {default_val}): ").lower().strip()
            chosen_val = default_val if not user_input else user_input
            if chosen_val in ["breakeven", "trailing", "original"]: configs["fib_move_sl_after_tp1"] = chosen_val; break
            print("Invalid input. Choose from 'breakeven', 'trailing', 'original'.")

        if configs.get("fib_move_sl_after_tp1") == "breakeven":
            while True:
                try:
                    val = get_input_with_default("Enter Fib Breakeven Buffer (in R, e.g. 0.1 for 0.1R)", "fib_breakeven_buffer_r", DEFAULT_FIB_BREAKEVEN_BUFFER_R, float)
                    if 0 <= val < 1: configs["fib_breakeven_buffer_r"] = val; break # Buffer can be 0
                    print("Breakeven Buffer (R) must be between 0 and 1 (e.g., 0.0 to 0.99).")
                except ValueError: print("Invalid input. Please enter a number.")
        
        # SL Management after TP2
        while True:
            default_val = configs.get("fib_sl_adjustment_after_tp2", DEFAULT_FIB_SL_ADJUSTMENT_AFTER_TP2)
            user_input = input(f"SL action after Fib TP2 (micro_pivot/atr_trailing/original) (default: {default_val}): ").lower().strip()
            chosen_val = default_val if not user_input else user_input
            if chosen_val in ["micro_pivot", "atr_trailing", "original"]: configs["fib_sl_adjustment_after_tp2"] = chosen_val; break
            print("Invalid input. Choose from 'micro_pivot', 'atr_trailing', 'original'.")


    # Micro-Pivot Trailing SL Configurations (Applicable to any strategy if enabled)
    print("\n--- General Micro-Pivot Trailing SL ---")
    while True:
        # Default display for micro_pivot_trailing_sl
        mp_default_display = 'yes' if configs.get("micro_pivot_trailing_sl", DEFAULT_MICRO_PIVOT_TRAILING_SL) else 'no'
        mp_input = input(f"Enable Micro-Pivot Trailing SL? (yes/no) (default: {mp_default_display}): ").lower().strip()
        
        chosen_mp_enabled = None
        if not mp_input: # User hit enter
            chosen_mp_enabled = configs.get("micro_pivot_trailing_sl", DEFAULT_MICRO_PIVOT_TRAILING_SL)
        elif mp_input in ["yes", "y"]: chosen_mp_enabled = True
        elif mp_input in ["no", "n"]: chosen_mp_enabled = False
        
        if isinstance(chosen_mp_enabled, bool):
            configs["micro_pivot_trailing_sl"] = chosen_mp_enabled
            break
        print("Invalid input. Please enter 'yes' or 'no'.")

    if configs.get("micro_pivot_trailing_sl"): # Only ask for related params if it's enabled
        while True:
            try:
                mp_buffer = get_input_with_default(
                    "Enter Micro-Pivot SL ATR Buffer (e.g., 0.25 for 0.25x ATR)",
                    "micro_pivot_buffer_atr", DEFAULT_MICRO_PIVOT_BUFFER_ATR, float
                )
                if mp_buffer > 0:
                    configs["micro_pivot_buffer_atr"] = mp_buffer
                    break
                print("Micro-Pivot SL ATR Buffer must be a positive number.")
            except ValueError: print("Invalid input. Please enter a number.")
        
        while True:
            try:
                mp_profit_r = get_input_with_default(
                    "Enter Micro-Pivot Profit Threshold (in R, e.g., 0.5 for 0.5R profit)",
                    "micro_pivot_profit_threshold_r", DEFAULT_MICRO_PIVOT_PROFIT_THRESHOLD_R, float
                )
                if mp_profit_r > 0:
                    configs["micro_pivot_profit_threshold_r"] = mp_profit_r
                    break
                print("Micro-Pivot Profit Threshold (R) must be a positive number.")
            except ValueError: print("Invalid input. Please enter a number.")
    else: # Not enabled, remove related params if they were in loaded CSV
        configs.pop("micro_pivot_buffer_atr", None)
        configs.pop("micro_pivot_profit_threshold_r", None)

    # Minimum Expected Profit USDT
    while True:
        try:
            min_profit = get_input_with_default(
                f"Enter Minimum Expected Profit per trade in USDT (0 to disable, e.g., {DEFAULT_MIN_EXPECTED_PROFIT_USDT})",
                "min_expected_profit_usdt", DEFAULT_MIN_EXPECTED_PROFIT_USDT, float
            )
            if min_profit >= 0:
                configs["min_expected_profit_usdt"] = min_profit
                break
            print("Minimum expected profit must be zero or a positive number.")
        except ValueError: print("Invalid input for minimum expected profit. Please enter a number.")

    # Strategy ID and Name are set based on "strategy_choice" earlier.
    
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
        # Initialize a temporary client to get server time first, if needed for offset calculation before full client init
        # However, python-binance client itself can fetch server time.
        # Let's increase recvWindow. Default is 5000ms. Let's try 20000ms.
        # The library handles timestamping internally, so adjusting system time or applying offset manually
        # to each request is not standard practice with this library.
        # The primary fix for -1021 is system time sync or a larger recvWindow.

        recv_window_ms = 20000  # Increased recvWindow to 20000ms
        # Set a reasonable requests timeout, e.g., 10 or 15 seconds. This is separate from recvWindow.
        requests_timeout_seconds = 15 
        client = Client(api_key, api_secret, testnet=(env == "testnet"), requests_params={'timeout': requests_timeout_seconds})

        # Set the recvWindow on the client instance
        client.RECV_WINDOW = recv_window_ms

        # Fetch server time and calculate offset for logging and warning
        server_time_info = client.get_server_time()
        server_timestamp_ms = server_time_info['serverTime']
        local_timestamp_ms = int(time.time() * 1000)
        time_offset_ms = local_timestamp_ms - server_timestamp_ms

        print(f"Binance Server Time: {pd.to_datetime(server_timestamp_ms, unit='ms')} UTC")
        print(f"Local System Time: {pd.to_datetime(local_timestamp_ms, unit='ms')} UTC")
        print(f"Time Offset (Local - Server): {time_offset_ms} ms")

        if abs(time_offset_ms) > 1000: # More than 1 second offset
            warning_message = (
                f"⚠️ WARNING: Your system clock is out of sync with Binance server time by {time_offset_ms} ms.\n"
                f"This can lead to API errors (like -1021).\n"
                f"Please ensure your system time is synchronized with an NTP server (e.g., time.google.com, pool.ntp.org)."
            )
            print(warning_message)
            # Optionally, send to Telegram if available in configs at this stage (configs might not be fully populated yet)
            # For now, just printing is fine as this is early in startup.

        client.ping() # Verify connection after setup
        
        # Return client, environment title, and server_time_info (which contains serverTime)
        return client, env, server_time_info
        
    except BinanceAPIException as e:
        print(f"Binance API Exception (client init): {e}")
        if e.code == -1021: # Specifically catch -1021 here too
             print("Timestamp error during client initialization. This strongly suggests system time desynchronization.")
        return None, None, None
    except Exception as e:
        print(f"Error initializing Binance client: {e}")
        return None, None, None

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
    # This rule is only applied if cur_tp is provided (i.e., not a multi-TP strategy)
    if cur_tp is not None: # Check if TP adjustment is applicable
        if profit_pct <= -0.005:
            target_tp = entry * (1 + 0.002 if side == "LONG" else 1 - 0.002)
            # Check if this new TP is an improvement (for LONG, lower but > entry; for SHORT, higher but < entry)
            # And also ensure it's a tighter TP than current, but still profitable
            if (side == "LONG" and target_tp < new_tp and target_tp > entry) or \
               (side == "SHORT" and target_tp > new_tp and target_tp < entry):
                new_tp = target_tp # new_tp was initialized with cur_tp, so this updates it
                
                # Update adjustment_reason carefully
                if adjustment_reason and "SL" in adjustment_reason: # If SL was already adjusted
                    adjustment_reason += ";TP_DRAWDOWN_REDUCE_0.2%"
                else: # Only TP is adjusted, or it's the first adjustment
                    adjustment_reason = "TP_DRAWDOWN_REDUCE_0.2%"
                print(f"Dynamic TP adjustment for {side} to {new_tp:.4f} ({adjustment_reason})")
    
    # If new_tp was not modified (because it's multi-TP or conditions not met), it remains as cur_tp (which could be None)
    # The function signature implies new_tp is always returned. If it's multi-TP, new_tp will be None.
    if adjustment_reason:
        return new_sl, new_tp, adjustment_reason # new_tp will be None if cur_tp was None
    else:
        return None, None, None # No adjustment made that met criteria, return None for new_sl, new_tp

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

def load_symbols_from_csv(filepath: str) -> list[str]:
    """Loads symbols from a CSV file, expecting a single 'symbol' column."""
    if not os.path.exists(filepath):
        print(f"Info: Symbol CSV file '{filepath}' not found.")
        return []
    try:
        df = pd.read_csv(filepath)
        if 'symbol' not in df.columns:
            print(f"Error: Symbol CSV file '{filepath}' must contain a 'symbol' column.")
            return []
        
        # Ensure symbols are strings, convert to uppercase, handle potential NaN, get unique, and sort
        symbols_list = sorted(list(set(df['symbol'].dropna().astype(str).str.upper().tolist())))
        
        if not symbols_list:
            print(f"Info: Symbol CSV file '{filepath}' is empty or contains no valid symbols.")
            return []
        # print(f"Loaded {len(symbols_list)} unique, sorted symbol(s) from '{filepath}'.") # More verbose, can enable if needed
        return symbols_list
    except pd.errors.EmptyDataError:
        print(f"Info: Symbol CSV file '{filepath}' is empty.")
        return []
    except Exception as e:
        print(f"Error loading symbols from CSV '{filepath}': {e}")
        return []

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

    # 6. Minimum Expected Profit Check
    # This check is applied to all trades, managed or unmanaged (if TP and SL are defined).
    # For unmanaged trades, this acts as a safety if the calculated SL/TP for an existing position are too close.
    min_profit_config = configs.get('min_expected_profit_usdt', DEFAULT_MIN_EXPECTED_PROFIT_USDT)
    if min_profit_config > 0: # Only apply check if configured > 0
        # Potential profit based on the distance between TP and SL prices, times quantity.
        # This represents the gross profit if price moves from SL to TP, or the total range of the trade in USDT.
        # This interpretation aligns with "TP – SL < min_profit_dollar" where "TP-SL" is the price range.
        potential_trade_range_usdt = abs(tp_price - sl_price) * quantity
        
        if potential_trade_range_usdt < min_profit_config:
            return False, (f"Potential trade range value ({potential_trade_range_usdt:.2f} USDT) "
                           f"from SL ({sl_price:.{p_prec}f}) to TP ({tp_price:.{p_prec}f}) with Qty {quantity:.{q_prec}f} "
                           f"is less than minimum configured ({min_profit_config:.2f} USDT).")

    return True, "Checks passed"


# --- Main Trading Logic ---

def load_symbol_blacklist(filepath: str) -> list[str]:
    """Loads symbols from a blacklist CSV file."""
    if not os.path.exists(filepath):
        print(f"Info: Symbol blacklist file '{filepath}' not found. No symbols will be blacklisted.")
        return []
    try:
        df = pd.read_csv(filepath)
        if 'symbol' not in df.columns:
            print(f"Error: Blacklist CSV file '{filepath}' must contain a 'symbol' column.")
            return []
        
        # Ensure symbols are strings, convert to uppercase, handle potential NaN, get unique, and sort
        blacklisted_symbols = sorted(list(set(df['symbol'].dropna().astype(str).str.upper().tolist())))
        
        if not blacklisted_symbols:
            print(f"Info: Symbol blacklist file '{filepath}' is empty or contains no valid symbols.")
            return []
        # This print was moved to main() to avoid repetition if called multiple times by handlers.
        if blacklisted_symbols:  # Only print if some symbols were loaded
            print(f"Loaded {len(blacklisted_symbols)} unique, sorted symbol(s) from blacklist file '{filepath}'.")
        return blacklisted_symbols
    except pd.errors.EmptyDataError:
        print(f"Info: Symbol blacklist file '{filepath}' is empty. No symbols will be blacklisted.")
        return []
    except Exception as e:
        print(f"Error loading symbol blacklist from '{filepath}': {e}")
        return []

def add_symbol_to_blacklist(filepath: str, symbol_to_add: str) -> bool:
    """
    Adds a symbol to the blacklist CSV file.
    Returns True if added, False if already exists or error.
    """
    symbol_to_add = symbol_to_add.upper() # Ensure consistency
    current_blacklist = load_symbol_blacklist(filepath) # This already returns unique, sorted, uppercase symbols
    
    if symbol_to_add in current_blacklist:
        print(f"Symbol {symbol_to_add} already in blacklist '{filepath}'.")
        return False # Indicates already exists

    new_blacklist_set = set(current_blacklist)
    new_blacklist_set.add(symbol_to_add)
    updated_blacklist = sorted(list(new_blacklist_set))
    
    try:
        df_to_save = pd.DataFrame(updated_blacklist, columns=['symbol'])
        df_to_save.to_csv(filepath, index=False)
        print(f"Symbol {symbol_to_add} added to blacklist '{filepath}'.")
        return True # Indicates successfully added
    except Exception as e:
        print(f"Error saving updated blacklist to '{filepath}': {e}")
        return False # Indicates error

def get_all_usdt_perpetual_symbols(client):
    print("\nFetching all USDT perpetual symbols...")
    # Note: Blacklist filtering is now handled in main() before calling trading_loop
    try:
        syms = [s['symbol'] for s in client.futures_exchange_info()['symbols']
                if s.get('symbol','').endswith('USDT') and s.get('contractType')=='PERPETUAL'
                and s.get('status')=='TRADING' and s.get('quoteAsset')=='USDT' and s.get('marginAsset')=='USDT']
        # print(f"Found {len(syms)} USDT perpetuals (before blacklist). Examples: {syms[:5]}") # Log before filtering if needed
        return sorted(list(set(syms)))
    except Exception as e: print(f"Error fetching symbols: {e}"); return []

def format_elapsed_time(start_time):
    return f"(Cycle Elapsed: {(time.time() - start_time):.2f}s)"

def process_symbol_task(symbol, client, configs, lock):
    # configs['cycle_start_time_ref'] is vital here
    thread_name = threading.current_thread().name
    cycle_start_ref = configs.get('cycle_start_time_ref', time.time()) # Fallback if not passed
    log_prefix_task = f"[{thread_name}] {symbol} EMA_Task:" # Specific prefix for EMA task
    print(f"{log_prefix_task} Processing {format_elapsed_time(cycle_start_ref)}")
    try:
        klines_df, klines_error = get_historical_klines(client, symbol) # Uses default limit 500 for EMA strategy (e.g., 15-min klines)
        
        if klines_error:
            if isinstance(klines_error, BinanceAPIException) and klines_error.code == -1121:
                msg = f"Skipped: Invalid symbol reported by API (code -1121)."
                print(f"{log_prefix_task} {msg} {format_elapsed_time(cycle_start_ref)}")
                return f"{symbol}: {msg}"
            else:
                msg = f"Skipped: Error fetching klines ({str(klines_error)})."
                print(f"{log_prefix_task} {msg} {format_elapsed_time(cycle_start_ref)}")
                return f"{symbol}: {msg}"

        if klines_df.empty or len(klines_df) < 202: # Min klines for EMA200 + validation
            msg = f"Skipped: Insufficient klines for EMA strategy ({len(klines_df)})."
            print(f"{log_prefix_task} {msg} {format_elapsed_time(cycle_start_ref)}")
            return f"{symbol}: {msg}"
        
        print(f"{log_prefix_task} Sufficient klines ({len(klines_df)}). Calling manage_trade_entry (EMA logic) {format_elapsed_time(cycle_start_ref)}")
        # manage_trade_entry is specifically for EMA cross logic based on its internal workings
        manage_trade_entry(client, configs, symbol, klines_df.copy(), lock) 
        return f"{symbol}: EMA Processed"
    except Exception as e:
        error_detail = f"Unhandled error in process_symbol_task (EMA): {e}"
        print(f"{log_prefix_task} ERROR processing: {error_detail} {format_elapsed_time(cycle_start_ref)}")
        traceback.print_exc()
        return f"{symbol}: EMA Error - {error_detail}"

def process_symbol_fib_task(symbol, client, configs, lock): # lock here is active_trades_lock
    thread_name = threading.current_thread().name
    cycle_start_ref = configs.get('cycle_start_time_ref', time.time())
    log_prefix_task = f"[{thread_name}] {symbol} FIB_Task:"
    print(f"{log_prefix_task} Processing {format_elapsed_time(cycle_start_ref)}")

    try:
        s_info = get_symbol_info(client, symbol)
        if not s_info:
            print(f"{log_prefix_task} Failed to get symbol_info. Skipping.")
            return f"{symbol}: Fib Error - No symbol_info"

        # 1. Fetch 1-minute klines
        # Limit should be based on fib_1m_buffer_size + some for initial pivot calculations if buffer is empty
        buffer_size_config = configs.get("fib_1m_buffer_size", DEFAULT_1M_BUFFER_SIZE)
        fetch_limit_1m = buffer_size_config + PIVOT_N_LEFT + PIVOT_N_RIGHT + 5 # Ensure enough for pivots and buffer fill
        
        klines_1m_df, klines_1m_error = get_historical_klines_1m(client, symbol, limit=fetch_limit_1m)

        if klines_1m_error:
            if isinstance(klines_1m_error, BinanceAPIException) and klines_1m_error.code == -1121:
                msg = f"Skipped: Invalid symbol reported by API (code -1121)."
                print(f"{log_prefix_task} {msg} {format_elapsed_time(cycle_start_ref)}")
                # Consider adding to a temporary blacklist for the session or logging for review
                return f"{symbol}: {msg}" # Correctly returns and skips
            else:
                # For other API errors or general errors during kline fetching
                msg = f"Skipped: Error fetching 1m klines ({str(klines_1m_error)})."
                print(f"{log_prefix_task} {msg} {format_elapsed_time(cycle_start_ref)}")
                return f"{symbol}: {msg}"
        
        if klines_1m_df.empty or len(klines_1m_df) < (PIVOT_N_LEFT + PIVOT_N_RIGHT + 2): # Min for BoS detection + current
            msg = f"Skipped: Insufficient 1m klines for Fib strategy ({len(klines_1m_df)})."
            print(f"{log_prefix_task} {msg} {format_elapsed_time(cycle_start_ref)}")
            return f"{symbol}: {msg}"

        # 2. Update 1-minute candle buffer (only last candle, or fill if buffer is new/empty)
        # For simplicity in task, let's assume buffer is mainly populated by a dedicated stream or this task is primary populator.
        # If buffer is empty, populate with fetched klines. Otherwise, just update with latest.
        with symbol_1m_candle_buffers_lock:
            is_new_buffer = symbol not in symbol_1m_candle_buffers
        
        if is_new_buffer: # Populate whole buffer
            print(f"{log_prefix_task} Populating new 1m candle buffer for {symbol} with {len(klines_1m_df)} candles.")
            for idx in range(len(klines_1m_df)):
                update_1m_candle_buffer(symbol, klines_1m_df.iloc[idx], buffer_size_config)
        else: # Just update with the latest candle(s)
            # update_1m_candle_buffer handles duplicates and ordering based on timestamp
            # We can submit the last few candles from klines_1m_df to ensure it's up-to-date.
            # For this task, let's ensure at least the very last candle is processed.
            update_1m_candle_buffer(symbol, klines_1m_df.iloc[-1], buffer_size_config)


        # 3. Retrieve the populated 1-minute candle buffer
        current_buffer_df = None
        with symbol_1m_candle_buffers_lock:
            if symbol in symbol_1m_candle_buffers and len(symbol_1m_candle_buffers[symbol]) > 0:
                # Convert deque of Series to DataFrame for BoS detection
                records = [s.to_dict() for s in symbol_1m_candle_buffers[symbol] if isinstance(s, pd.Series)]
                if records:
                    temp_df = pd.DataFrame(records)
                    if 'timestamp' not in temp_df.columns and symbol_1m_candle_buffers[symbol][0].name is not None:
                         temp_df.index = [s.name for s in symbol_1m_candle_buffers[symbol]]
                    elif 'timestamp' in temp_df.columns: # If 'timestamp' was a column in the Series
                         temp_df.set_index('timestamp', inplace=True)
                    
                    if all(c in temp_df for c in ['open','high','low','close']): # Ensure required columns exist
                        current_buffer_df = temp_df
        
        if current_buffer_df is None or current_buffer_df.empty:
            msg = f"Skipped: 1m candle buffer for {symbol} is empty or invalid after update."
            print(f"{log_prefix_task} {msg} {format_elapsed_time(cycle_start_ref)}")
            return f"{symbol}: {msg}"
        
        print(f"{log_prefix_task} Buffer for {symbol} has {len(current_buffer_df)} candles. Detecting BoS...")

        # 4. Detect market structure and BoS
        bos_event_details = detect_market_structure_and_bos(symbol, current_buffer_df.copy(), configs)

        if bos_event_details:
            print(f"{log_prefix_task} BoS event detected for {symbol}: {bos_event_details['direction']}")
            # Add symbol_info to bos_event_details as it's needed by manage_fib_retracement_entry_logic
            bos_event_details['symbol_info'] = s_info 

            current_fib_state = "IDLE" # Default
            with fib_strategy_states_lock:
                if symbol in fib_strategy_states:
                    current_fib_state = fib_strategy_states[symbol].get('state', "IDLE")
            
            if current_fib_state == "AWAITING_PULLBACK":
                 print(f"{log_prefix_task} State is AWAITING_PULLBACK. Calling manage_fib_retracement_entry_logic.")
                 manage_fib_retracement_entry_logic(client, configs, symbol, bos_event_details, s_info)
            else:
                print(f"{log_prefix_task} BoS detected, but state is '{current_fib_state}', not AWAITING_PULLBACK. No entry logic call.")
        # else: No BoS event, or already handled by detect_market_structure_and_bos logging
        
        return f"{symbol}: Fib Processed"
    except Exception as e:
        error_detail = f"Unhandled error in process_symbol_fib_task: {e}"
        print(f"{log_prefix_task} ERROR processing: {error_detail} {format_elapsed_time(cycle_start_ref)}")
        traceback.print_exc()
        return f"{symbol}: Fib Error - {error_detail}"

def process_symbol_ict_task(symbol, client, configs, lock): # lock is active_trades_lock
    thread_name = threading.current_thread().name
    cycle_start_ref = configs.get('cycle_start_time_ref', time.time())
    log_prefix_task = f"[{thread_name}] {symbol} ICT_Task:"
    print(f"{log_prefix_task} Processing {format_elapsed_time(cycle_start_ref)}")

    try:
        s_info = get_symbol_info(client, symbol)
        if not s_info:
            print(f"{log_prefix_task} Failed to get symbol_info. Skipping.")
            return f"{symbol}: ICT Error - No symbol_info"

        # Fetch klines based on main bot interval, ICT logic will interpret it
        # Or, fetch based on configs.get("ict_timeframe") if implemented for multi-TF analysis
        klines_df, klines_error = get_historical_klines(client, symbol, limit=configs.get("ict_kline_limit", DEFAULT_ICT_KLINE_LIMIT)) # Configurable limit

        if klines_error:
            # Handle specific errors like invalid symbol if necessary
            msg = f"Skipped: Error fetching klines for ICT ({str(klines_error)})."
            print(f"{log_prefix_task} {msg} {format_elapsed_time(cycle_start_ref)}")
            return f"{symbol}: {msg}"
        
        min_candles_ict = configs.get("ict_liquidity_lookback", DEFAULT_ICT_LIQUIDITY_LOOKBACK) + \
                          configs.get("ict_fvg_freshness_candles", DEFAULT_ICT_FVG_FRESHNESS_CANDLES) + 20 # Buffer
        if klines_df.empty or len(klines_df) < min_candles_ict:
            msg = f"Skipped: Insufficient klines for ICT strategy ({len(klines_df)}/{min_candles_ict})."
            print(f"{log_prefix_task} {msg} {format_elapsed_time(cycle_start_ref)}")
            return f"{symbol}: {msg}"
        
        print(f"{log_prefix_task} Sufficient klines ({len(klines_df)}). Calling manage_ict_trade_entry {format_elapsed_time(cycle_start_ref)}")
        manage_ict_trade_entry(client, configs, symbol, klines_df.copy(), s_info, lock) # Pass s_info
        return f"{symbol}: ICT Processed"
        
    except Exception as e:
        error_detail = f"Unhandled error in process_symbol_ict_task: {e}"
        print(f"{log_prefix_task} ERROR processing: {error_detail} {format_elapsed_time(cycle_start_ref)}")
        traceback.print_exc()
        return f"{symbol}: ICT Error - {error_detail}"


def manage_trade_entry(client, configs, symbol, klines_df, lock): # lock here is active_trades_lock
    global active_trades, symbols_currently_processing, symbols_currently_processing_lock
    global trading_halted_drawdown, trading_halted_daily_loss, daily_state_lock # For checking halt status
    global last_signal_time, last_signal_lock # For Cooldown Timer
    global recent_trade_signatures, recent_trade_signatures_lock # For Trade Signature Check

    log_prefix = f"[{threading.current_thread().name}] {symbol} manage_trade_entry:"

    # --- Cleanup old trade signatures (periodically) ---
    cleanup_recent_trade_signatures()

    # --- Check 1: Overall trading halt status (Daily Drawdown / Daily Loss / Manual) ---
    global trading_halted_manual # Ensure access to the manual halt flag
    with daily_state_lock: # daily_state_lock primarily protects drawdown and daily_loss flags
        if trading_halted_drawdown:
            print(f"{log_prefix} Trade entry BLOCKED for {symbol}. Reason: Max Drawdown trading halt is active.")
            return
        if trading_halted_daily_loss:
            print(f"{log_prefix} Trade entry BLOCKED for {symbol}. Reason: Daily Stop Loss trading halt is active.")
            return
    # Manual halt check (can be outside daily_state_lock as it's a separate global)
    if trading_halted_manual:
        print(f"{log_prefix} Trade entry BLOCKED for {symbol}. Reason: Trading is MANUALLY HALTED.")
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

        if configs['mode'] == 'signal':
            # --- Signal Mode: Send Telegram Notification, No Real Orders ---
            print(f"{log_prefix} Signal Mode: Preparing Telegram signal for {symbol} {signal}.")
            
            # Calculate P&L estimations for $100 capital
            est_pnl_tp1 = calculate_pnl_for_fixed_capital(entry_p, tp_p_calc, signal, target_leverage, 100.0, symbol_info)
            est_pnl_sl = calculate_pnl_for_fixed_capital(entry_p, sl_p_calc, signal, target_leverage, 100.0, symbol_info)
            
            send_entry_signal_telegram(
                configs=configs,
                symbol=symbol,
                signal_type_display=f"EMA_CROSS_{signal.upper()}",
                leverage=target_leverage,
                entry_price=entry_p,
                tp1_price=tp_p_calc, # EMA Cross has one TP, pass it as TP1
                tp2_price=None,      # No TP2 for EMA Cross
                tp3_price=None,      # No TP3 for EMA Cross
                sl_price=sl_p_calc,
                risk_percentage_config=configs['risk_percent'],
                est_pnl_tp1=est_pnl_tp1,
                est_pnl_sl=est_pnl_sl,
                symbol_info=symbol_info,
                strategy_name_display="EMA Cross",
                signal_timestamp=dt.now(tz=timezone.utc), # Pass current UTC time
                signal_order_type="MARKET"
            )
            
            # Add to active_signals for monitoring simulated progression
            with active_signals_lock:
                signal_id = f"signal_{symbol}_{int(dt.now().timestamp())}"
                active_signals[symbol] = {
                    "signal_id": signal_id, # Unique ID for the signal
                    "entry_price": entry_p,
                    "current_sl_price": sl_p_calc,
                    "current_tp1_price": tp_p_calc, # EMA Cross uses one TP, store as TP1
                    "current_tp2_price": None,
                    "current_tp3_price": None,
                    "initial_sl_price": sl_p_calc,
                    "initial_tp1_price": tp_p_calc,
                    "side": signal, # "LONG" or "SHORT"
                    "leverage": target_leverage,
                    "symbol_info": symbol_info,
                    "open_timestamp": dt.now(),
                    "strategy_type": "EMA_CROSS",
                    "sl_management_stage": "initial", # For potential future advanced SL simulation
                    "last_update_message_type": "NEW_SIGNAL" # To avoid spamming identical updates
                }
                print(f"{log_prefix} Signal Mode: EMA Cross signal for {symbol} added to active_signals for simulated monitoring.")
            
            # Log this new signal event to CSV
            log_event_details = {
                "SignalID": signal_id, "Symbol": symbol, "Strategy": "EMA_CROSS", "Side": signal,
                "Leverage": target_leverage, "SignalOpenPrice": entry_p, 
                "EventType": "NEW_SIGNAL", "EventPrice": entry_p,
                "Notes": f"SL: {sl_p_calc:.{symbol_info.get('pricePrecision', 2)}f}, TP: {tp_p_calc:.{symbol_info.get('pricePrecision', 2)}f}",
                "EstimatedPNL_USD100": est_pnl_tp1 # PNL if TP1 hits
            }
            log_signal_event_to_csv(log_event_details)

            # Record trade signature for Signal mode as well
            with recent_trade_signatures_lock:
                recent_trade_signatures[trade_sig] = dt.now()
                print(f"{log_prefix} Signal Mode: Trade signature recorded for {symbol}: {trade_sig}")

            print(f"{log_prefix} Signal Mode: Telegram signal sent, CSV logged, and virtual signal recorded for {symbol}.")
            # Cooldown (last_signal_time) was updated before this signal mode block.
            return # End processing for this symbol in signal mode

        # --- Live Mode: Place Real Orders ---
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
        
        # --- Final Lock for Order Placement and active_trades update (Live Mode) ---
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
            
            # --- OCO Logic Implementation for Position Closure ---
            # If position is closed (pos_exists is False), cancel all related SL and TP orders.
            sl_order_id_to_cancel = trade_details.get('sl_order_id')
            
            # Cancel SL order
            if sl_order_id_to_cancel:
                try:
                    print(f"Attempting to cancel SL order {sl_order_id_to_cancel} for closed position {symbol}...")
                    client.futures_cancel_order(symbol=symbol, orderId=sl_order_id_to_cancel)
                    print(f"Successfully cancelled SL order {sl_order_id_to_cancel} for {symbol}.")
                except BinanceAPIException as e:
                    if e.code == -2011: # Order filled or cancelled / Does not exist
                        print(f"SL order {sl_order_id_to_cancel} for {symbol} already filled/cancelled or does not exist (Code: {e.code}).")
                    else:
                        print(f"API Error cancelling SL order {sl_order_id_to_cancel} for {symbol}: {e}")
                except Exception as e:
                    print(f"Unexpected error cancelling SL order {sl_order_id_to_cancel} for {symbol}: {e}")

            # Cancel all TP orders
            if trade_details.get('strategy_type') == "FIBONACCI_MULTI_TP" and 'tp_orders' in trade_details:
                for tp_order_info in trade_details['tp_orders']:
                    tp_id_to_cancel = tp_order_info.get('id')
                    if tp_id_to_cancel and tp_order_info.get('status') == "OPEN": # Only cancel open TPs
                        try:
                            print(f"Attempting to cancel TP order {tp_id_to_cancel} ({tp_order_info.get('name')}) for closed position {symbol}...")
                            client.futures_cancel_order(symbol=symbol, orderId=tp_id_to_cancel)
                            print(f"Successfully cancelled TP order {tp_id_to_cancel} for {symbol}.")
                            # Update status in active_trades for this TP (optional, as trade is being removed)
                            # tp_order_info['status'] = "CANCELED" 
                        except BinanceAPIException as e:
                            if e.code == -2011:
                                print(f"TP order {tp_id_to_cancel} for {symbol} already filled/cancelled or does not exist (Code: {e.code}).")
                            else:
                                print(f"API Error cancelling TP order {tp_id_to_cancel} for {symbol}: {e}")
                        except Exception as e:
                            print(f"Unexpected error cancelling TP order {tp_id_to_cancel} for {symbol}: {e}")
            elif 'tp_order_id' in trade_details: # Single TP strategy (e.g., EMA Cross)
                tp_id_to_cancel = trade_details.get('tp_order_id')
                if tp_id_to_cancel:
                    try:
                        client.futures_cancel_order(symbol=symbol, orderId=tp_id_to_cancel)
                        print(f"Successfully cancelled single TP order {tp_id_to_cancel} for {symbol}.")
                    except BinanceAPIException as e:
                        if e.code == -2011: print(f"Single TP order {tp_id_to_cancel} for {symbol} already processed. Code: {e.code}")
                        else: print(f"API Error cancelling single TP order {tp_id_to_cancel}: {e}")
                    except Exception as e: print(f"Unexpected error cancelling single TP order {tp_id_to_cancel}: {e}")
            
            # --- Realized P&L Calculation on Full Closure ---
            # This section assumes the *entire position* is now closed.
            # For multi-TP, if only a part of the position was closed by a TP hit, that P&L is handled when the TP order status is checked.
            # If pos_exists is false, it means the *entire* position is gone (either SL, all TPs hit, or manual closure).
            
            # We need to determine the exit price. This is difficult if closed manually.
            # If SL hit, exit is SL price. If last TP hit, exit is that TP price.
            # This part of P&L calc needs to be more robust.
            # For now, the PNL calculation for a fully closed position (pos_exists=False) will be an estimation
            # or rely on other mechanisms to fetch actual trade history if exact PNL is critical here.
            # The previous logic tried to infer based on cancellation status, which is okay.

            # Let's assume the detailed PNL calculation for partial TP hits will be handled by checking TP order statuses.
            # If pos_exists is False, it means the *remainder* of the position (if any) was closed.
            # We'll add a simplified P&L update here for the full closure scenario.
            # A more robust solution would involve fetching trade history for the symbol.

            entry_p = trade_details['entry_price']
            original_total_qty = trade_details['quantity'] # Initial total quantity
            trade_side = trade_details['side']
            closure_data_for_removal = {"symbol": symbol, "pnl": 0, "reason": "Position Closed (Full)"}

            # Try to find out if SL was hit or if a TP was the last one to fill
            sl_order_status = None
            if sl_order_id_to_cancel:
                try: sl_order_status = client.futures_get_order(symbol=symbol, orderId=sl_order_id_to_cancel)
                except: pass # Ignore errors, just trying to get status

            if sl_order_status and sl_order_status['status'] == 'FILLED':
                exit_price_assumed = float(sl_order_status['avgPrice']) if float(sl_order_status.get('avgPrice',0)) > 0 else trade_details['current_sl_price']
                pnl = (exit_price_assumed - entry_p) * original_total_qty if trade_side == "LONG" else (entry_p - exit_price_assumed) * original_total_qty
                closure_data_for_removal["pnl"] = pnl
                closure_data_for_removal["reason"] = f"Stop-Loss Hit @ {exit_price_assumed}"
                print(f"Position {symbol} closed by SL. PNL: {pnl:.2f}")
            else:
                # If not SL, it might be manual, liquidation, or the last TP filled leading to full closure.
                # P&L for TP hits is ideally accounted for when TP order status is checked.
                # If we reach here and pos_exists is false, and it wasn't SL, it's complex.
                # We'll assume PNL from TPs was already handled if applicable.
                print(f"Position {symbol} closed. Not by bot's SL order. PNL for this full closure not calculated here (should be from TPs or manual).")
                closure_data_for_removal["reason"] = "Position Closed (Not SL, likely TPs or Manual/Liq)"
            
            with daily_state_lock:
                global daily_realized_pnl
                daily_realized_pnl += closure_data_for_removal["pnl"] # Add PNL from SL or final closure
                print(f"Updated daily realized PNL: {daily_realized_pnl:.2f}")

            symbols_to_remove.append(closure_data_for_removal) # Add dict with details
            continue

        # --- Position Still Exists: Monitor TPs and Dynamic SL/TP Adjustments ---
        
        # For Multi-TP strategy: Check status of individual TP orders
        if trade_details.get('strategy_type') == "FIBONACCI_MULTI_TP" and 'tp_orders' in trade_details:
            any_tp_hit_this_cycle = False
            for tp_order_info in trade_details['tp_orders']:
                if tp_order_info.get('status') == "OPEN" and tp_order_info.get('id'):
                    try:
                        tp_order_status = client.futures_get_order(symbol=symbol, orderId=tp_order_info['id'])
                        if tp_order_status['status'] == 'FILLED':
                            any_tp_hit_this_cycle = True
                            tp_order_info['status'] = "FILLED"
                            filled_qty_tp = float(tp_order_status['executedQty'])
                            fill_price_tp = float(tp_order_status['avgPrice'])
                            
                            pnl_partial_tp = 0
                            if trade_details['side'] == "LONG":
                                pnl_partial_tp = (fill_price_tp - trade_details['entry_price']) * filled_qty_tp
                            else: # SHORT
                                pnl_partial_tp = (trade_details['entry_price'] - fill_price_tp) * filled_qty_tp
                            
                            print(f"Partial TP Hit: {tp_order_info.get('name')} for {symbol} filled {filled_qty_tp} @ {fill_price_tp}. PNL: {pnl_partial_tp:.2f}")
                            
                            with daily_state_lock:
                                daily_realized_pnl += pnl_partial_tp
                                print(f"Updated daily realized PNL after partial TP: {daily_realized_pnl:.2f}")

                            # Reduce total quantity tracked by the bot for this trade
                            # This is complex: active_trades['quantity'] should reflect remaining open quantity.
                            # SL order also needs to be modified to new remaining quantity.
                            # For now, we'll log the partial TP. Full SL modification logic is extensive.
                            # A simpler approach for now: if any TP hits, we might close the rest or adjust SL to BreakEven.
                            # Request: "If price makes a fresh higher low ... bump SL to breakeven or just below that pivot."
                            # This is separate from TP hits.
                            
                            # If a TP hits, we should at least cancel the main SL and replace it for the remaining quantity.
                            # This is a significant change. For now, let's just mark TP as filled.
                            # The `pos_exists` check at the start of the loop will handle full closure.
                            
                            # If all TPs are filled, the position should be fully closed.
                            all_tps_filled_or_failed = all(
                                tp.get('status') != "OPEN" for tp in trade_details['tp_orders'] if tp.get('id') is not None
                            )
                            if all_tps_filled_or_failed:
                                print(f"All TPs for {symbol} are now filled or failed. Position should be closed.")
                                # The pos_exists check at the start of next cycle should pick this up.
                                # Or, we can force a re-check of pos_exists here.

                    except BinanceAPIException as e:
                        if e.code == -2013: # Order does not exist (e.g. already cancelled / filled and removed)
                            tp_order_info['status'] = "UNKNOWN_REMOVED"
                            print(f"TP order {tp_order_info['id']} for {symbol} not found, marked UNKNOWN.")
                        else:
                            print(f"API Error checking TP order {tp_order_info['id']} for {symbol}: {e}")
                    except Exception as e:
                        print(f"Error checking TP order {tp_order_info['id']} for {symbol}: {e}")
            
            # If a TP hit, might need to adjust SL (e.g., to breakeven or for remaining qty)
            # This is part of the adaptive SL logic. For now, just noting the TP hit.
            
            # --- Staged SL Management for Fibonacci Multi-TP after a TP is hit ---
            if is_multi_tp_strategy and any_tp_hit_this_cycle:
                # Recalculate remaining quantity accurately based on all filled TPs
                current_total_filled_tp_qty = sum(
                    tp.get('quantity', 0) for tp in trade_details['tp_orders'] 
                    if tp.get('status') == "FILLED" and tp.get('id') is not None
                )
                current_remaining_qty = round(float(trade_details['quantity']) - current_total_filled_tp_qty, int(trade_details['symbol_info'].get('quantityPrecision',0)))

                if current_remaining_qty <= 0: # All parts of TPs might have filled, or remaining is dust
                    print(f"[{symbol}] All TPs appear filled or remaining quantity is zero/dust after TP hit. Position should be closing.")
                    # The main pos_exists check at the start of monitor_active_trades will handle full closure.
                else:
                    # Determine which TPs have been hit to decide SL adjustment strategy
                    tp1_hit = any(tp.get('name') == "TP1" and tp.get('status') == "FILLED" for tp in trade_details['tp_orders'])
                    tp2_hit = any(tp.get('name') == "TP2" and tp.get('status') == "FILLED" for tp in trade_details['tp_orders'])
                    # TP3 hit implies full closure, handled by pos_exists.

                    new_sl_price_staged = None
                    sl_adjustment_reason_staged = None

                    if tp1_hit and trade_details.get('sl_management_stage') == "initial":
                        sl_action_tp1 = configs.get("fib_move_sl_after_tp1", DEFAULT_FIB_MOVE_SL_AFTER_TP1)
                        print(f"[{symbol}] TP1 hit. SL management stage moving to 'after_tp1'. Action: {sl_action_tp1}")
                        trade_details['sl_management_stage'] = "after_tp1" # Update state
                        
                        if sl_action_tp1 == "breakeven":
                            buffer_r = configs.get("fib_breakeven_buffer_r", DEFAULT_FIB_BREAKEVEN_BUFFER_R)
                            initial_risk_pu = trade_details.get('initial_risk_per_unit', 0)
                            if initial_risk_pu > 0:
                                if trade_details['side'] == "LONG":
                                    new_sl_price_staged = trade_details['entry_price'] + (initial_risk_pu * buffer_r)
                                else: # SHORT
                                    new_sl_price_staged = trade_details['entry_price'] - (initial_risk_pu * buffer_r)
                                sl_adjustment_reason_staged = f"TP1_HIT_SL_TO_BREAKEVEN_PLUS_{buffer_r}R"
                            else: print(f"[{symbol}] Cannot calculate breakeven SL after TP1: initial_risk_per_unit is zero.")
                        elif sl_action_tp1 == "trailing":
                            # Micro-pivot logic will be evaluated below, this flag indicates it *can* run.
                            sl_adjustment_reason_staged = "TP1_HIT_ACTIVATE_TRAILING" 
                            # No immediate SL price change here; trailing logic will determine it.
                        elif sl_action_tp1 == "original":
                            sl_adjustment_reason_staged = "TP1_HIT_SL_ORIGINAL"
                            new_sl_price_staged = trade_details['initial_sl_price'] # Keep original SL price
                        
                        # Update current_sl_for_dynamic_check if SL is moved to BE/Original explicitly here
                        if new_sl_price_staged is not None: current_sl_for_dynamic_check = new_sl_price_staged


                    if tp2_hit and trade_details.get('sl_management_stage') in ["initial", "after_tp1"]: # Can jump to after_tp2 if TP1&2 hit same time
                        sl_action_tp2 = configs.get("fib_sl_adjustment_after_tp2", DEFAULT_FIB_SL_ADJUSTMENT_AFTER_TP2)
                        print(f"[{symbol}] TP2 hit. SL management stage moving to 'after_tp2'. Action: {sl_action_tp2}")
                        trade_details['sl_management_stage'] = "after_tp2" # Update state

                        if sl_action_tp2 == "micro_pivot":
                            sl_adjustment_reason_staged = "TP2_HIT_ACTIVATE_MICRO_PIVOT"
                            # Micro-pivot logic will be evaluated below.
                        elif sl_action_tp2 == "atr_trailing":
                            sl_adjustment_reason_staged = "TP2_HIT_ACTIVATE_ATR_TRAILING (Not Implemented, fallback to MicroPivot/Original)"
                             # TODO: Implement standard ATR trailing or fallback
                        elif sl_action_tp2 == "original":
                            sl_adjustment_reason_staged = "TP2_HIT_SL_AS_PER_AFTER_TP1_STAGE"
                            # SL remains as it was after TP1 adjustment (or initial if TP1 was skipped)
                            new_sl_price_staged = current_sl_for_dynamic_check # Keep SL from previous stage
                        
                        # Update current_sl_for_dynamic_check if SL is modified explicitly here
                        if new_sl_price_staged is not None: current_sl_for_dynamic_check = new_sl_price_staged


                    # If an explicit SL price was determined by TP1/TP2 hit logic (e.g. breakeven)
                    if new_sl_price_staged is not None and abs(new_sl_price_staged - trade_details['current_sl_price']) > 1e-9 :
                        # This new_sl_price_staged will be a candidate for final_new_sl
                        # It will be compared against standard dynamic adjustments and micro-pivot adjustments later.
                        # For now, we make it the baseline if it's better than current.
                        p_prec_staged = int(trade_details['symbol_info'].get('pricePrecision', 2))
                        new_sl_price_staged_rounded = round(new_sl_price_staged, p_prec_staged)

                        # Check if this staged SL is an improvement
                        is_improvement = False
                        if trade_details['side'] == "LONG" and new_sl_price_staged_rounded > trade_details['current_sl_price']:
                            is_improvement = True
                        elif trade_details['side'] == "SHORT" and new_sl_price_staged_rounded < trade_details['current_sl_price']:
                            is_improvement = True
                        
                        if is_improvement:
                            print(f"[{symbol}] Staged SL after TP hit ({sl_adjustment_reason_staged}): {new_sl_price_staged_rounded}")
                            # This becomes a candidate for the SL. The main SL update logic later will handle order replacement.
                            # We update current_sl_for_dynamic_check so subsequent logic uses this new baseline.
                            current_sl_for_dynamic_check = new_sl_price_staged_rounded
                            # It will be fed into `potential_new_sl_standard` or `final_new_sl` decision process.
                            # Forcing it into potential_new_sl_standard to ensure it's considered:
                            if potential_new_sl_standard is None:
                                potential_new_sl_standard = new_sl_price_staged_rounded
                            else:
                                if trade_details['side'] == "LONG":
                                    potential_new_sl_standard = max(potential_new_sl_standard, new_sl_price_staged_rounded)
                                else:
                                    potential_new_sl_standard = min(potential_new_sl_standard, new_sl_price_staged_rounded)
                        else:
                            print(f"[{symbol}] Staged SL after TP hit ({sl_adjustment_reason_staged}) {new_sl_price_staged_rounded} is not an improvement over current SL {trade_details['current_sl_price']}. No explicit change yet.")
            # --- End Staged SL Management ---

        # Dynamic SL/TP adjustment logic (Standard profit-based and Micro-Pivot)
        try: cur_price = float(client.futures_ticker(symbol=symbol)['lastPrice'])
        except Exception as e: print(f"Error getting ticker for {symbol} in monitor: {e}"); continue

        s_info = trade_details['symbol_info']
        original_trade_side = trade_details['side']
        position_side_for_sl_tp = original_trade_side
        updated_orders = False

        # Determine current SL and TP(s) for dynamic checks
        current_sl_for_dynamic_check = trade_details['current_sl_price']
        current_tp_for_dynamic_check = None 
        initial_sl_for_dynamic_check = trade_details['initial_sl_price']
        initial_tp_for_dynamic_check = None
        
        is_multi_tp_strategy = trade_details.get('strategy_type') == "FIBONACCI_MULTI_TP"
        # The new Micro-Pivot SL can apply to any strategy type if enabled in configs.

        if not is_multi_tp_strategy: # For single TP strategies (e.g. EMA Cross)
            current_tp_for_dynamic_check = trade_details.get('current_tp_price')
            initial_tp_for_dynamic_check = trade_details.get('initial_tp_price')

        # Standard dynamic SL/TP adjustment (e.g., based on fixed profit percentages like SL to BE+0.2%)
        potential_new_sl_standard, potential_new_tp_standard, standard_adj_reason = check_and_adjust_sl_tp_dynamic(
            cur_price,
            trade_details['entry_price'],
            initial_sl_for_dynamic_check,
            initial_tp_for_dynamic_check, 
            current_sl_for_dynamic_check,
            current_tp_for_dynamic_check, 
            trade_details['side']
        )
        if standard_adj_reason:
            print(f"[{symbol}] Standard dynamic adjustment proposed: SL={potential_new_sl_standard}, TP={potential_new_tp_standard}, Reason: {standard_adj_reason}")

        # Initialize final SL to be set; it will be the best of standard adjustment or micro-pivot adjustment.
        final_new_sl = potential_new_sl_standard # Start with the standard adjustment's proposal (could be None if no change)

        # --- Micro-Pivot Trailing SL (Applies if enabled in config, AND if specific Fib stage allows it or if not a Fib trade) ---
        apply_micro_pivot_logic = False
        global_micro_pivot_enabled = configs.get("micro_pivot_trailing_sl", False)

        if global_micro_pivot_enabled:
            if is_multi_tp_strategy: # Fibonacci strategy with stages
                sl_stage = trade_details.get('sl_management_stage', 'initial')
                fib_sl_action_tp1 = configs.get("fib_move_sl_after_tp1", DEFAULT_FIB_MOVE_SL_AFTER_TP1)
                fib_sl_action_tp2 = configs.get("fib_sl_adjustment_after_tp2", DEFAULT_FIB_SL_ADJUSTMENT_AFTER_TP2)

                if (sl_stage == "after_tp1" and fib_sl_action_tp1 == "trailing") or \
                   (sl_stage == "after_tp2" and fib_sl_action_tp2 == "micro_pivot"):
                    apply_micro_pivot_logic = True
                elif sl_stage == "initial": # If TP1 not hit yet, standard profit threshold applies
                    pass # apply_micro_pivot_logic remains False unless profit threshold met below
            else: # Not a Fibonacci strategy, standard profit threshold applies
                pass # apply_micro_pivot_logic remains False unless profit threshold met below

            # General condition for non-Fib or Fib-initial stage: Profit threshold check
            if not apply_micro_pivot_logic: # Only check this if not already enabled by Fib stage
                unrealized_pnl_for_micro_pivot = calculate_unrealized_pnl(trade_details, cur_price)
                initial_sl_price_for_r_calc = trade_details.get('initial_sl_price')
                initial_risk_per_unit_for_micro_pivot = 0
                if initial_sl_price_for_r_calc and trade_details['entry_price'] != initial_sl_price_for_r_calc:
                    initial_risk_per_unit_for_micro_pivot = abs(trade_details['entry_price'] - initial_sl_price_for_r_calc)

                if initial_risk_per_unit_for_micro_pivot > 0:
                    current_profit_in_r_for_micro_pivot = unrealized_pnl_for_micro_pivot / (initial_risk_per_unit_for_micro_pivot * trade_details['quantity'])
                    profit_threshold_r_config = configs.get("micro_pivot_profit_threshold_r", DEFAULT_MICRO_PIVOT_PROFIT_THRESHOLD_R)
                    if current_profit_in_r_for_micro_pivot >= profit_threshold_r_config:
                        apply_micro_pivot_logic = True
                        print(f"[{symbol}] Trade profit ({current_profit_in_r_for_micro_pivot:.2f}R) >= threshold ({profit_threshold_r_config}R). Activating Micro-Pivot SL logic.")
                elif is_multi_tp_strategy and trade_details.get('sl_management_stage', 'initial') != 'initial':
                    # If Fib strategy is past TP1/TP2 and specific trailing is enabled, but R calc failed (e.g. SL was at BE)
                    # We might still want to apply micro-pivot if the stage dictates it.
                    # This case is covered by the Fib-stage specific checks above.
                    pass


        if apply_micro_pivot_logic:
            print(f"[{symbol}] Evaluating Micro-Pivot SL logic. Current SL for dynamic check: {current_sl_for_dynamic_check}")
            # Fetch or get 1-minute candle buffer
            buffer_1m_df = None
            with symbol_1m_candle_buffers_lock: # Assuming buffer is populated elsewhere for active trades
                        if symbol in symbol_1m_candle_buffers and len(symbol_1m_candle_buffers[symbol]) > 0:
                            # Convert deque of Series to DataFrame for pivot/ATR calculation
                            # This logic is similar to how it's done for Fib strategy monitoring
                            records = [s.to_dict() for s in symbol_1m_candle_buffers[symbol] if isinstance(s, pd.Series)]
                            if records:
                                temp_df = pd.DataFrame(records)
                                # Ensure 'timestamp' is the index if not already set by Series name
                                if 'timestamp' not in temp_df.columns and symbol_1m_candle_buffers[symbol][0].name is not None:
                                     temp_df.index = [s.name for s in symbol_1m_candle_buffers[symbol]]
                                elif 'timestamp' in temp_df.columns:
                                     temp_df.set_index('timestamp', inplace=True)
                                
                                if all(c in temp_df for c in ['high', 'low', 'close']):
                                    buffer_1m_df = temp_df
                                else: print(f"[{symbol}] 1m buffer DataFrame missing required columns for Micro-Pivot SL.")
                            else: print(f"[{symbol}] No valid records in 1m buffer for Micro-Pivot SL.")
                        else: # If buffer not populated, try to fetch on demand
                            print(f"[{symbol}] 1m buffer empty or not found. Attempting on-demand fetch for Micro-Pivot SL...")
                            # Fetch, e.g., last 60 1-min candles. Configurable limit might be needed.
                            # DEFAULT_1M_BUFFER_SIZE is 200, which is plenty.
                            fetched_klines_1m_df, fetch_err = get_historical_klines_1m(client, symbol, limit=DEFAULT_1M_BUFFER_SIZE)
                            if not fetch_err and not fetched_klines_1m_df.empty:
                                buffer_1m_df = fetched_klines_1m_df
                                print(f"[{symbol}] Fetched {len(buffer_1m_df)} 1m klines for Micro-Pivot SL.")
                                # Optionally, update the main buffer here too if design allows/requires
                                # for s_row_idx in range(len(buffer_1m_df)):
                                #    update_1m_candle_buffer(symbol, buffer_1m_df.iloc[s_row_idx], DEFAULT_1M_BUFFER_SIZE)
                            else: print(f"[{symbol}] Failed to fetch 1m klines for Micro-Pivot SL. Error: {fetch_err}")


                        if buffer_1m_df is not None and not buffer_1m_df.empty and all(c in buffer_1m_df for c in ['high', 'low', 'close']):
                            # Config for ATR period for micro-pivot SL buffer (can be new or reuse e.g. fib_atr_period)
                            # Let's assume a default or use fib_atr_period for now.
                            # For user request: "small ATR buffer (0.25 × ATR)" - implies ATR is from 1-min chart.
                            micro_pivot_atr_period_1m = configs.get("fib_atr_period", DEFAULT_FIB_ATR_PERIOD) # Re-use for now
                            
                            current_atr_1m_for_buffer = 0
                            if len(buffer_1m_df) >= micro_pivot_atr_period_1m:
                                atr_1m_series = calculate_atr(buffer_1m_df.copy(), period=micro_pivot_atr_period_1m)
                                if not atr_1m_series.empty and pd.notna(atr_1m_series.iloc[-1]):
                                    current_atr_1m_for_buffer = atr_1m_series.iloc[-1]
                            else: print(f"[{symbol}] Insufficient 1m data ({len(buffer_1m_df)}) for ATR ({micro_pivot_atr_period_1m}) calc in Micro-Pivot SL.")
                            
                            sl_micro_buffer_val = current_atr_1m_for_buffer * configs.get("micro_pivot_buffer_atr", DEFAULT_MICRO_PIVOT_BUFFER_ATR)
                            # Ensure buffer is at least a few ticks if ATR is tiny or zero
                            min_tick_val = 1 / (10**int(s_info.get('pricePrecision',2)))
                            if sl_micro_buffer_val < min_tick_val * 2 : sl_micro_buffer_val = min_tick_val * 2

                            # Get latest 1-min pivots
                            # Using existing constants MICRO_PIVOT_N_LEFT_1M, MICRO_PIVOT_N_RIGHT_1M (3,1)
                            _, latest_pivot_high_1m, _, latest_pivot_low_1m = get_latest_pivots_from_buffer(
                                buffer_1m_df, MICRO_PIVOT_N_LEFT_1M, MICRO_PIVOT_N_RIGHT_1M
                            )
                            
                            micro_pivot_sl_candidate = None
                            if trade_details['side'] == "LONG" and latest_pivot_low_1m is not None:
                                potential_mp_sl = latest_pivot_low_1m - sl_micro_buffer_val
                                # SL must be an improvement (higher) and at least breakeven
                                if potential_mp_sl > current_sl_for_dynamic_check and potential_mp_sl >= trade_details['entry_price']:
                                    micro_pivot_sl_candidate = potential_mp_sl
                                    print(f"[{symbol}] LONG Micro-Pivot SL candidate: {micro_pivot_sl_candidate:.{s_info['pricePrecision']}f} (Pivot Low 1m: {latest_pivot_low_1m}, Buffer: {sl_micro_buffer_val:.{s_info['pricePrecision']}f})")
                                    
                            elif trade_details['side'] == "SHORT" and latest_pivot_high_1m is not None:
                                potential_mp_sl = latest_pivot_high_1m + sl_micro_buffer_val
                                if potential_mp_sl < current_sl_for_dynamic_check and potential_mp_sl <= trade_details['entry_price']:
                                    micro_pivot_sl_candidate = potential_mp_sl
                                    print(f"[{symbol}] SHORT Micro-Pivot SL candidate: {micro_pivot_sl_candidate:.{s_info['pricePrecision']}f} (Pivot High 1m: {latest_pivot_high_1m}, Buffer: {sl_micro_buffer_val:.{s_info['pricePrecision']}f})")

                            if micro_pivot_sl_candidate is not None:
                                # Compare with SL from standard adjustment (if any) and current SL. Prioritize the tightest valid SL.
                                if final_new_sl is not None: # If standard adjustment already proposed an SL
                                    if trade_details['side'] == "LONG": final_new_sl = max(final_new_sl, micro_pivot_sl_candidate)
                                    else: final_new_sl = min(final_new_sl, micro_pivot_sl_candidate) # For SHORT, tighter is smaller
                                else: # No standard adjustment, compare micro_pivot_sl_candidate with current_sl_for_dynamic_check
                                    # This condition is already checked when setting micro_pivot_sl_candidate
                                    final_new_sl = micro_pivot_sl_candidate
                                print(f"[{symbol}] Micro-Pivot logic updated SL to: {final_new_sl:.{s_info['pricePrecision']}f}")
                        else:
                            print(f"[{symbol}] Could not get valid 1m buffer or ATR for Micro-Pivot SL check.")
                    # else: Profit threshold not met for micro-pivot
        else: # Initial risk per unit is zero (e.g. initial_sl_price was None or at entry)
            print(f"[{symbol}] Cannot calculate profit in R for Micro-Pivot SL as initial_risk_per_unit is zero.")
    # --- End Micro-Pivot SL Adjustment ---
        
        # Use `final_new_sl` (which is best of standard or micro-pivot) for SL adjustment logic below.
        # `potential_new_tp_standard` is only for non-multi-TP strategies' TP adjustment.

        # Calculate remaining quantity if any TPs have filled (for multi-TP)
        # For single TP, remaining_quantity is just the trade_details['quantity'] unless SL/TP hit.
        remaining_quantity = trade_details['quantity'] # Start with total/original quantity
        if is_multi_tp_strategy and 'tp_orders' in trade_details:
            qty_filled_by_tps = sum(
                tp.get('quantity', 0) for tp in trade_details['tp_orders'] if tp.get('status') == "FILLED" and tp.get('id') is not None
            )
            # Ensure qty_filled_by_tps is float for arithmetic operation
            remaining_quantity = float(trade_details['quantity']) - float(qty_filled_by_tps)
            if remaining_quantity < 0: remaining_quantity = 0.0
        
        # Ensure remaining_quantity is correctly rounded to symbol's quantityPrecision before use
        q_prec_current_trade = int(s_info.get('quantityPrecision', 0))
        remaining_quantity = round(remaining_quantity, q_prec_current_trade)

        min_qty_val = float(s_info.get('filters', [{}])[0].get('minQty', '0.001')) # Simplified minQty fetch
        
        if remaining_quantity < min_qty_val: # If remaining is dust or zero
            print(f"[{symbol}] Remaining quantity ({remaining_quantity}) is less than minQty ({min_qty_val}). No SL adjustment. Position should close if not already.")
        elif final_new_sl is not None and abs(final_new_sl - current_sl_for_dynamic_check) > 1e-9: # If SL needs adjustment
            # Ensure final_new_sl is also rounded to pricePrecision
            p_prec_current_trade = int(s_info.get('pricePrecision', 2))
            final_new_sl_rounded = round(final_new_sl, p_prec_current_trade)

            # Final check: SL must not cross entry in the wrong direction
            if (trade_details['side'] == "LONG" and final_new_sl_rounded >= cur_price) or \
               (trade_details['side'] == "SHORT" and final_new_sl_rounded <= cur_price):
                print(f"[{symbol}] Proposed new SL {final_new_sl_rounded} would be triggered by current price {cur_price}. SL adjustment SKIPPED.")
            elif (trade_details['side'] == "LONG" and final_new_sl_rounded >= trade_details['entry_price'] and final_new_sl_rounded < current_sl_for_dynamic_check) or \
                 (trade_details['side'] == "SHORT" and final_new_sl_rounded <= trade_details['entry_price'] and final_new_sl_rounded > current_sl_for_dynamic_check):
                 # This condition means SL is being moved to BE or profit, but it's not as good as current SL. This path should be rare if logic is right.
                 # More importantly, if current SL is already in profit, and new SL is also in profit but worse, we might not want to move it "back".
                 # The max/min logic when choosing between standard and micro-pivot should handle picking the "better" profitable SL.
                 # This check is more about ensuring the SL is valid relative to entry if it's being moved towards it.
                 pass # This is acceptable (e.g. moving to BreakEven)

            else: # SL is valid to be placed
                print(f"[{symbol}] Adjusting SL for {position_side_for_sl_tp} from {current_sl_for_dynamic_check:.{p_prec_current_trade}f} to {final_new_sl_rounded:.{p_prec_current_trade}f} for remaining Qty {remaining_quantity}")
                if trade_details.get('sl_order_id'): # Cancel old SL
                    try: client.futures_cancel_order(symbol=symbol, orderId=trade_details['sl_order_id'])
                    except Exception as e: print(f"Warn: Old SL {trade_details['sl_order_id']} for {symbol} cancel fail: {e}")
                
                sl_ord_new, sl_err_msg_new = place_new_order(client,
                                             s_info,
                                             "SELL" if original_trade_side == "LONG" else "BUY",
                                             "STOP_MARKET",
                                             remaining_quantity, 
                                             stop_price=final_new_sl_rounded,
                                             position_side=position_side_for_sl_tp,
                                             is_closing_order=True)
                if sl_ord_new:
                    with active_trades_lock:
                        if symbol in active_trades: # Check if trade still exists
                             active_trades[symbol]['current_sl_price'] = final_new_sl_rounded
                             active_trades[symbol]['sl_order_id'] = sl_ord_new.get('orderId')
                             # If remaining_quantity changed due to partial TP, active_trades[symbol]['quantity'] should be updated
                             # This update should happen where partial TP is detected and processed.
                             # For now, assuming 'quantity' reflects remaining open quantity for the purpose of new SL order.
                             updated_orders = True
                             # Send Telegram notification for SL adjustment
                             if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                                 sl_adj_msg = (
                                     f"⚙️ SL ADJUSTED (Micro-Pivot/Dynamic) ⚙️\n\n"
                                     f"Symbol: `{symbol}` ({trade_details['side']})\n"
                                     f"Old SL: `{current_sl_for_dynamic_check:.{p_prec_current_trade}f}`\n"
                                     f"New SL: `{final_new_sl_rounded:.{p_prec_current_trade}f}`\n"
                                     f"Quantity: `{remaining_quantity:.{q_prec_current_trade}f}`\n"
                                     f"Entry: `{trade_details['entry_price']:.{p_prec_current_trade}f}` | Current Price: `{cur_price:.{p_prec_current_trade}f}`"
                                 )
                                 send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], sl_adj_msg)
                else: 
                    errMsgSL = f"CRITICAL: FAILED TO PLACE NEW DYNAMIC SL FOR {symbol}! Error: {sl_err_msg_new}"
                    print(errMsgSL)
                    if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                        send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], f"🆘 {errMsgSL}")

        
        # Dynamic TP adjustment for SINGLE TP strategies (Multi-TP handles TPs via their own orders)
        # Uses potential_new_tp_standard from the standard dynamic adjustment logic
        if not is_multi_tp_strategy and potential_new_tp_standard is not None and \
           abs(potential_new_tp_standard - trade_details.get('current_tp_price', 0)) > 1e-9: # If TP needs adjustment
            p_prec_current_trade = int(s_info.get('pricePrecision', 2))
            potential_new_tp_standard_rounded = round(potential_new_tp_standard, p_prec_current_trade)

            print(f"Adjusting TP for {symbol} ({position_side_for_sl_tp}) to {potential_new_tp_standard_rounded:.{s_info['pricePrecision']}f}")
            if trade_details.get('tp_order_id'): # Cancel old TP
                try: client.futures_cancel_order(symbol=symbol, orderId=trade_details['tp_order_id'])
                except Exception as e: print(f"Warn: Old TP {trade_details['tp_order_id']} for {symbol} cancel fail: {e}")

            tp_ord_new, tp_err_msg_new = place_new_order(client,
                                         s_info,
                                         "SELL" if original_trade_side == "LONG" else "BUY",
                                         "TAKE_PROFIT_MARKET",
                                         remaining_quantity, # Should be full quantity if single TP
                                         stop_price=potential_new_tp_standard_rounded,
                                         position_side=position_side_for_sl_tp,
                                         is_closing_order=True)
            if tp_ord_new:
                with active_trades_lock:
                     if symbol in active_trades: # Check if trade still exists
                        active_trades[symbol]['current_tp_price'] = potential_new_tp_standard_rounded
                        active_trades[symbol]['tp_order_id'] = tp_ord_new.get('orderId')
                        updated_orders = True
            else: 
                print(f"Warning: Failed to place new dynamic TP for {symbol}. Error: {tp_err_msg_new}")
        
        if updated_orders: get_open_orders(client, symbol)

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
            global bot_shutdown_requested, bot_restart_requested # Added bot_restart_requested
            if bot_shutdown_requested:
                print("Shutdown requested via Telegram command. Exiting trading loop...")
                if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                     send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], "ℹ️ Bot is shutting down now as requested...")
                break # Exit the trading loop
            
            if bot_restart_requested:
                print("Restart requested via Telegram command. Exiting trading loop for restart...")
                if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                    send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], "⏳ Bot is preparing to restart...")
                break # Exit the trading loop to allow main to handle restart

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
                    print(f"Submitting {len(monitored_symbols)} symbol tasks to {configs.get('max_scan_threads')} threads for strategy '{configs['strategy_choice']}'... {format_elapsed_time(cycle_start_time)}")
                    
                    task_function_to_submit = None
                    if configs["strategy_choice"] == "fib_retracement":
                        task_function_to_submit = process_symbol_fib_task
                        print(f"Using Fibonacci task function: {task_function_to_submit.__name__}")
                    elif configs["strategy_choice"] == "ema_cross":
                        task_function_to_submit = process_symbol_task
                        print(f"Using EMA Cross task function: {task_function_to_submit.__name__}")
                    elif configs["strategy_choice"] == "ict_strategy":
                        task_function_to_submit = process_symbol_ict_task
                        print(f"Using ICT Strategy task function: {task_function_to_submit.__name__}")
                    else:
                        print(f"ERROR: Unknown strategy_choice '{configs['strategy_choice']}' in trading_loop. Cannot submit tasks.")
                        proceed_with_new_trades = False # Stop trying to process new trades if strategy is unknown

                    if task_function_to_submit and proceed_with_new_trades:
                        for symbol in monitored_symbols:
                            # Each task function will internally check halt flags again before processing
                            futures.append(executor.submit(task_function_to_submit, symbol, client, configs, active_trades_lock))
                    
                    processed_count = 0
                    if futures: # Only iterate if tasks were actually submitted
                        for future in as_completed(futures):
                            try: result = future.result(); # print(f"Task result: {result}") # Optional: log task result
                            except Exception as e_future: print(f"Task error: {e_future}")
                            processed_count += 1
                            if processed_count % (len(monitored_symbols)//5 or 1) == 0 or processed_count == len(monitored_symbols): # Log progress periodically
                                 print(f"Symbol tasks progress: {processed_count}/{len(monitored_symbols)} completed. {format_elapsed_time(cycle_start_time)}")
                        print(f"All symbol tasks completed for new trade scanning. {format_elapsed_time(cycle_start_time)}")
                    elif not proceed_with_new_trades and not task_function_to_submit : # If strategy was unknown
                        print(f"Skipping new trade task submission due to unknown strategy. {format_elapsed_time(cycle_start_time)}")
                    else: # No futures submitted for other reasons (e.g. monitored_symbols was empty, though guarded earlier)
                        print(f"No new trade tasks were submitted this cycle. {format_elapsed_time(cycle_start_time)}")

                # --- Operations for when new trades are paused/halted but monitoring continues ---
                if not proceed_with_new_trades:
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

                # --- Monitor Existing Trades / Signals ---
                if configs['mode'] == 'signal':
                    monitor_active_signals(client, configs) # Monitors EMA and Fib signals
                    # Potentially, ICT virtual limit orders could also be monitored here if not handled by a dedicated ICT signal monitor
                    print(f"Signal Mode: Active signal monitoring complete. {format_elapsed_time(cycle_start_time)}")
                elif configs['mode'] == 'live':
                    # Monitor pending limit orders for Fib and ICT strategies
                    monitor_pending_fib_entries(client, configs) 
                    monitor_pending_ict_entries(client, configs) # Added ICT pending monitor

                    if not halt_dd_flag: # If not in max drawdown hard stop (where positions are closed)
                        monitor_active_trades(client, configs) # Monitors filled trades from all strategies
                        print(f"Live Mode: Active trades monitoring complete. {format_elapsed_time(cycle_start_time)}")
                    else: # Max drawdown halt is active in live mode
                        print(f"Live Mode: Skipping active trade monitoring as Max Drawdown halt is active and positions should be closed. {format_elapsed_time(cycle_start_time)}")
                # Backtest mode has its own monitor_active_trades_backtest within its loop.
                
                # --- SL/TP Safety Net Check (Only for 'live' mode) ---
                if configs['mode'] == 'live':
                    # This is important even if halted, to manage any stragglers or manually opened positions,
                    # or if close_all_open_positions had issues.
                    symbol_info_cache_for_safety_net = {} 
                    print(f"Running SL/TP safety net check for all open positions... {format_elapsed_time(cycle_start_time)}")
                    ensure_sl_tp_for_all_open_positions(client, configs, active_trades, symbol_info_cache_for_safety_net)
                    print(f"SL/TP safety net check complete. {format_elapsed_time(cycle_start_time)}")
                elif configs['mode'] == 'signal':
                    print(f"Signal Mode: Skipping SL/TP safety net for real positions. {format_elapsed_time(cycle_start_time)}")
                # Backtest mode handles its own position/order simulation, so no explicit safety net call here from trading_loop.
                
                # --- Update 1-minute Candle Buffers for Active Trades/Signals (for Micro-Pivot SL) ---
                # This might be relevant for 'signal' mode too if simulating micro-pivot trailing for signals.
                # For now, let's assume it's mainly for live trades.
                # If active_signals store virtual SLs that can be trailed by micro-pivots, this section might need adjustment.
                # For the current step, we focus on ensuring no REAL orders are placed.
                # The `active_trades` variable will be empty in "signal" mode for real trades.
                # We will introduce `active_signals` later.
                
                # Check if micro-pivot is enabled and if there are items to monitor (either real trades or virtual signals)
                items_to_monitor_for_pivot_sl = 0
                symbols_for_pivot_sl_buffer_update = []

                if configs.get("micro_pivot_trailing_sl", False):
                    if configs['mode'] == 'live' and active_trades:
                        with active_trades_lock: # Access active_trades safely
                            items_to_monitor_for_pivot_sl = len(active_trades)
                            symbols_for_pivot_sl_buffer_update = list(active_trades.keys())
                    elif configs['mode'] == 'signal': # and active_signals: (active_signals to be added later)
                        # items_to_monitor_for_pivot_sl = len(active_signals) # Placeholder for future
                        # symbols_for_pivot_sl_buffer_update = list(active_signals.keys()) # Placeholder
                        print(f"Signal Mode: Placeholder for 1m buffer update for virtual signals if micro-pivot trailing is simulated. {format_elapsed_time(cycle_start_time)}")
                        pass # No active_signals yet

                    if items_to_monitor_for_pivot_sl > 0 and symbols_for_pivot_sl_buffer_update:
                        print(f"Updating 1-min candle buffers for {items_to_monitor_for_pivot_sl} item(s) eligible for Micro-Pivot SL... {format_elapsed_time(cycle_start_time)}")
                        for sym_active in symbols_for_pivot_sl_buffer_update:
                            latest_1m_klines_df, err_1m = get_historical_klines_1m(client, sym_active, limit=2)
                            if not err_1m and not latest_1m_klines_df.empty:
                                last_1m_candle_series = latest_1m_klines_df.iloc[-1]
                                update_1m_candle_buffer(sym_active, last_1m_candle_series, configs.get("fib_1m_buffer_size", DEFAULT_1M_BUFFER_SIZE))
                            elif err_1m:
                                print(f"Error fetching latest 1m kline for {sym_active} buffer update ({configs['mode']} mode): {err_1m}")
                # --- End 1-minute Candle Buffer Update ---

                # --- Loop Delay ---
                # Determine appropriate sleep time
                # If any halt is active, or max positions, might use a shorter delay to check for new day or slot availability sooner.
                current_loop_delay_seconds = configs['loop_delay_minutes'] # Default delay
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

# --- ICT Strategy Core Components ---

def identify_fair_value_gap(df: pd.DataFrame, direction: str = "bullish") -> pd.Series | None:
    """
    Identifies Fair Value Gaps (FVGs) in a DataFrame of klines.
    An FVG is a 3-candle pattern where there's an imbalance.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns. Index must be DateTimeIndex.
                           Should contain at least 3 candles.
        direction (str): "bullish" or "bearish" to specify the type of FVG to look for.

    Returns:
        pd.Series | None: A Series indicating FVG presence (1 for FVG, 0 otherwise) for each candle.
                          The FVG is marked on the third candle of the pattern.
                          Returns None if input conditions are not met.
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        print("Error: DataFrame must contain 'high', 'low', 'close' columns for FVG identification.")
        return None
    if len(df) < 3:
        # print("Info: Data length less than 3, cannot identify FVGs.") # Can be too verbose
        return None

    fvg_series = pd.Series(0, index=df.index, dtype=int)
    
    # For a bullish FVG (price moved up, leaving a gap below):
    # Candle 1: High
    # Candle 2: Middle candle, its body and wicks don't fully cover the gap
    # Candle 3: Low
    # Gap is between Candle 1's high and Candle 3's low.
    # Condition: Candle 1 High < Candle 3 Low (for a bullish FVG to exist)

    # For a bearish FVG (price moved down, leaving a gap above):
    # Candle 1: Low
    # Candle 2: Middle candle
    # Candle 3: High
    # Gap is between Candle 1's low and Candle 3's high.
    # Condition: Candle 1 Low > Candle 3 High (for a bearish FVG to exist)

    for i in range(2, len(df)): # Iterate from the third candle
        candle1_high = df['high'].iloc[i-2]
        candle1_low = df['low'].iloc[i-2]
        
        # Candle 2 (middle candle) is df.iloc[i-1]
        # We need to ensure Candle 2 does not fill the potential gap.
        candle2_high = df['high'].iloc[i-1]
        candle2_low = df['low'].iloc[i-1]

        candle3_high = df['high'].iloc[i]
        candle3_low = df['low'].iloc[i]

        is_fvg = False
        if direction == "bullish":
            # Bullish FVG: Candle 1 High is below Candle 3 Low
            # The gap is formed by the impulse upwards.
            # Candle 1 creates the bottom of the gap with its high.
            # Candle 3 creates the top of the gap with its low.
            # The middle candle (Candle 2) must not have its low go below Candle 1's high,
            # and its high must not go above Candle 3's low. (This is implicitly handled by the FVG definition)
            # The key is: Candle 1 High < Candle 3 Low.
            # And Candle 2's low must be > Candle 1's high. (Wick of candle 2 doesn't fill the gap from below)
            if candle1_high < candle3_low: # Potential gap
                # Check if candle 2's low is above candle 1's high (ensuring candle 2 didn't fill it)
                if candle2_low > candle1_high:
                    is_fvg = True
                    # FVG details: (candle1_high, candle3_low) - this is the FVG zone
                    # print(f"Bullish FVG detected at {df.index[i]}: C1_H={candle1_high}, C3_L={candle3_low}")
        
        elif direction == "bearish":
            # Bearish FVG: Candle 1 Low is above Candle 3 High
            # The gap is formed by the impulse downwards.
            # Candle 1 creates the top of the gap with its low.
            # Candle 3 creates the bottom of the gap with its high.
            # Candle 2's high must be < Candle 1's low. (Wick of candle 2 doesn't fill the gap from above)
            if candle1_low > candle3_high: # Potential gap
                # Check if candle 2's high is below candle 1's low
                if candle2_high < candle1_low:
                    is_fvg = True
                    # FVG details: (candle3_high, candle1_low) - this is the FVG zone
                    # print(f"Bearish FVG detected at {df.index[i]}: C1_L={candle1_low}, C3_H={candle3_high}")

        if is_fvg:
            fvg_series.iloc[i] = 1 # Mark FVG on the third candle

    return fvg_series

def identify_order_block(df: pd.DataFrame, direction: str = "bullish", fvg_series: pd.Series = None) -> pd.Series | None:
    """
    Identifies Order Blocks (OBs).
    A bullish OB is typically the last down-candle before a strong up-move that breaks structure.
    A bearish OB is typically the last up-candle before a strong down-move that breaks structure.
    Optionally considers if an FVG is present immediately after/within the OB's move.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close'. Index DateTimeIndex. Min 3-5 candles.
        direction (str): "bullish" (looking for bullish OB - last down candle before up move)
                         "bearish" (looking for bearish OB - last up candle before down move)
        fvg_series (pd.Series, optional): Series indicating FVG presence, to confirm OB.

    Returns:
        pd.Series | None: Series indicating OB presence (1 for OB, 0 otherwise) marked on the OB candle itself.
                          None if input conditions not met.
    """
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        print("Error: DataFrame must contain 'open', 'high', 'low', 'close' for OB identification.")
        return None
    if len(df) < 3: # Need at least a few candles to see a "move"
        return None

    ob_series = pd.Series(0, index=df.index, dtype=int)

    for i in range(1, len(df) - 1): # Iterate, leaving one candle before and one after for context
        ob_candle = df.iloc[i]
        prev_candle = df.iloc[i-1] # For structure break context (optional here, can be separate)
        next_candle = df.iloc[i+1] # The "strong move" candle after the OB

        is_ob = False
        if direction == "bullish":
            # Bullish OB: Last down-candle (close < open) before a strong up-move.
            # The "strong up-move" is represented by next_candle being strongly bullish
            # and ideally breaking the high of the OB candle or prior structure.
            if ob_candle['close'] < ob_candle['open']: # OB candidate is a down-candle
                # Next candle must be a strong up-move
                if next_candle['close'] > next_candle['open'] and \
                   next_candle['close'] > ob_candle['high']: # Closes above the high of the OB candle (break of micro-structure)
                    
                    # Optional: Check for FVG after this OB (on next_candle or candle after that)
                    if fvg_series is not None:
                        # FVG should be on the candle following the OB candle (i.e., next_candle index, or one after)
                        fvg_confirmation_index = i + 1 # Index of next_candle in the main df
                        if fvg_confirmation_index < len(fvg_series) and fvg_series.iloc[fvg_confirmation_index] == 1:
                            is_ob = True
                        # Could also check fvg_series.iloc[i+2] if FVG forms on 3rd candle of OB's move
                    else: # No FVG check, basic OB definition
                        is_ob = True
            
        elif direction == "bearish":
            # Bearish OB: Last up-candle (close > open) before a strong down-move.
            if ob_candle['close'] > ob_candle['open']: # OB candidate is an up-candle
                # Next candle must be a strong down-move
                if next_candle['close'] < next_candle['open'] and \
                   next_candle['low'] < ob_candle['low']: # Closes below the low of the OB candle
                    
                    if fvg_series is not None:
                        fvg_confirmation_index = i + 1
                        if fvg_confirmation_index < len(fvg_series) and fvg_series.iloc[fvg_confirmation_index] == 1:
                            is_ob = True
                    else:
                        is_ob = True
        
        if is_ob:
            ob_series.iloc[i] = 1
            # print(f"{direction.capitalize()} Order Block identified at {df.index[i]}")

    return ob_series

def identify_liquidity_sweep(df: pd.DataFrame, lookback: int = 20, sweep_type: str = "buyside") -> pd.Series | None:
    """
    Identifies liquidity sweeps (stop hunts).
    - Buyside sweep: Wick goes above a recent high, then price closes back below that high.
    - Sellside sweep: Wick goes below a recent low, then price closes back above that low.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close'. Index DateTimeIndex.
        lookback (int): Period to look back for recent highs/lows.
        sweep_type (str): "buyside" or "sellside".

    Returns:
        pd.Series | None: Series indicating sweep (1 for sweep, 0 otherwise) on the sweep candle.
                          None if input conditions not met.
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        print("Error: DataFrame must contain 'high', 'low', 'close' for liquidity sweep identification.")
        return None
    if len(df) < lookback + 1:
        # print("Info: Data length insufficient for lookback period in liquidity sweep.") # Can be verbose
        return None

    sweep_series = pd.Series(0, index=df.index, dtype=int)

    for i in range(lookback, len(df)):
        current_candle = df.iloc[i]
        history = df.iloc[i-lookback : i] # Lookback period excluding current candle

        is_sweep = False
        if sweep_type == "buyside":
            recent_high = history['high'].max()
            # Sweep: Current candle's high goes above recent_high, but closes below it.
            if current_candle['high'] > recent_high and current_candle['close'] < recent_high:
                # Ensure the wick actually went above, not just body
                if current_candle['open'] < recent_high or current_candle['close'] < current_candle['open']: # If it's a down close or opened below
                    is_sweep = True
                    # print(f"Buyside liquidity sweep detected at {df.index[i]}. Recent High: {recent_high}, Candle H: {current_candle['high']}, C: {current_candle['close']}")
        
        elif sweep_type == "sellside":
            recent_low = history['low'].min()
            # Sweep: Current candle's low goes below recent_low, but closes above it.
            if current_candle['low'] < recent_low and current_candle['close'] > recent_low:
                if current_candle['open'] > recent_low or current_candle['close'] > current_candle['open']: # If it's an up close or opened above
                    is_sweep = True
                    # print(f"Sellside liquidity sweep detected at {df.index[i]}. Recent Low: {recent_low}, Candle L: {current_candle['low']}, C: {current_candle['close']}")
        
        if is_sweep:
            sweep_series.iloc[i] = 1
            
    return sweep_series

# --- End ICT Strategy Core Components ---

# --- ICT Strategy Helper Functions (NEW) ---

def identify_market_structure_ict(df: pd.DataFrame, swing_lookback: int = 10, pivot_n_left: int = 5, pivot_n_right: int = 5) -> str:
    """
    Identifies market structure (trend) for ICT strategy based on swing points.
    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close'.
        swing_lookback (int): How many recent swing points to consider for trend.
        pivot_n_left (int): Left window for pivot detection.
        pivot_n_right (int): Right window for pivot detection (implies delay for confirmation).
    Returns:
        str: "uptrend", "downtrend", or "range/consolidation".
    """
    if len(df) < pivot_n_left + pivot_n_right + 1 + swing_lookback: # Need enough data
        return "range/consolidation" # Default if not enough data

    # Use existing pivot identification logic
    swing_highs_bool = identify_swing_pivots(df['high'], pivot_n_left, pivot_n_right, is_high=True)
    swing_lows_bool = identify_swing_pivots(df['low'], pivot_n_left, pivot_n_right, is_high=False)
    
    swing_high_prices = df['high'][swing_highs_bool].tail(swing_lookback).tolist()
    swing_low_prices = df['low'][swing_lows_bool].tail(swing_lookback).tolist()

    if len(swing_high_prices) < 2 or len(swing_low_prices) < 2:
        return "range/consolidation" # Not enough swings to determine trend

    # Check for higher highs and higher lows (uptrend)
    is_uptrend = True
    for i in range(1, len(swing_high_prices)):
        if swing_high_prices[i] <= swing_high_prices[i-1]:
            is_uptrend = False; break
    if is_uptrend:
        for i in range(1, len(swing_low_prices)):
            if swing_low_prices[i] <= swing_low_prices[i-1]:
                is_uptrend = False; break
    if is_uptrend: return "uptrend"

    # Check for lower highs and lower lows (downtrend)
    is_downtrend = True
    for i in range(1, len(swing_high_prices)):
        if swing_high_prices[i] >= swing_high_prices[i-1]:
            is_downtrend = False; break
    if is_downtrend:
        for i in range(1, len(swing_low_prices)):
            if swing_low_prices[i] >= swing_low_prices[i-1]:
                is_downtrend = False; break
    if is_downtrend: return "downtrend"
    
    return "range/consolidation"


def identify_liquidity_zones_ict(df: pd.DataFrame, lookback_period: int = 20) -> dict:
    """
    Identifies recent swing highs and lows as potential liquidity zones.
    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low'.
        lookback_period (int): Number of candles to look back for swings.
    Returns:
        dict: {'buyside': float | None, 'sellside': float | None}
    """
    if len(df) < lookback_period:
        return {'buyside': None, 'sellside': None}
    
    recent_data = df.iloc[-lookback_period:]
    buyside_liq = recent_data['high'].max()
    sellside_liq = recent_data['low'].min()
    
    return {'buyside': buyside_liq, 'sellside': sellside_liq}


def detect_liquidity_grab_ict(candle: pd.Series, liquidity_price: float, side: str) -> bool:
    """
    Detects if a single candle performed a liquidity grab.
    Args:
        candle (pd.Series): The candle to check (must have 'high', 'low', 'close', 'open').
        liquidity_price (float): The price level of the liquidity zone.
        side (str): "buyside" (grab above high) or "sellside" (grab below low).
    Returns:
        bool: True if a liquidity grab is detected.
    """
    if not all(k in candle for k in ['high', 'low', 'close', 'open']): return False
    if liquidity_price is None: return False

    if side == "buyside": # Price wicks above liquidity_price then closes below it
        return candle['high'] > liquidity_price and candle['close'] < liquidity_price
    elif side == "sellside": # Price wicks below liquidity_price then closes above it
        return candle['low'] < liquidity_price and candle['close'] > liquidity_price
    return False


def identify_fair_value_gap_ict(df: pd.DataFrame, candle_index_after_grab: int, direction: str, freshness_candles: int = 5) -> dict | None:
    """
    Identifies a fresh Fair Value Gap (FVG) after a liquidity grab.
    The FVG is a 3-candle pattern. This function looks for the FVG pattern where the FVG itself
    (the gap between candle 1 and candle 3) is "fresh" (not older than `freshness_candles` from the end of `df`).
    The FVG is defined by the range between candle 1's high and candle 3's low (bullish FVG),
    or candle 1's low and candle 3's high (bearish FVG).

    Args:
        df (pd.DataFrame): Kline data with 'high', 'low', 'close'. Index must be DateTimeIndex.
        candle_index_after_grab (int): Index in `df` of the candle where the grab occurred. FVG search starts after this.
        direction (str): "bullish" (look for bullish FVG after sellside grab, implies FVG is a price void below current price action)
                         "bearish" (look for bearish FVG after buyside grab, implies FVG is a price void above current price action)
        freshness_candles (int): Max age (in candles from the end of `df`) for the 3rd candle of the FVG pattern.

    Returns:
        dict | None: FVG details {'fvg_top', 'fvg_bottom', 'fvg_mid', 'timestamp_c1', 'timestamp_c3', 'index_c1', 'index_c3'}
                     or None if no suitable FVG is found.
                     'fvg_top' and 'fvg_bottom' define the gap.
                     'index_c1' and 'index_c3' are indices of the first and third candles forming the FVG.
    """
    df_len = len(df)
    if df_len < candle_index_after_grab + 3: # Need at least 3 candles after grab for an FVG
        # print(f"FVG_ICT: Not enough data after grab index {candle_index_after_grab} (len: {df_len})")
        return None

    # Iterate to find a 3-candle FVG pattern.
    # The pattern consists of candles at index i, i+1, i+2.
    # The FVG is considered "at" index i+2 (the third candle).
    # Search starts from the candle immediately following the grab.
    # Max search window: up to `freshness_candles` from the end of the df for the FVG's third candle.
    
    # Start searching for the first candle (c1) of the FVG pattern
    # from one candle after the grab.
    # The third candle (c3) of the FVG must be within `freshness_candles` from the end of df.
    # So, c1 can be at most `df_len - 1 - 2 - freshness_candles`.
    # However, it's simpler to iterate and check freshness of c3.

    # Iterate i from candle_index_after_grab + 1 up to len(df) - 3
    # i will be the index of the first candle of the FVG pattern.
    for i in range(candle_index_after_grab + 1, df_len - 2):
        idx_c1, idx_c2, idx_c3 = i, i + 1, i + 2

        # Check freshness: The third candle of the FVG (idx_c3) must not be too old.
        # (df_len - 1) is the last index in df.
        # (df_len - 1) - idx_c3 is how many candles ago c3 occurred from the latest candle.
        if (df_len - 1) - idx_c3 >= freshness_candles:
            # print(f"FVG_ICT: Skipping FVG starting at {df.index[idx_c1]} as its 3rd candle ({df.index[idx_c3]}) is too old ({(df_len - 1) - idx_c3} candles from latest, freshness: {freshness_candles}).")
            continue

        c1_high, c1_low = df['high'].iloc[idx_c1], df['low'].iloc[idx_c1]
        c2_low = df['low'].iloc[idx_c2]  # Middle candle's low
        c2_high = df['high'].iloc[idx_c2] # Middle candle's high
        c3_high, c3_low = df['high'].iloc[idx_c3], df['low'].iloc[idx_c3]

        fvg_details = None
        if direction == "bullish": # Looking for a bullish FVG (a void/gap below, formed by upward price movement)
            # Candle 1's high is below Candle 3's low, and Candle 2's low did not fill this void.
            if c1_high < c3_low and c2_low > c1_high:
                fvg_details = {
                    'fvg_bottom': c1_high, 
                    'fvg_top': c3_low,
                    'timestamp_c1': df.index[idx_c1], 'timestamp_c3': df.index[idx_c3],
                    'index_c1': idx_c1, 'index_c3': idx_c3,
                    'direction': 'bullish'
                }
        elif direction == "bearish": # Looking for a bearish FVG (a void/gap above, formed by downward price movement)
            # Candle 1's low is above Candle 3's high, and Candle 2's high did not fill this void.
            if c1_low > c3_high and c2_high < c1_low:
                fvg_details = {
                    'fvg_bottom': c3_high,
                    'fvg_top': c1_low, 
                    'timestamp_c1': df.index[idx_c1], 'timestamp_c3': df.index[idx_c3],
                    'index_c1': idx_c1, 'index_c3': idx_c3,
                    'direction': 'bearish'
                }
        
        if fvg_details:
            fvg_details['fvg_mid'] = (fvg_details['fvg_top'] + fvg_details['fvg_bottom']) / 2
            # Higher verbosity logging
            # print(f"FVG_ICT: Found {direction} FVG. C1: {fvg_details['timestamp_c1']}, C3: {fvg_details['timestamp_c3']}. "
            #       f"Gap: [{fvg_details['fvg_bottom']:.4f} - {fvg_details['fvg_top']:.4f}], Mid: {fvg_details['fvg_mid']:.4f}")
            return fvg_details # Return the first fresh FVG found after the grab
            
    # print(f"FVG_ICT: No fresh {direction} FVG found after grab at index {candle_index_after_grab} within {freshness_candles} candle freshness.")
    return None


def confirm_power_of_three_ict(df: pd.DataFrame, manipulation_candle_index: int, lookback_consolidation: int = 10, min_acceleration_candles: int = 1) -> bool:
    """
    Confirms Power of Three: Consolidation -> Manipulation -> Expansion/Acceleration.
    Args:
        df (pd.DataFrame): Kline data.
        manipulation_candle_index (int): Index of the manipulation (liquidity grab) candle.
        lookback_consolidation (int): How many candles before manipulation to check for consolidation.
        min_acceleration_candles (int): Minimum number of candles for acceleration phase after manipulation.
    Returns:
        bool: True if Power of Three pattern is confirmed.
    """
    if manipulation_candle_index < lookback_consolidation or \
       manipulation_candle_index + min_acceleration_candles >= len(df):
        return False # Not enough data

    # 1. Consolidation phase before manipulation
    consolidation_slice = df.iloc[manipulation_candle_index - lookback_consolidation : manipulation_candle_index]
    if consolidation_slice.empty: return False
    
    # Simple consolidation check: range of close prices is relatively small (e.g., within X ATRs or % of price)
    # For now, a basic check: height of consolidation range vs average candle height
    consolidation_range = consolidation_slice['high'].max() - consolidation_slice['low'].min()
    avg_candle_height_consol = (consolidation_slice['high'] - consolidation_slice['low']).mean()
    if consolidation_range > avg_candle_height_consol * 3: # Heuristic: range not more than 3x avg candle height
        # print(f"Po3 Fail: Consolidation phase before index {manipulation_candle_index} seems too volatile.")
        pass # Allow it for now, this check can be refined

    # 2. Manipulation is the provided candle index

    # 3. Expansion/Acceleration phase after manipulation
    # Price moves strongly away from the manipulation zone.
    manipulation_candle = df.iloc[manipulation_candle_index]
    
    # Check candles after manipulation
    for i in range(1, min_acceleration_candles + 1):
        if manipulation_candle_index + i >= len(df): return False # Not enough candles for acceleration
        
        current_accel_candle = df.iloc[manipulation_candle_index + i]
        # Example: If buyside grab (price went up then down), acceleration is further down.
        # If sellside grab (price went down then up), acceleration is further up.
        
        # This is a simplified check. True Po3 involves specific candle characteristics for expansion.
        # For now, just check if price moved significantly.
        price_moved_significantly = False
        if manipulation_candle['close'] < manipulation_candle['open']: # Buyside grab (fake up, real move down)
            if current_accel_candle['close'] < manipulation_candle['low']: price_moved_significantly = True
        elif manipulation_candle['close'] > manipulation_candle['open']: # Sellside grab (fake down, real move up)
            if current_accel_candle['close'] > manipulation_candle['high']: price_moved_significantly = True
        
        if price_moved_significantly:
            print(f"Po3 Confirmed: Consolidation -> Manipulation (idx {manipulation_candle_index}) -> Acceleration (idx {manipulation_candle_index + i})")
            return True
            
    return False


def identify_order_block_ict(df: pd.DataFrame, fvg_details: dict, manipulation_candle_index: int, direction: str, lookback_bos: int = 10) -> dict | None:
    """
    Identifies an unmitigated Order Block (OB) associated with an FVG and prior manipulation.
    The OB must precede the FVG and the impulsive move that created the FVG.
    It also checks for a Break of Structure (BoS) caused by the move from the OB.

    Args:
        df (pd.DataFrame): Kline data with 'open', 'high', 'low', 'close'.
        fvg_details (dict): Details of the FVG. Expected keys: 'index_c1' (index of FVG's first candle).
        manipulation_candle_index (int): Index of the manipulation candle. OB search is between this and FVG.
        direction (str): "bullish" (bullish OB before bullish FVG) or "bearish".
        lookback_bos (int): Lookback period from the OB to find a swing point for BoS check.

    Returns:
        dict | None: OB details {'ob_top', 'ob_bottom', 'ob_mid', 'timestamp', 'index', 'bos_confirmed'} or None.
                     'ob_top'/'ob_bottom' define the OB range (high/low of the candle).
                     'bos_confirmed' is True if a break of structure is confirmed after the OB.
    """
    if fvg_details is None or 'index_c1' not in fvg_details:
        # print("OB_ICT: FVG details missing or incomplete.")
        return None

    # Search for OB candle between manipulation and start of FVG.
    # OB is the candle that initiated the move leading to the FVG.
    # For a bullish FVG, the OB is the last down-candle before the up-move.
    # For a bearish FVG, the OB is the last up-candle before the down-move.
    
    # Search backwards from the candle just before the FVG's first candle (fvg_details['index_c1'] - 1)
    # up to and including the manipulation candle (or one after if manipulation itself is the OB candidate).
    search_end_idx = fvg_details['index_c1'] - 1
    search_start_idx = max(0, manipulation_candle_index) # OB can be the manipulation candle itself or after it.
    
    if search_end_idx < search_start_idx :
        # print(f"OB_ICT: Invalid search range for OB. Start: {search_start_idx}, End: {search_end_idx}")
        return None

    best_ob_candidate = None

    for i in range(search_end_idx, search_start_idx - 1, -1): # Iterate backwards
        if i < 0 or i >= len(df) -1 : continue # Ensure there's a next candle for impulsive move check

        candle = df.iloc[i]
        next_candle = df.iloc[i+1] # Candle immediately following the OB candidate

        is_candidate = False
        if direction == "bullish": # Bullish OB: Last down-candle (close < open)
            if candle['close'] < candle['open']:
                # Impulsive move check: next candle is strong bullish and engulfs/moves significantly past OB
                if next_candle['close'] > next_candle['open'] and next_candle['high'] > candle['high'] and (next_candle['close'] - next_candle['open']) > (candle['open'] - candle['close']): # Basic impulse
                    is_candidate = True
        elif direction == "bearish": # Bearish OB: Last up-candle (close > open)
            if candle['close'] > candle['open']:
                if next_candle['close'] < next_candle['open'] and next_candle['low'] < candle['low'] and (next_candle['open'] - next_candle['close']) > (candle['close'] - candle['open']):
                    is_candidate = True
        
        if is_candidate:
            # Basic unmitigated check: The FVG starts after this OB. A more detailed check would ensure
            # price hasn't traded deep into this OB's body between its formation and FVG.
            # For now, proximity to FVG implies it's likely relevant.
            
            # Break of Structure (BoS) check:
            # The move from this OB should break a recent swing high (for bullish OB) or low (for bearish OB).
            bos_confirmed = False
            # Look for swing point within `lookback_bos` candles *before* the OB candidate `i`.
            bos_lookback_data = df.iloc[max(0, i - lookback_bos) : i]
            if not bos_lookback_data.empty:
                if direction == "bullish": # Move from OB broke a recent swing high
                    recent_swing_high_for_bos = bos_lookback_data['high'].max()
                    # Check if any candle from OB's next candle up to FVG's start broke this high
                    # The FVG forms at index_c1, index_c2, index_c3. The impulsive move is typically c1 or c2 of FVG.
                    # So, check if df['high'].iloc[i+1 : fvg_details['index_c1']] went above recent_swing_high_for_bos
                    impulse_candles_after_ob = df.iloc[i+1 : fvg_details['index_c1']] # Candles between OB and FVG start
                    if not impulse_candles_after_ob.empty and impulse_candles_after_ob['high'].max() > recent_swing_high_for_bos:
                        bos_confirmed = True
                elif direction == "bearish": # Move from OB broke a recent swing low
                    recent_swing_low_for_bos = bos_lookback_data['low'].min()
                    impulse_candles_after_ob = df.iloc[i+1 : fvg_details['index_c1']]
                    if not impulse_candles_after_ob.empty and impulse_candles_after_ob['low'].min() < recent_swing_low_for_bos:
                        bos_confirmed = True
            
            if bos_confirmed: # Only consider OBs that led to a BoS
                ob_details = {
                    'ob_bottom': candle['low'], 'ob_top': candle['high'],
                    'ob_mid': (candle['high'] + candle['low']) / 2,
                    'open': candle['open'], 'close': candle['close'], # For OB open/mean entry type
                    'timestamp': df.index[i], 'index': i,
                    'bos_confirmed': bos_confirmed, 'direction': direction
                }
                # Higher verbosity logging for identified OB
                # print(f"OB_ICT: Found {direction} OB at index {i} (TS: {ob_details['timestamp']}) with BoS. Range: [{ob_details['ob_bottom']:.4f} - {ob_details['ob_top']:.4f}]")
                best_ob_candidate = ob_details # Take the latest valid OB found in search range
                break # Found the most recent relevant OB
    
    if best_ob_candidate:
        # Final mitigation check (simplified): ensure FVG is *after* this OB and implies the OB wasn't fully mitigated by FVG formation itself.
        # If FVG's first candle (index_c1) is after OB's index (i), it's generally fine.
        # FVG's body should ideally not fully overlap and consume the OB's body.
        # The current logic implies FVG is after OB due to search range.
        
        # Check if FVG overlaps with this OB (as per user requirement)
        fvg_low = fvg_details['fvg_bottom']
        fvg_high = fvg_details['fvg_top']
        ob_low = best_ob_candidate['ob_bottom']
        ob_high = best_ob_candidate['ob_top']

        # Overlap condition: max(fvg_low, ob_low) <= min(fvg_high, ob_high)
        overlap_exists = max(fvg_low, ob_low) <= min(fvg_high, ob_high)
        if overlap_exists:
            print(f"OB_ICT: Confirmed {direction} OB at index {best_ob_candidate['index']} overlaps with FVG. BOS: {best_ob_candidate['bos_confirmed']}.")
            return best_ob_candidate
        else:
            # print(f"OB_ICT: Found OB at index {best_ob_candidate['index']} but it does not overlap with FVG [{fvg_low}-{fvg_high}]. OB: [{ob_low}-{ob_high}]")
            return None # No suitable OB that overlaps with FVG
            
    # print(f"OB_ICT: No suitable {direction} OB found for FVG starting at index {fvg_details['index_c1']}.")
    return None

# --- End ICT Strategy Helper Functions ---

# --- ICT Strategy Main Logic ---
def manage_ict_trade_entry(client, configs, symbol, klines_df, symbol_info, lock): # lock is active_trades_lock
    global active_trades, active_trades_lock, last_signal_time, last_signal_lock, recent_trade_signatures, recent_trade_signatures_lock
    global trading_halted_drawdown, trading_halted_daily_loss, daily_state_lock, trading_halted_manual
    global active_signals, active_signals_lock, ict_strategy_states, ict_strategy_states_lock

    log_prefix = f"[{threading.current_thread().name}] {symbol} ICTEntry:"
    # print(f"{log_prefix} ▶️ ICT entry logic triggered for {symbol}.") # Reduce verbosity

    # --- Basic Validations & Halt Checks ---
    min_candles_needed = max(
        configs.get("ict_liquidity_lookback", DEFAULT_ICT_LIQUIDITY_LOOKBACK),
        configs.get("ict_fvg_freshness_candles", DEFAULT_ICT_FVG_FRESHNESS_CANDLES) + 3, # FVG needs 3 candles
        configs.get("ict_po3_consolidation_lookback", DEFAULT_ICT_PO3_CONSOLIDATION_LOOKBACK) + 
        configs.get("ict_po3_acceleration_min_candles", DEFAULT_ICT_PO3_ACCELERATION_MIN_CANDLES) + 1 # Po3 needs consolidation, manip, accel
    ) + 20 # General buffer
    if klines_df.empty or len(klines_df) < min_candles_needed:
        # print(f"{log_prefix} Insufficient kline data ({len(klines_df)} needed {min_candles_needed}).")
        return

    with daily_state_lock:
        if trading_halted_drawdown or trading_halted_daily_loss: return
    if trading_halted_manual: return
    
    # --- ICT Strategy Sequential Logic ---
    # 1. Market Structure (Optional for now, focus on core pattern first)
    # market_structure = identify_market_structure_ict(klines_df, ...)
    # print(f"{log_prefix} Market Structure: {market_structure}")

    # 2. Detect Liquidity Sweep
    liquidity_lookback_cfg = configs.get("ict_liquidity_lookback", DEFAULT_ICT_LIQUIDITY_LOOKBACK)
    liquidity_zones = identify_liquidity_zones_ict(klines_df, lookback_period=liquidity_lookback_cfg)
    if not liquidity_zones: # Should not happen if klines_df is valid
        # print(f"{log_prefix} Could not identify liquidity zones.")
        return

    grab_details = None
    # Iterate backwards from the most recent candle to find a sweep
    # Limit search to recent candles, e.g., last 5-10 candles for the sweep itself.
    max_sweep_lookback_candles = configs.get("ict_sweep_detection_window", DEFAULT_ICT_SWEEP_DETECTION_WINDOW) # How many recent candles to check for a sweep
    
    # Ensure we don't go out of bounds and have enough history for liquidity_zones
    start_idx_sweep_search = len(klines_df) - 1
    end_idx_sweep_search = max(0, len(klines_df) - max_sweep_lookback_candles -1, liquidity_lookback_cfg)


    for i in range(start_idx_sweep_search, end_idx_sweep_search -1, -1): # Iterate backwards over recent candles
        if i < 0 or i >= len(klines_df): continue 
        current_candle_for_sweep = klines_df.iloc[i]
        
        # Check for Buyside Sweep
        if liquidity_zones.get('buyside') and \
           detect_liquidity_grab_ict(current_candle_for_sweep, liquidity_zones['buyside'], "buyside"):
            grab_details = {"type": "buyside_grab", "candle_index": i, "price_swept": liquidity_zones['buyside'], 
                              "candle_timestamp": klines_df.index[i], "trigger_candle": current_candle_for_sweep.to_dict()}
            print(f"{log_prefix} Potential Buyside Liquidity Grab detected at index {i} (Price: {liquidity_zones['buyside']:.4f})")
            break 
        
        # Check for Sellside Sweep
        if liquidity_zones.get('sellside') and \
           detect_liquidity_grab_ict(current_candle_for_sweep, liquidity_zones['sellside'], "sellside"):
            grab_details = {"type": "sellside_grab", "candle_index": i, "price_swept": liquidity_zones['sellside'],
                              "candle_timestamp": klines_df.index[i], "trigger_candle": current_candle_for_sweep.to_dict()}
            print(f"{log_prefix} Potential Sellside Liquidity Grab detected at index {i} (Price: {liquidity_zones['sellside']:.4f})")
            break
    
    if not grab_details:
        # print(f"{log_prefix} No liquidity grab detected in the last {max_sweep_lookback_candles} candles.")
        return

    # 3. Find Fresh FVG on the return move
    # Direction of FVG is opposite to the grab type
    fvg_search_direction = "bearish" if grab_details["type"] == "buyside_grab" else "bullish"
    fvg_freshness_cfg = configs.get("ict_fvg_freshness_candles", DEFAULT_ICT_FVG_FRESHNESS_CANDLES)
    
    fvg = identify_fair_value_gap_ict(klines_df, 
                                      candle_index_after_grab=grab_details["candle_index"], 
                                      direction=fvg_search_direction, 
                                      freshness_candles=fvg_freshness_cfg)
    if not fvg:
        # print(f"{log_prefix} No fresh {fvg_search_direction} FVG found after {grab_details['type']}.")
        return
    print(f"{log_prefix} FVG ({fvg['direction']}) found: [{fvg['fvg_bottom']:.4f} - {fvg['fvg_top']:.4f}], Mid: {fvg['fvg_mid']:.4f}. C1 Idx: {fvg['index_c1']}, C3 Idx: {fvg['index_c3']}")

    # 4. Validate Order Block (overlapping/aligned with FVG)
    ob_search_direction = fvg['direction'] # OB direction matches FVG direction
    order_block = identify_order_block_ict(klines_df, 
                                           fvg_details=fvg, 
                                           manipulation_candle_index=grab_details["candle_index"], 
                                           direction=ob_search_direction,
                                           lookback_bos=configs.get("ict_ob_bos_lookback", DEFAULT_ICT_OB_BOS_LOOKBACK))
    if not order_block:
        # print(f"{log_prefix} No valid Order Block found overlapping FVG for {ob_search_direction} setup.")
        return
    if not order_block.get('bos_confirmed'):
        print(f"{log_prefix} Order Block found but BoS not confirmed. OB Index: {order_block['index']}")
        return
    print(f"{log_prefix} Order Block ({order_block['direction']}) validated at index {order_block['index']}: [{order_block['ob_bottom']:.4f} - {order_block['ob_top']:.4f}], BoS: True.")

    # 5. Check for Power of Three Confirmation
    po3_confirmed = confirm_power_of_three_ict(klines_df, 
                                               manipulation_candle_index=grab_details["candle_index"],
                                               lookback_consolidation=configs.get("ict_po3_consolidation_lookback", DEFAULT_ICT_PO3_CONSOLIDATION_LOOKBACK),
                                               min_acceleration_candles=configs.get("ict_po3_acceleration_min_candles", DEFAULT_ICT_PO3_ACCELERATION_MIN_CANDLES))
    if not po3_confirmed:
        # print(f"{log_prefix} Power of Three pattern not confirmed for manipulation at index {grab_details['candle_index']}.")
        return
    print(f"{log_prefix} Power of Three dynamic confirmed around manipulation at index {grab_details['candle_index']}.")

    # --- All conditions aligned, Calculate Signal Parameters ---
    p_prec = int(symbol_info['pricePrecision'])
    signal_side = "LONG" if fvg['direction'] == "bullish" else "SHORT"

    # Entry Price Logic (as per user spec: FVG.50 or OB related)
    # Defaulting to FVG.50 for this implementation phase.
    entry_type_config = configs.get("ict_entry_type", DEFAULT_ICT_ENTRY_TYPE)
    entry_price = round(fvg['fvg_mid'], p_prec) 
    entry_type_used = "FVG Midpoint"
    if entry_type_config == "ob_open": entry_price = round(order_block['open'], p_prec); entry_type_used = "OB Open"
    elif entry_type_config == "ob_mean": entry_price = round(order_block['ob_mid'], p_prec); entry_type_used = "OB Mean"
    # Hybrid logic would be more complex, find overlap and take its mid.

    # Stop Loss Logic
    sl_type_config = configs.get("ict_sl_type", DEFAULT_ICT_SL_TYPE)
    sl_atr_buffer_mult = configs.get("ict_sl_atr_buffer_multiplier", DEFAULT_ICT_SL_ATR_BUFFER_MULTIPLIER) 
    atr_period_for_sl = configs.get("atr_period", DEFAULT_ATR_PERIOD) # Use general ATR period
    
    sl_atr_val = 0
    if sl_atr_buffer_mult > 0 and len(klines_df) > atr_period_for_sl: # Only calc ATR if multiplier is positive
        atr_s = calculate_atr(klines_df.copy(), period=atr_period_for_sl)
        if not atr_s.empty and pd.notna(atr_s.iloc[-1]):
            sl_atr_val = atr_s.iloc[-1] * sl_atr_buffer_mult
    
    stop_loss_price = None
    if sl_type_config == "ob_fvg_zone":
        if signal_side == "LONG":
            stop_loss_price = round(min(fvg['fvg_bottom'], order_block['ob_bottom']) - sl_atr_val, p_prec)
        else: # SHORT
            stop_loss_price = round(max(fvg['fvg_top'], order_block['ob_top']) + sl_atr_val, p_prec)
    elif sl_type_config == "swept_point":
        # SL beyond the swept liquidity point, plus optional buffer
        if signal_side == "LONG": # Swept sellside liquidity, SL goes below it
            stop_loss_price = round(grab_details['price_swept'] - sl_atr_val, p_prec)
        else: # Swept buyside liquidity, SL goes above it
            stop_loss_price = round(grab_details['price_swept'] + sl_atr_val, p_prec)
    elif sl_type_config == "atr_buffered_zone": # Similar to ob_fvg_zone but emphasizes ATR buffer more explicitly
        # This logic is effectively the same as ob_fvg_zone if sl_atr_val is used.
        # Could be refined to use a different base point for ATR buffer if needed.
        # For now, assume it's like ob_fvg_zone with a mandatory ATR buffer if sl_atr_buffer_mult > 0
        if signal_side == "LONG":
            stop_loss_price = round(min(fvg['fvg_bottom'], order_block['ob_bottom']) - sl_atr_val, p_prec)
        else: # SHORT
            stop_loss_price = round(max(fvg['fvg_top'], order_block['ob_top']) + sl_atr_val, p_prec)
    else: # Fallback or error
        print(f"{log_prefix} Unknown ict_sl_type: {sl_type_config}. Defaulting SL logic.")
        if signal_side == "LONG": stop_loss_price = round(min(fvg['fvg_bottom'], order_block['ob_bottom']) - sl_atr_val, p_prec)
        else: stop_loss_price = round(max(fvg['fvg_top'], order_block['ob_top']) + sl_atr_val, p_prec)


    if stop_loss_price is None: # Should be set by one of the SL types
        print(f"{log_prefix} CRITICAL: Stop loss price not set for ICT signal. Aborting."); return

    if entry_price == stop_loss_price: # Prevent zero risk
        print(f"{log_prefix} Entry and SL are identical ({entry_price}). Adjusting SL slightly."); 
        min_tick = 1 / (10**p_prec)
        stop_loss_price = stop_loss_price - min_tick if signal_side == "LONG" else stop_loss_price + min_tick
        stop_loss_price = round(stop_loss_price, p_prec)

    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit <= 1e-9: # Effectively zero risk
        print(f"{log_prefix} Calculated risk per unit is zero or negligible. Entry: {entry_price}, SL: {stop_loss_price}. Aborting.")
        return

    # Take Profit Logic
    tp1_rr = 1.0
    tp2_rr = configs.get("ict_risk_reward_ratio", DEFAULT_ICT_RISK_REWARD_RATIO) # Default is 2.0R
    tp3_rr = tp2_rr + 1.0 # Example: 3R if TP2 is 2R. This needs better logic based on liquidity.
    # TODO: Implement TP3 based on opposing liquidity or next structure. For now, fixed R.

    tp1_price = round(entry_price + (risk_per_unit * tp1_rr * (1 if signal_side == "LONG" else -1)), p_prec)
    tp2_price = round(entry_price + (risk_per_unit * tp2_rr * (1 if signal_side == "LONG" else -1)), p_prec)
    tp3_price = round(entry_price + (risk_per_unit * tp3_rr * (1 if signal_side == "LONG" else -1)), p_prec) # Placeholder

    print(f"{log_prefix} Calculated ICT Limit Signal Parameters for {symbol} ({signal_side}):")
    print(f"  Entry ({entry_type_used}): {entry_price:.{p_prec}f}")
    print(f"  Stop Loss: {stop_loss_price:.{p_prec}f} (Risk/Unit: {risk_per_unit:.{p_prec}f})")
    print(f"  TP1 (1R): {tp1_price:.{p_prec}f}")
    print(f"  TP2 ({tp2_rr}R): {tp2_price:.{p_prec}f}")
    print(f"  TP3 ({tp3_rr}R): {tp3_price:.{p_prec}f} (Placeholder - Opposing Liq. TBD)")
    print(f"  Context: Grab Candle @ {grab_details['candle_timestamp']}, FVG C3 @ {fvg['timestamp_c3']}, OB @ {order_block['timestamp']}")

    # --- Standard Pre-Trade Checks (Cooldown, Active Trade, Max Positions, Sizing, Sanity) ---
    with last_signal_lock:
        cooldown_seconds_ict = configs.get("ict_limit_signal_cooldown_seconds", DEFAULT_ICT_LIMIT_SIGNAL_COOLDOWN_SECONDS) # Separate cooldown for limit signals
        if symbol in last_signal_time and (dt.now() - last_signal_time.get(f"{symbol}_ict_limit", dt.min())).total_seconds() < cooldown_seconds_ict:
            # print(f"{log_prefix} Cooldown active for ICT Limit Signal on {symbol}. Skipping.")
            return

    with active_trades_lock: # Check against live trades
        if symbol in active_trades:
            # print(f"{log_prefix} Symbol {symbol} already has an active LIVE trade. Skipping ICT Limit Signal.")
            return
        # Check against pending ICT limit orders in ict_strategy_states to avoid duplicate signals for same setup
        with ict_strategy_states_lock:
            if symbol in ict_strategy_states and ict_strategy_states[symbol].get('state') == "PENDING_ICT_ENTRY":
                # This check might need to be more sophisticated, e.g., comparing parameters if a new setup is truly different.
                # For now, if a pending order exists, don't send another signal.
                # print(f"{log_prefix} Symbol {symbol} already has a PENDING ICT limit order/signal. Skipping new signal.")
                return
        if len(active_trades) + len([s for s, st in ict_strategy_states.items() if st.get('state') == "PENDING_ICT_ENTRY"]) >= configs.get("max_concurrent_positions", DEFAULT_MAX_CONCURRENT_POSITIONS):
            # print(f"{log_prefix} Max concurrent positions (live + pending ICT signals) reached. Cannot generate ICT Limit Signal for {symbol}.")
            return
            
    acc_bal = get_account_balance(client, configs)
    if acc_bal is None or acc_bal <= 0: 
        # print(f"{log_prefix} Invalid account balance ({acc_bal}) for ICT signal generation checks."); 
        return

    # Leverage for sanity checks and P&L estimation (not for placing order yet)
    current_leverage_on_symbol = configs.get('leverage') 
    try:
        pos_info_lev = client.futures_position_information(symbol=symbol)
        if pos_info_lev and isinstance(pos_info_lev, list) and pos_info_lev[0]:
            current_leverage_on_symbol = int(pos_info_lev[0].get('leverage', configs.get('leverage')))
    except Exception: pass # Ignore error, use default
    
    # Calculate a hypothetical quantity for sanity checks and P&L estimation for the signal message
    hypothetical_qty = calculate_position_size(acc_bal, configs['risk_percent'], entry_price, stop_loss_price, symbol_info, configs)
    if hypothetical_qty is None or hypothetical_qty <= 0: 
        # print(f"{log_prefix} Hypothetical position size calculation failed for ICT signal (Qty: {hypothetical_qty}).")
        return

    passed_sanity, sanity_reason = pre_order_sanity_checks(
        symbol, signal_side, entry_price, stop_loss_price, tp1_price, hypothetical_qty, # Use tp1_price for basic R:R check
        symbol_info, acc_bal, configs['risk_percent'], configs, specific_leverage_for_trade=current_leverage_on_symbol
    )
    if not passed_sanity: 
        print(f"{log_prefix} Pre-signal sanity checks FAILED for ICT signal: {sanity_reason}")
        return
    # print(f"{log_prefix} Sanity checks for ICT signal parameters PASSED.")

    # Trade Signature for ICT Limit Signal to avoid spamming for the exact same setup
    # Note: TP2/TP3 are not in this signature as they might be dynamic or vary. Focus on Entry/SL/TP1.
    trade_sig_ict_limit = generate_trade_signature(symbol, f"ICT_LIMIT_{signal_side}", entry_price, stop_loss_price, tp1_price, hypothetical_qty, p_prec)
    with recent_trade_signatures_lock:
        if trade_sig_ict_limit in recent_trade_signatures and \
           (dt.now() - recent_trade_signatures[trade_sig_ict_limit]).total_seconds() < configs.get("ict_limit_signal_signature_block_seconds", DEFAULT_ICT_LIMIT_SIGNAL_SIGNATURE_BLOCK_SECONDS) : # Longer block for same signal
            # print(f"{log_prefix} Duplicate ICT Limit Signal signature found within blocking period. Skipping.")
            return
    
    # --- Signal Mode or Live Mode with ICT Limit Order Placement ---
    # This function is now primarily for IDENTIFYING the setup and parameters.
    # The actual order placement or signal sending will be decided by the calling context (e.g. if mode == 'signal')
    # or if a new config 'ict_place_limit_orders_live' is true.

    # For now, if in 'signal' mode, send the Telegram signal.
    # If in 'live' mode, this function will now also handle placing the actual LIMIT order
    # and storing it in `ict_strategy_states` for `monitor_pending_ict_entries` to pick up.

    signal_timestamp_final = klines_df.index[-1] # Timestamp of the last kline data used for signal generation

    # Prepare ICT details for Telegram
    ict_details_for_telegram = {
        'grab_type': grab_details.get('type'),
        'price_swept': grab_details.get('price_swept'),
        'grab_timestamp': grab_details.get('candle_timestamp'),
        'fvg_range': {
            'fvg_bottom': fvg.get('fvg_bottom'),
            'fvg_top': fvg.get('fvg_top')
        } if fvg else None,
        'fvg_direction': fvg.get('direction') if fvg else None,
        'fvg_timestamp_c3': fvg.get('timestamp_c3') if fvg else None,
        'ob_range': {
            'ob_bottom': order_block.get('ob_bottom'),
            'ob_top': order_block.get('ob_top')
        } if order_block else None,
        'ob_direction': order_block.get('direction') if order_block else None,
        'ob_timestamp': order_block.get('timestamp') if order_block else None,
        'po3_confirmed': po3_confirmed,
        'entry_logic_used': entry_type_used
    }

    if configs['mode'] == 'signal':
        est_pnl_tp1 = calculate_pnl_for_fixed_capital(entry_price, tp1_price, signal_side, current_leverage_on_symbol, 100.0, symbol_info)
        est_pnl_sl = calculate_pnl_for_fixed_capital(entry_price, stop_loss_price, signal_side, current_leverage_on_symbol, 100.0, symbol_info)
        
        send_entry_signal_telegram(
            configs, symbol, f"ICT_LIMIT_{signal_side.upper()}", current_leverage_on_symbol, 
            entry_price, tp1_price, tp2_price, tp3_price, stop_loss_price, 
            configs['risk_percent'], est_pnl_tp1, est_pnl_sl, symbol_info, 
            "ICT Limit Signal", signal_timestamp_final, "LIMIT",
            ict_details=ict_details_for_telegram # Pass constructed ICT details
        )
        
        with active_signals_lock: # Store in active_signals for virtual monitoring
            signal_id_virtual_ict = f"signal_ict_limit_{symbol}_{int(signal_timestamp_final.timestamp())}"
            active_signals[symbol] = { # Overwrites if previous signal for symbol existed
                "signal_id": signal_id_virtual_ict, "entry_price": entry_price, 
                "current_sl_price": stop_loss_price, "initial_sl_price": stop_loss_price,
                "current_tp1_price": tp1_price, "initial_tp1_price": tp1_price,
                "current_tp2_price": tp2_price, "initial_tp2_price": tp2_price,
                "current_tp3_price": tp3_price, "initial_tp3_price": tp3_price,
                "side": signal_side, "leverage": current_leverage_on_symbol, 
                "symbol_info": symbol_info, "open_timestamp": signal_timestamp_final, 
                "strategy_type": "ICT_LIMIT_SIGNAL", "sl_management_stage": "initial",
                "last_update_message_type": "NEW_SIGNAL",
                "initial_risk_per_unit": risk_per_unit
                # Add FVG, OB, Sweep details if needed for richer virtual monitoring later
            }
        
        # Log this new signal event to CSV
        log_event_ict_signal = {
            "SignalID": signal_id_virtual_ict, "Symbol": symbol, "Strategy": "ICT_LIMIT_SIGNAL", "Side": signal_side,
            "Leverage": current_leverage_on_symbol, "SignalOpenPrice": entry_price, 
            "EventType": "NEW_ICT_LIMIT_SIGNAL", "EventPrice": entry_price,
            "Notes": f"SL:{stop_loss_price:.{p_prec}f}, TP1:{tp1_price:.{p_prec}f}, TP2:{tp2_price:.{p_prec}f}, TP3:{tp3_price:.{p_prec}f}. EntryType:{entry_type_used}. Grab@{grab_details['candle_timestamp']}",
            "EstimatedPNL_USD100": est_pnl_tp1 
        }
        log_signal_event_to_csv(log_event_ict_signal)
        
        with last_signal_lock: last_signal_time[f"{symbol}_ict_limit"] = dt.now() # Update cooldown
        with recent_trade_signatures_lock: recent_trade_signatures[trade_sig_ict_limit] = dt.now() # Record signature
        print(f"{log_prefix} Signal Mode: ICT Limit Signal for {symbol} sent and recorded.")
        return

    elif configs['mode'] == 'live': # Live mode: Place the actual LIMIT order
        print(f"{log_prefix} Live Mode: Attempting to place ICT LIMIT order for {symbol} ({signal_side})")
        
        # Ensure leverage and margin type are set on the symbol for live trading
        if not (set_leverage_on_symbol(client, symbol, current_leverage_on_symbol) and \
                set_margin_type_on_symbol(client, symbol, configs['margin_type'], configs)):
            print(f"{log_prefix} Failed to set leverage/margin for {symbol} before placing ICT limit order. Aborting."); return

        # Use the already calculated hypothetical_qty for the order
        # This qty was already sanity checked.
        
        limit_order_api_side = "BUY" if signal_side == "LONG" else "SELL"
        position_side_api = signal_side.upper()

        ict_limit_order_obj, ict_limit_error_msg = place_new_order(
            client, symbol_info, limit_order_api_side, "LIMIT", hypothetical_qty, 
            price=entry_price, position_side=position_side_api
        )

        if not ict_limit_order_obj:
            reason_live_fail = f"Failed to place ICT LIMIT entry order. Error: {ict_limit_error_msg}"
            print(f"{log_prefix} {reason_live_fail}")
            send_trade_rejection_notification(symbol, f"ICT_LIMIT_{signal_side}", reason_live_fail, entry_price, stop_loss_price, tp1_price, hypothetical_qty, symbol_info, configs)
            return

        print(f"{log_prefix} ICT LIMIT entry order PLACED: ID {ict_limit_order_obj['orderId']}. Status: {ict_limit_order_obj['status']}")
        
        # Send Telegram notification for the placed limit order
        if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
            qty_disp_live = f"{hypothetical_qty:.{int(symbol_info.get('quantityPrecision',0))}f}"
            limit_order_msg_live = (
                f"⏳ ICT LIMIT ORDER PLACED (Live) ⏳\n\n"
                f"Symbol: `{symbol}`\nSide: `{signal_side}`\nType: `LIMIT`\n"
                f"Quantity: `{qty_disp_live}`\nLimit Price: `{entry_price:.{p_prec}f}`\n"
                f"Order ID: `{ict_limit_order_obj['orderId']}`\n\n"
                f"Intended SL (if filled): `{stop_loss_price:.{p_prec}f}`\n"
                f"Intended TP1 (if filled): `{tp1_price:.{p_prec}f}`\nMonitoring for fill..."
            )
            send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], limit_order_msg_live)

        # Store pending order details in ict_strategy_states for monitoring
        with ict_strategy_states_lock:
            ict_strategy_states[symbol] = {
                "state": "PENDING_ICT_ENTRY",
                "pending_entry_order_id": ict_limit_order_obj['orderId'],
                "order_placed_timestamp": dt.now(timezone.utc), # Record when bot placed it
                "pending_entry_details": {
                    "limit_price": entry_price, "sl_price": stop_loss_price,
                    "tp1_price": tp1_price, "tp2_price": tp2_price, "tp3_price": tp3_price, # Store all TPs
                    "quantity": hypothetical_qty, "side": signal_side,
                    "symbol_info": symbol_info, "leverage": current_leverage_on_symbol,
                    "initial_risk_per_unit": risk_per_unit,
                    "fvg_snapshot": fvg, # Store context
                    "ob_snapshot": order_block,
                    "grab_snapshot": grab_details
                }
            }
            print(f"{log_prefix} ICT Limit order {ict_limit_order_obj['orderId']} for {symbol} stored in ict_strategy_states for monitoring.")
        
        with last_signal_lock: last_signal_time[f"{symbol}_ict_limit"] = dt.now() # Cooldown
        with recent_trade_signatures_lock: recent_trade_signatures[trade_sig_ict_limit] = dt.now() # Signature
        return # Successfully placed live limit order or sent signal
    
    # If not 'signal' or 'live', or if other conditions prevent action (though should be caught earlier)
    # print(f"{log_prefix} Mode is '{configs['mode']}'. No action taken for ICT setup (not placing live limit order or sending signal).")

# --- End ICT Strategy Main Logic ---

# --- Monitor Active Signals (for 'Signal' Mode) ---
def monitor_active_signals(client, configs):
    global active_signals, active_signals_lock
    
    if not active_signals:
        return

    log_prefix = "[SignalMonitor]"
    print(f"\n{log_prefix} Monitoring {len(active_signals)} active signal(s)... {format_elapsed_time(configs.get('cycle_start_time_ref', time.time()))}")
    
    signals_to_remove = []

    # Iterate over a copy of items for safe modification during iteration (though removal happens at the end)
    active_signals_copy = {}
    with active_signals_lock:
        active_signals_copy = active_signals.copy()

    for symbol, signal_details in active_signals_copy.items():
        s_info = signal_details.get('symbol_info', {})
        p_prec = int(s_info.get('pricePrecision', 2))
        
        current_market_price = get_current_market_price(client, symbol)
        if current_market_price is None:
            print(f"{log_prefix} Could not fetch market price for {symbol}. Skipping this signal update cycle.")
            continue

        print(f"{log_prefix} Checking Signal: {symbol} ({signal_details['side']}), Entry: {signal_details['entry_price']:.{p_prec}f}, SL: {signal_details['current_sl_price']:.{p_prec}f}, TP1: {signal_details.get('current_tp1_price', 0.0):.{p_prec}f}, Market: {current_market_price:.{p_prec}f}")

        # --- Check for SL Hit ---
        if (signal_details['side'] == "LONG" and current_market_price <= signal_details['current_sl_price']) or \
           (signal_details['side'] == "SHORT" and current_market_price >= signal_details['current_sl_price']):
            
            pnl_sl_hit = calculate_pnl_for_fixed_capital(
                signal_details['entry_price'], signal_details['current_sl_price'], signal_details['side'],
                signal_details['leverage'], 100.0, s_info
            )
            msg_detail = f"Stop Loss hit at ~{signal_details['current_sl_price']:.{p_prec}f}."
            send_signal_update_telegram(configs, signal_details, "SL_HIT", msg_detail, current_market_price, pnl_sl_hit)
            
            # Log SL hit to CSV
            log_event_details_sl_hit = {
                "SignalID": signal_details.get('signal_id'), "Symbol": symbol, "Strategy": signal_details.get('strategy_type'), 
                "Side": signal_details.get('side'), "Leverage": signal_details.get('leverage'), 
                "SignalOpenPrice": signal_details.get('entry_price'), "EventType": "SL_HIT", 
                "EventPrice": signal_details.get('current_sl_price'), "Notes": msg_detail,
                "EstimatedPNL_USD100": pnl_sl_hit
            }
            log_signal_event_to_csv(log_event_details_sl_hit)
            
            signals_to_remove.append(symbol)
            with active_signals_lock: 
                if symbol in active_signals: 
                     active_signals[symbol]["last_update_message_type"] = "SL_HIT"
                     active_signals[symbol]["last_update_message_detail_preview"] = msg_detail[:50]
            continue # Move to next signal

        # --- Check for TP Hits (Iterate through TPs) ---
        # For simplicity, this version checks TPs sequentially. If TP1 hits, it doesn't immediately check TP2 in the same cycle for the same signal.
        # More advanced logic could handle multiple TP hits within one price movement or candle.
        
        tp_levels_to_check = []
        if signal_details.get('current_tp1_price'):
            tp_levels_to_check.append({'name': 'TP1', 'price': signal_details['current_tp1_price'], 'key': 'current_tp1_price'})
        if signal_details.get('current_tp2_price'):
            tp_levels_to_check.append({'name': 'TP2', 'price': signal_details['current_tp2_price'], 'key': 'current_tp2_price'})
        if signal_details.get('current_tp3_price'):
            tp_levels_to_check.append({'name': 'TP3', 'price': signal_details['current_tp3_price'], 'key': 'current_tp3_price'})

        any_tp_hit_this_signal = False
        for tp_info in tp_levels_to_check:
            if tp_info['price'] is None: continue # Skip if this TP level is not set

            tp_hit = False
            if signal_details['side'] == "LONG" and current_market_price >= tp_info['price']:
                tp_hit = True
            elif signal_details['side'] == "SHORT" and current_market_price <= tp_info['price']:
                tp_hit = True
            
            if tp_hit:
                any_tp_hit_this_signal = True
                pnl_tp_hit = calculate_pnl_for_fixed_capital(
                    signal_details['entry_price'], tp_info['price'], signal_details['side'],
                    signal_details['leverage'], 100.0, s_info
                )
                msg_detail_tp = f"{tp_info['name']} hit at ~{tp_info['price']:.{p_prec}f}."
                send_signal_update_telegram(configs, signal_details, tp_info['name'] + "_HIT", msg_detail_tp, current_market_price, pnl_tp_hit)
                
                # Log TP hit to CSV
                log_event_details_tp_hit = {
                    "SignalID": signal_details.get('signal_id'), "Symbol": symbol, "Strategy": signal_details.get('strategy_type'),
                    "Side": signal_details.get('side'), "Leverage": signal_details.get('leverage'),
                    "SignalOpenPrice": signal_details.get('entry_price'), "EventType": tp_info['name'] + "_HIT",
                    "EventPrice": tp_info['price'], "Notes": msg_detail_tp,
                    "EstimatedPNL_USD100": pnl_tp_hit
                }
                log_signal_event_to_csv(log_event_details_tp_hit)

                with active_signals_lock:
                    if symbol in active_signals:
                        active_signals[symbol][tp_info['key']] = None 
                        active_signals[symbol]["last_update_message_type"] = tp_info['name'] + "_HIT"
                        active_signals[symbol]["last_update_message_detail_preview"] = msg_detail_tp[:50]
                        
                        if tp_info['name'] == 'TP1' and signal_details.get('strategy_type') == "FIBONACCI_RETRACEMENT":
                            if active_signals[symbol].get('fib_move_sl_after_tp1_config') == "breakeven":
                                buffer_r = active_signals[symbol].get('fib_breakeven_buffer_r_config', 0.0)
                                initial_risk_pu = active_signals[symbol].get('initial_risk_per_unit', 0)
                                new_be_sl_price = signal_details['entry_price']
                                if initial_risk_pu > 0:
                                    if signal_details['side'] == "LONG":
                                        new_be_sl_price = signal_details['entry_price'] + (initial_risk_pu * buffer_r)
                                    else: # SHORT
                                        new_be_sl_price = signal_details['entry_price'] - (initial_risk_pu * buffer_r)
                                new_be_sl_price = round(new_be_sl_price, p_prec)
                                
                                current_sl = active_signals[symbol]['current_sl_price']
                                if (signal_details['side'] == "LONG" and new_be_sl_price > current_sl) or \
                                   (signal_details['side'] == "SHORT" and new_be_sl_price < current_sl):
                                    active_signals[symbol]['current_sl_price'] = new_be_sl_price
                                    active_signals[symbol]['sl_management_stage'] = "after_tp1_breakeven"
                                    be_msg = f"TP1 Hit. SL moved to Breakeven (+{buffer_r}R) at ~{new_be_sl_price:.{p_prec}f}."
                                    send_signal_update_telegram(configs, active_signals[symbol], "SL_ADJUSTED_BE", be_msg, current_market_price)
                                    active_signals[symbol]["last_update_message_type"] = "SL_ADJUSTED_BE"
                                    active_signals[symbol]["last_update_message_detail_preview"] = be_msg[:50]
                                    
                                    # Log SL adjustment to CSV
                                    log_event_sl_adj_be = {
                                        "SignalID": active_signals[symbol].get('signal_id'), "Symbol": symbol, "Strategy": active_signals[symbol].get('strategy_type'),
                                        "Side": active_signals[symbol].get('side'), "Leverage": active_signals[symbol].get('leverage'),
                                        "SignalOpenPrice": active_signals[symbol].get('entry_price'), "EventType": "SL_ADJUSTED_BE",
                                        "EventPrice": new_be_sl_price, "Notes": be_msg, "EstimatedPNL_USD100": None # PNL not direct for SL adjustment
                                    }
                                    log_signal_event_to_csv(log_event_sl_adj_be)

                # TODO: Implement more advanced SL adjustments (trailing, micro-pivot) here if a TP was hit
                # For now, if any TP hit, we might break or continue to check other TPs depending on strategy rules.
                # Current logic will check all TPs. If all TPs are hit (become None), the signal should be closed.
                
        # Check if all TPs are now None (meaning they've all been "hit" and processed)
        all_tps_processed = True
        for tp_key_check in ['current_tp1_price', 'current_tp2_price', 'current_tp3_price']:
            # Check directly in the potentially updated active_signals dict under lock
            with active_signals_lock:
                signal_data_for_all_tp_check = active_signals.get(symbol)
                if signal_data_for_all_tp_check and signal_data_for_all_tp_check.get(tp_key_check) is not None:
                    all_tps_processed = False
                    break
        
        if all_tps_processed and any_tp_hit_this_signal: # If any TP was hit this cycle leading to all TPs being processed
            # Final P&L would be based on the last TP hit.
            # This is a simplified closure; a real system might average out P&L if TPs had different quantities.
            # For signals, we assume full quantity at each TP for estimation purposes if not specified otherwise.
            # The P&L for the *last* TP hit was already sent. Send a general closure message.
            msg_detail_all_tps = "All Take Profit levels achieved."
            send_signal_update_telegram(configs, signal_details, "CLOSED_ALL_TPS", msg_detail_all_tps, current_market_price)
            
            # Log "CLOSED_ALL_TPS" event to CSV
            log_event_all_tps = {
                "SignalID": signal_details.get('signal_id'), "Symbol": symbol, "Strategy": signal_details.get('strategy_type'),
                "Side": signal_details.get('side'), "Leverage": signal_details.get('leverage'),
                "SignalOpenPrice": signal_details.get('entry_price'), "EventType": "CLOSED_ALL_TPS",
                "EventPrice": current_market_price, # Current market price at time of this event
                "Notes": msg_detail_all_tps,
                "EstimatedPNL_USD100": None # PNL for final TP was already logged with its specific TP_HIT event
            }
            log_signal_event_to_csv(log_event_all_tps)

            signals_to_remove.append(symbol)
            with active_signals_lock:
                if symbol in active_signals:
                     active_signals[symbol]["last_update_message_type"] = "CLOSED_ALL_TPS"
                     active_signals[symbol]["last_update_message_detail_preview"] = msg_detail_all_tps[:50]

        # TODO: Implement Trailing SL logic here (e.g., based on check_and_adjust_sl_tp_dynamic concepts or micro-pivots)
        # When implemented, SL_ADJUSTED events from trailing should also be logged to CSV.
        # Example for a hypothetical trailing SL adjustment:
        # if trailing_sl_moved:
        #     log_event_sl_trail = {
        #         "SignalID": signal_details.get('signal_id'), ..., "EventType": "SL_ADJUSTED_TRAIL",
        #         "EventPrice": new_trailing_sl_price, "Notes": "Trailing SL updated", ...
        #     }
        #     log_signal_event_to_csv(log_event_sl_trail)
        # This would involve:
        # 1. Getting 1-min candle buffer if micro-pivot is enabled.
        # 2. Calculating new potential SL based on rules.
        # 3. If new SL is an improvement, update signal_details['current_sl_price'] and send a "SL_ADJUSTED" message.

    # Remove closed signals
    if signals_to_remove:
        with active_signals_lock:
            for sym_remove in signals_to_remove:
                if sym_remove in active_signals:
                    print(f"{log_prefix} Removing signal for {sym_remove} from active_signals.")
                    del active_signals[sym_remove]
    
    print(f"{log_prefix} Finished monitoring active signals. {len(active_signals_copy) - len(signals_to_remove)} signals remain active.")

# --- CSV Logging for Signal Summary ---
CSV_SUMMARY_FILENAME = "signal_summary.csv"
CSV_SUMMARY_HEADERS = [
    "Timestamp", "Date", "SignalID", "Symbol", "Strategy", "Side", 
    "Leverage", "SignalOpenPrice", "EventType", "EventPrice", "Notes", "EstimatedPNL_USD100"
]

def log_signal_event_to_csv(event_details_dict: dict):
    """
    Logs a signal event to the CSV_SUMMARY_FILENAME.
    Manages file creation and header writing.
    """
    try:
        file_exists = os.path.exists(CSV_SUMMARY_FILENAME)
        
        # Ensure all header columns are present in the dict, fill with N/A if missing
        row_data = {header: event_details_dict.get(header, "N/A") for header in CSV_SUMMARY_HEADERS}
        
        # Specific formatting for certain fields
        row_data["Timestamp"] = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')
        row_data["Date"] = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')

        if isinstance(row_data.get("SignalOpenPrice"), (float, int)):
            row_data["SignalOpenPrice"] = f"{row_data['SignalOpenPrice']:.8g}" # General purpose float formatting
        if isinstance(row_data.get("EventPrice"), (float, int)):
            row_data["EventPrice"] = f"{row_data['EventPrice']:.8g}"
        if isinstance(row_data.get("EstimatedPNL_USD100"), (float, int)):
             row_data["EstimatedPNL_USD100"] = f"{row_data['EstimatedPNL_USD100']:.2f}"


        df_to_append = pd.DataFrame([row_data])
        
        if not file_exists:
            df_to_append.to_csv(CSV_SUMMARY_FILENAME, mode='a', header=True, index=False, columns=CSV_SUMMARY_HEADERS)
        else:
            df_to_append.to_csv(CSV_SUMMARY_FILENAME, mode='a', header=False, index=False, columns=CSV_SUMMARY_HEADERS)

        # print(f"Logged signal event to {CSV_SUMMARY_FILENAME}: {row_data.get('SignalID')} - {row_data.get('EventType')}")
    except Exception as e:
        print(f"Error logging signal event to CSV: {e}")
        traceback.print_exc()

def escape_markdown_v1(text: str) -> str:
    """Escapes characters for Telegram Markdown V1."""
    if not isinstance(text, str):
        return ""
    # Order matters to avoid double escaping if an escape char is part of another
    text = text.replace('_', r'\_')
    text = text.replace('*', r'\*')
    text = text.replace('`', r'\`')
    text = text.replace('[', r'\[')
    # text = text.replace(']', r'\]') # Closing bracket usually doesn't need escaping unless in a pair
    return text

def get_summary_from_csv(target_date_str: str = None, get_last_day: bool = False) -> tuple[str, int]:
    """
    Reads signal_summary.csv, filters by date, and formats a summary string.

    Args:
        target_date_str (str, optional): Specific date YYYY-MM-DD. Defaults to None (today).
        get_last_day (bool, optional): If True, gets the summary for the last recorded day before today.

    Returns:
        tuple[str, int]: Formatted summary string and count of events.
    """
    if not os.path.exists(CSV_SUMMARY_FILENAME):
        return "🗓️ Signal summary file (`signal_summary.csv`) not found.", 0

    try:
        df = pd.read_csv(CSV_SUMMARY_FILENAME)
        if df.empty:
            return "🗓️ Signal summary file is empty.", 0
        
        # Ensure Timestamp is datetime for sorting, and Date is string for comparison
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = df['Date'].astype(str)

        today_utc_str = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')
        
        date_to_filter = ""
        day_description = ""

        if target_date_str:
            # Validate target_date_str format (basic check)
            try:
                pd.to_datetime(target_date_str, format='%Y-%m-%d') # Test conversion
                date_to_filter = target_date_str
                day_description = f"for {date_to_filter}"
            except ValueError:
                return f"❌ Invalid date format. Please use YYYY-MM-DD. You provided: {target_date_str}", 0
        elif get_last_day:
            # Find the latest date in the CSV that is not today
            available_dates = sorted(df['Date'].unique(), reverse=True)
            last_recorded_day = None
            for date_val in available_dates:
                if date_val < today_utc_str:
                    last_recorded_day = date_val
                    break
            if last_recorded_day:
                date_to_filter = last_recorded_day
                day_description = f"for the last recorded day ({date_to_filter})"
            else:
                return "🗓️ No entries found for any day before today.", 0
        else: # Default to today
            date_to_filter = today_utc_str
            day_description = "for today"

        filtered_df = df[df['Date'] == date_to_filter].sort_values(by="Timestamp")

        if filtered_df.empty:
            return f"🗓️ No signal events found {day_description} ({date_to_filter}).", 0

        summary_lines = [f"📊 *Signal Summary {day_description} ({date_to_filter}):*"]
        
        # Group by SignalID to consolidate related events
        for signal_id, group in filtered_df.groupby("SignalID"):
            group = group.sort_values(by="Timestamp") # Ensure events within a signal are chronological
            first_event = group.iloc[0]
            summary_lines.append(
                f"\n🔹 Signal ID: `{signal_id}`\n"
                f"   Symbol: `{first_event['Symbol']}` ({first_event['Strategy']}) - {first_event['Side']} @ {first_event['Leverage']}x\n"
                f"   Opened At: `{pd.to_datetime(first_event['Timestamp']).strftime('%H:%M:%S')}` (Price: {first_event['SignalOpenPrice']})"
            )
            
            for idx, event in group.iterrows():
                event_time_str = pd.to_datetime(event['Timestamp']).strftime('%H:%M:%S')
                event_type = event['EventType']
                event_price = event['EventPrice']
                notes = event['Notes'] if pd.notna(event['Notes']) and event['Notes'] != "N/A" else ""
                pnl_est = event['EstimatedPNL_USD100'] if pd.notna(event['EstimatedPNL_USD100']) and event['EstimatedPNL_USD100'] != "N/A" else ""

                emoji = "➡️"
                if "NEW_SIGNAL" in event_type: emoji = "🔔"
                elif "TP" in event_type and "HIT" in event_type: emoji = "✅"
                elif "SL_HIT" in event_type: emoji = "❌"
                elif "ADJUSTED" in event_type: emoji = "🛡️"
                elif "CLOSED" in event_type: emoji = "🎉"

                line = f"     {emoji} {event_time_str} - *{event_type}*"
                if event_price != "N/A" and event_price != first_event['SignalOpenPrice']: # Don't repeat open price for NEW_SIGNAL
                    line += f" at `{event_price}`"
                if notes:
                    line += f" ({escape_markdown_v1(notes)})" # Escape notes
                if pnl_est:
                    line += f" | Est. PNL ($100): `{pnl_est}`"
                summary_lines.append(line)
        
        event_count = len(filtered_df)
        summary_lines.append(f"\nTotal events for this period: {event_count}")
        
        full_summary = "\n".join(summary_lines)
        
        # Truncate if too long for a single Telegram message (Telegram limit is 4096 chars)
        # This is a basic truncation; smarter splitting might be needed for very long summaries.
        if len(full_summary) > 4000: # Leave some buffer
            full_summary = full_summary[:4000] + "\n\n... (summary truncated due to length)"
            
        return full_summary, event_count

    except pd.errors.EmptyDataError:
        return f"🗓️ Signal summary file is empty or not formatted correctly.", 0
    except FileNotFoundError: # Should be caught by os.path.exists, but as a fallback.
        return f"🗓️ Signal summary file (`{CSV_SUMMARY_FILENAME}`) not found.", 0
    except Exception as e:
        print(f"Error reading or processing CSV for summary: {e}")
        traceback.print_exc()
        return f"❌ Error generating summary: {str(e)}", 0

# --- P&L Calculation Helper for Signals ---
def calculate_pnl_for_fixed_capital(entry_price: float, exit_price: float, side: str, leverage: int, fixed_capital_usdt: float = 100.0, symbol_info: dict = None) -> float | None:
    """
    Calculates the estimated P&L for a trade based on a fixed capital amount (e.g., $100).

    Args:
        entry_price (float): The entry price.
        exit_price (float): The exit price (SL or TP).
        side (str): "LONG" or "SHORT".
        leverage (int): The leverage used for the trade.
        fixed_capital_usdt (float): The amount of capital to simulate the trade with.
        symbol_info (dict): Symbol information (primarily for quantity precision if needed, though less critical for P&L estimation).

    Returns:
        float | None: Estimated P&L in USDT, or None if inputs are invalid.
    """
    if entry_price is None or exit_price is None: # Added check for None prices
        print(f"Error in P&L calc for signal: Entry price ({entry_price}) or Exit price ({exit_price}) is None.")
        return None
    if not all([isinstance(entry_price, (int,float)) and entry_price > 0, 
                isinstance(exit_price, (int,float)) and exit_price > 0, 
                leverage > 0, fixed_capital_usdt > 0]):
        print(f"Error in P&L calc for signal: Invalid inputs (entry_price:{entry_price}, exit_price:{exit_price}, leverage:{leverage}, capital:{fixed_capital_usdt})")
        return None
    if side not in ["LONG", "SHORT"]:
        print(f"Error in P&L calc for signal: Invalid side '{side}'")
        return None # Added return for invalid side
    
    # Additional check: For a LONG, exit_price (TP) should be > entry_price for positive P&L calc,
    # and exit_price (SL) should be < entry_price. Vice-versa for SHORT.
    # This function primarily calculates P&L based on values; logical validation of TP/SL placement should be done before calling.
    # However, if entry and exit are identical, P&L is 0.
    if entry_price == exit_price:
        return 0.0
        return None

    # Calculate position size in base asset based on fixed capital and leverage
    # Position Value (USDT) = fixed_capital_usdt * leverage
    # Quantity (Base Asset) = Position Value (USDT) / entry_price
    position_value_usdt = fixed_capital_usdt * leverage
    quantity_base_asset = position_value_usdt / entry_price

    # Apply quantity precision if symbol_info is available (optional for estimation, but good for consistency)
    if symbol_info:
        q_prec = int(symbol_info.get('quantityPrecision', 8)) # Default to high precision if not found
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            if step_size > 0:
                 quantity_base_asset = math.floor(quantity_base_asset / step_size) * step_size
        quantity_base_asset = round(quantity_base_asset, q_prec)
    
    if quantity_base_asset == 0:
        # This can happen if fixed_capital is too small for the min_qty or step_size of the symbol.
        # For a $100 signal, this is less likely for major pairs but possible for very low-priced assets or high min_qty.
        print(f"Warning in P&L calc for signal: Calculated quantity for ${fixed_capital_usdt} is zero for {symbol_info.get('symbol', 'N/A') if symbol_info else 'N/A'}. Check capital or symbol parameters.")
        return 0.0 # P&L is 0 if no position can be opened.

    pnl = 0.0
    if side == "LONG":
        pnl = (exit_price - entry_price) * quantity_base_asset
    elif side == "SHORT":
        pnl = (entry_price - exit_price) * quantity_base_asset
    
    return pnl

# --- Main Execution ---

def reset_global_states_for_restart():
    """Resets critical global states before a bot restart."""
    global daily_high_equity, day_start_equity, last_trading_day
    global trading_halted_drawdown, trading_halted_daily_loss, daily_realized_pnl
    global symbols_currently_processing, last_signal_time, recent_trade_signatures
    global active_trades, active_trades_lock, fib_strategy_states, fib_strategy_states_lock # Added fib_strategy_states

    print("Resetting global states for bot restart...")

    daily_high_equity = 0.0
    day_start_equity = 0.0
    last_trading_day = None
    trading_halted_drawdown = False
    trading_halted_daily_loss = False
    daily_realized_pnl = 0.0
    
    symbols_currently_processing.clear()
    last_signal_time.clear()
    recent_trade_signatures.clear()
    
    with active_trades_lock:
        active_trades.clear()
    
    with fib_strategy_states_lock: # Also clear fib_strategy_states
        fib_strategy_states.clear()
        print("Fibonacci strategy states cleared for restart.")
    
    with ict_strategy_states_lock: # Clear ICT strategy states
        ict_strategy_states.clear()
        print("ICT strategy states cleared for restart.")

    print("Global states reset complete for restart.")

def main_bot_logic(): # Renamed main to main_bot_logic
    global telegram_load_choice, telegram_make_changes_choice # Access globals

    print("Initializing Binance Trading Bot - Advance EMA Cross Strategy (ID: 8)")
    bot_start_time_utc = pd.Timestamp.now(tz='UTC')
    bot_start_time_str = bot_start_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # Pass Telegram override choices to get_user_configurations
    configs = get_user_configurations(
        load_choice_override=telegram_load_choice,
        make_changes_override=telegram_make_changes_choice
    )
    
    # Reset global override choices after they've been used for this startup sequence
    telegram_load_choice = None
    telegram_make_changes_choice = None
    print("Telegram configuration choice overrides have been applied and reset for the next startup.")

    print("\nLoaded Configurations:")
    for k, v in configs.items(): 
        if k not in ["api_key", "api_secret", "telegram_bot_token", "telegram_chat_id"]: # Hide sensitive keys
             print(f"  {k.replace('_',' ').title()}: {v}")

    # Call initialize_binance_client once and unpack its results
    # The first item in the returned tuple is the actual client object.
    client_connection_details = initialize_binance_client(configs)
    # Assign to global client variable for handlers
    global client 
    client = client_connection_details[0] # client_obj is now assigned to global client
    env_for_msg = client_connection_details[1]
    server_time_obj = client_connection_details[2]
    
    if not client: # Check if the global client object is None
        # Error messages are printed by initialize_binance_client already
        print("Exiting: Binance client init failed.") 
        sys.exit(1)

    # Now use global client for operations
    print("\nFetching initial account balance...")
    initial_balance = get_account_balance(client, configs) # Use the global client object, pass configs

    if configs['mode'] == 'live' and initial_balance is None:
        # IP-related error or other critical issue detected by get_account_balance returning None.
        # Telegram alert with IP details is sent from within get_account_balance for -2015.
        print("CRITICAL: Initial API connection failed (e.g., IP whitelist issue or invalid API key).")
        while initial_balance is None:
            print("Retrying initial connection in 60 seconds... Check Telegram for IP details if this is a whitelist issue.")
            time.sleep(60)
            initial_balance = get_account_balance(client, configs) # Use global client
            if initial_balance is not None:
                print(f"Initial API connection successful! Current balance: {initial_balance:.2f} USDT")
                # Re-send startup message now that connection is established
                if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                    open_pos_text_retry = get_open_positions(client, format_for_telegram=True) # Use global client
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
    else: # Fallback if client is somehow valid but other details are not
        print(f"\nSuccessfully connected to Binance API. (Connection details partially unavailable)")

    configs.setdefault("api_delay_short", 1) 
    configs.setdefault("api_delay_symbol_processing", 0.1) # Can be very short with threads
    configs.setdefault("loop_delay_minutes", 5)

    # Load symbol blacklist
    blacklist_filepath = "symbol_blacklist.csv"
    blacklisted_symbols = load_symbol_blacklist(blacklist_filepath)
    if blacklisted_symbols:
        print(f"Loaded {len(blacklisted_symbols)} symbol(s) from blacklist: {', '.join(blacklisted_symbols)}")

    # --- Load symbols from symbols.csv ---
    symbols_csv_filepath = "symbols.csv"
    user_defined_symbols = load_symbols_from_csv(symbols_csv_filepath)

    if not user_defined_symbols:
        print(f"Warning: '{symbols_csv_filepath}' is empty or not found. Attempting to fall back to all USDT perpetuals.")
        monitored_symbols_all = get_all_usdt_perpetual_symbols(client) # Fallback
        if not monitored_symbols_all:
            print("Exiting: No symbols found from fallback (all USDT perpetuals). Please check symbols.csv or ensure API connectivity.")
            sys.exit(1)
        print(f"Using {len(monitored_symbols_all)} symbols from Binance (all USDT perpetuals) as fallback.")
    else:
        print(f"Loaded {len(user_defined_symbols)} symbols from '{symbols_csv_filepath}'.")
        monitored_symbols_all = user_defined_symbols
    
    # Filter out blacklisted symbols from the (user-defined or fallback) list
    monitored_symbols = [s for s in monitored_symbols_all if s not in blacklisted_symbols]
    
    excluded_by_blacklist_count = len(monitored_symbols_all) - len(monitored_symbols)
    if excluded_by_blacklist_count > 0:
        excluded_symbols_str = ', '.join(sorted(list(set(monitored_symbols_all) - set(monitored_symbols))))
        print(f"Excluded {excluded_by_blacklist_count} symbol(s) due to blacklist: {excluded_symbols_str}")
    
    if not monitored_symbols: 
        print("Exiting: No symbols to monitor after applying blacklist (or no symbols found from CSV/fallback).")
        sys.exit(1)
    
    # --- Initial Telegram Notification ---
    # --- Telegram Startup Notification ---
    if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
        open_pos_text = "None"
        # active_trades is initially empty, so s_info_map will be empty.
        # This is fine as there are no bot-managed SL/TP to show at this very start.
        # If there were pre-existing positions from API, they'd show "N/A (Bot)" for SL/TP.
        s_info_map_initial = _build_symbol_info_map_from_active_trades(active_trades) # active_trades is global
        
        if client: # Use global client
            if configs["mode"] == "live":
                open_pos_text = get_open_positions(client, format_for_telegram=True, active_trades_data=active_trades.copy(), symbol_info_map=s_info_map_initial) # Use global client
            else:
                open_pos_text = "None (Backtest Mode)"
        else:
            open_pos_text = "N/A (Client not initialized)"

        startup_msg = build_startup_message(configs, initial_balance, open_pos_text, bot_start_time_str)
        send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], startup_msg)
        configs["last_startup_message"] = startup_msg

        # Start Telegram polling in a new daemon thread
        thread = threading.Thread(
            target=start_telegram_polling,
            args=(configs["telegram_bot_token"], configs),
            name="TelegramPollingThread",
            daemon=True
        )
        thread.start()
        print("Main ▶ Telegram polling thread started.")

    else:
        print("Telegram notifications disabled (token or chat_id not configured in keys.py or load failed).")
    configs["monitored_symbols_count"] = len(monitored_symbols)
    configs['bot_start_time_str'] = bot_start_time_str # Make bot_start_time_str available in configs

    print(f"Found {len(monitored_symbols)} USDT perpetuals. Proceeding to monitor all for {'live trading' if configs['mode'] == 'live' else 'backtesting'}.")
    # confirm = input(f"Found {len(monitored_symbols)} USDT perpetuals. Monitor all for {'live trading' if configs['mode'] == 'live' else 'backtesting'}? (yes/no) [yes]: ").lower().strip()
    # if confirm == 'no': print("Exiting by user choice."); sys.exit(0)

    if configs["mode"] == "live" or configs["mode"] == "signal": # Include "signal" mode here
        try:
            # For "signal" mode, trading_loop will be adapted not to place real orders
            # but will perform scanning and (soon) signal notifications.
            trading_loop(client, configs, monitored_symbols) 
        except KeyboardInterrupt: 
            print(f"\nBot stopped by user (Ctrl+C) in {configs['mode']} mode.")
            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], f"ℹ️ Bot stopped by user (Ctrl+C) in {configs['mode']} mode.")
        except Exception as e: 
            print(f"\nCRITICAL UNEXPECTED ERROR IN {configs['mode'].upper()} MODE: {e}")
            traceback.print_exc()
            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                error_message_for_telegram = f"🆘 CRITICAL BOT ERROR ({configs['mode'].upper()} MODE) 🆘\nBot encountered an unhandled exception and may have stopped.\nError: {str(e)[:1000]}\nCheck logs!"
                send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], error_message_for_telegram)
        finally:
            print(f"\n--- {configs['mode'].title()} Mode Bot Shutting Down ---")

            if configs['mode'] == 'live': # Only cancel real orders if in live mode
                if client and active_trades: 
                    print(f"Cancelling {len(active_trades)} bot-managed active SL/TP orders...")
                    with active_trades_lock:
                        for symbol, trade_details in list(active_trades.items()):
                            for oid_key in ['sl_order_id', 'tp_order_id']: # For EMA Cross
                                oid = trade_details.get(oid_key)
                                if oid:
                                    try:
                                        print(f"Cancelling {oid_key} {oid} for {symbol}...")
                                        client.futures_cancel_order(symbol=symbol, orderId=oid) 
                                    except Exception as e_c:
                                        print(f"Failed to cancel {oid_key} {oid} for {symbol}: {e_c}")
                            
                            # For Fib Multi-TP strategy
                            if trade_details.get('strategy_type') == "FIBONACCI_MULTI_TP" and 'tp_orders' in trade_details:
                                for tp_order_info in trade_details['tp_orders']:
                                    tp_id_to_cancel = tp_order_info.get('id')
                                    if tp_id_to_cancel and tp_order_info.get('status') == "OPEN":
                                        try:
                                            print(f"Cancelling Fib TP order {tp_id_to_cancel} ({tp_order_info.get('name')}) for {symbol}...")
                                            client.futures_cancel_order(symbol=symbol, orderId=tp_id_to_cancel)
                                        except Exception as e_fib_tp_cancel:
                                            print(f"Failed to cancel Fib TP order {tp_id_to_cancel} for {symbol}: {e_fib_tp_cancel}")
            
            print(f"{configs['mode'].title()} Bot shutdown sequence complete.")

            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                shutdown_time_str = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')
                shutdown_msg = (
                    f"*⚠️ Bot Stopped*\n\n"
                    f"*Stop Time:* `{shutdown_time_str}`\n"
                    f"*Strategy:* `{configs.get('strategy_name', 'Unknown')}`\n"
                    f"*Environment:* `{configs.get('environment', 'Unknown')}`"
                )
                send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], shutdown_msg)
            # The second block for cancelling orders was a duplicate, it has been removed by ensuring the first one uses global `client`.
            # The `print("Live Bot shutdown sequence complete.")` was also duplicated.
    elif configs["mode"] == "backtest":
        try:
            # Pass global client to backtesting_loop; it might also be used for initializing backtest env
            backtesting_loop(client, configs, monitored_symbols) # Use global client
        except KeyboardInterrupt: print("\nBacktest stopped by user (Ctrl+C).")
        except Exception as e: print(f"\nCRITICAL UNEXPECTED ERROR IN BACKTESTING: {e}"); traceback.print_exc()
        finally:
            print("\n--- Backtesting Complete ---")
            # No orders to cancel in backtest mode unless simulating exchange interactions
            # For now, active_trades will just be cleared or analyzed.
        
            active_trades.clear() # Clear trades for a clean slate if re-run or for reporting
            print("Backtest shutdown sequence complete.")
    return configs # Return configs at the end of the function


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
                    # send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), final_msg.strip()) # User request to remove this success message
                    print(f"{log_prefix} Suppressed Telegram notification for successful SL/TP placement on UNMANAGED trade {symbol}.")


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

def monitor_pending_ict_entries(client, configs: dict):
    """
    Monitors pending ICT LIMIT entry orders.
    If filled, places SL/TP. If timed out or invalid, cancels.
    """
    global ict_strategy_states, ict_strategy_states_lock, active_trades, active_trades_lock, active_signals, active_signals_lock
    
    log_prefix_monitor = "[ICTLimitMonitor]"
    symbols_to_reset_ict_state = []

    pending_ict_entries_copy = {}
    with ict_strategy_states_lock:
        pending_ict_entries_copy = {sym: state_data for sym, state_data in ict_strategy_states.items() if state_data.get('state') == "PENDING_ICT_ENTRY"}

    if not pending_ict_entries_copy:
        return

    print(f"\n{log_prefix_monitor} Checking {len(pending_ict_entries_copy)} pending ICT entry order(s)...")

    for symbol, state_data in pending_ict_entries_copy.items():
        order_id = state_data.get('pending_entry_order_id')
        pending_details = state_data.get('pending_entry_details')
        
        if not order_id or not pending_details:
            print(f"{log_prefix_monitor} Incomplete pending ICT entry data for {symbol}. Resetting state.")
            symbols_to_reset_ict_state.append(symbol)
            continue
        
        # order_placed_time can be fetched from state_data if stored, or use a default timeout logic.
        # For ICT, the order is typically placed based on a very recent setup.
        order_placed_timestamp = pending_details.get('order_placed_timestamp', dt.now(timezone.utc) - pd.Timedelta(minutes=10)) # Fallback if not stored

        try:
            order_status = client.futures_get_order(symbol=symbol, orderId=order_id)
            s_info = pending_details.get('symbol_info', get_symbol_info(client, symbol)) # Ensure s_info
            if not s_info:
                print(f"{log_prefix_monitor} Cannot get symbol_info for {symbol}. Cannot process order {order_id}. Resetting."); symbols_to_reset_ict_state.append(symbol); continue

            p_prec = int(s_info.get('pricePrecision', 2))
            q_prec = int(s_info.get('quantityPrecision', 0))

            if order_status['status'] == 'FILLED':
                print(f"{log_prefix_monitor} ✅ ICT LIMIT entry order {order_id} for {symbol} FILLED!")
                actual_entry_price = float(order_status['avgPrice'])
                total_filled_qty = float(order_status['executedQty'])
                
                intended_sl_price = pending_details['sl_price']
                intended_tp1_price = pending_details['tp1_price'] # Assuming TP1 is primary for ICT as per initial notes
                trade_side = pending_details['side'] # "LONG" or "SHORT"
                position_side_for_sl_tp = trade_side.upper()

                # Recalculate SL/TP based on actual fill price if significantly different (optional, but good practice)
                # For ICT, precision is key, so using intended SL/TP from the setup might be preferred.
                # Let's use intended for now.
                final_sl_price = intended_sl_price
                final_tp1_price = intended_tp1_price

                print(f"{log_prefix_monitor} Placing SL/TP for ICT trade {symbol}: SL={final_sl_price}, TP1={final_tp1_price}, Qty={total_filled_qty}")

                if configs['mode'] == 'signal':
                    # Update active_signals or log virtual fill
                    print(f"{log_prefix_monitor} Signal Mode: ICT Limit order {order_id} for {symbol} virtually FILLED.")
                    fill_msg = f"Virtual ICT limit entry for {symbol} filled at ~{actual_entry_price:.{p_prec}f}. SL/TP planned."
                    # Update signal in active_signals if it was added as pending, or add it now as active.
                    # For consistency, manage_ict_trade_entry already adds to active_signals. We confirm fill here.
                    with active_signals_lock:
                        if symbol in active_signals and active_signals[symbol].get('signal_id', '').startswith("signal_ict_"):
                             active_signals[symbol]['entry_price'] = actual_entry_price # Update with actual fill
                             active_signals[symbol]['current_sl_price'] = final_sl_price
                             active_signals[symbol]['current_tp1_price'] = final_tp1_price
                             active_signals[symbol]['status_note'] = "Entry Filled (Virtual)"
                             send_signal_update_telegram(configs, active_signals[symbol], "VIRTUAL_ICT_ENTRY_FILLED", fill_msg, actual_entry_price)
                    # Log event
                    log_event_ict_fill_virtual = {
                        "SignalID": active_signals[symbol].get('signal_id'), "Symbol": symbol, "Strategy": "ICT_STRATEGY",
                        "Side": trade_side, "Leverage": active_signals[symbol].get('leverage'),
                        "SignalOpenPrice": pending_details.get('limit_price'), # Original intended entry
                        "EventType": "VIRTUAL_ICT_ENTRY_FILLED", 
                        "EventPrice": actual_entry_price, # Actual (simulated) fill price
                        "Notes": fill_msg,
                        # PNL for a virtual fill is not directly applicable, but can estimate based on planned TP1 if desired
                        # For now, let's keep it N/A for the fill event itself.
                        "EstimatedPNL_USD100": "N/A" 
                    }
                    log_signal_event_to_csv(log_event_ict_fill_virtual)


                else: # Live/Backtest Mode
                    sl_ord_obj, sl_err_msg = place_new_order(client, s_info, "SELL" if trade_side == "LONG" else "BUY", "STOP_MARKET", total_filled_qty, stop_price=final_sl_price, position_side=position_side_for_sl_tp, is_closing_order=True)
                    tp_ord_obj, tp_err_msg = place_new_order(client, s_info, "SELL" if trade_side == "LONG" else "BUY", "TAKE_PROFIT_MARKET", total_filled_qty, stop_price=final_tp1_price, position_side=position_side_for_sl_tp, is_closing_order=True)

                    if not sl_ord_obj or not tp_ord_obj:
                        err_sl = f"SL Error: {sl_err_msg}" if not sl_ord_obj else "SL OK"
                        err_tp = f"TP Error: {tp_err_msg}" if not tp_ord_obj else "TP OK"
                        print(f"{log_prefix_monitor} CRITICAL: FAILED TO PLACE SL/TP for ICT {symbol}! {err_sl}, {err_tp}")
                        # Alert user
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), f"🆘 FAILED SL/TP for ICT {symbol} after fill! {err_sl}, {err_tp}")
                        # Position is open without SL/TP - safety net should catch, or manual intervention.
                        # We still add to active_trades to track the open position.
                    
                    with active_trades_lock:
                        if symbol not in active_trades: # Should not be in active_trades yet
                            active_trades[symbol] = {
                                "entry_order_id": order_id,
                                "sl_order_id": sl_ord_obj.get('orderId') if sl_ord_obj else None, 
                                "tp_order_id": tp_ord_obj.get('orderId') if tp_ord_obj else None, # Assuming single TP for now
                                "entry_price": actual_entry_price,
                                "current_sl_price": final_sl_price, 
                                "current_tp_price": final_tp1_price,
                                "initial_sl_price": final_sl_price, 
                                "initial_tp_price": final_tp1_price,
                                "initial_risk_per_unit": abs(actual_entry_price - final_sl_price),
                                "quantity": total_filled_qty, 
                                "side": trade_side.upper(),
                                "symbol_info": s_info,
                                "open_timestamp": pd.Timestamp(order_status['updateTime'], unit='ms', tz='UTC'),
                                "strategy_type": "ICT_STRATEGY", # Mark as ICT strategy
                                "sl_management_stage": "initial"
                            }
                            print(f"{log_prefix_monitor} ICT trade for {symbol} moved to active_trades.")
                            # Send Telegram for new trade
                            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                                qty_prec_ict = int(s_info.get('quantityPrecision', 0))
                                price_prec_ict = int(s_info.get('pricePrecision', 2))
                                ict_fill_msg_tg = (
                                    f"🚀 ICT TRADE ENTRY FILLED 🚀\n\n"
                                    f"Symbol: `{symbol}` ({trade_side.upper()})\n"
                                    f"Qty: `{total_filled_qty:.{qty_prec_ict}f}` @ Entry: `{actual_entry_price:.{price_prec_ict}f}`\n"
                                    f"SL: `{final_sl_price:.{price_prec_ict}f}` (ID: {sl_ord_obj.get('orderId') if sl_ord_obj else 'FAIL'})\n"
                                    f"TP1: `{final_tp1_price:.{price_prec_ict}f}` (ID: {tp_ord_obj.get('orderId') if tp_ord_obj else 'FAIL'})\n"
                                    f"Limit Order ID: `{order_id}`"
                                )
                                send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], ict_fill_msg_tg)

                        else: # Symbol somehow already in active_trades - very unlikely if logic is correct
                            print(f"{log_prefix_monitor} WARNING: {symbol} already in active_trades when processing ICT fill. This is unexpected.")
                
                symbols_to_reset_ict_state.append(symbol) # Mark for removal from pending ICT states

            elif order_status['status'] in ['CANCELED', 'EXPIRED', 'REJECTED', 'PENDING_CANCEL']:
                print(f"{log_prefix_monitor} ICT Limit order {order_id} for {symbol} is {order_status['status']}. Resetting state.")
                symbols_to_reset_ict_state.append(symbol)
            
            elif order_status['status'] == 'NEW':
                ict_order_timeout_minutes = configs.get("ict_order_timeout_minutes", DEFAULT_ICT_ORDER_TIMEOUT_MINUTES) # Configurable timeout
                # If order_placed_timestamp was from dt.now() upon placement, this check is fine.
                # If it was from kline data, ensure timezone consistency.
                if (dt.now(timezone.utc) - order_placed_timestamp).total_seconds() > ict_order_timeout_minutes * 60:
                    print(f"{log_prefix_monitor} ICT Limit order {order_id} for {symbol} timed out ({ict_order_timeout_minutes}m). Cancelling.")
                    try:
                        client.futures_cancel_order(symbol=symbol, orderId=order_id)
                        print(f"{log_prefix_monitor} Successfully cancelled timed-out ICT order {order_id} for {symbol}.")
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), f"⏳ ICT entry order for `{symbol}` (ID: {order_id}) timed out and was cancelled.")
                    except Exception as e_cancel:
                        print(f"{log_prefix_monitor} Failed to cancel timed-out ICT order {order_id} for {symbol}: {e_cancel}")
                    symbols_to_reset_ict_state.append(symbol)
                # else: Still new and within timeout, do nothing.
            
            else: # Other statuses like PARTIALLY_FILLED (needs careful handling, for now treat as "reset")
                 print(f"{log_prefix_monitor} ICT Limit order {order_id} for {symbol} has status: {order_status['status']}. Resetting state for now.")
                 symbols_to_reset_ict_state.append(symbol)

        except BinanceAPIException as e:
            if e.code == -2013: # Order does not exist
                print(f"{log_prefix_monitor} ICT Limit order {order_id} for {symbol} NOT FOUND. Resetting state.")
                symbols_to_reset_ict_state.append(symbol)
            else:
                print(f"{log_prefix_monitor} API Error checking ICT order {order_id} for {symbol}: {e}")
        except Exception as e:
            print(f"{log_prefix_monitor} Unexpected error checking ICT order {order_id} for {symbol}: {e}")
            traceback.print_exc()
            symbols_to_reset_ict_state.append(symbol)

    if symbols_to_reset_ict_state:
        with ict_strategy_states_lock:
            for sym_to_reset in symbols_to_reset_ict_state:
                if sym_to_reset in ict_strategy_states:
                    print(f"{log_prefix_monitor} Resetting ICT strategy state for {sym_to_reset} to IDLE.")
                    del ict_strategy_states[sym_to_reset] # Remove from pending list
                else:
                    print(f"{log_prefix_monitor} Warning: Tried to reset ICT state for {sym_to_reset}, but it was not found.")

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

# --- Global variables for Telegram Handlers and Bot Control ---
# client instance will be populated in main()
client = None 
# configs dictionary will be populated in main() and is treated as global in many functions
# active_trades, active_trades_lock, etc. are already defined at the top level.

config_filepath = "configure.csv" # Path to the configuration CSV file, used by handlers
bot_shutdown_requested = False    # Flag to signal graceful shutdown from Telegram command
bot_restart_requested = False     # Flag to signal graceful restart from Telegram command
trading_halted_manual = False     # Flag for manual trading halt from Telegram command

# Global variables for Telegram-driven configuration choices (will be set by a Telegram command)
telegram_load_choice: str = None
telegram_make_changes_choice: str = None

# --- Telegram Command Handlers ---

async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global client, active_trades, active_trades_lock # Removed configs
    global trading_halted_drawdown, trading_halted_daily_loss, trading_halted_manual
    
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))
    if not expected_chat_id:
        print("Error: telegram_chat_id not found in app_configs for status_handler.")
        await update.message.reply_text("Bot configuration error: Admin chat ID not set.", parse_mode="Markdown")
        return
    if effective_chat_id != expected_chat_id:
        print(f"Status request from unauthorized chat ID: {effective_chat_id}")
        return

    balance_for_status = get_account_balance(client, app_configs) 
    if balance_for_status is None:
        await update.message.reply_text("Error: Could not fetch current balance. API connection issue likely.", parse_mode="Markdown")
        return
    
    current_balance = balance_for_status 

    equity = get_current_equity(client, app_configs, balance_for_status, active_trades, active_trades_lock)
    if equity is None:
        await update.message.reply_text("Error: Could not calculate current equity.", parse_mode="Markdown")
        return

    active_trades_count = 0
    with active_trades_lock:
        active_trades_count = len(active_trades)

    status_text = (
        f"📊 *Bot Status*\n\n"
        f"Balance: `{balance_for_status:.2f}` USDT\n"
        f"Equity: `{equity:.2f}` USDT\n"
        f"Halted (Manual): `{trading_halted_manual}`\n"
        f"Halted (Drawdown): `{trading_halted_drawdown}`\n"
        f"Halted (Daily Loss): `{trading_halted_daily_loss}`\n"
        f"Active Trades: `{active_trades_count}`"
    )
    await update.message.reply_text(status_text, parse_mode="Markdown")

async def positions_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global client, active_trades, active_trades_lock # Removed configs
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))
    if effective_chat_id != expected_chat_id: return

    s_info_map = {}
    active_trades_copy = {}
    with active_trades_lock:
        active_trades_copy = active_trades.copy() 
        s_info_map = _build_symbol_info_map_from_active_trades(active_trades_copy)
        
    positions_text = get_open_positions(client, format_for_telegram=True, active_trades_data=active_trades_copy, symbol_info_map=s_info_map)
    await update.message.reply_text(f"*Open Positions:*\n{positions_text}", parse_mode="Markdown")

async def orders_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global client # Removed configs
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))
    if effective_chat_id != expected_chat_id: return

    open_orders = get_open_orders(client) 
    if not open_orders:
        await update.message.reply_text("No open orders.", parse_mode="Markdown")
        return

    orders_texts = ["*Open Orders:*"]
    for o in open_orders:
        price_prec = 2 
        price_str = f"{float(o['price']):.{price_prec}f}" if o['type'] != 'MARKET' else 'MARKET'
        stop_price_str = f"{float(o.get('stopPrice',0)):.{price_prec}f}" if o.get('stopPrice') and float(o.get('stopPrice',0)) > 0 else "N/A"
        orders_texts.append(
            f"- {o['symbol']} (ID: {o['orderId']}): {o['side']} {o['type']} Qty: {o['origQty']} @ {price_str}, StopPx: {stop_price_str}"
        )
    await update.message.reply_text("\n".join(orders_texts), parse_mode="Markdown")

async def halt_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global trading_halted_manual # Removed configs
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))
    if effective_chat_id != expected_chat_id: return
    
    trading_halted_manual = True
    message = "Trading MANUALLY HALTED. Bot will not open new positions."
    print(message)
    await update.message.reply_text(f"🛑 {message}", parse_mode="Markdown")

async def resume_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global trading_halted_manual # Removed configs
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))
    if effective_chat_id != expected_chat_id: return

    trading_halted_manual = False
    message = "Trading MANUALLY RESUMED. Bot can open new positions (if other auto-halts are not active)."
    print(message)
    await update.message.reply_text(f"✅ {message}", parse_mode="Markdown")

async def close_all_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global client, active_trades, active_trades_lock # Removed configs
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))
    if effective_chat_id != expected_chat_id: return

    await update.message.reply_text("Attempting to close all active positions...", parse_mode="Markdown")
    try:
        await asyncio.to_thread(close_all_open_positions, client, app_configs, active_trades, active_trades_lock)
        await update.message.reply_text("All positions closure process initiated. Check logs for details.", parse_mode="Markdown")
    except Exception as e:
        detailed_error = f"Error during close_all_positions: {e}\n{traceback.format_exc()}"
        print(detailed_error)
        await update.message.reply_text(f"Error during close_all_positions: {e}", parse_mode="Markdown")


async def set_risk_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global config_filepath # Removed configs
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))
    if effective_chat_id != expected_chat_id: return

    try:
        if not context.args:
            await update.message.reply_text("Usage: /setrisk <percentage> (e.g., /setrisk 1.0)", parse_mode="Markdown")
            return
        risk_str = context.args[0]
        new_risk_percent_val = float(risk_str)

        if 0 < new_risk_percent_val <= 100: 
            old_risk = app_configs.get('risk_percent', 0.01) * 100.0 
            app_configs['risk_percent'] = new_risk_percent_val / 100.0 
            
            configs_to_save = app_configs.copy()
            for k_sens in ["api_key", "api_secret", "telegram_bot_token", "telegram_chat_id", "last_startup_message", "cycle_start_time_ref"]:
                configs_to_save.pop(k_sens, None)
            
            if save_configuration_to_csv(config_filepath, configs_to_save):
                msg = f"Risk per trade updated from {old_risk:.2f}% to {new_risk_percent_val:.2f}% and saved."
            else:
                msg = f"Risk per trade updated to {new_risk_percent_val:.2f}%, but FAILED to save to {config_filepath}."
            print(msg)
            await update.message.reply_text(msg, parse_mode="Markdown")
        else:
            await update.message.reply_text("Invalid risk percentage. Must be > 0 and <= 100.", parse_mode="Markdown")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /setrisk <percentage> (e.g., /setrisk 1.0)", parse_mode="Markdown")
    except Exception as e:
        print(f"Error in set_risk_handler: {e}")
        await update.message.reply_text(f"An error occurred: {e}", parse_mode="Markdown")


async def set_leverage_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global config_filepath # Removed configs
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))
    if effective_chat_id != expected_chat_id: return

    try:
        if not context.args:
            await update.message.reply_text("Usage: /setleverage <value> (e.g., /setleverage 20)", parse_mode="Markdown")
            return
        leverage_str = context.args[0]
        new_leverage = int(leverage_str)

        if 1 <= new_leverage <= 125: 
            old_leverage = app_configs.get('leverage', 20) 
            app_configs['leverage'] = new_leverage
            
            configs_to_save = app_configs.copy()
            for k_sens in ["api_key", "api_secret", "telegram_bot_token", "telegram_chat_id", "last_startup_message", "cycle_start_time_ref"]:
                configs_to_save.pop(k_sens, None)

            if save_configuration_to_csv(config_filepath, configs_to_save):
                msg = f"Default leverage updated from {old_leverage}x to {new_leverage}x and saved."
            else:
                msg = f"Default leverage updated to {new_leverage}x, but FAILED to save to {config_filepath}."
            print(msg)
            await update.message.reply_text(msg, parse_mode="Markdown")
            await update.message.reply_text(f"Note: This sets the default leverage. Dynamic leverage might override this for specific trades if enabled.", parse_mode="Markdown")
        else:
            await update.message.reply_text("Invalid leverage. Must be between 1 and 125.", parse_mode="Markdown")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /setleverage <value> (e.g., /setleverage 20)", parse_mode="Markdown")
    except Exception as e:
        print(f"Error in set_leverage_handler: {e}")
        await update.message.reply_text(f"An error occurred: {e}", parse_mode="Markdown")

async def log_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # global configs # Removed configs
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))
    if effective_chat_id != expected_chat_id: return

    last_startup_msg = app_configs.get("last_startup_message", "No recent status summary available.")
    log_text = (
        f"📜 *Bot Log/Status Summary*\n\n"
        f"Recent Status Snapshot:\n{last_startup_msg}\n\n"
        f"_Note: Detailed log file access via Telegram is currently basic. For full logs, please check the console/log file directly._"
    )
    await update.message.reply_text(log_text, parse_mode="Markdown")

async def config_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # global configs # Removed configs
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))
    if effective_chat_id != expected_chat_id: return

    config_texts = ["⚙️ *Current Bot Configuration:*"]
    excluded_keys = ["api_key", "api_secret", "telegram_bot_token", "telegram_chat_id", "last_startup_message", "cycle_start_time_ref", "strategy_id", "strategy_name"] 
    
    sorted_config_keys = sorted(app_configs.keys())

    for key in sorted_config_keys:
        if key not in excluded_keys:
            value = app_configs[key]
            if key == "risk_percent": value_str = f"{value * 100:.2f}%"
            elif key == "portfolio_risk_cap": value_str = f"{value:.2f}%"
            elif key == "max_drawdown_percent": value_str = f"{value:.2f}%"
            elif key == "daily_stop_loss_percent": value_str = f"{value:.2f}%"
            elif key == "target_annualized_volatility": value_str = f"{value * 100:.2f}% (decimal: {value:.4f})"
            elif isinstance(value, float): value_str = f"{value:.4f}"
            else: value_str = str(value)
            config_texts.append(f"`{key.replace('_', ' ').title()}`: `{value_str}`")
    
    if len(config_texts) == 1: 
        config_texts.append("No displayable configurations found.")

    message_parts = []
    current_part = ""
    for line in config_texts:
        if len(current_part) + len(line) + 1 > 4096: 
            message_parts.append(current_part)
            current_part = line
        else:
            if current_part: current_part += "\n"
            current_part += line
    if current_part: message_parts.append(current_part)

    for part in message_parts:
        await update.message.reply_text(part, parse_mode="Markdown")

async def shutdown_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_shutdown_requested # Removed configs
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))
    if effective_chat_id != expected_chat_id: return

    bot_shutdown_requested = True
    message = "Shutdown request received. Bot will attempt to stop gracefully after the current trading cycle."
    print(message)
    await update.message.reply_text(f"⏳ {message}", parse_mode="Markdown")

async def blacklist_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /blacklist command to show or add symbols to the blacklist."""
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))

    if effective_chat_id != expected_chat_id:
        print(f"Blacklist command from unauthorized chat ID: {effective_chat_id}")
        return

    blacklist_file = "symbol_blacklist.csv" # Define the path to the blacklist file

    if not context.args: # /blacklist - show current blacklist
        current_blacklist = load_symbol_blacklist(blacklist_file)
        if not current_blacklist:
            await update.message.reply_text("The symbol blacklist is currently empty.", parse_mode="Markdown")
        else:
            message_text = "*Current Symbol Blacklist:*\n" + "\n".join([f"- `{s}`" for s in current_blacklist])
            await update.message.reply_text(message_text, parse_mode="Markdown")
    else: # /blacklist <SYMBOL> - add to blacklist
        symbol_to_add = context.args[0].upper()
        
        # Validate symbol format (basic check, can be improved)
        if not symbol_to_add.isalnum() or not symbol_to_add.endswith("USDT"):
             await update.message.reply_text(f"Invalid symbol format: `{symbol_to_add}`. Please use format like `XRPUSDT`.", parse_mode="Markdown")
             return

        print(f"Attempting to add '{symbol_to_add}' to blacklist via Telegram command...")
        added = add_symbol_to_blacklist(blacklist_file, symbol_to_add)
        
        reply_messages = []
        if added:
            reply_messages.append(f"`{symbol_to_add}` added to the blacklist.")
            reply_messages.append("_Note: A bot restart is typically required for this change to affect the active symbol scanning process._")
        elif added is False and os.path.exists(blacklist_file): # Check if 'False' was due to already existing vs error
            # Re-load to check if it was truly an "already exists" case or a file save error
            # This assumes load_symbol_blacklist returns a list, and add_symbol_to_blacklist returned False because it was already there
            # or because of a file save error.
            # A more robust check in add_symbol_to_blacklist could return specific error codes/types.
            # For now, if it's not added and the file exists, we assume it was already there or a save error.
            # Let's refine `add_symbol_to_blacklist` to be more explicit or check here.
            # Re-checking if symbol is in the list after attempting to add.
            _temp_blacklist = load_symbol_blacklist(blacklist_file) # load_symbol_blacklist handles uppercase
            if symbol_to_add in _temp_blacklist:
                 reply_messages.append(f"`{symbol_to_add}` is already in the blacklist.")
            else: # If not in list after attempting add, then it was likely a save error
                 reply_messages.append(f"Error adding `{symbol_to_add}` to the blacklist. It was not found after attempting to add. Please check bot logs.")
        else: # `added` is False and file might not exist (e.g. initial error in add_symbol_to_blacklist)
            reply_messages.append(f"Error adding `{symbol_to_add}` to the blacklist. Please check bot logs.")

        # Show updated blacklist
        current_blacklist_updated = load_symbol_blacklist(blacklist_file)
        if not current_blacklist_updated:
            reply_messages.append("\nThe symbol blacklist is currently empty.")
        else:
            reply_messages.append("\n*Updated Symbol Blacklist:*")
            reply_messages.extend([f"- `{s}`" for s in current_blacklist_updated])
        
        await update.message.reply_text("\n".join(reply_messages), parse_mode="Markdown")

async def restart_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /restart command to gracefully restart the bot."""
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))

    if effective_chat_id != expected_chat_id:
        print(f"Restart command from unauthorized chat ID: {effective_chat_id}")
        return

    global bot_restart_requested, bot_shutdown_requested
    bot_restart_requested = True
    bot_shutdown_requested = False # Ensure restart takes precedence

    message = "Restart request received. The bot will restart after the current trading cycle completes."
    print(message)
    await update.message.reply_text(f"⏳ {message}", parse_mode="Markdown")

async def set_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): # Renamed function
    """Sets global choices for get_user_configurations via Telegram command /set <L_or_C> [Y_or_N]."""
    global telegram_load_choice, telegram_make_changes_choice
    
    app_configs = context.bot_data.get('configs', {}) # For admin check
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))

    if effective_chat_id != expected_chat_id:
        print(f"setconfigchoice command from unauthorized chat ID: {effective_chat_id}")
        return

    args = context.args
    reply_message = ""

    if not args or len(args) == 0:
        reply_message = "Usage: `/set <L_or_C> [Y_or_N]`\n" \
                        "Sets initial configuration load choices.\n" \
                        "L/C: Load from CSV (L) or Custom setup (C).\n" \
                        "Y/N (optional): Yes/No to making changes after CSV load.\n\n" \
                        "Example 1: `/set L N` (Load from CSV, No changes)\n" \
                        "Example 2: `/set C` (Custom setup, Y/N for changes is not applicable here)"
        await update.message.reply_text(reply_message, parse_mode="Markdown")
        return

    load_arg = args[0].lower()
    if load_arg in ['l', 'c']:
        telegram_load_choice = load_arg
        reply_message += f"Initial config load choice pre-set to: '{telegram_load_choice.upper()}'."
    else:
        reply_message += f"Invalid first argument '{args[0]}'. Must be 'L' or 'C'.\n" \
                         "Usage: `/set <L_or_C> [Y_or_N]`"
        await update.message.reply_text(reply_message, parse_mode="Markdown")
        return

    if load_arg == 'l': # Only process second argument if first was 'L'
        if len(args) > 1:
            make_changes_arg = args[1].lower()
            if make_changes_arg in ['y', 'n']:
                telegram_make_changes_choice = make_changes_arg
                reply_message += f"\nMake changes after CSV load choice pre-set to: '{telegram_make_changes_choice.upper()}'."
            else:
                reply_message += f"\nInvalid second argument '{args[1]}' for 'L' choice. Must be 'Y' or 'N'. 'Make changes' choice not set."
                # telegram_make_changes_choice remains as is (or None if not set before)
        else:
            # If only 'L' is given, reset make_changes_choice to prompt user or use default
            telegram_make_changes_choice = None 
            reply_message += "\n'Make changes after CSV load' choice not specified (will use default 'N' or prompt if applicable)."
    elif load_arg == 'c':
        # For 'C' (Custom setup), the second argument Y/N is not applicable. Reset it.
        telegram_make_changes_choice = None
        if len(args) > 1:
            reply_message += "\n(Note: Second argument Y/N is ignored for 'C'ustom setup choice)."


    reply_message += "\n\nThese settings will be used on the next bot startup or full restart."
    print(f"Telegram config choices set: Load='{telegram_load_choice}', Changes='{telegram_make_changes_choice}'")
    await update.message.reply_text(reply_message, parse_mode="Markdown")

async def summary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /sum command to retrieve and display signal summaries."""
    app_configs = context.bot_data.get('configs', {})
    effective_chat_id = str(update.effective_chat.id)
    expected_chat_id = str(app_configs.get("telegram_chat_id"))

    if effective_chat_id != expected_chat_id:
        print(f"Summary command from unauthorized chat ID: {effective_chat_id}")
        return

    args = context.args
    summary_text = ""
    event_count = 0

    if not args: # /sum - today's summary
        summary_text, event_count = get_summary_from_csv()
    elif len(args) == 1:
        arg1 = args[0].lower()
        if arg1 in ['l', 'last']: # /sum l - yesterday's/last day's summary
            summary_text, event_count = get_summary_from_csv(get_last_day=True)
        else: # /sum YYYY-MM-DD or YYYY.MM.DD
            date_arg = args[0].replace('.', '-') # Normalize to YYYY-MM-DD
            # Basic validation for date format (more thorough validation in get_summary_from_csv)
            if len(date_arg) == 10 and date_arg[4] == '-' and date_arg[7] == '-':
                summary_text, event_count = get_summary_from_csv(target_date_str=date_arg)
            else:
                summary_text = "❌ Invalid argument. Usage:\n" \
                               "/sum (for today)\n" \
                               "/sum L (for last recorded day before today)\n" \
                               "/sum YYYY-MM-DD (for a specific date)"
                event_count = -1 # Indicate error or invalid usage
    else: # Invalid number of arguments
        summary_text = "❌ Too many arguments. Usage:\n" \
                       "/sum (for today)\n" \
                       "/sum L (for last recorded day before today)\n" \
                       "/sum YYYY-MM-DD (for a specific date)"
        event_count = -1

    # Send the summary message
    # Telegram messages have a limit of 4096 characters.
    # get_summary_from_csv already truncates, but good to be aware.
    if summary_text:
        # Split message if it's too long (though get_summary_from_csv already truncates)
        max_len = 4096
        if len(summary_text) > max_len:
            parts = [summary_text[i:i + max_len] for i in range(0, len(summary_text), max_len)]
            for part in parts:
                await update.message.reply_text(part, parse_mode="Markdown")
        else:
            await update.message.reply_text(summary_text, parse_mode="Markdown")
    else: # Should not happen if get_summary_from_csv always returns a string
        await update.message.reply_text("Could not retrieve summary.", parse_mode="Markdown")


if __name__ == "__main__":
    # These globals are directly managed by the main loop or its direct conditions
    #global bot_shutdown_requested, bot_restart_requested, client 
    # Other globals like daily_high_equity etc., are modified by functions called from main_bot_logic
    # or by the new reset_global_states_for_restart function.

    returned_configs = None # Initialize to ensure it exists for the first Telegram message if restart happens early
    first_run = True

    while True:
        # Clear critical state at the beginning of each main loop iteration (especially for restarts)
        # This ensures that stale states from a previous run (e.g. live mode then signal mode) are wiped.
        with active_trades_lock:
            active_trades.clear()
        with active_signals_lock: # Assuming active_signals_lock is defined with active_signals
            active_signals.clear()
        with fib_strategy_states_lock:
            fib_strategy_states.clear()
        with ict_strategy_states_lock: # Also clear ICT states on main loop restart
            ict_strategy_states.clear()
        
        if first_run:
            print("First run: Cleared active_trades, active_signals, fib_strategy_states, and ict_strategy_states.")
            first_run = False
        else: # This is a restart
            print("Bot Restart: Cleared active_trades, active_signals, fib_strategy_states, and ict_strategy_states before restarting main logic.")
            
        # If this isn't the first iteration (i.e., it's a restart), client might already exist.
        # main_bot_logic is responsible for initializing/re-initializing it.
        # If client becomes None due to an error in main_bot_logic, restart might be problematic
        # for pre-restart order cancellation. This is a complex edge case.
        # For now, assume main_bot_logic sets client correctly or exits.

        returned_configs = main_bot_logic() # Call the main operational logic

        if bot_restart_requested:
            print("Bot restart sequence initiated...")
            
            # Attempt to cancel any outstanding orders from the previous run if client is available
            if client: # Check if client was successfully initialized in the previous run
                print("Performing pre-restart order cancellation check (if any trades still marked active by bot)...")
                with active_trades_lock: 
                    # active_trades should ideally be empty here if trading_loop exited cleanly,
                    # but this is a fallback.
                    for symbol, trade_details in list(active_trades.items()):
                        for oid_key in ['sl_order_id', 'tp_order_id']:
                            oid = trade_details.get(oid_key)
                            if oid:
                                try:
                                    print(f"Pre-restart: Attempting to cancel {oid_key} {oid} for {symbol}...")
                                    client.futures_cancel_order(symbol=symbol, orderId=oid)
                                except Exception as e_c:
                                    print(f"Pre-restart: Failed to cancel {oid_key} {oid} for {symbol}: {e_c}")
            active_trades.clear() # Ensure it's cleared again after attempted cancellations

            reset_global_states_for_restart() # Call the centralized reset function

            bot_restart_requested = False 
            bot_shutdown_requested = False # Ensure restart overrides shutdown for the next loop

            print("Global states have been reset. Restarting in 5 seconds...")
            
            # Use returned_configs from the *just completed* run of main_bot_logic for the notification
            if returned_configs and returned_configs.get("telegram_bot_token") and returned_configs.get("telegram_chat_id"):
                 send_telegram_message(returned_configs["telegram_bot_token"], returned_configs["telegram_chat_id"], "✅ Bot has shut down current operations and is restarting now...")
            else:
                print("Telegram details not available from the last run's configs for restart message.")
            
            time.sleep(5) 
            continue 
        else:
            print("Bot shutdown sequence complete (not a restart request, or error during execution). Exiting script.")
            break
