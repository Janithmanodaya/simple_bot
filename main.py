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

# --- Configuration Defaults ---
DEFAULT_RISK_PERCENT = 1.0       # Default account risk percentage per trade (e.g., 1.0 for 1%)
DEFAULT_LEVERAGE = 20            # Default leverage (e.g., 20x)
DEFAULT_MAX_CONCURRENT_POSITIONS = 5 # Default maximum number of concurrent open positions
DEFAULT_MARGIN_TYPE = "ISOLATED" # Default margin type: "ISOLATED" or "CROSS"
DEFAULT_MAX_SCAN_THREADS = 10    # Default threads for scanning symbols

# --- Global State Variables ---
# Stores details of active trades. Key: symbol (e.g., "BTCUSDT")
# Value: dict with trade info like order IDs, entry/SL/TP prices, quantity, side.
active_trades = {}
active_trades_lock = threading.Lock() # Lock for synchronizing access to active_trades

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

        placeholders = ["<your-testnet-api-key>", "<your-testnet-secret>",
                        "<your-mainnet-api-key>", "<your-mainnet-secret>"]
        if not api_key or not api_secret or api_key in placeholders or api_secret in placeholders:
            print(f"Error: API key/secret for {env} not found or not configured in keys.py.")
            print("Please open keys.py and replace placeholder values.")
            sys.exit(1)
        return api_key, api_secret
    except FileNotFoundError:
        print("Error: keys.py not found. Please create it and add your API credentials.")
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
    while True:
        env_input = input("Select environment (testnet/mainnet): ").lower().strip()
        if env_input in ["testnet", "mainnet"]:
            configs["environment"] = env_input
            break
        print("Invalid environment. Please enter 'testnet' or 'mainnet'.")
    configs["api_key"], configs["api_secret"] = load_api_keys(configs["environment"])

    while True:
        mode_input = input("Select mode (1:live / 2:backtest): ").strip()
        if mode_input in ["1", "2"]:
            configs["mode"] = "live" if mode_input == "1" else "backtest"
            break
        print("Invalid mode. Please enter '1' for live or '2' for backtest.")

    if configs["mode"] == "backtest":
        while True:
            try:
                days_input = input("Enter number of days for backtesting (e.g., 30): ")
                days = int(days_input)
                if days > 0:
                    configs["backtest_days"] = days
                    break
                print("Number of days must be a positive integer.")
            except ValueError:
                print("Invalid input. Please enter an integer for the number of days.")
        
        while True:
            balance_choice = input("For backtest, use current account balance or set a custom start balance? (current/custom) [current]: ").lower().strip()
            if not balance_choice: balance_choice = "current" # Default if user presses Enter

            if balance_choice in ["current", "custom"]:
                configs["backtest_start_balance_type"] = balance_choice
                break
            print("Invalid choice. Please enter 'current' or 'custom'.")

        if configs["backtest_start_balance_type"] == "custom":
            while True:
                try:
                    custom_bal_input = input("Enter custom start balance for backtest (e.g., 10000): ")
                    custom_bal = float(custom_bal_input)
                    if custom_bal > 0:
                        configs["backtest_custom_start_balance"] = custom_bal
                        break
                    print("Custom balance must be a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a number for the custom balance.")

    while True:
        try:
            risk_input = input(f"Enter account risk % per trade (e.g., 1 for 1%, default: {DEFAULT_RISK_PERCENT}%): ")
            risk_percent = float(risk_input or DEFAULT_RISK_PERCENT)
            if 0 < risk_percent <= 100:
                configs["risk_percent"] = risk_percent / 100
                break
            print("Risk percentage must be a positive value (e.g., 0.5, 1, up to 100).")
        except ValueError:
            print("Invalid input for risk percentage. Please enter a number.")
    while True:
        try:
            leverage_input = input(f"Enter leverage (e.g., 10 for 10x, default: {DEFAULT_LEVERAGE}x): ")
            leverage = int(leverage_input or DEFAULT_LEVERAGE)
            if 1 <= leverage <= 125:
                configs["leverage"] = leverage
                break
            print("Leverage must be an integer between 1 and 125.")
        except ValueError:
            print("Invalid input for leverage. Please enter an integer.")
    while True:
        try:
            max_pos_input = input(f"Enter max concurrent positions (default: {DEFAULT_MAX_CONCURRENT_POSITIONS}): ")
            max_concurrent_positions = int(max_pos_input or DEFAULT_MAX_CONCURRENT_POSITIONS)
            if max_concurrent_positions > 0:
                configs["max_concurrent_positions"] = max_concurrent_positions
                break
            print("Max concurrent positions must be a positive integer.")
        except ValueError:
            print("Invalid input for max positions. Please enter an integer.")
    while True:
        margin_input = input(f"Enter margin type (ISOLATED/CROSS, default: {DEFAULT_MARGIN_TYPE}): ").upper().strip()
        margin_type = margin_input or DEFAULT_MARGIN_TYPE
        if margin_type in ["ISOLATED", "CROSS"]:
            configs["margin_type"] = margin_type
            break
        print("Invalid margin type. Please enter 'ISOLATED' or 'CROSS'.")
    while True:
        try:
            threads_input = input(f"Enter maximum symbol scan threads (1-50, default: {DEFAULT_MAX_SCAN_THREADS}): ")
            max_scan_threads = int(threads_input or DEFAULT_MAX_SCAN_THREADS)
            if 1 <= max_scan_threads <= 50:
                configs["max_scan_threads"] = max_scan_threads
                break
            print("Max scan threads must be an integer between 1 and 50.")
        except ValueError:
            print("Invalid input for scan threads. Please enter an integer.")
    configs["strategy_id"] = 8
    configs["strategy_name"] = "Advance EMA Cross"
    print("--- Configuration Complete ---")
    return configs

# --- Binance API Interaction Functions (Error handling included) ---

def initialize_binance_client(configs):
    api_key, api_secret, env = configs["api_key"], configs["api_secret"], configs["environment"]
    try:
        client = Client(api_key, api_secret, testnet=(env == "testnet"))
        client.ping()
        server_time = client.get_server_time()
        print(f"\nSuccessfully connected to Binance {env.title()} API. Server Time: {pd.to_datetime(server_time['serverTime'], unit='ms')} UTC")
        return client
    except BinanceAPIException as e: print(f"Binance API Exception (client init): {e}"); return None
    except Exception as e: print(f"Error initializing Binance client: {e}"); return None

def get_historical_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=500, backtest_days=None):
    """
    Fetches historical klines. If backtest_days is specified, it fetches data for that many days.
    Otherwise, it fetches the most recent 'limit' klines.
    """
    start_time = time.time()
    if backtest_days:
        # Calculate how many klines are needed based on the interval and days
        # Binance API limit is 1000-1500 klines per request depending on endpoint (futures vs spot)
        # For 15min interval: 4 klines/hour * 24 hours/day = 96 klines/day
        # Let's use a known limit like 1000 for client.get_historical_klines
        klines_per_day = (24 * 60) // int(interval.replace('m','').replace('h','').replace('d','')) # Approximate
        if 'h' in interval: klines_per_day = 24 // int(interval.replace('h',''))
        if 'd' in interval: klines_per_day = 1

        total_klines_needed = klines_per_day * backtest_days
        print(f"Fetching klines for {symbol}, interval {interval}, for {backtest_days} days (approx {total_klines_needed} klines)...")
        
        # get_historical_klines fetches in batches of 1000 (or its internal limit)
        # The "days ago" string needs to be like "X days ago UTC"
        # Ensure client.get_historical_klines exists and is the correct method for this.
        # The standard client.get_historical_klines is what we need.
        start_str = f"{backtest_days + 1} days ago UTC" # Fetch a bit more to ensure enough data
        try:
            klines = client.get_historical_klines(symbol, interval, start_str)
        except Exception as e: # Catch if get_historical_klines fails with start_str
             print(f"Failed to fetch historical klines with start_str: {e}. Trying with limit.")
             # Fallback or alternative logic if the above fails or is not desired
             # For simplicity, this example will stick to the intended method
             # but a real implementation might need robust fetching in chunks if start_str is problematic
             # or if total_klines_needed > API max per call (which get_historical_klines handles by pagination)
             klines = [] # Ensure klines is defined

    else: # Original behavior for live mode
        print(f"Fetching klines for {symbol}, interval {interval}, limit {limit}...")
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        except BinanceAPIException as e:
            print(f"API Error fetching klines for {symbol}: {e}"); return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching klines for {symbol}: {e}"); return pd.DataFrame()

    duration = time.time() - start_time
    try: # Moved common processing into this try block
        if not klines:
            print(f"No kline data for {symbol} (fetch duration: {duration:.2f}s).")
            return pd.DataFrame()
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        print(f"Fetched {len(df)} klines for {symbol} (runtime: {duration:.2f}s).")
        if len(df) < 200: print(f"Warning: Low kline count for {symbol} ({len(df)}), EMA200 may be inaccurate.")
        return df
    except BinanceAPIException as e: print(f"API Error fetching klines for {symbol} ({time.time()-start_time:.2f}s): {e}"); return pd.DataFrame()
    except Exception as e: print(f"Error processing klines for {symbol}: {e}"); return pd.DataFrame()

def get_account_balance(client, asset="USDT"):
    try:
        balances = client.futures_account_balance()
        for b in balances:
            if b['asset'] == asset: print(f"Account Balance ({asset}): {b['balance']}"); return float(b['balance'])
        print(f"{asset} not found in futures balance."); return 0.0
    except Exception as e: print(f"Error getting balance: {e}"); return 0.0

def get_open_positions(client):
    try:
        positions = [p for p in client.futures_position_information() if float(p.get('positionAmt',0)) != 0]
        if not positions: print("No open positions."); return []
        print("Current Open Positions:")
        for p in positions: print(f"  {p['symbol']}: Amt={p['positionAmt']}, Entry={p['entryPrice']}, PnL={p['unRealizedProfit']}")
        return positions
    except Exception as e: print(f"Error getting positions: {e}"); return []

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
        if e.code == -4046: print(f"Margin for {symbol} already {margin_type}."); return True
        print(f"API Error setting margin for {symbol}: {e}"); return False
    except Exception as e: print(f"Error setting margin for {symbol}: {e}"); return False

def place_new_order(client, symbol_info, side, order_type, quantity, price=None, stop_price=None, reduce_only=None):
    symbol, p_prec, q_prec = symbol_info['symbol'], int(symbol_info['pricePrecision']), int(symbol_info['quantityPrecision'])
    params = {"symbol": symbol, "side": side.upper(), "type": order_type.upper(), "quantity": f"{quantity:.{q_prec}f}"}
    if order_type.upper() in ["LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if price is None: print(f"Price needed for {order_type} on {symbol}"); return None
        params.update({"price": f"{price:.{p_prec}f}", "timeInForce": "GTC"})
    if order_type.upper() in ["STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if stop_price is None: print(f"Stop price needed for {order_type} on {symbol}"); return None
        params["stopPrice"] = f"{stop_price:.{p_prec}f}"
    if reduce_only is not None: params["reduceOnly"] = str(reduce_only).lower()
    try:
        order = client.futures_create_order(**params)
        print(f"Order PLACED: {order['symbol']} ID {order['orderId']} {order['side']} {order['type']} {order['origQty']} @ {order.get('price','MARKET')} SP:{order.get('stopPrice','N/A')} AvgP:{order.get('avgPrice','N/A')} Status:{order['status']}")
        return order
    except Exception as e: print(f"ORDER FAILED for {symbol} {side} {quantity} {order_type}: {e}"); return None

# --- Indicator, Strategy, SL/TP, Sizing ---
def calculate_ema(df, period, column='close'):
    if column not in df or len(df) < period: return None # Basic checks
    return df[column].ewm(span=period, adjust=False).mean()

def check_ema_crossover_conditions(df, short_ema_col='EMA100', long_ema_col='EMA200', validation_candles=20):
    if not all(c in df for c in [short_ema_col, long_ema_col, 'low', 'high']) or len(df) < validation_candles + 2 \
       or df[[short_ema_col, long_ema_col]].iloc[-(validation_candles + 2):].isnull().values.any(): return None # Not enough data or EMAs are NaN, no signal

    prev_short, curr_short = df[short_ema_col].iloc[-2], df[short_ema_col].iloc[-1]
    prev_long, curr_long = df[long_ema_col].iloc[-2], df[long_ema_col].iloc[-1]
    
    signal_type = None
    # Basic crossover check
    if prev_short <= prev_long and curr_short > curr_long: signal_type = "LONG_CROSS"
    elif prev_short >= prev_long and curr_short < curr_long: signal_type = "SHORT_CROSS"
    else: return None # No crossover event this candle

    # Validation part
    val_df = df.iloc[-(validation_candles + 1) : -1] # Look at candles *before* the current one for validation
    if len(val_df) < validation_candles:
        # This case implies we have a crossover, but not enough preceding candles for full validation.
        # Depending on strictness, this could be a 'NO_SIGNAL_INSUFFICIENT_VALIDATION_HISTORY'
        # For now, let's treat it as unable to validate, so no actionable signal.
        # Or, we can log this specific state if desired. For now, let's return a distinct status.
        print(f"Potential {signal_type.replace('_CROSS','')} signal, but insufficient validation history ({len(val_df)}/{validation_candles} candles).")
        return "INSUFFICIENT_VALIDATION_HISTORY" 
        
    print(f"Potential {signal_type.replace('_CROSS','')} signal: {short_ema_col} {curr_short:.4f} vs {long_ema_col} {curr_long:.4f}.")

    for i in range(len(val_df)):
        c = val_df.iloc[i]; e100, e200 = val_df[short_ema_col].iloc[i], val_df[long_ema_col].iloc[i]
        if (c['low']<=e100<=c['high']) or (c['low']<=e200<=c['high']):
            print(f"Validation FAILED for {signal_type.replace('_CROSS','')} at {val_df.index[i]}: Price touched EMAs during validation period.")
            return "VALIDATION_FAILED" # Crossover happened, but validation failed
            
    print(f"{signal_type.replace('_CROSS','')} signal VALIDATED.")
    return signal_type.replace('_CROSS','') # Return "LONG" or "SHORT" if validated

def calculate_swing_high_low(df, window=20, idx=-1):
    if len(df) < window + abs(idx) or idx - window < -len(df): return None, None
    chunk = df.iloc[idx - window : idx]
    if chunk.empty: return (df['high'].iloc[idx-1], df['low'].iloc[idx-1]) if len(df) >= abs(idx)+1 else (None,None)
    return chunk['high'].max(), chunk['low'].min()

def calculate_sl_tp_values(entry, side, ema100, df_klines, idx=-1):
    tp_pct, sl_max_pct = 0.01, 0.01
    tp = entry * (1 + tp_pct if side == "LONG" else 1 - tp_pct)
    sl_ema = ema100 * (1 - 0.0005 if side == "LONG" else 1 + 0.0005)
    if abs(entry - sl_ema) / entry <= sl_max_pct: final_sl = sl_ema
    else:
        print(f"SL from EMA >{sl_max_pct*100}%. Using swing point.")
        sw_h, sw_l = calculate_swing_high_low(df_klines, 20, idx)
        if side == "LONG": final_sl = (sw_l * (1 - 0.0005)) if sw_l else entry * (1 - sl_max_pct)
        else: final_sl = (sw_h * (1 + 0.0005)) if sw_h else entry * (1 + sl_max_pct)
        if abs(entry - final_sl) / entry > sl_max_pct * 1.5: # Cap swing SL
            final_sl = entry * (1 - sl_max_pct*1.5 if side == "LONG" else 1 + sl_max_pct*1.5)
            print(f"Swing SL too far, capped to: {final_sl:.4f}")
    if (side=="LONG" and final_sl >= entry) or (side=="SHORT" and final_sl <= entry): # Validate SL
        final_sl = entry * (1 - sl_max_pct if side == "LONG" else 1 + sl_max_pct)
        print(f"Invalid SL, adjusted to: {final_sl:.4f}")
    print(f"Initial SL: {final_sl:.4f}, TP: {tp:.4f} for {side} from {entry:.4f}")
    return final_sl, tp

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

def calculate_position_size(balance, risk_pct, entry, sl, symbol_info):
    if not symbol_info or balance <= 0 or entry <= 0 or sl <= 0 or abs(entry-sl)<1e-9 : return None
    q_prec = int(symbol_info['quantityPrecision'])
    lot_f = next((f for f in symbol_info['filters'] if f['filterType']=='LOT_SIZE'),None)
    if not lot_f or float(lot_f['stepSize'])==0: print(f"No LOT_SIZE/stepSize for {symbol_info['symbol']}"); return None
    min_qty, step = float(lot_f['minQty']), float(lot_f['stepSize'])
    
    pos_size = (balance * risk_pct) / abs(entry - sl)
    adj_size = math.floor(pos_size / step) * step
    adj_size = round(adj_size, q_prec)

    if adj_size < min_qty: print(f"Risk calc size {adj_size} < min_qty {min_qty}. No trade."); return None
    
    min_not_f = next((f for f in symbol_info['filters'] if f['filterType']=='MIN_NOTIONAL'),None)
    if min_not_f and (adj_size * entry) < float(min_not_f['notional']):
        print(f"Notional for {adj_size} too low. Trying min notional qty.")
        qty_min_not = math.ceil((float(min_not_f['notional']) / entry) / step) * step
        qty_min_not = round(max(qty_min_not, min_qty), q_prec) # Ensure meets min_qty too
        if (qty_min_not * abs(entry-sl) / balance) > (risk_pct * 1.5): # Risk check
            print(f"Risk for min_notional_qty too high. No trade."); return None
        adj_size = qty_min_not
        print(f"Adjusted size to {adj_size} for min_notional. New risk: {(adj_size*abs(entry-sl)/balance)*100:.2f}%")
    
    if adj_size <=0: print(f"Final size {adj_size} is zero. No trade."); return None
    print(f"Calc Pos Size: {adj_size} for {symbol_info['symbol']} (Risk: ${ (adj_size*abs(entry-sl)):.2f})")
    return adj_size

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
        klines_df = get_historical_klines(client, symbol) # Uses default limit 500
        if klines_df.empty or len(klines_df) < 202:
            return f"{symbol}: Skipped - Insufficient klines ({len(klines_df)})"
        manage_trade_entry(client, configs, symbol, klines_df.copy(), lock)
        return f"{symbol}: Processed"
    except Exception as e:
        print(f"[{thread_name}] ERROR processing {symbol}: {e} {format_elapsed_time(cycle_start_ref)}")
        traceback.print_exc() # Print full traceback for thread errors
        return f"{symbol}: Error - {e}"

def manage_trade_entry(client, configs, symbol, klines_df, lock):
    global active_trades
    was_existing_trade = False # Flag to see if we are replacing a trade
    with lock:
        if symbol in active_trades and active_trades[symbol]:
            existing_details = active_trades[symbol]
            open_ts = existing_details.get('open_timestamp')
            if open_ts and isinstance(open_ts, pd.Timestamp):
                if (pd.Timestamp.now(tz='UTC') - open_ts).total_seconds() <= 7200: # 2 hours
                    print(f"New signal for {symbol} IGNORED. Existing trade too recent ({open_ts}).")
                    return
                print(f"Existing trade for {symbol} >2hrs old. Processing new signal, cancelling old SL/TP.")
                was_existing_trade = True
                for oid_key in ['sl_order_id', 'tp_order_id']:
                    oid = existing_details.get(oid_key)
                    if oid:
                        try: client.futures_cancel_order(symbol=symbol, orderId=oid)
                        except BinanceAPIException as e:
                            if e.code != -2011: print(f"ERROR cancelling old {oid_key} {oid} for {symbol}: {e}"); return
                            else: print(f"Old {oid_key} {oid} for {symbol} already gone.")
                        except Exception as e: print(f"ERROR cancelling old {oid_key} {oid} for {symbol}: {e}"); return
            else: print(f"Warning: {symbol} in active_trades but no valid open_timestamp.")
        
        # Max concurrent position check - only if it's a truly new symbol slot
        if not was_existing_trade and len(active_trades) >= configs["max_concurrent_positions"]:
            print(f"Max concurrent positions ({configs['max_concurrent_positions']}) reached. Cannot open for new symbol {symbol}.")
            return

    # --- Calculations and non-critical API calls (outside initial lock) ---
    klines_df['EMA100'] = calculate_ema(klines_df, 100)
    klines_df['EMA200'] = calculate_ema(klines_df, 200)
    if klines_df['EMA100'] is None or klines_df['EMA200'] is None or \
       klines_df['EMA100'].isnull().all() or klines_df['EMA200'].isnull().all() or \
       len(klines_df) < 202: # Check for valid EMAs and sufficient length
        print(f"EMA calculation failed or insufficient data for {symbol}."); return

    signal = check_ema_crossover_conditions(klines_df)
    if not signal: return

    print(f"\n--- New Trade Signal for {symbol}: {signal} ---")
    symbol_info = get_symbol_info(client, symbol)
    if not symbol_info: print(f"No symbol info for {symbol}. Abort."); return

    entry_p = klines_df['close'].iloc[-1]
    ema100_val = klines_df['EMA100'].iloc[-1]
    
    if not (set_leverage_on_symbol(client, symbol, configs['leverage']) and \
            set_margin_type_on_symbol(client, symbol, configs['margin_type'])):
        print(f"Failed to set leverage/margin for {symbol}. Abort."); return

    sl_p, tp_p = calculate_sl_tp_values(entry_p, signal, ema100_val, klines_df)
    if sl_p is None or tp_p is None: print(f"SL/TP calc failed for {symbol}. Abort."); return
    
    acc_bal = get_account_balance(client)
    if acc_bal <= 0: print("Zero/unavailable balance. Abort."); return

    qty = calculate_position_size(acc_bal, configs['risk_percent'], entry_p, sl_p, symbol_info)
    if qty is None or qty <= 0: print(f"Invalid position size for {symbol}. Abort."); return
    if round(qty, int(symbol_info['quantityPrecision'])) == 0.0: print(f"Qty for {symbol} rounds to 0. Abort."); return

    print(f"Attempting {signal} {qty} {symbol} @MKT (EP:{entry_p:.4f}), SL:{sl_p:.4f}, TP:{tp_p:.4f}")
    entry_order = place_new_order(client, symbol_info, "BUY" if signal=="LONG" else "SELL", "MARKET", qty)

    if entry_order and entry_order.get('status') == 'FILLED':
        actual_ep = float(entry_order.get('avgPrice', entry_p))
        if entry_p > 0 : # Recalculate SL/TP based on actual fill if original signal price was valid
            sl_dist_pct = abs(entry_p - sl_p) / entry_p
            tp_dist_pct = abs(entry_p - tp_p) / entry_p
            sl_p = actual_ep * (1 - sl_dist_pct if signal == "LONG" else 1 + sl_dist_pct)
            tp_p = actual_ep * (1 + tp_dist_pct if signal == "LONG" else 1 - tp_dist_pct)
            print(f"SL/TP adjusted for actual fill {actual_ep:.4f}: SL {sl_p:.4f}, TP {tp_p:.4f}")

        sl_ord = place_new_order(client, symbol_info, "SELL" if signal=="LONG" else "BUY", "STOP_MARKET", qty, stop_price=sl_p, reduce_only=True)
        if not sl_ord: print(f"CRITICAL: FAILED TO PLACE SL FOR {symbol}!"); # Consider emergency close
        tp_ord = place_new_order(client, symbol_info, "SELL" if signal=="LONG" else "BUY", "TAKE_PROFIT_MARKET", qty, stop_price=tp_p, reduce_only=True)
        if not tp_ord: print(f"Warning: Failed to place TP for {symbol}.")

        with lock: # Final update to shared active_trades
            active_trades[symbol] = {
                "entry_order_id": entry_order['orderId'], "sl_order_id": sl_ord.get('orderId') if sl_ord else None,
                "tp_order_id": tp_ord.get('orderId') if tp_ord else None, "entry_price": actual_ep,
                "current_sl_price": sl_p, "current_tp_price": tp_p, "initial_sl_price": sl_p, "initial_tp_price": tp_p,
                "quantity": qty, "side": signal, "symbol_info": symbol_info, "open_timestamp": pd.Timestamp.now(tz='UTC')
            }
            print(f"Trade for {symbol} recorded at {active_trades[symbol]['open_timestamp']}.")
        get_open_positions(client); get_open_orders(client, symbol)
    else: print(f"Market order for {symbol} failed or not filled: {entry_order}")

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
            print(f"Position for {symbol} closed/zero. Removing from active list.")
            symbols_to_remove.append(symbol)
            # Cancel any residual SL/TP (already done if bot closed it, but good for external close)
            for oid_key in ['sl_order_id', 'tp_order_id']:
                oid = trade_details.get(oid_key)
                if oid:
                    try: client.futures_cancel_order(symbol=symbol, orderId=oid)
                    except Exception: pass # Ignore errors, order might be gone
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
        side = trade_details['side']
        updated_orders = False

        if new_sl is not None and abs(new_sl - trade_details['current_sl_price']) > 1e-9:
            print(f"Adjusting SL for {symbol} to {new_sl:.4f}")
            if trade_details.get('sl_order_id'): # Cancel old
                try: client.futures_cancel_order(symbol=symbol, orderId=trade_details['sl_order_id'])
                except Exception as e: print(f"Warn: Old SL {trade_details['sl_order_id']} for {symbol} cancel fail: {e}")
            # Place new
            sl_ord_new = place_new_order(client, s_info, "SELL" if side=="LONG" else "BUY", "STOP_MARKET", qty, stop_price=new_sl, reduce_only=True)
            if sl_ord_new: 
                with active_trades_lock: # Lock for update
                    if symbol in active_trades: # Check if still there (might have been removed if pos closed rapidly)
                         active_trades[symbol]['current_sl_price'] = new_sl
                         active_trades[symbol]['sl_order_id'] = sl_ord_new.get('orderId')
                         updated_orders = True
            else: print(f"CRITICAL: FAILED TO PLACE NEW SL FOR {symbol}!")
        
        if new_tp is not None and abs(new_tp - trade_details['current_tp_price']) > 1e-9:
            print(f"Adjusting TP for {symbol} to {new_tp:.4f}")
            if trade_details.get('tp_order_id'): # Cancel old
                try: client.futures_cancel_order(symbol=symbol, orderId=trade_details['tp_order_id'])
                except Exception as e: print(f"Warn: Old TP {trade_details['tp_order_id']} for {symbol} cancel fail: {e}")
            # Place new
            tp_ord_new = place_new_order(client, s_info, "SELL" if side=="LONG" else "BUY", "TAKE_PROFIT_MARKET", qty, stop_price=new_tp, reduce_only=True)
            if tp_ord_new: 
                with active_trades_lock: # Lock for update
                     if symbol in active_trades:
                        active_trades[symbol]['current_tp_price'] = new_tp
                        active_trades[symbol]['tp_order_id'] = tp_ord_new.get('orderId')
                        updated_orders = True
            else: print(f"Warning: Failed to place new TP for {symbol}.")
        
        if updated_orders: get_open_orders(client, symbol) # Show updated orders

    if symbols_to_remove:
        with active_trades_lock: # Lock for deleting from shared dict
            for sym in symbols_to_remove:
                if sym in active_trades: del active_trades[sym]; print(f"Removed {sym} from bot's active trades.")

def trading_loop(client, configs, monitored_symbols):
    print("\n--- Starting Trading Loop ---")
    if not monitored_symbols: print("No symbols to monitor. Exiting."); return
    print(f"Monitoring {len(monitored_symbols)} symbols. Examples: {monitored_symbols[:5]}")

    current_cycle_number = 0
    executor = ThreadPoolExecutor(max_workers=configs.get('max_scan_threads', DEFAULT_MAX_SCAN_THREADS))
    
    try:
        while True:
            current_cycle_number += 1
            cycle_start_time = time.time()
            configs['cycle_start_time_ref'] = cycle_start_time # For threads to use consistent base
            iter_ts = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')
            print(f"\n--- Starting Scan Cycle #{current_cycle_number}: {iter_ts} {format_elapsed_time(cycle_start_time)} ---")

            try:
                print(f"Fetching account status... {format_elapsed_time(cycle_start_time)}")
                get_account_balance(client)
                get_open_positions(client)
                print(f"Account status updated. {format_elapsed_time(cycle_start_time)}")

                futures = []
                print(f"Submitting {len(monitored_symbols)} symbol tasks to {configs.get('max_scan_threads')} threads... {format_elapsed_time(cycle_start_time)}")
                for symbol in monitored_symbols:
                    futures.append(executor.submit(process_symbol_task, symbol, client, configs, active_trades_lock))
                
                processed_count = 0
                for future in as_completed(futures):
                    try: result = future.result(); # print(f"Task result: {result}") # Optional: log task result
                    except Exception as e_future: print(f"Task error: {e_future}")
                    processed_count += 1
                    if processed_count % (len(monitored_symbols)//5 or 1) == 0 or processed_count == len(monitored_symbols): # Log progress periodically
                         print(f"Symbol tasks progress: {processed_count}/{len(monitored_symbols)} completed. {format_elapsed_time(cycle_start_time)}")
                print(f"All symbol tasks completed for cycle. {format_elapsed_time(cycle_start_time)}")

                monitor_active_trades(client, configs) # Monitor after scan
                print(f"Active trades monitoring complete. {format_elapsed_time(cycle_start_time)}")

            except Exception as loop_err: # Catch errors within the try block of the loop
                print(f"ERROR in trading loop cycle: {loop_err} {format_elapsed_time(cycle_start_time)}")
                traceback.print_exc()

            cycle_dur_s = time.time() - cycle_start_time
            print(f"\n--- Scan Cycle #{current_cycle_number} Completed (Runtime: {cycle_dur_s:.2f}s / {(cycle_dur_s/60):.2f}min). Waiting for {configs['loop_delay_minutes']}m... ---")
            time.sleep(configs['loop_delay_minutes'] * 60)
    finally:
        print("Shutting down thread pool executor...")
        executor.shutdown(wait=True) # Ensure all threads finish before exiting loop/program
        print("Thread pool executor shut down.")


# --- Main Execution ---
def main():
    print("Initializing Binance Trading Bot - Advance EMA Cross Strategy (ID: 8)")
    configs = get_user_configurations()
    print("\nLoaded Configurations:")
    for k, v in configs.items(): 
        if k not in ["api_key", "api_secret"]: print(f"  {k.replace('_',' ').title()}: {v}")

    client = initialize_binance_client(configs)
    if not client: print("Exiting: Binance client init failed."); sys.exit(1)

    configs.setdefault("api_delay_short", 1) 
    configs.setdefault("api_delay_symbol_processing", 0.1) # Can be very short with threads
    configs.setdefault("loop_delay_minutes", 5)

    monitored_symbols = get_all_usdt_perpetual_symbols(client)
    if not monitored_symbols: print("Exiting: No symbols to monitor."); sys.exit(1)
    
    confirm = input(f"Found {len(monitored_symbols)} USDT perpetuals. Monitor all for {'live trading' if configs['mode'] == 'live' else 'backtesting'}? (yes/no) [yes]: ").lower().strip()
    if confirm == 'no': print("Exiting by user choice."); sys.exit(0)

    if configs["mode"] == "live":
        try:
            trading_loop(client, configs, monitored_symbols)
        except KeyboardInterrupt: print("\nBot stopped by user (Ctrl+C).")
        except Exception as e: print(f"\nCRITICAL UNEXPECTED ERROR IN LIVE TRADING: {e}"); traceback.print_exc()
        finally:
            print("\n--- Live Trading Bot Shutting Down ---")
            if client and active_trades:
                print(f"Cancelling {len(active_trades)} bot-managed active SL/TP orders...")
                with active_trades_lock:
                    for symbol, trade_details in list(active_trades.items()):
                        for oid_key in ['sl_order_id', 'tp_order_id']:
                            oid = trade_details.get(oid_key)
                            if oid:
                                try:
                                    print(f"Cancelling {oid_key} {oid} for {symbol}...")
                                    client.futures_cancel_order(symbol=symbol, orderId=oid)
                                except Exception as e_c: print(f"Failed to cancel {oid_key} {oid} for {symbol}: {e_c}")
            print("Live Bot shutdown sequence complete.")
    elif configs["mode"] == "backtest":
        try:
            backtesting_loop(client, configs, monitored_symbols)
        except KeyboardInterrupt: print("\nBacktest stopped by user (Ctrl+C).")
        except Exception as e: print(f"\nCRITICAL UNEXPECTED ERROR IN BACKTESTING: {e}"); traceback.print_exc()
        finally:
            print("\n--- Backtesting Complete ---")
            # No orders to cancel in backtest mode unless simulating exchange interactions
            # For now, active_trades will just be cleared or analyzed.
        
            active_trades.clear() # Clear trades for a clean slate if re-run or for reporting
            print("Backtest shutdown sequence complete.")


# --- Backtesting Specific Functions ---

# Global state for backtesting simulation
backtest_current_time = None
backtest_simulated_balance = None # Will be initialized from actual balance or a preset
backtest_simulated_orders = {} # symbol -> list of order dicts
backtest_simulated_positions = {} # symbol -> position dict
backtest_trade_log = [] # Log of all simulated trade actions

def initialize_backtest_environment(client, configs):
    global backtest_simulated_balance, active_trades, backtest_trade_log, backtest_simulated_orders, backtest_simulated_positions
    active_trades.clear()
    backtest_trade_log = []
    backtest_simulated_orders = {}
    backtest_simulated_positions = {}
    
    start_balance_type = configs.get("backtest_start_balance_type", "current") # Default to current if not set
    
    if start_balance_type == "custom":
        backtest_simulated_balance = configs.get("backtest_custom_start_balance", 10000) # Default custom to 10k if somehow not set
        print(f"Backtest initialized with CUSTOM starting balance: {backtest_simulated_balance:.2f} USDT")
    else: # 'current' or default
        live_balance = get_account_balance(client) # This hits the API
        backtest_simulated_balance = live_balance if live_balance > 0 else 10000 # Default to 10k if no live balance
        print(f"Backtest initialized with CURRENT account balance: {backtest_simulated_balance:.2f} USDT (or default if zero)")

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
                
                backtest_simulated_balance += pnl # Add PnL to balance
                current_pos['positionAmt'] += actual_reduce_qty if side == "BUY" else -actual_reduce_qty # Reduce position
                
                sim_order.update({"status": "FILLED", "executedQty": actual_reduce_qty, "avgPrice": fill_price})
                backtest_trade_log.append({
                    "time": backtest_current_time, "symbol": symbol, "type": "MARKET_CLOSE", "side": side,
                    "qty": actual_reduce_qty, "price": fill_price, "pnl": pnl, "balance": backtest_simulated_balance,
                    "order_id": order_id
                })
                print(f"[SIM-{backtest_current_time}] {side} {actual_reduce_qty} {symbol} CLOSED @ {fill_price:.4f}. PnL: {pnl:.2f}. New Bal: {backtest_simulated_balance:.2f}")

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
                
                global backtest_simulated_balance
                backtest_simulated_balance += pnl
                current_pos['positionAmt'] += actual_filled_qty if order['side'] == "BUY" else -actual_filled_qty
                
                order.update({"status": "FILLED", "executedQty": actual_filled_qty, "avgPrice": trigger_price})
                backtest_trade_log.append({
                    "time": backtest_current_time, "symbol": symbol, "type": f"{order['type']}_FILL", "side": order['side'],
                    "qty": actual_filled_qty, "price": trigger_price, "pnl": pnl, "balance": backtest_simulated_balance,
                    "triggered_order_id": order['orderId']
                })
                print(f"[SIM-{backtest_current_time}] {order['side']} {actual_filled_qty} {symbol} (from {order['type']}) FILLED @ {trigger_price:.4f}. PnL: {pnl:.2f}. New Bal: {backtest_simulated_balance:.2f}")

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


def manage_trade_entry_backtest(client_dummy, configs, symbol, klines_df_current_slice, symbol_info_map):
    # This function is a wrapper around the core strategy logic for backtesting.
    # It uses simulated order placement and state management.
    global active_trades, active_trades_lock, backtest_current_time, backtest_simulated_positions

    # The lock might not be strictly necessary if backtesting is single-threaded for decision making per symbol
    # but good to keep if any part of the original manage_trade_entry expects it.
    # For pure backtesting, we can simplify and remove the lock if threading is not used for symbol processing.
    
    # --- Simulate `active_trades` logic for backtesting ---
    # In backtesting, `active_trades` would store the state of trades as if they were live.
    # Max concurrent positions check
    # Simplified: count entries in `backtest_simulated_positions` that are not fully closed.
    # A more robust check would be against `active_trades` which should mirror `backtest_simulated_positions` states.

    current_candle_close = klines_df_current_slice['close'].iloc[-1]
    current_candle_high = klines_df_current_slice['high'].iloc[-1]
    current_candle_low = klines_df_current_slice['low'].iloc[-1]

    # Check if we are already in a trade for this symbol (based on backtest_simulated_positions)
    if symbol in backtest_simulated_positions and backtest_simulated_positions[symbol]['positionAmt'] != 0:
        # print(f"[SIM-{backtest_current_time}] Already in position for {symbol}. Skipping new entry signal check.")
        return # Already in a position, monitoring will handle SL/TP.

    # Max concurrent positions check based on `active_trades` which should be kept in sync
    # with `backtest_simulated_positions`
    if len(active_trades) >= configs["max_concurrent_positions"] and symbol not in active_trades:
         print(f"[SIM-{backtest_current_time}] Max concurrent positions ({configs['max_concurrent_positions']}) reached. Cannot open for new symbol {symbol}.")
         return

    # --- Calculations (same as live) ---
    klines_df_current_slice['EMA100'] = calculate_ema(klines_df_current_slice, 100)
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
    qty = calculate_position_size(current_sim_balance, configs['risk_percent'], entry_p_signal, sl_p, symbol_info)
    
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

        if i % 100 == 0 : # Log progress
            print(f"\n[SIM] Processing Candle {i - simulation_start_index + 1} / {len(master_klines_df) - simulation_start_index} | Time: {backtest_current_time} | Balance: {backtest_simulated_balance:.2f} USDT")

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
            manage_trade_entry_backtest(client, configs, symbol, klines_slice_for_symbol, symbol_info_map)

        # Step D: Perform dynamic SL/TP adjustments (after all entries and SL/TP hits for the current candle)
        monitor_active_trades_backtest(configs)


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
