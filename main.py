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

def get_historical_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=500):
    print(f"Fetching klines for {symbol}, interval {interval}, limit {limit}...")
    start_time = time.time()
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        duration = time.time() - start_time
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
       or df[[short_ema_col, long_ema_col]].iloc[-(validation_candles + 2):].isnull().values.any(): return None
    prev_short, curr_short = df[short_ema_col].iloc[-2], df[short_ema_col].iloc[-1]
    prev_long, curr_long = df[long_ema_col].iloc[-2], df[long_ema_col].iloc[-1]
    val_df = df.iloc[-(validation_candles + 1) : -1]
    if len(val_df) < validation_candles: return None
    
    signal_type = None
    if prev_short <= prev_long and curr_short > curr_long: signal_type = "LONG"
    elif prev_short >= prev_long and curr_short < curr_long: signal_type = "SHORT"
    else: return None
    print(f"Potential {signal_type} signal: {short_ema_col} {curr_short:.4f} vs {long_ema_col} {curr_long:.4f}.")

    for i in range(len(val_df)):
        c = val_df.iloc[i]; e100, e200 = val_df[short_ema_col].iloc[i], val_df[long_ema_col].iloc[i]
        if (c['low']<=e100<=c['high']) or (c['low']<=e200<=c['high']):
            print(f"Validation FAILED for {signal_type} at {val_df.index[i]}: Price touched EMAs."); return None
    print(f"{signal_type} signal VALIDATED."); return signal_type

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
    if entry == 0: return None, None
    profit_pct = (cur_price - entry) / entry if side == "LONG" else (entry - cur_price) / entry
    new_sl, new_tp, adj = cur_sl, cur_tp, False
    if profit_pct >= 0.005: # +0.5% profit -> SL to +0.2%
        target_sl = entry * (1 + 0.002 if side == "LONG" else 1 - 0.002)
        if (side=="LONG" and target_sl > new_sl) or (side=="SHORT" and target_sl < new_sl):
            new_sl, adj = target_sl, True; print(f"Dynamic SL: {side} to {new_sl:.4f} (+0.2% lock)")
    if profit_pct <= -0.005: # -0.5% loss -> TP to +0.2%
        target_tp = entry * (1 + 0.002 if side == "LONG" else 1 - 0.002)
        if (side=="LONG" and target_tp < new_tp and target_tp > entry) or \
           (side=="SHORT" and target_tp > new_tp and target_tp < entry):
            new_tp, adj = target_tp, True; print(f"Dynamic TP: {side} to {new_tp:.4f} (+0.2% target after drawdown)")
    return (new_sl, new_tp) if adj else (None, None)

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
    
    confirm = input(f"Found {len(monitored_symbols)} USDT perpetuals. Monitor all? (yes/no) [yes]: ").lower().strip()
    if confirm == 'no': print("Exiting by user choice."); sys.exit(0)

    try:
        trading_loop(client, configs, monitored_symbols)
    except KeyboardInterrupt: print("\nBot stopped by user (Ctrl+C).")
    except Exception as e: print(f"\nCRITICAL UNEXPECTED ERROR: {e}"); traceback.print_exc()
    finally:
        print("\n--- Trading Bot Shutting Down ---")
        # Cancel open SL/TP orders for trades managed by this bot
        if client and active_trades: # Check if client was initialized
            print(f"Cancelling {len(active_trades)} bot-managed active SL/TP orders...")
            with active_trades_lock: # Ensure exclusive access for final cleanup
                for symbol, trade_details in list(active_trades.items()):
                    for oid_key in ['sl_order_id', 'tp_order_id']:
                        oid = trade_details.get(oid_key)
                        if oid:
                            try:
                                print(f"Cancelling {oid_key} {oid} for {symbol}...")
                                client.futures_cancel_order(symbol=symbol, orderId=oid)
                            except Exception as e_c: print(f"Failed to cancel {oid_key} {oid} for {symbol}: {e_c}")
        print("Bot shutdown sequence complete.")

if __name__ == "__main__":
    main()
