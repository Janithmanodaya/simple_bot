# app.py
# Contains ICT Trading Strategy Logic and Telegram Messaging Functionality

import asyncio
import os
import time
import traceback
from collections import deque
from datetime import datetime as dt
from datetime import timezone

import pandas as pd
import requests
import telegram
from binance.client import Client
from binance.exceptions import BinanceAPIException
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# --- Configuration Defaults (ICT Specific) ---
DEFAULT_ICT_TIMEFRAME_PRIMARY = "1h"
DEFAULT_ICT_TIMEFRAME_ENTRY = "1m"
DEFAULT_ICT_MIN_LIQUIDITY_SWEEP_SIZE_PIPS = 10
DEFAULT_ICT_MIN_LIQUIDITY_SWEEP_SIZE_ATR = 0.5
DEFAULT_ICT_FVG_MIN_WIDTH_PIPS = 5
DEFAULT_ICT_FVG_MIN_WIDTH_PERCENT_RANGE = 0.1
DEFAULT_ICT_ORDERBLOCK_LOOKBACK = 10
DEFAULT_ICT_SESSION_FILTER = ["London", "NewYork"]
DEFAULT_ICT_ENTRY_ORDER_TYPE = "LIMIT"
DEFAULT_ICT_SL_BUFFER_ATR_MULT = 0.2
DEFAULT_ICT_TP1_QTY_PCT = 0.33
DEFAULT_ICT_TP2_QTY_PCT = 0.33
DEFAULT_ICT_TP3_QTY_PCT = 0.34
DEFAULT_ICT_BREAKEVEN_BUFFER_R = 0.1
DEFAULT_ICT_SIGNAL_COOLDOWN_SECONDS = 3600
DEFAULT_ICT_ORDER_TIMEOUT_MINUTES = 10
DEFAULT_PRIMARY_TF_BUFFER_SIZE = 100

# Common defaults that might be shared or needed by ICT/Telegram
DEFAULT_ATR_PERIOD = 14 # Used in ICT sweep detection and SL/TP calc

# --- Global State Variables (ICT Specific) ---
symbol_primary_tf_candle_buffers = {}
symbol_primary_tf_candle_buffers_lock = asyncio.Lock() # Use asyncio.Lock for async context

ict_strategy_states = {}
ict_strategy_states_lock = asyncio.Lock() # Use asyncio.Lock

# Globals that might be shared or referenced
# These might need to be passed as arguments or handled via a shared context if app.py is run independently
active_trades = {} # Placeholder if needed by any utility, ideally ICT logic uses its own state
active_trades_lock = asyncio.Lock()
last_signal_time = {} # For cooldowns
last_signal_lock = asyncio.Lock()
recent_trade_signatures = {}
recent_trade_signatures_lock = asyncio.Lock()
trading_halted_drawdown = False # Placeholder
trading_halted_daily_loss = False # Placeholder
trading_halted_manual = False # Placeholder
daily_state_lock = asyncio.Lock() # Placeholder for daily state variables if any are used by ICT directly

# Placeholder for client and configs if this module is run standalone or needs them directly
# In a real application, these would be passed or managed by a central component.
binance_client_instance: Client | None = None
app_configurations: dict = {}


# --- Utility Functions (Potentially Shared or Core) ---

def get_public_ip():
    """Fetches the current public IP address of the machine."""
    try:
        response = requests.get("https://api.ipify.org?format=json", timeout=10)
        response.raise_for_status()
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

def escape_markdown_v1(text: str) -> str:
    """Escapes characters for Telegram Markdown V1."""
    if not isinstance(text, str):
        return ""
    text = text.replace('_', r'\_')
    text = text.replace('*', r'\*')
    text = text.replace('`', r'\`')
    text = text.replace('[', r'\[')
    return text

# --- Telegram Messaging Functions ---

async def send_telegram_message(bot_token, chat_id, message):
    """Sends a message to a specified Telegram chat using async."""
    if not bot_token or not chat_id:
        print(f"TELEGRAM_MSG_SKIPPED: Token/ChatID missing. Message: '{message[:100]}...'")
        return False
    try:
        bot = telegram.Bot(token=bot_token)
        await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
        print(f"Telegram message sent successfully to chat ID {chat_id}.")
        return True
    except telegram.error.TelegramError as e:
        print(f"Error sending Telegram message: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while sending Telegram message: {e}")
        return False

async def build_startup_message(configs, balance, open_positions_text, bot_start_time_str):
    # This function is kept for direct use if app.py sends startup messages
    # Or it can be called by the main application.
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

async def send_entry_signal_telegram(configs: dict, symbol: str, signal_type_display: str, leverage: int, entry_price: float,
                               tp1_price: float, tp2_price: float | None, tp3_price: float | None, sl_price: float,
                               risk_percentage_config: float, est_pnl_tp1: float | None, est_pnl_sl: float | None,
                               symbol_info: dict, strategy_name_display: str = "Signal",
                               signal_timestamp: dt = None, signal_order_type: str = "N/A"):
    if signal_timestamp is None:
        signal_timestamp = dt.now(tz=timezone.utc)

    if not configs.get("telegram_bot_token") or not configs.get("telegram_chat_id"):
        print(f"Telegram not configured. Cannot send signal notification for {symbol}.")
        return

    p_prec = int(symbol_info.get('pricePrecision', 2))
    tp1_str = f"{tp1_price:.{p_prec}f}" if tp1_price is not None else "N/A"
    tp2_str = f"{tp2_price:.{p_prec}f}" if tp2_price is not None else "N/A"
    tp3_str = f"{tp3_price:.{p_prec}f}" if tp3_price is not None else "N/A"
    sl_str = f"{sl_price:.{p_prec}f}" if sl_price is not None else "N/A"
    pnl_tp1_str = f"{est_pnl_tp1:.2f} USDT" if est_pnl_tp1 is not None else "Not Calculated"
    pnl_sl_str = f"{est_pnl_sl:.2f} USDT" if est_pnl_sl is not None else "Not Calculated"
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
    if tp1_price is not None: tps_message_part += f"🎯 Take Profit 1: `{tp1_str}`\n"
    if tp2_price is not None: tps_message_part += f"🎯 Take Profit 2: `{tp2_str}`\n"
    if tp3_price is not None: tps_message_part += f"🎯 Take Profit 3: `{tp3_str}`\n"
    if not tps_message_part and tp1_price is None: tps_message_part = "🎯 Take Profit Levels: `N/A`\n"
    message += tps_message_part
    message += (
        f"\n📊 Configured Risk: `{risk_percentage_config * 100:.2f}%`\n\n"
        f"💰 *Est. P&L ($100 Capital Trade):*\n"
        f"  - TP1 Hit: `{pnl_tp1_str}`\n"
        f"  - SL Hit: `{pnl_sl_str}`\n\n"
        f"⚠️ _This is a signal only. No order has been placed._"
    )
    print(f"Sending TRADE SIGNAL notification for {symbol} ({signal_type_display}).")
    await send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], message)


async def send_signal_update_telegram(configs: dict, signal_details: dict, update_type: str, message_detail: str,
                                current_market_price: float, pnl_estimation_fixed_capital: float | None = None):
    if not configs.get("telegram_bot_token") or not configs.get("telegram_chat_id"):
        print(f"Telegram not configured. Cannot send signal update for {signal_details.get('symbol')}.")
        return

    symbol = signal_details.get('symbol', 'N/A')
    side = signal_details.get('side', 'N/A')
    entry_price = signal_details.get('entry_price', 0.0)
    s_info = signal_details.get('symbol_info', {})
    p_prec = int(s_info.get('pricePrecision', 2))
    title_emoji = "⚙️"
    if update_type.startswith("TP"): title_emoji = "✅"
    elif update_type == "SL_HIT": title_emoji = "❌"
    elif update_type == "SL_ADJUSTED": title_emoji = "🛡️"
    elif update_type == "CLOSED_ALL_TPS": title_emoji = "🎉"
    pnl_info_str = f"\nEst. P&L ($100 Capital): `{pnl_estimation_fixed_capital:.2f} USDT`" if pnl_estimation_fixed_capital is not None else ""
    message = (
        f"{title_emoji} *SIGNAL UPDATE* ({signal_details.get('strategy_type', 'Signal')}) {title_emoji}\n\n"
        f"Symbol: `{symbol}` ({side})\n"
        f"Entry: `{entry_price:.{p_prec}f}`\n"
        f"Update Type: `{update_type}`\n"
        f"Details: _{message_detail}_\n"
        f"Current Market Price: `{current_market_price:.{p_prec}f}`"
        f"{pnl_info_str}"
    )
    print(f"Sending SIGNAL UPDATE notification for {symbol}: {update_type} - {message_detail}")
    await send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], message)

async def send_trade_rejection_notification(symbol, signal_type, reason, entry_price, sl_price, tp_price, quantity, symbol_info, configs):
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
    await send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], message)

async def send_ict_telegram_alert(configs: dict, message_type: str, symbol: str, details: dict = None, symbol_info: dict = None):
    if not configs.get("telegram_bot_token") or not configs.get("telegram_chat_id"):
        return

    if details is None: details = {}
    p_prec = 2
    if symbol_info and isinstance(symbol_info.get('pricePrecision'), (str, int)):
        try: p_prec = int(symbol_info['pricePrecision'])
        except ValueError: pass

    message = f"🔔 *[ICT] {symbol.upper()}*"
    if message_type == "SWEEP":
        sweep_type_display = details.get('type', 'Unknown Sweep').replace('_', ' ').title()
        price_swept_str = f"{details.get('price_swept', 0):.{p_prec}f}"
        wick_str = f"{details.get('sweep_wick', 0):.{p_prec}f}"
        close_str = f"{details.get('closing_price', 0):.{p_prec}f}"
        message += f": {sweep_type_display} @ {price_swept_str} (Wick: {wick_str}, Close: {close_str})"
    elif message_type == "FVG_CREATED":
        fvg_type_display = details.get('type', 'Unknown FVG').replace('_', ' ').title()
        lower_band_str = f"{details.get('lower_band', 0):.{p_prec}f}"
        upper_band_str = f"{details.get('upper_band', 0):.{p_prec}f}"
        message += f": {fvg_type_display} {lower_band_str} - {upper_band_str}"
        if details.get("timestamp_created"):
             message += f" (created @ {details['timestamp_created'].strftime('%H:%M:%S')})"
    elif message_type == "ZONE_VALIDATED":
        zone_type_display = details.get('type', 'Zone').replace('_zone', '').title() + " Zone"
        fvg_l = f"{details.get('fvg_lower_orig', 0):.{p_prec}f}"
        fvg_u = f"{details.get('fvg_upper_orig', 0):.{p_prec}f}"
        ob_l = f"{details.get('ob_low', 0):.{p_prec}f}"
        ob_h = f"{details.get('ob_high', 0):.{p_prec}f}"
        message += f": {zone_type_display} Validated!\n  FVG: `{fvg_l} - {fvg_u}`\n  OB: `{ob_l} - {ob_h}`"
    elif message_type == "ENTRY":
        side = details.get('side', 'N/A').upper()
        entry_p_str = f"{details.get('entry_price', 0):.{p_prec}f}"
        sl_p_str = f"{details.get('sl_price', 0):.{p_prec}f}"
        message += f": {side} Entry @ {entry_p_str}\n  SL: `{sl_p_str}`"
        tp_lines = []
        for i in range(1, 4):
            tp_price = details.get(f'tp{i}_price')
            tp_qty_pct = details.get(f'tp{i}_qty_pct', 0) * 100
            if tp_price is not None and tp_qty_pct > 0:
                tp_lines.append(f"  TP{i}: `{tp_price:.{p_prec}f}` ({tp_qty_pct:.0f}%)")
        if tp_lines: message += "\n" + "\n".join(tp_lines)
        else: message += "\n  TPs: Not Set or All Zero Qty"
        if details.get("order_id"): message += f"\n  Entry Order ID: `{details['order_id']}`"
    elif message_type == "EXIT":
        exit_reason = details.get('reason', 'Unknown Exit')
        exit_p_str = f"{details.get('exit_price', 0):.{p_prec}f}"
        pnl_str = f"{details.get('pnl', 0):.2f} USDT"
        qty_str = f"{details.get('quantity', 0):.{symbol_info.get('quantityPrecision', 2) if symbol_info else 2}f}"
        emoji = "✅" if "TP" in exit_reason.upper() else "❌" if "SL" in exit_reason.upper() else "ℹ️"
        message = f"{emoji} *[ICT] {symbol.upper()}*: Position Closed ({details.get('side', 'N/A')})\n" \
                  f"  Reason: `{exit_reason}` @ `{exit_p_str}`\n" \
                  f"  Qty: `{qty_str}` | PNL: `{pnl_str}`"
        if details.get("order_id"): message += f"\n  Order ID: `{details['order_id']}`"
    else:
        message += f": Unknown Event Type '{message_type}'"
        if details: message += f"\n  Details: `{str(details)[:200]}`"
    await send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], message)

# --- Telegram Command Handlers (Minimal stubs, actual implementation in main.py or passed context) ---
# These handlers will need access to the main application's state (client, configs, active_trades etc.)
# This can be done by passing them via context.bot_data or making them accessible globally if this module runs in the same process.

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello from app.py! ICT & Telegram module.")

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Commands available (app.py context):\n/start, /help, /status_ict (placeholder)")

async def status_ict_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Placeholder for ICT specific status
    global ict_strategy_states, ict_strategy_states_lock
    status_lines = ["*ICT Strategy Status:*"]
    async with ict_strategy_states_lock:
        if not ict_strategy_states:
            status_lines.append("No active ICT states.")
        else:
            for symbol, state in ict_strategy_states.items():
                status_lines.append(f"\nSymbol: `{symbol}`")
                if state.get('last_sweep'): status_lines.append(f"  Last Sweep: {state['last_sweep']['type']} @ {state['last_sweep']['timestamp']}")
                if state.get('active_fvgs'): status_lines.append(f"  Active FVGs: {len(state['active_fvgs'])}")
                if state.get('active_order_blocks'): status_lines.append(f"  Active OBs: {len(state['active_order_blocks'])}")
                if state.get('active_ict_trade_zones'): status_lines.append(f"  Active Trade Zones: {len(state['active_ict_trade_zones'])}")
                if state.get('pending_ict_entries'): status_lines.append(f"  Pending Entries: {len(state['pending_ict_entries'])}")
    await update.message.reply_text("\n".join(status_lines), parse_mode="Markdown")


async def start_telegram_polling(bot_token: str, app_configs_polling: dict):
    # This function is designed to be run in a separate thread if app.py is part of a larger application.
    # If app.py is run standalone, this can be called directly.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application = Application.builder().token(bot_token).concurrent_updates(True).build()
    application.bot_data['configs'] = app_configs_polling # Store configs for handlers

    # Register handlers (basic ones for now, can be expanded)
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("status_ict", status_ict_handler))
    # Add other relevant handlers from main.py if they are to be managed here.
    # For example, generic status, positions, orders, halt, resume, etc.
    # This requires careful consideration of shared state (client, active_trades, etc.)

    print("Telegram Polling (app.py context) starting...")
    await application.initialize()
    await application.run_polling() # Changed from application.run_polling() to await
    print("Telegram Polling (app.py context) exited.")


# --- ICT Strategy Data and Logic Functions ---

async def update_primary_tf_candle_buffer(symbol: str, new_candle_df_row: pd.Series, buffer_size: int):
    global symbol_primary_tf_candle_buffers, symbol_primary_tf_candle_buffers_lock
    async with symbol_primary_tf_candle_buffers_lock:
        if symbol not in symbol_primary_tf_candle_buffers:
            symbol_primary_tf_candle_buffers[symbol] = deque(maxlen=buffer_size)
        if symbol_primary_tf_candle_buffers[symbol]:
            last_buffered_candle_time = symbol_primary_tf_candle_buffers[symbol][-1].name
            if new_candle_df_row.name == last_buffered_candle_time:
                symbol_primary_tf_candle_buffers[symbol][-1] = new_candle_df_row
                return
            elif new_candle_df_row.name < last_buffered_candle_time:
                return
        symbol_primary_tf_candle_buffers[symbol].append(new_candle_df_row)

async def get_historical_klines_primary_tf(client: Client, symbol: str, interval_str: str, limit: int = 200):
    # This function needs to be async if called from async ICT logic, or run in executor
    # For now, keeping it synchronous as Binance client calls are blocking.
    # Consider using asyncio.to_thread for blocking calls in an async context.
    start_time = time.time()
    api_error = None
    interval_mapping = {
        "1m": Client.KLINE_INTERVAL_1MINUTE, "3m": Client.KLINE_INTERVAL_3MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE, "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE, "1h": Client.KLINE_INTERVAL_1HOUR,
        "2h": Client.KLINE_INTERVAL_2HOUR, "4h": Client.KLINE_INTERVAL_4HOUR,
        "6h": Client.KLINE_INTERVAL_6HOUR, "8h": Client.KLINE_INTERVAL_8HOUR,
        "12h": Client.KLINE_INTERVAL_12HOUR, "1d": Client.KLINE_INTERVAL_1DAY,
    }
    api_interval = interval_mapping.get(interval_str.lower())
    if not api_interval:
        return pd.DataFrame(), ValueError(f"Invalid interval string: {interval_str}")

    print(f"Fetching Primary TF ('{interval_str}') klines for {symbol}, limit {limit}...")
    try:
        # Blocking call, consider to_thread if in async event loop
        klines_primary = await asyncio.to_thread(client.get_klines, symbol=symbol, interval=api_interval, limit=limit)
    except BinanceAPIException as e:
        api_error = e; return pd.DataFrame(), api_error
    except Exception as e:
        api_error = e; return pd.DataFrame(), api_error
    
    duration = time.time() - start_time
    if not klines_primary: return pd.DataFrame(), api_error
    df = pd.DataFrame(klines_primary, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    print(f"Fetched {len(df)} Primary TF klines for {symbol} (runtime: {duration:.2f}s).")
    return df, None

import math # For rounding, floor, ceil operations
import numpy as np # For numerical operations

# --- ICT CSV Journaling ---
ICT_JOURNAL_FILENAME = "ict_trade_journal.csv"
ICT_JOURNAL_HEADERS = [
    "Timestamp", "Date", "Symbol", "EventType", "Direction",
    "Price", "PriceSwept", "SweepWick",
    "FVG_Upper", "FVG_Lower", "FVG_Created_TS",
    "OB_High", "OB_Low", "OB_Timestamp",
    "Zone_Upper", "Zone_Lower",
    "EntryPrice", "SL_Price",
    "TP1_Price", "TP1_Qty_Pct",
    "TP2_Price", "TP2_Qty_Pct",
    "TP3_Price", "TP3_Qty_Pct",
    "OrderID", "ExitReason", "PNL", "Notes"
]

# Constants for pivot detection (used by ICT and potentially other strategies if general enough)
PIVOT_N_LEFT = 5
PIVOT_N_RIGHT = 5

# --- Helper Functions (needed by ICT logic, adapted from main.py) ---

async def get_symbol_info(client: Client, symbol: str):
    # This would typically be cached in a real application.
    # Kept async if other parts of app.py are heavily async.
    # Binance client calls are blocking, so use to_thread.
    if client is None:
        print(f"get_symbol_info: Binance client not initialized for {symbol}.")
        return None
    try:
        exchange_info = await asyncio.to_thread(client.futures_exchange_info)
        for s_info in exchange_info['symbols']:
            if s_info['symbol'] == symbol:
                return s_info
        print(f"No symbol_info found for {symbol}.")
        return None
    except Exception as e:
        print(f"Error getting symbol_info for {symbol}: {e}")
        return None

async def get_account_balance(client: Client, configs: dict, asset="USDT"):
    if client is None:
        print(f"get_account_balance: Binance client not initialized.")
        return 0.0 # Cannot fetch if client is None
    try:
        balances = await asyncio.to_thread(client.futures_account_balance)
        for b in balances:
            if b['asset'] == asset:
                return float(b['balance'])
        return 0.0
    except BinanceAPIException as e:
        if e.code == -2015: # IP whitelist issue
            public_ip = get_public_ip()
            ip_msg = f"Bot's public IP: {public_ip}" if public_ip else "Could not get bot's public IP."
            error_message = f"⚠️ CRITICAL BINANCE API ERROR (-2015) ⚠️\nLikely IP Whitelist Issue for account balance check.\n{ip_msg}"
            if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                await send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], error_message)
            return None # Indicate critical failure
        print(f"API Error getting balance: {e}")
        return 0.0
    except Exception as e:
        print(f"Unexpected error getting balance: {e}")
        return 0.0

async def place_new_order(client: Client, symbol_info: dict, side: str, order_type: str, quantity: float,
                          price: float | None = None, stop_price: float | None = None,
                          position_side: str | None = None, is_closing_order: bool = False):
    if client is None:
        print(f"place_new_order: Binance client not initialized for {symbol_info.get('symbol')}.")
        return None, "Binance client not initialized."

    symbol, p_prec, q_prec = symbol_info['symbol'], int(symbol_info['pricePrecision']), int(symbol_info['quantityPrecision'])
    params = {"symbol": symbol, "side": side.upper(), "type": order_type.upper(), "quantity": f"{quantity:.{q_prec}f}"}

    if position_side: params["positionSide"] = position_side.upper()
    if order_type.upper() in ["LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if price is None: return None, f"Price needed for {order_type}"
        params.update({"price": f"{price:.{p_prec}f}", "timeInForce": "GTC"})
    if order_type.upper() in ["STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
        if stop_price is None: return None, f"Stop price needed for {order_type}"
        params["stopPrice"] = f"{stop_price:.{p_prec}f}"
        if is_closing_order: params["closePosition"] = "true"

    try:
        order = await asyncio.to_thread(client.futures_create_order, **params)
        print(f"Order PLACED (app.py): {order['symbol']} ID {order['orderId']} ... Status:{order['status']}")
        return order, None
    except Exception as e:
        error_msg = f"ORDER FAILED (app.py) for {symbol} {side} {quantity} {order_type}: {str(e)}"
        print(error_msg)
        return None, str(e)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if not all(col in df.columns for col in ['high', 'low', 'close']) or len(df) < period:
        return pd.Series(dtype='float64')
    df_copy = df.copy() # Avoid modifying original DataFrame
    df_copy['prev_close'] = df_copy['close'].shift(1)
    df_copy['tr0'] = abs(df_copy['high'] - df_copy['low'])
    df_copy['tr1'] = abs(df_copy['high'] - df_copy['prev_close'])
    df_copy['tr2'] = abs(df_copy['low'] - df_copy['prev_close'])
    df_copy['tr'] = df_copy[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr_series = df_copy['tr'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr_series.astype(float)

def identify_swing_pivots(data: pd.Series, n_left: int, n_right: int, is_high: bool) -> pd.Series:
    pivots = pd.Series(False, index=data.index)
    for i in range(n_left, len(data) - n_right):
        current_val = data.iloc[i]
        left_window = data.iloc[i-n_left : i]
        right_window = data.iloc[i+1 : i+1+n_right]
        if is_high:
            if (left_window < current_val).all() and (right_window <= current_val).all():
                pivots.iloc[i] = True
        else: # is_low
            if (left_window > current_val).all() and (right_window >= current_val).all():
                pivots.iloc[i] = True
    return pivots

def get_latest_pivots_from_buffer(candle_buffer_df: pd.DataFrame, n_left: int, n_right: int) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None, pd.Series | None]:
    if candle_buffer_df.empty or len(candle_buffer_df) < n_left + n_right + 1:
        return None, None, None, None
    relevant_data = candle_buffer_df.iloc[:-(n_right+1)] if n_right > 0 else candle_buffer_df
    if relevant_data.empty: return None, None, None, None

    swing_highs_bool = identify_swing_pivots(relevant_data['high'], n_left, n_right, is_high=True)
    swing_lows_bool = identify_swing_pivots(relevant_data['low'], n_left, n_right, is_high=False)
    confirmed_highs = relevant_data[swing_highs_bool]
    confirmed_lows = relevant_data[swing_lows_bool]

    latest_high_price, latest_high_time = (confirmed_highs['high'].iloc[-1], confirmed_highs.index[-1]) if not confirmed_highs.empty else (None, None)
    latest_low_price, latest_low_time = (confirmed_lows['low'].iloc[-1], confirmed_lows.index[-1]) if not confirmed_lows.empty else (None, None)
    return latest_high_time, latest_high_price, latest_low_time, latest_low_price

def calculate_position_size(balance, risk_pct, entry, sl, symbol_info, configs=None):
    if not symbol_info or balance <= 0 or entry <= 0 or sl <= 0 or abs(entry-sl)<1e-9 : return None
    q_prec = int(symbol_info['quantityPrecision'])
    lot_f = next((f for f in symbol_info['filters'] if f['filterType']=='LOT_SIZE'),None)
    if not lot_f or float(lot_f['stepSize'])==0: return None
    min_qty, step = float(lot_f['minQty']), float(lot_f['stepSize'])
    pos_size = (balance * risk_pct) / abs(entry - sl)
    adj_size = math.floor(pos_size / step) * step
    adj_size = round(adj_size, q_prec)

    if adj_size < min_qty:
        risk_for_min_qty = (min_qty * abs(entry - sl)) / balance
        allow_exceed = configs.get('allow_exceed_risk_for_min_notional', False) if configs else False
        if allow_exceed or risk_for_min_qty <= (risk_pct * 1.5) : # Simplified condition
            adj_size = min_qty
        else: return None

    min_not_f = next((f for f in symbol_info['filters'] if f['filterType']=='MIN_NOTIONAL'),None)
    if min_not_f and (adj_size * entry) < float(min_not_f['notional']):
        qty_min_not = math.ceil((float(min_not_f['notional']) / entry) / step) * step
        qty_min_not = round(max(qty_min_not, min_qty), q_prec)
        risk_for_min_notional_qty = (qty_min_not * abs(entry-sl) / balance)
        allow_exceed = configs.get('allow_exceed_risk_for_min_notional', False) if configs else False
        if allow_exceed or risk_for_min_notional_qty <= (risk_pct * 1.5):
            adj_size = qty_min_not
        else: return None
            
    if adj_size <= 0 or adj_size < min_qty: return None
    return adj_size

def pre_order_sanity_checks(symbol, signal, entry_price, sl_price, tp_price, quantity,
                            symbol_info, current_balance, risk_percent_config, configs,
                            specific_leverage_for_trade: int, klines_df_for_debug=None, is_unmanaged_check=False):
    # Simplified version for app.py, full checks in main.py if needed
    if not all(isinstance(p, (int, float)) and p > 0 for p in [entry_price, sl_price, tp_price, quantity, current_balance, specific_leverage_for_trade]):
        return False, "Invalid numeric inputs."
    if (signal == "LONG" and not (sl_price < entry_price < tp_price)) or \
       (signal == "SHORT" and not (sl_price > entry_price > tp_price)):
        return False, "SL/TP not logical for signal direction."
    # Basic risk check (more detailed in main.py version)
    if not is_unmanaged_check and current_balance > 0:
        risk_amount_abs = quantity * abs(entry_price - sl_price)
        if (risk_amount_abs / current_balance) > (risk_percent_config * 2.0): # Allow up to 2x configured for simplicity here
            return False, "Risk significantly exceeds configuration."
    return True, "Checks passed (app.py simplified)"

def generate_trade_signature(symbol: str, signal_type: str, entry_price: float, sl_price: float, tp_price: float, quantity: float, precision: int = 4) -> str:
    return f"{symbol}_{signal_type}_{entry_price:.{precision}f}_{sl_price:.{precision}f}_{tp_price:.{precision}f}_{quantity:.{precision}f}"

def calculate_pnl_for_fixed_capital(entry_price: float, exit_price: float, side: str, leverage: int, fixed_capital_usdt: float = 100.0, symbol_info: dict = None) -> float | None:
    if entry_price is None or exit_price is None or not all([isinstance(entry_price, (int,float)) and entry_price > 0, isinstance(exit_price, (int,float)) and exit_price > 0, leverage > 0, fixed_capital_usdt > 0]) or side not in ["LONG", "SHORT"]:
        return None
    if entry_price == exit_price: return 0.0
    position_value_usdt = fixed_capital_usdt * leverage
    quantity_base_asset = position_value_usdt / entry_price
    if symbol_info:
        q_prec = int(symbol_info.get('quantityPrecision', 8))
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter and float(lot_size_filter['stepSize']) > 0:
            quantity_base_asset = math.floor(quantity_base_asset / float(lot_size_filter['stepSize'])) * float(lot_size_filter['stepSize'])
        quantity_base_asset = round(quantity_base_asset, q_prec)
    if quantity_base_asset == 0: return 0.0
    return (exit_price - entry_price) * quantity_base_asset if side == "LONG" else (entry_price - exit_price) * quantity_base_asset

async def log_ict_event_to_csv(event_details_dict: dict):
    # Simplified async wrapper for CSV logging.
    # In a real app, consider a dedicated async logging library or queue.
    await asyncio.to_thread(_log_ict_event_to_csv_sync, event_details_dict)

def _log_ict_event_to_csv_sync(event_details_dict: dict):
    try:
        file_exists = os.path.exists(ICT_JOURNAL_FILENAME)
        row_data = {header: event_details_dict.get(header) for header in ICT_JOURNAL_HEADERS}
        row_data["Timestamp"] = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')
        row_data["Date"] = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')
        df_to_append = pd.DataFrame([row_data])
        if not file_exists:
            df_to_append.to_csv(ICT_JOURNAL_FILENAME, mode='a', header=True, index=False, columns=ICT_JOURNAL_HEADERS)
        else:
            df_to_append.to_csv(ICT_JOURNAL_FILENAME, mode='a', header=False, index=False, columns=ICT_JOURNAL_HEADERS)
    except Exception as e:
        print(f"Error logging ICT event to CSV (sync part): {e}")

# --- ICT Core Strategy Functions (Adapted from main.py) ---

async def detect_liquidity_sweep(symbol: str, primary_tf_df: pd.DataFrame, configs: dict, symbol_info: dict, client: Client) -> dict | None:
    global ict_strategy_states, ict_strategy_states_lock
    log_prefix = f"[{symbol} ICT_Sweep_App]"
    if primary_tf_df.empty or len(primary_tf_df) < PIVOT_N_LEFT + PIVOT_N_RIGHT + 2: return None

    current_candle = primary_tf_df.iloc[-1]
    historical_data_for_pivots = primary_tf_df.iloc[:-1]
    if len(historical_data_for_pivots) < PIVOT_N_LEFT + PIVOT_N_RIGHT + 1: return None

    prev_high_time, prev_swing_high, prev_low_time, prev_swing_low = get_latest_pivots_from_buffer(
        historical_data_for_pivots, PIVOT_N_LEFT, PIVOT_N_RIGHT
    )
    sweep_event = None
    min_sweep_pips = configs.get("ict_min_liquidity_sweep_size_pips", DEFAULT_ICT_MIN_LIQUIDITY_SWEEP_SIZE_PIPS)
    min_sweep_atr_mult = configs.get("ict_min_liquidity_sweep_size_atr", DEFAULT_ICT_MIN_LIQUIDITY_SWEEP_SIZE_ATR)
    price_precision = int(symbol_info.get('pricePrecision', 2))
    current_atr_primary_tf = 0
    if min_sweep_atr_mult > 0:
        atr_series_primary = calculate_atr(primary_tf_df.copy(), period=configs.get("atr_period", DEFAULT_ATR_PERIOD))
        if not atr_series_primary.empty and pd.notna(atr_series_primary.iloc[-1]):
            current_atr_primary_tf = atr_series_primary.iloc[-1]

    if prev_swing_high is not None and current_candle['high'] > prev_swing_high:
        sweep_size_actual = current_candle['high'] - prev_swing_high
        pip_value = 1 / (10**price_precision)
        min_sweep_val_pips = min_sweep_pips * pip_value
        min_sweep_val_atr = current_atr_primary_tf * min_sweep_atr_mult if current_atr_primary_tf > 0 else 0
        if (min_sweep_pips > 0 and sweep_size_actual >= min_sweep_val_pips) or \
           (min_sweep_atr_mult > 0 and current_atr_primary_tf > 0 and sweep_size_actual >= min_sweep_val_atr):
            if current_candle['close'] < prev_swing_high:
                sweep_event = {"type": "bullish_sweep", "timestamp": current_candle.name, "price_swept": prev_swing_high, "sweep_wick": current_candle['high'], "closing_price": current_candle['close']}

    if not sweep_event and prev_swing_low is not None and current_candle['low'] < prev_swing_low:
        sweep_size_actual = prev_swing_low - current_candle['low']
        pip_value = 1 / (10**price_precision)
        min_sweep_val_pips = min_sweep_pips * pip_value
        min_sweep_val_atr = current_atr_primary_tf * min_sweep_atr_mult if current_atr_primary_tf > 0 else 0
        if (min_sweep_pips > 0 and sweep_size_actual >= min_sweep_val_pips) or \
           (min_sweep_atr_mult > 0 and current_atr_primary_tf > 0 and sweep_size_actual >= min_sweep_val_atr):
            if current_candle['close'] > prev_swing_low:
                sweep_event = {"type": "bearish_sweep", "timestamp": current_candle.name, "price_swept": prev_swing_low, "sweep_wick": current_candle['low'], "closing_price": current_candle['close']}

    if sweep_event:
        async with ict_strategy_states_lock:
            ict_strategy_states.setdefault(symbol, {}).update({
                'last_sweep': sweep_event, 'active_fvgs': [], 'active_order_blocks': [], 'active_ict_trade_zones': []
            })
        await log_ict_event_to_csv({"Symbol": symbol, "EventType": "SWEEP_DETECTED", "Direction": sweep_event['type'], "PriceSwept": sweep_event['price_swept'], "SweepWick": sweep_event['sweep_wick'], "Price": sweep_event['closing_price']})
        await send_ict_telegram_alert(configs, "SWEEP", symbol, sweep_event, symbol_info)
    return sweep_event

async def find_fair_value_gap(symbol: str, primary_tf_df: pd.DataFrame, sweep_event: dict, configs: dict, symbol_info: dict) -> dict | None:
    global ict_strategy_states, ict_strategy_states_lock
    log_prefix = f"[{symbol} ICT_FVG_App]"
    if primary_tf_df.empty or len(primary_tf_df) < 3: return None
    try: sweep_candle_index = primary_tf_df.index.get_loc(sweep_event['timestamp'])
    except KeyError: return None

    fvg_event = None
    for k in range(sweep_candle_index + 1, len(primary_tf_df) - 2):
        c1, c2, c3 = primary_tf_df.iloc[k], primary_tf_df.iloc[k+1], primary_tf_df.iloc[k+2]
        fvg_upper, fvg_lower, fvg_type = None, None, None
        if c1['low'] > c3['high']: fvg_type, fvg_lower, fvg_upper = "bullish_fvg", c3['high'], c1['low']
        elif c1['high'] < c3['low']: fvg_type, fvg_lower, fvg_upper = "bearish_fvg", c1['high'], c3['low']

        if fvg_type:
            fvg_width = abs(fvg_upper - fvg_lower)
            min_fvg_pips = configs.get("ict_fvg_min_width_pips", DEFAULT_ICT_FVG_MIN_WIDTH_PIPS)
            min_fvg_pct_range = configs.get("ict_fvg_min_width_percent_range", DEFAULT_ICT_FVG_MIN_WIDTH_PERCENT_RANGE)
            price_precision = int(symbol_info.get('pricePrecision', 2))
            pip_value = 1 / (10**price_precision)
            min_fvg_val_pips = min_fvg_pips * pip_value
            three_candle_range = max(c1['high'],c2['high'],c3['high']) - min(c1['low'],c2['low'],c3['low'])
            min_fvg_val_pct = three_candle_range * min_fvg_pct_range if three_candle_range > 0 else 0
            if (min_fvg_pips > 0 and fvg_width >= min_fvg_val_pips) or \
               (min_fvg_pct_range > 0 and three_candle_range > 0 and fvg_width >= min_fvg_val_pct):
                fvg_event = {"type": fvg_type, "upper_band": fvg_upper, "lower_band": fvg_lower, "timestamp_created": c3.name, "triggering_sweep_timestamp": sweep_event['timestamp']}
                async with ict_strategy_states_lock:
                    ict_strategy_states.setdefault(symbol, {}).setdefault('active_fvgs', []).append(fvg_event)
                await log_ict_event_to_csv({"Symbol": symbol, "EventType": "FVG_CREATED", "Direction": fvg_event['type'], "FVG_Upper": fvg_event['upper_band'], "FVG_Lower": fvg_event['lower_band'], "FVG_Created_TS": fvg_event['timestamp_created']})
                await send_ict_telegram_alert(configs, "FVG_CREATED", symbol, fvg_event, symbol_info)
                return fvg_event
    return None

async def validate_order_block(symbol: str, primary_tf_df: pd.DataFrame, fvg_event: dict, configs: dict, symbol_info: dict) -> dict | None:
    global ict_strategy_states, ict_strategy_states_lock
    log_prefix = f"[{symbol} ICT_OB_App]"
    try: fvg_confirm_candle_idx = primary_tf_df.index.get_loc(fvg_event["timestamp_created"])
    except KeyError: return None
    
    lookback = configs.get("ict_orderblock_lookback", DEFAULT_ICT_ORDERBLOCK_LOOKBACK)
    ob_search_end_idx = fvg_confirm_candle_idx - 2
    if ob_search_end_idx < 0: return None
    ob_search_start_idx = max(0, ob_search_end_idx - lookback)
    
    candidate_ob_candle, ob_type_expected = None, None
    if fvg_event["type"] == "bearish_fvg": ob_type_expected = "bullish_ob"
    elif fvg_event["type"] == "bullish_fvg": ob_type_expected = "bearish_ob"
    else: return None

    for i in range(ob_search_end_idx, ob_search_start_idx -1, -1):
        if i < 0: break
        candle = primary_tf_df.iloc[i]
        if (ob_type_expected == "bullish_ob" and candle['close'] < candle['open']) or \
           (ob_type_expected == "bearish_ob" and candle['close'] > candle['open']):
            candidate_ob_candle = candle; break
    if candidate_ob_candle is None: return None

    ob_high, ob_low = candidate_ob_candle['high'], candidate_ob_candle['low']
    if not (max(ob_low, fvg_event['lower_band']) < min(ob_high, fvg_event['upper_band'])): return None # No overlap

    ob_event = {"type": ob_type_expected, "high": ob_high, "low": ob_low, "timestamp": candidate_ob_candle.name, "fvg_timestamp": fvg_event["timestamp_created"]}
    trade_zone_type = "long_zone" if ob_type_expected == "bullish_ob" else "short_zone"
    trade_zone = {"type": trade_zone_type, "zone_upper": fvg_event['upper_band'], "zone_lower": fvg_event['lower_band'], "ob_high": ob_high, "ob_low": ob_low, "fvg_upper_orig": fvg_event['upper_band'], "fvg_lower_orig": fvg_event['lower_band'], "timestamp_created": candidate_ob_candle.name}
    
    async with ict_strategy_states_lock:
        state_sym = ict_strategy_states.setdefault(symbol, {})
        state_sym.setdefault('active_order_blocks', []).append(ob_event)
        state_sym.setdefault('active_ict_trade_zones', []).append(trade_zone)
    
    await log_ict_event_to_csv({"Symbol": symbol, "EventType": "ZONE_VALIDATED", "Direction": trade_zone_type, "FVG_Upper": fvg_event['upper_band'], "FVG_Lower": fvg_event['lower_band'], "OB_High": ob_event['high'], "OB_Low": ob_event['low'], "Zone_Upper": trade_zone['zone_upper'], "Zone_Lower": trade_zone['zone_lower']})
    await send_ict_telegram_alert(configs, "ZONE_VALIDATED", symbol, trade_zone, symbol_info)
    return ob_event

async def calculate_ict_sl_tp(entry_price: float, side: str, ict_zone: dict, atr_1m_value: float | None, configs: dict, symbol_info: dict) -> tuple[float | None, list[dict]]:
    p_prec = int(symbol_info.get('pricePrecision', 2))
    sl_buffer_atr_mult = configs.get("ict_sl_buffer_atr_mult", DEFAULT_ICT_SL_BUFFER_ATR_MULT)
    sl_price = None
    if side == "LONG":
        protective_low = min(ict_zone.get("zone_lower", float('inf')), ict_zone.get("ob_low", float('inf')))
        if protective_low == float('inf'): return None, []
        sl_price_candidate = protective_low - (atr_1m_value * sl_buffer_atr_mult if atr_1m_value and atr_1m_value > 0 else 0)
        sl_price = round(sl_price_candidate, p_prec)
        if sl_price >= entry_price: return None, []
    elif side == "SHORT":
        protective_high = max(ict_zone.get("zone_upper", float('-inf')), ict_zone.get("ob_high", float('-inf')))
        if protective_high == float('-inf'): return None, []
        sl_price_candidate = protective_high + (atr_1m_value * sl_buffer_atr_mult if atr_1m_value and atr_1m_value > 0 else 0)
        sl_price = round(sl_price_candidate, p_prec)
        if sl_price <= entry_price: return None, []
    else: return None, []

    tp_levels = []
    if sl_price is None: return None, []
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit == 0: return sl_price, []
    
    r_multiples = [1.0, 2.0, 3.0]
    tp_qty_pcts = [configs.get("ict_tp1_qty_pct", DEFAULT_ICT_TP1_QTY_PCT), configs.get("ict_tp2_qty_pct", DEFAULT_ICT_TP2_QTY_PCT), configs.get("ict_tp3_qty_pct", DEFAULT_ICT_TP3_QTY_PCT)]
    # Simplified sum adjustment for brevity
    if abs(sum(tp_qty_pcts) - 1.0) > 1e-5 : tp_qty_pcts[2] = max(0, 1.0 - tp_qty_pcts[0] - tp_qty_pcts[1])

    for i, r_mult in enumerate(r_multiples):
        tp_price_cand = entry_price + (risk_per_unit * r_mult) if side == "LONG" else entry_price - (risk_per_unit * r_mult)
        tp_price_final = round(tp_price_cand, p_prec)
        if ((side == "LONG" and tp_price_final <= entry_price) or (side == "SHORT" and tp_price_final >= entry_price)): continue
        if tp_qty_pcts[i] > 0: tp_levels.append({"price": tp_price_final, "quantity_pct": tp_qty_pcts[i], "name": f"TP{i+1}"})
    return sl_price, tp_levels

async def manage_ict_entry_logic(client: Client, configs: dict, symbol: str, symbol_info: dict, current_entry_tf_candle_row: pd.Series, ict_zones: list[dict], symbol_1m_candle_buffer: pd.DataFrame | None):
    global active_trades, active_trades_lock, ict_strategy_states, ict_strategy_states_lock, last_signal_time, last_signal_lock, recent_trade_signatures, recent_trade_signatures_lock, trading_halted_drawdown, trading_halted_daily_loss, trading_halted_manual, daily_state_lock

    log_prefix = f"[{symbol} ICT_Entry_App]" # Simplified log prefix
    entry_order_type = configs.get("ict_entry_order_type", DEFAULT_ICT_ENTRY_ORDER_TYPE)
    async with daily_state_lock: # Assuming daily_state_lock is asyncio.Lock now
        if trading_halted_drawdown or trading_halted_daily_loss: return
    if trading_halted_manual: return
    if not ict_zones or current_entry_tf_candle_row is None or current_entry_tf_candle_row.empty: return

    current_entry_tf_close = current_entry_tf_candle_row['close']
    current_utc_time = pd.Timestamp.now(tz='UTC')
    if not is_time_in_ict_session(current_utc_time, [s.lower() for s in configs.get("ict_session_filter", DEFAULT_ICT_SESSION_FILTER)]): return

    for zone in ict_zones:
        trade_side = "LONG" if zone['type'] == "long_zone" else "SHORT" if zone['type'] == "short_zone" else None
        if not trade_side: continue
        p_prec = int(symbol_info['pricePrecision'])
        proposed_limit_entry_price = round((zone['zone_upper'] + zone['zone_lower']) / 2.0, p_prec)

        async with last_signal_lock:
            if (dt.now() - last_signal_time.get(f"{symbol}_ict", dt.min())).total_seconds() < configs.get("ict_signal_cooldown_seconds", DEFAULT_ICT_SIGNAL_COOLDOWN_SECONDS): continue
        async with active_trades_lock:
            if symbol in active_trades or len(active_trades) >= configs["max_concurrent_positions"]: continue

        atr_entry_tf_value = None
        if symbol_1m_candle_buffer is not None and not symbol_1m_candle_buffer.empty and len(symbol_1m_candle_buffer) >= configs.get("atr_period", DEFAULT_ATR_PERIOD):
            atr_1m_series = calculate_atr(symbol_1m_candle_buffer.copy(), period=configs.get("atr_period", DEFAULT_ATR_PERIOD))
            if not atr_1m_series.empty and pd.notna(atr_1m_series.iloc[-1]): atr_entry_tf_value = atr_1m_series.iloc[-1]
        if atr_entry_tf_value is None: atr_entry_tf_value = 0.0 # Default to zero buffer if ATR fails

        sl_price, tp_levels_details = await calculate_ict_sl_tp(proposed_limit_entry_price, trade_side, zone, atr_entry_tf_value, configs, symbol_info)
        if sl_price is None or not tp_levels_details: continue

        acc_bal = await get_account_balance(client, configs)
        if acc_bal is None or acc_bal <= 0: print(f"{log_prefix} Invalid account balance. Abort."); continue
        
        # Fetch current leverage
        current_leverage_on_symbol = configs.get('leverage')
        try:
            pos_info_lev_list = await asyncio.to_thread(client.futures_position_information, symbol=symbol)
            if pos_info_lev_list and isinstance(pos_info_lev_list, list) and pos_info_lev_list[0]:
                current_leverage_on_symbol = int(pos_info_lev_list[0].get('leverage', configs.get('leverage')))
        except Exception: pass # Use default if fetch fails

        qty_to_order_total = calculate_position_size(acc_bal, configs['risk_percent'], proposed_limit_entry_price, sl_price, symbol_info, configs)
        if qty_to_order_total is None or qty_to_order_total <= 0: continue
        
        passed_sanity, sanity_reason = pre_order_sanity_checks(symbol, trade_side, proposed_limit_entry_price, sl_price, tp_levels_details[0]['price'], qty_to_order_total, symbol_info, acc_bal, configs['risk_percent'], configs, current_leverage_on_symbol)
        if not passed_sanity: print(f"{log_prefix} Sanity FAILED: {sanity_reason}"); continue
        
        trade_sig_ict = generate_trade_signature(symbol, f"ICT_{trade_side}", proposed_limit_entry_price, sl_price, tp_levels_details[0]['price'], qty_to_order_total, p_prec)
        async with recent_trade_signatures_lock:
            if trade_sig_ict in recent_trade_signatures and (dt.now() - recent_trade_signatures[trade_sig_ict]).total_seconds() < 60: continue
        async with last_signal_lock: last_signal_time[f"{symbol}_ict"] = dt.now()

        if configs['mode'] == 'signal':
            # Simplified signal sending, PNL calc would require client for price
            await send_ict_telegram_alert(configs, "ENTRY", symbol, {"side": trade_side, "entry_price": proposed_limit_entry_price, "sl_price": sl_price, "tp1_price": tp_levels_details[0]['price'], "tp1_qty_pct": tp_levels_details[0]['quantity_pct']}, symbol_info)
            await log_ict_event_to_csv({"Symbol": symbol, "EventType": "LIMIT_ENTRY_SIGNAL", "Direction": trade_side, "EntryPrice": proposed_limit_entry_price, "SL_Price": sl_price})
            async with recent_trade_signatures_lock: recent_trade_signatures[trade_sig_ict] = dt.now()
            return

        if entry_order_type != "LIMIT": return # Enforce LIMIT for ICT
        limit_entry_order, entry_err_msg = await place_new_order(client, symbol_info, "BUY" if trade_side=="LONG" else "SELL", "LIMIT", qty_to_order_total, price=proposed_limit_entry_price, position_side=trade_side)
        if not limit_entry_order: await log_ict_event_to_csv({"Symbol": symbol, "EventType": "LIMIT_ENTRY_FAIL", "Direction": trade_side, "Notes": entry_err_msg or "Unknown"}); return

        pending_order_info = {"order_id": limit_entry_order['orderId'], "limit_price": proposed_limit_entry_price, "sl_price": sl_price, "tp_levels_details": tp_levels_details, "quantity": qty_to_order_total, "side": trade_side, "zone_snapshot": zone, "atr_1m_at_placement": atr_entry_tf_value, "order_placed_timestamp": current_utc_time, "symbol_info": symbol_info}
        async with ict_strategy_states_lock:
            ict_strategy_states.setdefault(symbol, {}).setdefault('pending_ict_entries', []).append(pending_order_info)
        async with recent_trade_signatures_lock: recent_trade_signatures[trade_sig_ict] = dt.now()
        await log_ict_event_to_csv({"Symbol": symbol, "EventType": "LIMIT_ENTRY_PLACED", "Direction": trade_side, "EntryPrice": proposed_limit_entry_price, "SL_Price": sl_price, "OrderID": limit_entry_order['orderId']})
        await send_ict_telegram_alert(configs, "ENTRY", symbol, {"side": trade_side, "entry_price": proposed_limit_entry_price, "sl_price": sl_price, "order_id": limit_entry_order['orderId'], "tp1_price": tp_levels_details[0]['price'],"tp1_qty_pct": tp_levels_details[0]['quantity_pct']}, symbol_info)
        return # Processed one zone

ICT_SESSION_TIMES_UTC = {"london": ("08:00", "16:00"), "newyork": ("13:30", "20:00"), "tokyo": ("00:00", "08:00"), "sydney": ("22:00", "06:00")}
def is_time_in_ict_session(current_time_utc: dt, ict_session_filter: list[str] | None) -> bool:
    if not ict_session_filter: return True
    current_time_obj = current_time_utc.time()
    for session_name_lower in ict_session_filter:
        if session_name_lower in ICT_SESSION_TIMES_UTC:
            open_str, close_str = ICT_SESSION_TIMES_UTC[session_name_lower]
            try:
                session_open_time = dt.strptime(open_str, "%H:%M").time()
                session_close_time = dt.strptime(close_str, "%H:%M").time()
            except ValueError: continue
            if (session_open_time > session_close_time and (current_time_obj >= session_open_time or current_time_obj < session_close_time)) or \
               (session_open_time <= session_close_time and session_open_time <= current_time_obj < session_close_time):
                return True
    return False

async def monitor_pending_ict_entries(client: Client, configs: dict):
    global ict_strategy_states, ict_strategy_states_lock, active_trades, active_trades_lock
    log_prefix = "[ICT_PendingMonitor_App]"
    entries_to_remove = [] # (symbol, order_id)

    pending_snapshot = {}
    async with ict_strategy_states_lock:
        for sym, state_data in list(ict_strategy_states.items()):
            if state_data.get('pending_ict_entries'):
                pending_snapshot[sym] = list(state_data['pending_ict_entries'])
    if not pending_snapshot: return

    for symbol, pending_list in pending_snapshot.items():
        for pending_entry in pending_list:
            order_id = pending_entry.get('order_id')
            s_info = pending_entry.get('symbol_info')
            if not all([order_id, s_info]): entries_to_remove.append((symbol, order_id)); continue
            
            try:
                order_status = await asyncio.to_thread(client.futures_get_order, symbol=symbol, orderId=order_id)
                if order_status['status'] == 'FILLED':
                    actual_ep = float(order_status['avgPrice'])
                    total_qty = float(order_status['executedQty'])
                    sl_orig, tp_details_orig = pending_entry['sl_price'], pending_entry['tp_levels_details']
                    trade_side = pending_entry['side']
                    sl_final, tp_details_final = sl_orig, tp_details_orig # Simplified: assume no re-calc on fill for now

                    sl_ord_obj, _ = await place_new_order(client, s_info, "SELL" if trade_side=="LONG" else "BUY", "STOP_MARKET", total_qty, stop_price=sl_final, position_side=trade_side, is_closing_order=True)
                    tp_orders_placed = []
                    # Simplified TP placement (assumes full quantity for each TP for now, needs proper partial logic)
                    for tp_info in tp_details_final:
                        tp_ord_obj, _ = await place_new_order(client, s_info, "SELL" if trade_side=="LONG" else "BUY", "TAKE_PROFIT_MARKET", total_qty * tp_info['quantity_pct'], stop_price=tp_info['price'], position_side=trade_side, is_closing_order=True) # Qty needs to be partial
                        if tp_ord_obj: tp_orders_placed.append({"id": tp_ord_obj.get('orderId'), "price": tp_info['price'], "quantity": total_qty * tp_info['quantity_pct'], "name": tp_info['name']})
                    
                    async with active_trades_lock:
                        if symbol not in active_trades: # Ensure no overwrite
                            active_trades[symbol] = {"entry_order_id": order_id, "sl_order_id": sl_ord_obj.get('orderId') if sl_ord_obj else None, "tp_orders": tp_orders_placed, "entry_price": actual_ep, "current_sl_price": sl_final, "initial_sl_price": sl_final, "quantity": total_qty, "side": trade_side, "symbol_info": s_info, "open_timestamp": pd.Timestamp(order_status['updateTime'], unit='ms', tz='UTC'), "strategy_type": "ICT_MULTI_TP", "ict_trade_zone_snapshot": pending_entry.get('zone_snapshot')}
                    await log_ict_event_to_csv({"Symbol": symbol, "EventType": "LIMIT_ENTRY_FILLED", "Direction": trade_side, "EntryPrice": actual_ep, "SL_Price": sl_final, "OrderID": order_id})
                    # Alert details for ENTRY type (modify as needed)
                    alert_details_fill = {"side": trade_side, "entry_price": actual_ep, "sl_price": sl_final, "order_id": order_id}
                    if tp_details_final: # Add first TP for summary
                        alert_details_fill["tp1_price"] = tp_details_final[0].get("price")
                        alert_details_fill["tp1_qty_pct"] = tp_details_final[0].get("quantity_pct",0)

                    await send_ict_telegram_alert(configs, "ENTRY", symbol, alert_details_fill, s_info)
                    entries_to_remove.append((symbol, order_id))

                elif order_status['status'] in ['CANCELED', 'EXPIRED', 'REJECTED', 'PENDING_CANCEL'] or \
                     (pd.Timestamp.now(tz='UTC') - pd.Timestamp(pending_entry.get('order_placed_timestamp'))).total_seconds() > configs.get("ict_order_timeout_minutes", DEFAULT_ICT_ORDER_TIMEOUT_MINUTES) * 60:
                    if order_status['status'] not in ['CANCELED', 'EXPIRED', 'REJECTED', 'PENDING_CANCEL']: # If timeout
                        try: await asyncio.to_thread(client.futures_cancel_order, symbol=symbol, orderId=order_id)
                        except Exception as e_cancel: print(f"{log_prefix} Failed to cancel timed-out ICT order {order_id}: {e_cancel}")
                    entries_to_remove.append((symbol, order_id))
                    await log_ict_event_to_csv({"Symbol": symbol, "EventType": f"LIMIT_ORDER_{order_status['status']}_OR_TIMEOUT", "OrderID": order_id})
            
            except BinanceAPIException as e:
                if e.code == -2013: entries_to_remove.append((symbol, order_id)) # Order not found
                else: print(f"{log_prefix} API Error checking ICT order {order_id}: {e}")
            except Exception as e:
                print(f"{log_prefix} Error checking ICT order {order_id}: {e}"); entries_to_remove.append((symbol, order_id))

    if entries_to_remove:
        async with ict_strategy_states_lock:
            for sym_clear, oid_clear in entries_to_remove:
                if sym_clear in ict_strategy_states and 'pending_ict_entries' in ict_strategy_states[sym_clear]:
                    ict_strategy_states[sym_clear]['pending_ict_entries'] = [p for p in ict_strategy_states[sym_clear]['pending_ict_entries'] if p.get('order_id') != oid_clear]

# --- Main execution example (if app.py is run standalone) ---
async def main():
    global app_configurations, binance_client_instance # Allow main to set these for testing
    print("app.py main() started (standalone execution)")

    # Attempt to import credentials directly from keys.py
    try:
        import keys
        imported_api_key = keys.api_testnet # Default to testnet for app.py standalone
        imported_api_secret = keys.secret_testnet
        imported_telegram_token = keys.telegram_bot_token
        imported_telegram_chat_id = keys.telegram_chat_id
        print("Successfully imported credentials from keys.py")
    except ImportError:
        print("Error: keys.py not found. Please ensure it exists in the same directory.")
        print("Telegram polling and Binance client interactions will not be available.")
        imported_api_key, imported_api_secret, imported_telegram_token, imported_telegram_chat_id = None, None, None, None
    except AttributeError as e:
        print(f"Error: One or more expected variables not found in keys.py: {e}")
        print("Please ensure keys.py defines: api_testnet, secret_testnet, telegram_bot_token, telegram_chat_id.")
        imported_api_key, imported_api_secret, imported_telegram_token, imported_telegram_chat_id = None, None, None, None

    if not all([imported_telegram_token, imported_telegram_chat_id]):
        print("Telegram token/chat_id not available from keys.py. Telegram polling will not start.")
    
    app_configurations = {
        "telegram_bot_token": imported_telegram_token,
        "telegram_chat_id": imported_telegram_chat_id,
        "risk_percent": 0.01, # Example
        "leverage": 20, # Example
        "ict_min_liquidity_sweep_size_pips": 5, # Example
        "mode": "signal", # Example mode
        "ict_timeframe_entry": "1m",
        "atr_period": 14,
        "ict_order_timeout_minutes": 10,
        "ict_entry_order_type": "LIMIT",
        "ict_sl_buffer_atr_mult": 0.2,
        "ict_tp1_qty_pct": 0.33,
        "ict_tp2_qty_pct": 0.33,
        "ict_tp3_qty_pct": 0.34,
        "ict_session_filter": ["london", "newyork"],
        "ict_signal_cooldown_seconds": 60,
        "max_concurrent_positions": 5,
        # Add any other configs required by ICT functions if they are called directly in this main
    }

    # Initialize Binance client instance if API keys are available
    if imported_api_key and imported_api_secret:
        try:
            # Using testnet=True for safety with keys from keys.py if they are testnet keys
            binance_client_instance = Client(imported_api_key, imported_api_secret, testnet=True) 
            await asyncio.to_thread(binance_client_instance.ping)
            print("Binance client initialized using keys.py (testnet) and ping successful.")
        except Exception as e:
            print(f"Failed to initialize Binance client using keys.py: {e}")
            binance_client_instance = None # Ensure it's None if init fails
    else:
        print("Binance API key/secret not available from keys.py. Binance client not initialized.")
        binance_client_instance = None

    # Start Telegram polling (if configured)
    telegram_task = None
    if imported_telegram_token and imported_telegram_chat_id:
        telegram_task = asyncio.create_task(start_telegram_polling(imported_telegram_token, app_configurations))
        print("Telegram polling task created.")
    
    print("app.py main() finished setup. If Telegram polling started, it runs in background.")

    # Keep alive if Telegram is running
    if telegram_task:
        try:
            await telegram_task
        except asyncio.CancelledError:
            print("Telegram polling task was cancelled.")
    else:
        print("Exiting app.py main as Telegram polling was not started.")


if __name__ == "__main__":
    asyncio.run(main())
