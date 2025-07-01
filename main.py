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

# --- Configuration Defaults ---
DEFAULT_RISK_PERCENT = 1.0       # Default account risk percentage per trade (e.g., 1.0 for 1%)
DEFAULT_LEVERAGE = 20            # Default leverage (e.g., 20x)
DEFAULT_MAX_CONCURRENT_POSITIONS = 5 # Default maximum number of concurrent open positions
DEFAULT_MARGIN_TYPE = "ISOLATED" # Default margin type: "ISOLATED" or "CROSS"
DEFAULT_MAX_SCAN_THREADS = 10    # Default threads for scanning symbols
DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL = False # Default for allowing higher risk to meet min notional

# --- Global State Variables ---
# Stores details of active trades. Key: symbol (e.g., "BTCUSDT")
# Value: dict with trade info like order IDs, entry/SL/TP prices, quantity, side.
active_trades = {}
active_trades_lock = threading.Lock() # Lock for synchronizing access to active_trades

# Stores the timestamp of the last trade initiation for each symbol to manage cooldowns.
# Key: symbol (e.g., "BTCUSDT"), Value: pd.Timestamp
symbol_last_trade_time = {}
symbol_last_trade_time_lock = threading.Lock() # Lock for synchronizing access to symbol_last_trade_time

# Set to keep track of symbols currently being processed by manage_trade_entry to prevent race conditions.
symbols_currently_processing = set()
symbols_currently_processing_lock = threading.Lock()

# --- Utility and Configuration Functions ---

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
    async def actual_bot_runner():
        # Create and configure the application within the async function
        # that will be the entry point for asyncio.run()
        application = Application.builder().token(bot_token).build()

        async def command3_handler(update, context: ContextTypes.DEFAULT_TYPE):
            if str(update.effective_chat.id) == str(chat_id):
                await context.bot.send_message(chat_id=chat_id, text=last_message_content, parse_mode="Markdown")
        
        application.add_handler(CommandHandler("command3", command3_handler))
        application.add_handler(CommandHandler("start", start_handler))
        application.add_handler(CommandHandler("help", help_handler))

        # run_polling() will block until the application is stopped.
        await application.run_polling()
        application.run_polling()

    def thread_starter():
        try:
            asyncio.run(actual_bot_runner())
        except RuntimeError as e:
            # This specific error can sometimes occur on Windows with ProactorEventLoop
            # during shutdown if the loop PTB tries to close is already being managed/closed.
            if "Cannot close a running event loop" in str(e) or "Event loop is closed" in str(e):
                print(f"Known asyncio loop issue during Telegram thread shutdown: {e}")
            else:
                # Re-raise other RuntimeErrors
                print(f"Unhandled RuntimeError in Telegram thread: {e}")
                traceback.print_exc()
                raise
        except Exception as e:
            print(f"Exception in Telegram thread_starter: {e}")
            traceback.print_exc()

    thread = threading.Thread(target=thread_starter)
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
    """
    print("\n--- Strategy Configuration ---")
    configs = {}
    while True:
        env_input = input("Select environment (1:testnet / 2:mainnet): ").strip()
        if env_input == "1":
            configs["environment"] = "testnet"
            break
        elif env_input == "2":
            configs["environment"] = "mainnet"
            break
        print("Invalid environment. Please enter '1' for testnet or '2' for mainnet.")
    # Load all keys including Telegram
    api_key, api_secret, telegram_token, telegram_chat_id = load_api_keys(configs["environment"])
    configs["api_key"] = api_key
    configs["api_secret"] = api_secret
    configs["telegram_bot_token"] = telegram_token
    configs["telegram_chat_id"] = telegram_chat_id

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
            balance_choice_input = input("For backtest, use (1:current account balance) or (2:set a custom start balance)? [1]: ").strip()
            if not balance_choice_input: balance_choice_input = "1" # Default if user presses Enter

            if balance_choice_input == "1":
                configs["backtest_start_balance_type"] = "current"
                break
            elif balance_choice_input == "2":
                configs["backtest_start_balance_type"] = "custom"
                break
            print("Invalid choice. Please enter '1' for current or '2' for custom.")

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
    
    while True:
        exceed_risk_input = input(f"Allow exceeding risk % to meet MIN_NOTIONAL? (yes/no, default: {'yes' if DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL else 'no'}): ").lower().strip()
        if not exceed_risk_input: # User pressed Enter, use default
            configs["allow_exceed_risk_for_min_notional"] = DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL
            break
        if exceed_risk_input in ["yes", "y"]:
            configs["allow_exceed_risk_for_min_notional"] = True
            break
        elif exceed_risk_input in ["no", "n"]:
            configs["allow_exceed_risk_for_min_notional"] = False
            break
        print("Invalid input. Please enter 'yes' or 'no'.")

    configs["strategy_id"] = 8
    configs["strategy_name"] = "Advance EMA Cross"
    print("--- Configuration Complete ---")
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

# --- Pre-Order Sanity Checks ---
def pre_order_sanity_checks(symbol, signal, entry_price, sl_price, tp_price, quantity, 
                            symbol_info, current_balance, risk_percent_config, configs, 
                            klines_df_for_debug=None, is_unmanaged_check=False): # Added is_unmanaged_check
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
        # Define a small tolerance for float comparisons of percentages
        strict_risk_tolerance = 0.0001 # Absolute 0.01% deviation from target risk percentage
        allow_exceed_risk = configs.get('allow_exceed_risk_for_min_notional', DEFAULT_ALLOW_EXCEED_RISK_FOR_MIN_NOTIONAL)

        if not allow_exceed_risk:
            # Strict check: actual risk should be very close to configured risk
            if abs(risk_percentage_actual - risk_percent_config) > strict_risk_tolerance:
                return False, (f"Actual risk ({risk_percentage_actual*100:.3f}%) deviates "
                               f"from configured risk ({risk_percent_config*100:.3f}%) by more than "
                               f"{strict_risk_tolerance*100:.3f}%. Stricter risk adherence is enabled.")
        else: # allow_exceed_risk is True
            # More lenient check: actual risk should not exceed a cap (e.g., 1.5x configured risk)
            max_permissible_risk = risk_percent_config * 1.5 
            if risk_percentage_actual > (max_permissible_risk + strict_risk_tolerance): 
                return False, (f"Actual risk ({risk_percentage_actual*100:.3f}%) exceeds "
                               f"the maximum permissible risk limit ({max_permissible_risk*100:.3f}%) "
                               f"even when 'allow_exceed_risk_for_min_notional' is enabled.")
    else: # For unmanaged checks, just ensure risk is not absurdly high, e.g. > 50% of balance for a single trade SL
        if risk_percentage_actual > 0.5: # Arbitrary high cap for unmanaged safety SL/TP
             return False, (f"Calculated SL for UNMANAGED trade implies extremely high risk ({risk_percentage_actual*100:.2f}% of balance). "
                            f"Entry: {entry_price}, SL: {sl_price}, Qty: {quantity}. Check position and SL logic.")


    # 5. Sufficient Balance (Basic Margin Check)
    # For unmanaged trades, this check is against current balance for an existing position's margin,
    # which is implicitly covered as the position exists.
    # For new trades, it's a pre-check.
    leverage = configs.get('leverage', DEFAULT_LEVERAGE) # This might be different from actual position leverage for unmanaged
    if leverage <= 0:
        return False, f"Leverage ({leverage}) must be positive."
    
    required_margin = (quantity * entry_price) / leverage
    if required_margin > current_balance: # This is a simplified check. Binance actual margin req might differ.
        return False, (f"Estimated required margin ({required_margin:.2f} USDT) exceeds "
                       f"current balance ({current_balance:.2f} USDT) for quantity {quantity:.{q_prec}f} at {entry_price:.{p_prec}f} with {leverage}x leverage.")

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
        klines_df = get_historical_klines(client, symbol) # Uses default limit 500
        if klines_df.empty or len(klines_df) < 202:
            print(f"[{thread_name}] {symbol}: Skipped calling manage_trade_entry - Insufficient klines ({len(klines_df)}) {format_elapsed_time(cycle_start_ref)}")
            return f"{symbol}: Skipped - Insufficient klines ({len(klines_df)})"
        
        print(f"[{thread_name}] {symbol}: Sufficient klines ({len(klines_df)}). Calling manage_trade_entry {format_elapsed_time(cycle_start_ref)}")
        manage_trade_entry(client, configs, symbol, klines_df.copy(), lock)
        return f"{symbol}: Processed"
    except Exception as e:
        print(f"[{thread_name}] ERROR processing {symbol}: {e} {format_elapsed_time(cycle_start_ref)}")
        traceback.print_exc() # Print full traceback for thread errors
        return f"{symbol}: Error - {e}"

def manage_trade_entry(client, configs, symbol, klines_df, lock): # lock here is active_trades_lock
    global active_trades, symbol_last_trade_time, symbol_last_trade_time_lock, symbols_currently_processing, symbols_currently_processing_lock

    log_prefix = f"[{threading.current_thread().name}] {symbol} manage_trade_entry:"

    # --- Symbol Processing Lock (to prevent concurrent execution for the same symbol) ---
    with symbols_currently_processing_lock:
        if symbol in symbols_currently_processing:
            print(f"{log_prefix} {symbol} is ALREADY BEING PROCESSED by another thread. Skipping this instance.")
            return
        symbols_currently_processing.add(symbol)
        print(f"{log_prefix} Acquired processing lock for {symbol}.")

    try:
        # --- Original start of function logic begins here ---
        was_existing_trade = False # Flag to see if we are replacing a trade
        
        # The 'signal' variable (new_signal) is determined later. 
        # The check for active_trades needs to be aware of this.
        # We first check general cooldown, then calculate signal, then check active_trades.

        # --- Symbol Cooldown Check (1-hour cooldown after a trade) ---
        with symbol_last_trade_time_lock: 
            last_trade_ts = symbol_last_trade_time.get(symbol)
        
        if last_trade_ts:
            time_since_last_trade_seconds = (pd.Timestamp.now(tz='UTC') - last_trade_ts).total_seconds()
            COOLDOWN_PERIOD_SECONDS = 3600 
            if time_since_last_trade_seconds < COOLDOWN_PERIOD_SECONDS:
                print(f"{log_prefix} Symbol is on 1-hour cooldown. Last trade was {time_since_last_trade_seconds:.0f}s ago. "
                      f"Remaining cooldown: {COOLDOWN_PERIOD_SECONDS - time_since_last_trade_seconds:.0f}s. Signal ignored.")
                return # Exit if symbol is on cooldown, finally block will release processing lock

        # Initial check for sufficient kline data 
        if klines_df.empty or len(klines_df) < 202: 
            print(f"{log_prefix} Insufficient kline data for {symbol} (Length: {len(klines_df)}). Aborting trade entry.")
            return # finally block will release processing lock

        klines_df['EMA100'] = calculate_ema(klines_df, 100)
        klines_df['EMA200'] = calculate_ema(klines_df, 200)

        # Verbose logging for EMAs can be enabled if needed by uncommenting below
        # if klines_df['EMA100'] is not None and not klines_df['EMA100'].empty:
        #     print(f"{log_prefix} Last EMA100 values: {klines_df['EMA100'].iloc[-3:].values}")
        # else:
        #     print(f"{log_prefix} EMA100 calculation resulted in None or empty series.")
        # if klines_df['EMA200'] is not None and not klines_df['EMA200'].empty:
        #     print(f"{log_prefix} Last EMA200 values: {klines_df['EMA200'].iloc[-3:].values}")
        # else:
        #     print(f"{log_prefix} EMA200 calculation resulted in None or empty series.")

        if klines_df['EMA100'] is None or klines_df['EMA200'] is None or \
           klines_df['EMA100'].isnull().all() or klines_df['EMA200'].isnull().all() or \
           len(klines_df) < 202: 
            print(f"{log_prefix} EMA calculation failed, EMAs are NaN, or insufficient data length ({len(klines_df)}). Aborting for {symbol}.")
            return # finally block will release processing lock

        # Determine the new signal based on current klines
        new_signal = check_ema_crossover_conditions(klines_df, symbol_for_logging=symbol)
        # The function check_ema_crossover_conditions itself prints the signal, so no need to double print here.

        # If the signal from crossover conditions is not actionable, exit early.
        if new_signal not in ["LONG", "SHORT"]:
            print(f"{log_prefix} No actionable trade signal ('{new_signal}') for {symbol}. Aborting further processing in manage_trade_entry.")
            # No trade rejection notification here specifically, as check_ema_crossover_conditions logs reasons for non-actionable signals.
            return # Exit manage_trade_entry if no valid LONG/SHORT signal

        # Initialize variables that might be conditionally assigned within the 'with lock' block
        open_ts = None
        existing_signal_side = None
        # was_existing_trade is already initialized to False at the start of the function.

        # Now, handle active_trades with the new_signal
        with lock: # This is active_trades_lock (passed as 'lock' argument)
            if symbol in active_trades and active_trades[symbol]:
                existing_details = active_trades[symbol]
                open_ts = existing_details.get('open_timestamp')
                existing_signal_side = existing_details.get('side') 

                if not open_ts or not isinstance(open_ts, pd.Timestamp):
                    print(f"{log_prefix} DEBUG: Problematic existing_details for {symbol}: {existing_details}") # LOGGING C
                    error_msg = f"CRITICAL ERROR: {symbol} has invalid/missing 'open_timestamp' (type: {type(open_ts)}, value: '{open_ts}'). Skipping trade."
                    print(error_msg)
                    try:
                        if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                            send_telegram_message(
                                configs["telegram_bot_token"], configs["telegram_chat_id"],
                                f"🆘 BOT ALERT: Symbol `{symbol}` has invalid `open_timestamp`. Type: {type(open_ts)}. Value: `{open_ts}`. Skipping."
                            )
                    except Exception as tel_ex:
                        print(f"Error sending Telegram notification for invalid open_ts for {symbol}: {tel_ex}")
                    return # Crucial: ensure return happens if open_ts is invalid

                # If we reach here, open_ts is a valid pd.Timestamp.
                time_since_last_trade = (pd.Timestamp.now(tz='UTC') - open_ts).total_seconds()
                
                # Check if same signal (new_signal) within cooldown window (1 hour)
                if time_since_last_trade < 3600 and existing_signal_side == new_signal: # Use new_signal here
                    print(f"Duplicate {new_signal} signal for {symbol} IGNORED. Last same-side trade was {time_since_last_trade:.0f}s ago at {open_ts}.")
                    return

                # Proceed to replace if older than 1 hour or signal direction changed
                print(f"Trade for {symbol} is eligible for replacement (age: {time_since_last_trade:.0f}s, previous: {existing_signal_side}, new: {new_signal}). Cancelling old SL/TP.") # Use new_signal
                was_existing_trade = True
                for oid_key in ['sl_order_id', 'tp_order_id']:
                    oid = existing_details.get(oid_key)
                    if oid:
                        try:
                            client.futures_cancel_order(symbol=symbol, orderId=oid)
                        except BinanceAPIException as e:
                            if e.code != -2011:
                                print(f"ERROR cancelling old {oid_key} {oid} for {symbol}: {e}")
                                return
                            else:
                                print(f"Old {oid_key} {oid} for {symbol} already gone.")
                        except Exception as e:
                            print(f"ERROR cancelling old {oid_key} {oid} for {symbol}: {e}")
                            return

            # First, check if the timestamp is valid.
            # This block seems redundant or misplaced if the first check for open_ts (around line 1136) is effective.
            # If open_ts was invalid, the function should have returned already.
            # If this block is reached, it implies the earlier check was passed, meaning open_ts is valid.
            # However, if this 'if' block is meant to be associated with the 'else' of 
            # 'if symbol in active_trades and active_trades[symbol]', then its placement and condition are incorrect.
            # For now, addressing only the indentation error. The logic of this block might need review later
            # if errors persist around open_ts handling.
            if not open_ts or not isinstance(open_ts, pd.Timestamp):
                # CRITICAL: Symbol is in active_trades but has an invalid or missing open_timestamp.
                # This prevents proper cooldown checks and could lead to re-trading.
                print(f"CRITICAL ERROR: {symbol} found in active_trades but has an invalid/missing 'open_timestamp': {open_ts}. Skipping trade processing for this symbol to prevent re-trading.")
                # Send a Telegram alert about this inconsistent state
                if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                    error_message = (
                        f"🆘 BOT ALERT: Invalid Timestamp 🆘\n\n"
                        f"Symbol: `{symbol}` is in `active_trades` but has a problematic `open_timestamp`.\n"
                        f"Value: `{open_ts}`\n"
                        f"This trade will be skipped in the current cycle to prevent errors or duplicate trades.\n"
                        f"Please investigate `active_trades` state if this persists."
                    )
                    send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], error_message)
                return # Exit to prevent re-trading due to inconsistent state
            
            # Timestamp is valid, proceed with 1-hour cooldown check
            if (pd.Timestamp.now(tz='UTC') - open_ts).total_seconds() < 3600: # Less than 1 hour
                print(f"New signal for {symbol} IGNORED. Existing trade is less than 1 hour old ({open_ts}).")
                return
            
            # If the trade is 1 hour old or older, proceed to process it as a potential replacement.
            print(f"Existing trade for {symbol} is 1hr old or older. Processing new signal, cancelling old SL/TP.")
            was_existing_trade = True # Mark that we are intending to replace an existing trade
            for oid_key in ['sl_order_id', 'tp_order_id']:
                oid = existing_details.get(oid_key)
                if oid:
                    try: client.futures_cancel_order(symbol=symbol, orderId=oid)
                    except BinanceAPIException as e:
                        if e.code != -2011: print(f"ERROR cancelling old {oid_key} {oid} for {symbol}: {e}"); return # Return if cancel fails critically
                        else: print(f"Old {oid_key} {oid} for {symbol} already gone.")
                    except Exception as e: print(f"ERROR cancelling old {oid_key} {oid} for {symbol}: {e}"); return # Return on other cancel errors
        
        # Max concurrent position check.
        # This applies if 'was_existing_trade' is False (i.e., it's a new symbol, not a replacement)
        if not was_existing_trade and len(active_trades) >= configs["max_concurrent_positions"]:
            print(f"Max concurrent positions ({configs['max_concurrent_positions']}) reached. Cannot open for new symbol {symbol}.")
            return

        # --- Calculations and non-critical API calls (now part of the main try block) ---
        klines_df['EMA100'] = calculate_ema(klines_df, 100)
        klines_df['EMA200'] = calculate_ema(klines_df, 200)

        # log_prefix is already defined at the start of the function if needed here,
        # but it's redefined based on thread name. Let's ensure it's current.
        # log_prefix = f"[{threading.current_thread().name}] {symbol} manage_trade_entry:" # Re-affirm or ensure it's correct

        # --- Symbol Cooldown Check (This was already done at the beginning of the try block) ---
        # The cooldown check at lines 1111-1118 should suffice.
        # Re-checking here might be redundant unless the klines_df processing took extremely long.
        # For now, assuming the initial cooldown check is the primary one.

        # Initial check for sufficient kline data (also done earlier)
        if klines_df.empty or len(klines_df) < 202: 
            print(f"{log_prefix} Insufficient kline data for {symbol} (Length: {len(klines_df)}) before final signal processing. Aborting.")
            return

        # Verbose EMA logging (original was commented out, can be re-enabled if needed)
        # if klines_df['EMA100'] is not None and not klines_df['EMA100'].empty:
        #     print(f"{log_prefix} Last EMA100 values (post-active-trade check): {klines_df['EMA100'].iloc[-3:].values}")
        # if klines_df['EMA200'] is not None and not klines_df['EMA200'].empty:
        #     print(f"{log_prefix} Last EMA200 values (post-active-trade check): {klines_df['EMA200'].iloc[-3:].values}")

        if klines_df['EMA100'] is None or klines_df['EMA200'] is None or \
           klines_df['EMA100'].isnull().all() or klines_df['EMA200'].isnull().all(): # Length check already done
            print(f"{log_prefix} EMA calculation failed or EMAs are NaN (post-active-trade check). Aborting for {symbol}.")
            return

        # Determine signal (this was `new_signal` before, now it's the main `signal` for execution)
        # The `check_ema_crossover_conditions` was already called to get `new_signal`.
        # If `new_signal` was not actionable (e.g., None, VALIDATION_FAILED), the function would have returned.
        # So, `new_signal` (which we can rename to `signal` for this phase) should be "LONG" or "SHORT".
        signal = new_signal # `new_signal` comes from line 1142
        
        if signal not in ["LONG", "SHORT"]: # Should have been caught earlier, but as a safeguard
            print(f"{log_prefix} No actionable signal ('{signal}') for {symbol} before execution phase. Aborting.")
            return

        print(f"\n{log_prefix} --- Proceeding with Validated Trade Signal for {symbol}: {signal} ---")
        symbol_info = get_symbol_info(client, symbol)
        entry_p = klines_df['close'].iloc[-1] 
        sl_p, tp_p, qty_calc = None, None, None 
        
        if not symbol_info:
            reason = "Failed to retrieve symbol information."
            print(f"{log_prefix} {reason} for {symbol}. Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p, tp_p, qty_calc, symbol_info, configs)
            return

        print(f"{log_prefix} Entry price (last close): {entry_p} for {symbol}")
        ema100_val = klines_df['EMA100'].iloc[-1]
        
        if not (set_leverage_on_symbol(client, symbol, configs['leverage']) and \
                set_margin_type_on_symbol(client, symbol, configs['margin_type'])):
            reason = "Failed to set leverage or margin type."
            print(f"{log_prefix} {reason} for {symbol}. Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p, tp_p, qty_calc, symbol_info, configs)
            return

        print(f"{log_prefix} Leverage and margin type set for {symbol}.")
        sl_p, tp_p = calculate_sl_tp_values(entry_p, signal, ema100_val, klines_df)
        if sl_p is None or tp_p is None:
            reason = "Stop Loss / Take Profit calculation failed."
            print(f"{log_prefix} {reason} for {symbol}. Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p, tp_p, qty_calc, symbol_info, configs)
            return
        print(f"{log_prefix} Calculated SL: {sl_p}, TP: {tp_p} for {symbol}.")
        
        acc_bal = get_account_balance(client, configs) 
        if acc_bal is None:
            reason = "Critical error fetching account balance."
            print(f"{log_prefix} {reason} Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p, tp_p, qty_calc, symbol_info, configs)
            return
        if acc_bal <= 0:
            reason = f"Zero or negative account balance ({acc_bal})."
            print(f"{log_prefix} {reason} Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p, tp_p, qty_calc, symbol_info, configs)
            return
        print(f"{log_prefix} Account balance: {acc_bal} for position sizing.")

        qty_calc = calculate_position_size(acc_bal, configs['risk_percent'], entry_p, sl_p, symbol_info, configs) 
        if qty_calc is None or qty_calc <= 0:
            reason = f"Invalid position size calculated (Qty: {qty_calc}). Check logs for details from calculate_position_size."
            print(f"{log_prefix} {reason} for {symbol}. Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p, tp_p, qty_calc, symbol_info, configs)
            return
        print(f"{log_prefix} Calculated position size: {qty_calc} for {symbol}.")
        
        final_qty_check = round(qty_calc, int(symbol_info['quantityPrecision']))
        if final_qty_check == 0.0:
            reason = f"Calculated quantity {qty_calc} rounds to zero."
            print(f"{log_prefix} {reason} for {symbol}. Abort.")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p, tp_p, qty_calc, symbol_info, configs)
            return
        print(f"{log_prefix} Final quantity check (rounded): {final_qty_check} for {symbol}.")

        qty_to_order = final_qty_check 

        passed_sanity_checks, sanity_check_reason = pre_order_sanity_checks(
            symbol=symbol,
            signal=signal,
            entry_price=entry_p,
            sl_price=sl_p,
            tp_price=tp_p,
            quantity=qty_to_order,
            symbol_info=symbol_info,
            current_balance=acc_bal, 
            risk_percent_config=configs['risk_percent'],
            configs=configs,
            klines_df_for_debug=klines_df 
        )

        if not passed_sanity_checks:
            print(f"{log_prefix} Pre-order sanity checks FAILED for {symbol}: {sanity_check_reason}")
            send_trade_rejection_notification(symbol, signal, f"Sanity Check Failed: {sanity_check_reason}", 
                                              entry_p, sl_p, tp_p, qty_to_order, symbol_info, configs)
            return 

        print(f"{log_prefix} Pre-order sanity checks PASSED for {symbol}.")

        print(f"{log_prefix} Attempting {signal} {qty_to_order} {symbol} @MKT (EP:{entry_p:.4f}), SL:{sl_p:.4f}, TP:{tp_p:.4f}")
        entry_order = place_new_order(client, symbol_info, "BUY" if signal=="LONG" else "SELL", "MARKET", qty_to_order)

        if not entry_order or entry_order.get('status') != 'FILLED':
            reason = f"Market entry order failed or not filled. Status: {entry_order.get('status') if entry_order else 'N/A'}."
            print(f"{log_prefix} {reason} for {symbol}: {entry_order}")
            send_trade_rejection_notification(symbol, signal, reason, entry_p, sl_p, tp_p, qty_to_order, symbol_info, configs)
            return
        
        print(f"{log_prefix} Market entry order FILLED for {symbol}. Order details: {entry_order}")
        actual_ep = float(entry_order.get('avgPrice', entry_p))
        
        if entry_p > 0 : 
            sl_dist_pct = abs(entry_p - sl_p) / entry_p
            tp_dist_pct = abs(entry_p - tp_p) / entry_p
            sl_p = actual_ep * (1 - sl_dist_pct if signal == "LONG" else 1 + sl_dist_pct)
            tp_p = actual_ep * (1 + tp_dist_pct if signal == "LONG" else 1 - tp_dist_pct)
            price_prec_adj = symbol_info.get('pricePrecision', 2) if symbol_info else 2
            print(f"SL/TP adjusted for actual fill {actual_ep:.{price_prec_adj}f}: SL {sl_p:.{price_prec_adj}f}, TP {tp_p:.{price_prec_adj}f}")

        sl_ord = place_new_order(client, symbol_info, "SELL" if signal=="LONG" else "BUY", "STOP_MARKET", qty_to_order, stop_price=sl_p, reduce_only=True)
        if not sl_ord: print(f"{log_prefix} CRITICAL: FAILED TO PLACE SL FOR {symbol}! Order details: {sl_ord}"); 
        else: print(f"{log_prefix} SL order placed for {symbol}. Order details: {sl_ord}")
        
        tp_ord = place_new_order(client, symbol_info, "SELL" if signal=="LONG" else "BUY", "TAKE_PROFIT_MARKET", qty_to_order, stop_price=tp_p, reduce_only=True)
        if not tp_ord: print(f"{log_prefix} Warning: Failed to place TP for {symbol}. Order details: {tp_ord}")
        else: print(f"{log_prefix} TP order placed for {symbol}. Order details: {tp_ord}")

        with lock: 
            current_pd_timestamp = pd.Timestamp.now(tz='UTC') # LOGGING A - Part 1
            print(f"{log_prefix} DEBUG: Timestamp to be used for new trade {symbol}: {current_pd_timestamp} (Type: {type(current_pd_timestamp)})") # LOGGING A - Part 2
            
            new_trade_data = {
                "entry_order_id": entry_order['orderId'], "sl_order_id": sl_ord.get('orderId') if sl_ord else None,
                "tp_order_id": tp_ord.get('orderId') if tp_ord else None, "entry_price": actual_ep,
                "current_sl_price": sl_p, "current_tp_price": tp_p, "initial_sl_price": sl_p, "initial_tp_price": tp_p,
                "quantity": qty_to_order, "side": signal, "symbol_info": symbol_info, "open_timestamp": current_pd_timestamp
            }
            active_trades[symbol] = new_trade_data
            print(f"{log_prefix} DEBUG: Full active_trades entry for {symbol} after creation: {active_trades[symbol]}") # LOGGING B
            print(f"{log_prefix} Trade for {symbol} recorded in active_trades at {active_trades[symbol]['open_timestamp']}.")
            
            with symbol_last_trade_time_lock:
                symbol_last_trade_time[symbol] = active_trades[symbol]['open_timestamp'] 
                print(f"{log_prefix} Updated last trade time for {symbol} to {symbol_last_trade_time[symbol]} for cooldown.")

        if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
            current_balance_for_msg = get_account_balance(client, configs)
            if current_balance_for_msg is None: current_balance_for_msg = "N/A (Error)"
            else: current_balance_for_msg = f"{current_balance_for_msg:.2f}"

            with active_trades_lock:
                s_info_map_for_new_trade_msg = _build_symbol_info_map_from_active_trades(active_trades)
            open_positions_str = get_open_positions(client, format_for_telegram=True, active_trades_data=active_trades.copy(), symbol_info_map=s_info_map_for_new_trade_msg)

            qty_prec_msg = symbol_info.get('quantityPrecision', 0) if symbol_info else 0
            price_prec_msg = symbol_info.get('pricePrecision', 2) if symbol_info else 2

            new_trade_message = (
                f"🚀 NEW TRADE PLACED 🚀\n\n"
                f"Symbol: {symbol}\n"
                f"Side: {signal}\n"
                f"Quantity: {qty_to_order:.{qty_prec_msg}f}\n"
                f"Entry Price: {actual_ep:.{price_prec_msg}f}\n"
                f"SL: {sl_p:.{price_prec_msg}f}\n"
                f"TP: {tp_p:.{price_prec_msg}f}\n\n"
                f"💰 Account Balance: {current_balance_for_msg} USDT\n"
                f"📊 Current Open Positions:\n{open_positions_str}"
            )
            send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], new_trade_message)
            print(f"{log_prefix} New trade Telegram notification sent for {symbol}.")

        get_open_positions(client); get_open_orders(client, symbol) 
    
    finally:
        # --- Release Symbol Processing Lock ---
        with symbols_currently_processing_lock:
            symbols_currently_processing.discard(symbol) # Use discard to avoid KeyError if not present
            print(f"{log_prefix} Released processing lock for {symbol}.")

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
            for sym_to_remove in symbols_to_remove: # Iterate over the list of symbols marked for removal
                if sym_to_remove in active_trades: # Check if it's still in active_trades (it should be)
                    closed_trade_details = active_trades[sym_to_remove] # Get details before deleting

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
                            f"(Reason: SL/TP hit or external closure detected)\n\n"
                            f"💰 Account Balance: {current_balance_for_msg} USDT\n"
                            f"📊 Current Open Positions:\n{open_positions_str}"
                        )
                        send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], closed_trade_message)
                        print(f"Trade closure Telegram notification sent for {sym_to_remove}.")
                    
                    del active_trades[sym_to_remove]
                    print(f"Removed {sym_to_remove} from bot's active trades.")

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
                # Ensure client_obj is used here if that's the correct variable name passed to trading_loop
                # Assuming 'client' is the parameter name for trading_loop, which holds client_obj from main()
                current_cycle_balance = get_account_balance(client, configs) # Pass configs

                if configs['mode'] == 'live' and current_cycle_balance is None:
                    # Message already sent by get_account_balance if it was -2015 (IP issue)
                    print("WARNING: Account balance could not be fetched in this cycle due to an API error (potentially IP whitelist).")
                    print("The bot will continue and retry in the next cycle. Check Telegram for IP alert if this was error -2015.")
                    # No break here, allowing the loop to continue to the next cycle.
                    # We might want to skip further processing for *this* cycle if balance is None.
                    # For now, it will proceed to try and process symbols, which might be okay as individual
                    # trade entries also check balance via manage_trade_entry->get_account_balance.
                    # If manage_trade_entry's get_account_balance also returns None, it will abort that trade.
                    # This seems like a reasonable approach: log cycle-level issue, continue, let symbol-level logic handle individual trades.
                else: # Only get positions and print status if balance fetch was successful or not critical
                    get_open_positions(client)
                    print(f"Account status updated. Current Balance: {current_cycle_balance} {format_elapsed_time(cycle_start_time)}")

                # The rest of the loop (symbol processing, monitoring) will still run.
                # manage_trade_entry has its own balance check.

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

                # --- SL/TP Safety Net Check for All Open Positions ---
                # Initialize a cache for symbol_info for this cycle of the safety net check
                # Alternatively, could try to pass/build upon the main symbol_info_map if available and up-to-date
                symbol_info_cache_for_safety_net = {} 
                print(f"Running SL/TP safety net check for all open positions... {format_elapsed_time(cycle_start_time)}")
                ensure_sl_tp_for_all_open_positions(client, configs, active_trades, symbol_info_cache_for_safety_net)
                print(f"SL/TP safety net check complete. {format_elapsed_time(cycle_start_time)}")
                # --- End SL/TP Safety Net Check ---

            except Exception as loop_err: # Catch errors within the try block of the loop
                print(f"ERROR in trading loop cycle: {loop_err} {format_elapsed_time(cycle_start_time)}")
                traceback.print_exc()

            cycle_dur_s = time.time() - cycle_start_time
            print(f"\n--- Scan Cycle #{current_cycle_number} Completed (Runtime: {cycle_dur_s:.2f}s / {(cycle_dur_s/60):.2f}min). ---")

            # Send status update every 100 cycles
            if current_cycle_number > 0 and current_cycle_number % 100 == 0:
                print(f"Scan cycle {current_cycle_number} reached. Sending status update to Telegram...")
                if configs.get("telegram_bot_token") and configs.get("telegram_chat_id"):
                    current_balance_for_update = get_account_balance(client, configs)
                    if current_balance_for_update is None: # Handle critical balance fetch error
                        current_balance_for_update = "Error fetching" # Placeholder for message

                    open_pos_text_update = "None"
                    if client: # Check if client is valid
                        if configs["mode"] == "live": # Ensure it's live mode for position check
                            # Use the new formatting option
                            with active_trades_lock: # Lock when reading active_trades
                                s_info_map_for_status_update = _build_symbol_info_map_from_active_trades(active_trades)
                                # Pass a copy of active_trades to avoid issues if it's modified elsewhere
                                # while get_open_positions is running, though the lock helps.
                                current_active_trades_copy = active_trades.copy()
                            open_pos_text_update = get_open_positions(client, format_for_telegram=True, active_trades_data=current_active_trades_copy, symbol_info_map=s_info_map_for_status_update)
                        else:
                            open_pos_text_update = "None (Backtest Mode)" # Should not happen if in live trading_loop
                    else:
                        open_pos_text_update = "N/A (Client not initialized)" # Should not happen

                    retrieved_bot_start_time_str = configs.get('bot_start_time_str', 'N/A')
                    
                    status_update_msg = build_startup_message(configs, current_balance_for_update, open_pos_text_update, retrieved_bot_start_time_str)
                    send_telegram_message(configs["telegram_bot_token"], configs["telegram_chat_id"], status_update_msg)
                    configs["last_startup_message"] = status_update_msg # Update for /command3
                    print("Telegram status update sent.")
                else:
                    print("Telegram not configured, skipping status update message.")

            print(f"Waiting for {configs['loop_delay_minutes']} m...")
            time.sleep(configs['loop_delay_minutes'] )
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
        except KeyboardInterrupt: print("\nBot stopped by user (Ctrl+C).")
        except Exception as e: print(f"\nCRITICAL UNEXPECTED ERROR IN LIVE TRADING: {e}"); traceback.print_exc()
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
                    print(f"{log_prefix} SL order for managed trade {symbol} is MISSING or incorrect. Attempting to re-place.")
                    new_sl_order = place_new_order(client, s_info_managed, 
                                                   "SELL" if side == "LONG" else "BUY", 
                                                   "STOP_MARKET", qty_for_new_sl_tp_orders, 
                                                   stop_price=target_sl_price, reduce_only=True)
                    if new_sl_order and new_sl_order.get('orderId'):
                        print(f"{log_prefix} Successfully re-placed SL order for {symbol}. New ID: {new_sl_order['orderId']}")
                        with active_trades_lock: # Lock to update shared active_trades
                             if symbol in active_trades_ref: # Check again as it might have been removed by another thread
                                active_trades_ref[symbol]['sl_order_id'] = new_sl_order['orderId']
                        # Send Telegram notification for successful re-placement
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"),
                                              f"✅ {log_prefix} Re-placed MISSING SL for managed {symbol} @ {target_sl_price:.{s_info_managed['pricePrecision']}f}")
                    else:
                        err_msg_sl = f"⚠️ {log_prefix} FAILED to re-place SL for managed {symbol}. Target SL: {target_sl_price:.{s_info_managed['pricePrecision']}f}. Details: {new_sl_order or 'No order object'}"
                        print(err_msg_sl)
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), err_msg_sl)

                if not tp_order_active:
                    print(f"{log_prefix} TP order for managed trade {symbol} is MISSING or incorrect. Attempting to re-place.")
                    new_tp_order = place_new_order(client, s_info_managed,
                                                   "SELL" if side == "LONG" else "BUY",
                                                   "TAKE_PROFIT_MARKET", qty_for_new_sl_tp_orders,
                                                   stop_price=target_tp_price, reduce_only=True)
                    if new_tp_order and new_tp_order.get('orderId'):
                        print(f"{log_prefix} Successfully re-placed TP order for {symbol}. New ID: {new_tp_order['orderId']}")
                        with active_trades_lock: # Lock to update shared active_trades
                            if symbol in active_trades_ref:
                                active_trades_ref[symbol]['tp_order_id'] = new_tp_order['orderId']
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"),
                                              f"✅ {log_prefix} Re-placed MISSING TP for managed {symbol} @ {target_tp_price:.{s_info_managed['pricePrecision']}f}")
                    else:
                        err_msg_tp = f"⚠️ {log_prefix} FAILED to re-place TP for managed {symbol}. Target TP: {target_tp_price:.{s_info_managed['pricePrecision']}f}. Details: {new_tp_order or 'No order object'}"
                        print(err_msg_tp)
                        send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), err_msg_tp)
            else:
                print(f"{log_prefix} {symbol} is UNMANAGED by the bot. Attempting to calculate and set SL/TP.")
                
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
                klines_df_unmanaged = get_historical_klines(client, symbol, limit=250) # Sufficient for EMA100/200
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
                # calculate_sl_tp_values(entry, side, ema100, df_klines, idx=-1)
                calc_sl_price, calc_tp_price = calculate_sl_tp_values(entry_price, side, ema100_val_unmanaged, klines_df_unmanaged)

                if calc_sl_price is None or calc_tp_price is None:
                    msg = (f"⚠️ {log_prefix} Failed to calculate SL/TP for UNMANAGED {symbol}. "
                           f"Entry: {entry_price:.{p_prec_unmanaged}f}, Side: {side}.")
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

                sanity_passed, sanity_reason = pre_order_sanity_checks(
                    symbol, side, entry_price, calc_sl_price, calc_tp_price, 
                    abs_position_qty, # Use actual position quantity
                    s_info_unmanaged, 
                    current_balance_for_check if current_balance_for_check is not None else 10000, # Use fetched balance or a placeholder
                    configs.get('risk_percent'), # This might need adjustment for unmanaged
                    configs, 
                    klines_df_unmanaged,
                    is_unmanaged_check=True # Pass True for unmanaged positions
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
                
                new_sl_unmanaged = place_new_order(client, s_info_unmanaged,
                                                   "SELL" if side == "LONG" else "BUY",
                                                   "STOP_MARKET", abs_position_qty,
                                                   stop_price=calc_sl_price, reduce_only=True)
                if new_sl_unmanaged and new_sl_unmanaged.get('orderId'):
                    sl_placed_unmanaged = True
                    print(f"{log_prefix} Successfully placed SL for UNMANAGED {symbol} @ {calc_sl_price:.{p_prec_unmanaged}f}")
                else:
                    msg = (f"⚠️ {log_prefix} FAILED to place SL for UNMANAGED {symbol}. "
                           f"Entry: {entry_price:.{p_prec_unmanaged}f}, Target SL: {calc_sl_price:.{p_prec_unmanaged}f}. Details: {new_sl_unmanaged or 'No order object'}")
                    print(msg)
                    send_telegram_message(configs.get("telegram_bot_token"), configs.get("telegram_chat_id"), msg)

                new_tp_unmanaged = place_new_order(client, s_info_unmanaged,
                                                   "SELL" if side == "LONG" else "BUY",
                                                   "TAKE_PROFIT_MARKET", abs_position_qty,
                                                   stop_price=calc_tp_price, reduce_only=True)
                if new_tp_unmanaged and new_tp_unmanaged.get('orderId'):
                    tp_placed_unmanaged = True
                    print(f"{log_prefix} Successfully placed TP for UNMANAGED {symbol} @ {calc_tp_price:.{p_prec_unmanaged}f}")
                else:
                    msg = (f"⚠️ {log_prefix} FAILED to place TP for UNMANAGED {symbol}. "
                           f"Entry: {entry_price:.{p_prec_unmanaged}f}, Target TP: {calc_tp_price:.{p_prec_unmanaged}f}. Details: {new_tp_unmanaged or 'No order object'}")
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
        # Pass configs, though for backtesting the -2015 error is less likely / relevant for IP alert
        live_balance = get_account_balance(client, configs) 
        # Handle if live_balance is None (critical error)
        if live_balance is None:
            print("Warning: Could not fetch live balance for backtest initialization due to an API error. Defaulting to 10000 USDT.")
            backtest_simulated_balance = 10000
        else:
            backtest_simulated_balance = live_balance if live_balance > 0 else 10000 # Default to 10k if no live balance
        print(f"Backtest initialized with starting balance: {backtest_simulated_balance:.2f} USDT")


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

    # --- Symbol Cooldown Check for Backtest ---
    # No lock needed for backtest symbol_last_trade_time if backtesting is sequential for this part.
    last_trade_ts_bt = symbol_last_trade_time.get(symbol)
    if last_trade_ts_bt:
        time_since_last_trade_seconds_bt = (backtest_current_time - last_trade_ts_bt).total_seconds()
        COOLDOWN_PERIOD_SECONDS_BT = 3600  # 1 hour
        if time_since_last_trade_seconds_bt < COOLDOWN_PERIOD_SECONDS_BT:
            # Don't print excessively in backtest, but log it.
            # print(f"[SIM-{backtest_current_time}] Symbol {symbol} on cooldown. Last trade {time_since_last_trade_seconds_bt:.0f}s ago. Ignored.")
            backtest_trade_log.append({
                "time": backtest_current_time, "symbol": symbol, "type": "COOLDOWN_SKIP",
                "reason": f"Last trade {time_since_last_trade_seconds_bt:.0f}s ago.",
                "remaining_cooldown_seconds": COOLDOWN_PERIOD_SECONDS_BT - time_since_last_trade_seconds_bt
            })
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
        
        # Update last trade time for cooldown in backtest
        # No lock needed for backtest if assuming single-threaded decision making per symbol iteration
        symbol_last_trade_time[symbol] = active_trades[symbol]['open_timestamp'] # Use the same timestamp
        print(f"[SIM-{backtest_current_time}] Updated last trade time for {symbol} to {symbol_last_trade_time[symbol]} for backtest cooldown.")
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
