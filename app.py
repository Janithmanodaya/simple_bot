import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import lightgbm as lgb
import optuna
import joblib
from binance.client import Client # Assuming you'll use python-binance
import time

# --- Configuration ---
# TODO: Move to a config file or pass as arguments
API_KEY = "YOUR_API_KEY"  # Replace with your Binance API key
API_SECRET = "YOUR_API_SECRET"  # Replace with your Binance API secret
PIVOT_N_LEFT = 3
PIVOT_N_RIGHT = 3
ATR_PERIOD = 14
MIN_ATR_DISTANCE = 1.0
MIN_BAR_GAP = 8
FIB_LEVEL_ENTRY = 0.618
FIB_LEVEL_TP1 = 0.382
FIB_LEVEL_TP2 = 0.618
FIB_LEVEL_TP3_EXTENSION = 1.618 # Example extension level

# --- 1. Data Collection & Labeling ---

def get_historical_bars(symbol, interval, start_str, end_str=None):
    """
    Pull historical OHLCV bars from Binance.
    """
    client = Client(API_KEY, API_SECRET)
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
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

def calculate_atr(df, period=ATR_PERIOD):
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

def prune_and_label_pivots(df, atr_distance_factor=MIN_ATR_DISTANCE, min_bar_gap=MIN_BAR_GAP):
    """
    Prunes candidate pivots and labels them.
    - Enforce ATR-distance >= 1 * ATR14 from the previous pivot.
    - Enforce bar gap >= 8 bars since the last confirmed pivot.
    - Label each pivot as is_swing_high, is_swing_low, or neither.
    """
    df['is_swing_high'] = 0 # 0: neither, 1: swing high, 2: swing low (for multiclass target)
    df['is_swing_low'] = 0  # For binary classifiers
    df['pivot_label'] = 0 # 0: none, 1: high, 2: low

    last_confirmed_pivot_idx = -1
    last_confirmed_pivot_price = 0
    last_confirmed_pivot_type = None # 'high' or 'low'

    # Ensure ATR is calculated
    if f'atr_{ATR_PERIOD}' not in df.columns:
        df = calculate_atr(df, ATR_PERIOD)

    # Iterate through candidate pivots
    for i in range(len(df)):
        atr_val = df[f'atr_{ATR_PERIOD}'].iloc[i]
        if pd.isna(atr_val):
            continue

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
        if last_confirmed_pivot_idx != -1:
            price_diff = abs(current_price - last_confirmed_pivot_price)
            if price_diff < (atr_distance_factor * df[f'atr_{ATR_PERIOD}'].iloc[last_confirmed_pivot_idx]): # Use ATR at time of last pivot
                 # Too close to the last pivot in terms of price * ATR
                continue


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


def simulate_fib_entries(df):
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

    # Ensure ATR is calculated
    if f'atr_{ATR_PERIOD}' not in df.columns:
        df = calculate_atr(df, ATR_PERIOD)

    pivots = df[(df['is_swing_high'] == 1) | (df['is_swing_low'] == 1)].copy()

    for i, pivot_row in pivots.iterrows():
        if pd.isna(df.loc[i, f'atr_{ATR_PERIOD}']):
            continue

        atr_at_pivot = df.loc[i, f'atr_{ATR_PERIOD}']
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

def engineer_pivot_features(df):
    """
    Engineers features for the pivot detection model.
    """
    # Ensure ATR is calculated
    if f'atr_{ATR_PERIOD}' not in df.columns:
        df = calculate_atr(df, ATR_PERIOD)
    atr_col = f'atr_{ATR_PERIOD}'

    # Volatility & Range
    df['range_atr_norm'] = (df['high'] - df['low']) / df[atr_col]

    # Trend & Momentum
    df['ema12'] = calculate_ema(df, 12)
    df['ema26'] = calculate_ema(df, 26)
    df['macd_line'] = df['ema12'] - df['ema26']
    df['macd_slope_atr_norm'] = df['macd_line'].diff() / df[atr_col]

    for n in [1, 3, 5]:
        df[f'return_{n}b_atr_norm'] = df['close'].pct_change(n) / df[atr_col] # (df['close'] / df['close'].shift(n) - 1)

    # Local Structure
    df['high_rank_7'] = df['high'].rolling(window=7).rank(pct=True) # Rank of current high among last 7 highs
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
    # Or 'is_swing_high', 'is_swing_low' for binary models
    feature_cols = [
        atr_col, 'range_atr_norm', 'macd_slope_atr_norm',
        'return_1b_atr_norm', 'return_3b_atr_norm', 'return_5b_atr_norm',
        'high_rank_7', 'bars_since_last_pivot', 'volume_spike_vs_avg', 'rsi_14'
    ]
    df.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle infinities from divisions
    return df, feature_cols

def engineer_entry_features(df, pivot_df_with_predictions):
    """
    Engineers features for the entry evaluation model.
    `pivot_df_with_predictions` should contain `P_swing` from the pivot model.
    This function assumes it's called for rows that are *potential* entries
    (i.e., after a pivot has been detected by the first model).
    """
    # This function will operate on a df that has potential entry signals
    # We need to merge P_swing from the pivot detection stage.
    # For simplicity, let's assume 'df' here is the original dataframe,
    # and we'll select rows corresponding to pivots later.

    atr_col = f'atr_{ATR_PERIOD}'
    if atr_col not in df.columns:
        df = calculate_atr(df, ATR_PERIOD) # Recalculate if not present

    # Features will be calculated at the time of the pivot.
    # We need: entry_price, pivot_price, SL_price (from simulation or actual)
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
    df['ema20_ema50_norm_atr'] = (df['ema20'] - df['ema50']) / df[atr_col]

    # Recent Behavior
    for n in [1, 3, 5]: # Returns *before* entry
        df[f'return_entry_{n}b'] = df['close'].pct_change(n) # Shift if these are prior to entry bar
    df[f'atr_{ATR_PERIOD}_change'] = df[atr_col].pct_change() # ATR change

    # Contextual Flags
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Regime cluster label (e.g. low/high vol from K-means) - Placeholder
    # This would require a separate clustering step on volatility or other features.
    df['vol_regime'] = 0 # Example: 0 for low, 1 for high. Needs actual implementation.

    # Meta-Feature: P_swing (This will be added when preparing data for the entry model)

    # Target: 'trade_outcome' (0=SL, 1=TP1, 2=TP2, 3=TP3)
    # Or binary: profit > 1R vs not (e.g., trade_outcome > 0)
    entry_feature_cols = [
        'ema20_ema50_norm_atr',
        'return_entry_1b', 'return_entry_3b', 'return_entry_5b',
        f'atr_{ATR_PERIOD}_change', 'hour_of_day', 'day_of_week', 'vol_regime',
        # These will be added specifically for pivot rows:
        # 'norm_dist_entry_pivot', 'norm_dist_entry_sl', 'P_swing'
    ]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df, entry_feature_cols


# --- 3. Model Training & Validation ---

def train_pivot_model(X_train, y_train, X_val, y_val, model_type='lgbm'):
    """Trains pivot detection model."""
    if model_type == 'lgbm':
        # Target: multiclass {none, high, low}
        # pivot_label: 0=none, 1=high, 2=low
        model = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
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
        model.fit(X_train, y_train_binary, eval_set=[(X_val, y_val_binary)], early_stopping_rounds=10, verbose=False)
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

    # --- Data Prep for Optuna Trial ---
    # Chronological split: 70% train, 15% validation, 15% test
    # For Optuna, we typically use train and validation. Test is for final hold-out.
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
        pivot_model.fit(X_pivot_train, y_pivot_train, eval_set=[(X_pivot_val, y_pivot_val)], early_stopping_rounds=5, verbose=False)
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
    entry_train_candidates['norm_dist_entry_pivot'] = (entry_train_candidates['entry_price_sim'] - entry_train_candidates.apply(lambda r: r['low'] if r['is_swing_low'] else r['high'], axis=1)) / entry_train_candidates[f'atr_{ATR_PERIOD}']
    entry_train_candidates['norm_dist_entry_sl'] = (entry_train_candidates['entry_price_sim'] - entry_train_candidates['sl_price_sim']).abs() / entry_train_candidates[f'atr_{ATR_PERIOD}']


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
                entry_model.fit(X_entry_train_sub, y_entry_train_sub, eval_set=[(X_entry_val_sub, y_entry_val_sub)], early_stopping_rounds=5, verbose=False)
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
    potential_pivots_val['norm_dist_entry_pivot'] = (potential_pivots_val['entry_price_sim'] - potential_pivots_val.apply(lambda r: r['low'] if r['is_swing_low'] else r['high'], axis=1)) / potential_pivots_val[f'atr_{ATR_PERIOD}']
    potential_pivots_val['norm_dist_entry_sl'] = (potential_pivots_val['entry_price_sim'] - potential_pivots_val['sl_price_sim']).abs() / potential_pivots_val[f'atr_{ATR_PERIOD}']

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


def run_optuna_tuning(df_processed, pivot_features, entry_features_base, n_trials=50):
    """Runs Optuna hyperparameter tuning."""
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_optuna(trial, df_processed, pivot_features, entry_features_base),
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
    potential_pivots_test['norm_dist_entry_pivot'] = (potential_pivots_test['entry_price_sim'] - potential_pivots_test.apply(lambda r: r['low'] if r['predicted_pivot_class'] == 2 else r['high'], axis=1)) / potential_pivots_test[f'atr_{ATR_PERIOD}']
    potential_pivots_test['norm_dist_entry_sl'] = (potential_pivots_test['entry_price_sim'] - potential_pivots_test['sl_price_sim']).abs() / potential_pivots_test[f'atr_{ATR_PERIOD}']

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
                          atr_col_name=f'atr_{ATR_PERIOD}'):
    """
    Runs a backtest for a specific scenario (Rule-based, ML Stage 1, Full ML).
    """
    print(f"\n--- Starting Backtest Scenario: {scenario_name} ---")

    # Determine test set (e.g., last 15% of data)
    train_val_size = int(0.85 * len(df_processed))
    df_test = df_processed.iloc[train_val_size:].copy()

    if df_test.empty:
        print(f"No data in test set for {scenario_name} backtest.")
        return {"scenario": scenario_name, "trades": 0, "win_rate": 0, "avg_r": 0, "profit_factor": 0, "max_dd_r": 0, "trade_frequency":0}

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
    test_period_days = (df_test.index.max() - df_test.index.min()).days +1 if not df_test.empty else 1
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
if __name__ == '__main__':
    # 1. Data Collection
    print("Fetching historical data...")
    # Use a longer period for real training, e.g., "1 Jan, 2022" to "1 Jan, 2023" for train
    # and "1 Jan, 2023" to "1 Apr, 2023" for validation/test like behavior initially
    # For quick test:
    historical_df = get_historical_bars("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE, "1 Jan, 2023", "1 May, 2023")
    if historical_df.empty:
        print("No data fetched. Exiting.")
        exit()

    print(f"Data fetched: {len(historical_df)} bars")

    # 2. Preprocessing & Labeling
    print("Calculating ATR...")
    historical_df = calculate_atr(historical_df)
    print("Generating candidate pivots...")
    historical_df = generate_candidate_pivots(historical_df)
    print("Pruning and labeling pivots...")
    historical_df = prune_and_label_pivots(historical_df)
    print("Simulating Fib entries and outcomes...")
    historical_df = simulate_fib_entries(historical_df) # This creates 'trade_outcome'
    historical_df.dropna(subset=[f'atr_{ATR_PERIOD}'], inplace=True) # Drop rows where ATR is NaN
    historical_df.reset_index(drop=True, inplace=True)

    print(f"Confirmed Swing Highs: {historical_df['is_swing_high'].sum()}")
    print(f"Confirmed Swing Lows: {historical_df['is_swing_low'].sum()}")
    print(f"Simulated Trade Outcomes:\n{historical_df['trade_outcome'].value_counts()}")


    # 3. Feature Engineering
    print("Engineering pivot features...")
    historical_df, pivot_feature_names = engineer_pivot_features(historical_df)
    print("Engineering entry features (base)...")
    # Entry features are engineered on the whole df, then selected for pivot rows
    historical_df, entry_feature_names_base = engineer_entry_features(historical_df, None) # P_swing added later

    # Drop rows with NaNs created by feature engineering (especially lookbacks)
    # A more robust approach would be to ensure enough data before feature engineering window
    # Or handle NaNs by imputation strategically.
    historical_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # print("NaNs per column after feature eng:")
    # print(historical_df[pivot_feature_names + entry_feature_names_base].isnull().sum())
    min_required_data_for_features = 30 # Example, depends on longest lookback
    df_processed = historical_df.iloc[min_required_data_for_features:].copy()
    df_processed.dropna(subset=pivot_feature_names, inplace=True) # Drop if any core pivot features are NaN
    # For entry features, NaNs are also problematic, handle during model prep.
    df_processed.reset_index(drop=True, inplace=True)

    if len(df_processed) < 100: # Arbitrary minimum
        print(f"Not enough processed data to proceed: {len(df_processed)} rows. Exiting.")
        exit()

    print(f"Processed data for modeling: {len(df_processed)} rows")

    # 4. Optuna Hyperparameter Tuning (Optional, can run separately or load best_params)
    print("Running Optuna for hyperparameter tuning...")
    # This can take time. For a quick run, reduce n_trials or use pre-defined params.
    # Ensure 'trade_outcome' and pivot labels are present in df_processed.
    # Ensure simulated entry/sl/tp prices are there for 'norm_dist' features in objective.
    try:
        best_hyperparams = run_optuna_tuning(df_processed.copy(), pivot_feature_names, entry_feature_names_base, n_trials=10) # Reduced for speed
        print("Best Hyperparameters from Optuna:", best_hyperparams)
    except Exception as e:
        print(f"Optuna tuning failed: {e}. Using default parameters.")
        # Define some default best_params if Optuna fails or is skipped
        best_hyperparams = {
            'pivot_model_type': 'lgbm', 'pivot_num_leaves': 30, 'pivot_learning_rate': 0.05,
            'pivot_max_depth': 7, 'entry_model_type': 'lgbm', 'entry_num_leaves': 30,
            'entry_learning_rate': 0.05, 'entry_max_depth': 7,
            'p_swing_threshold': 0.6, 'profit_threshold': 0.6
        }


    # 5. Train Final Models with Best Hyperparameters on Train+Val data
    print("Training final models with best hyperparameters...")
    # Split: 70% train, 15% val, 15% test. For final models, train on Train+Val (85%)
    final_train_size = int(0.85 * len(df_processed))
    df_final_train = df_processed.iloc[:final_train_size].copy()
    # df_final_test = df_processed.iloc[final_train_size:].copy() # Used in full_backtest

    # --- Pivot Model ---
    X_p_final_train = df_final_train[pivot_feature_names].fillna(-1)
    y_p_final_train = df_final_train['pivot_label']

    if best_hyperparams['pivot_model_type'] == 'lgbm':
        final_pivot_model = lgb.LGBMClassifier(
            num_leaves=best_hyperparams['pivot_num_leaves'],
            learning_rate=best_hyperparams['pivot_learning_rate'],
            max_depth=best_hyperparams['pivot_max_depth'],
            class_weight='balanced', random_state=42, n_estimators=150 # More estimators for final model
        )
    else: # rf
        final_pivot_model = RandomForestClassifier(
            n_estimators=150, max_depth=best_hyperparams['pivot_max_depth'],
            class_weight='balanced', random_state=42
        )
    final_pivot_model.fit(X_p_final_train, y_p_final_train)
    print("Final Pivot Model trained.")
    save_model(final_pivot_model, "pivot_detector_model.joblib")

    # --- Entry Model ---
    # Predict P_swing on the final_train_data to use as meta-feature
    p_swing_final_train_all_classes = final_pivot_model.predict_proba(X_p_final_train)
    df_final_train['P_swing'] = np.max(p_swing_final_train_all_classes[:,1:], axis=1)

    # Filter for actual pivots that had a simulated trade and would pass the P_swing threshold
    entry_final_train_candidates = df_final_train[
        (df_final_train['pivot_label'].isin([1, 2])) &
        (df_final_train['trade_outcome'] != -1) &
        (df_final_train['P_swing'] >= best_hyperparams['p_swing_threshold'])
    ].copy()

    if len(entry_final_train_candidates) < 50:
        print("Not enough candidates for final entry model training. Skipping entry model.")
        final_entry_model = None
    else:
        entry_final_train_candidates['norm_dist_entry_pivot'] = (entry_final_train_candidates['entry_price_sim'] - entry_final_train_candidates.apply(lambda r: r['low'] if r['is_swing_low'] else r['high'], axis=1)) / entry_final_train_candidates[f'atr_{ATR_PERIOD}']
        entry_final_train_candidates['norm_dist_entry_sl'] = (entry_final_train_candidates['entry_price_sim'] - entry_final_train_candidates['sl_price_sim']).abs() / entry_final_train_candidates[f'atr_{ATR_PERIOD}']

        X_e_final_train = entry_final_train_candidates[entry_feature_names_base + ['P_swing', 'norm_dist_entry_pivot', 'norm_dist_entry_sl']].fillna(-1)
        y_e_final_train = (entry_final_train_candidates['trade_outcome'] > 0).astype(int) # Binary profitable

        if len(X_e_final_train) > 0 and len(y_e_final_train.unique()) > 1:
            if best_hyperparams['entry_model_type'] == 'lgbm':
                final_entry_model = lgb.LGBMClassifier(
                    num_leaves=best_hyperparams['entry_num_leaves'],
                    learning_rate=best_hyperparams['entry_learning_rate'],
                    max_depth=best_hyperparams['entry_max_depth'],
                    class_weight='balanced', random_state=42, n_estimators=150
                )
            else: # rf
                final_entry_model = RandomForestClassifier(
                    n_estimators=150, max_depth=best_hyperparams['entry_max_depth'],
                    class_weight='balanced', random_state=42
                )
            final_entry_model.fit(X_e_final_train, y_e_final_train)
            print("Final Entry Model trained.")
            save_model(final_entry_model, "entry_evaluator_model.joblib")
        else:
            print("Not enough data or variance to train final entry model.")
            final_entry_model = None


    # 6. Full Backtest on Test Set (using models trained on Train+Val)
    # This section is now expanded to run multiple scenarios for comparison.

    backtest_results_summary = []

    # Scenario 1: Pure Rule-Based Baseline
    # For this, we don't need ML models. We need a way to simulate rule-based pivot trades.
    # The `run_backtest_scenario` is designed to handle this if `pivot_model` and `entry_model` are None.
    # However, it currently relies on `trade_outcome` which is from `simulate_fib_entries`.
    # This is acceptable for a conceptual backtest as `simulate_fib_entries` IS the rule-based simulation.
    print("\n--- Running Backtest Scenarios ---")
    baseline_results = run_backtest_scenario(
        scenario_name="Rule-Based Baseline",
        df_processed=df_processed.copy(), # Pass a copy
        pivot_model=None, # No ML model for baseline
        entry_model=None, # No ML model for baseline
        best_params=best_hyperparams, # Thresholds not used, but pass for consistency
        pivot_features=pivot_feature_names,
        entry_features_base=entry_feature_names_base
    )
    backtest_results_summary.append(baseline_results)

    # Scenario 2: Stage 1 ML Only (Pivot Filter)
    if final_pivot_model:
        stage1_ml_results = run_backtest_scenario(
            scenario_name="ML Stage 1 (Pivot Filter)",
            df_processed=df_processed.copy(),
            pivot_model=final_pivot_model, # Use trained pivot model
            entry_model=None, # No entry model for stage 1
            best_params=best_hyperparams, # Uses p_swing_threshold
            pivot_features=pivot_feature_names,
            entry_features_base=entry_feature_names_base
        )
        backtest_results_summary.append(stage1_ml_results)
    else:
        print("Skipping ML Stage 1 backtest as pivot model was not trained.")

    # Scenario 3: Full ML Pipeline
    if final_pivot_model and final_entry_model:
        full_ml_results = run_backtest_scenario(
            scenario_name="Full ML Pipeline",
            df_processed=df_processed.copy(),
            pivot_model=final_pivot_model,
            entry_model=final_entry_model,
            best_params=best_hyperparams, # Uses p_swing_threshold and profit_threshold
            pivot_features=pivot_feature_names,
            entry_features_base=entry_feature_names_base
        )
        backtest_results_summary.append(full_ml_results)
    else:
        print("Skipping Full ML Pipeline backtest as one or both models were not trained.")

    print("\n--- Backtest Scenarios Summary ---")
    for result in backtest_results_summary:
        print(f"\nScenario: {result['scenario']}")
        print(f"  Trades: {result['trades']}, Win Rate: {result['win_rate']:.3f}, Avg R: {result['avg_r']:.3f}")
        print(f"  Profit Factor: {result['profit_factor']:.3f}, Max Drawdown (R): {result['max_dd_r']:.3f}R, Trade Freq: {result['trade_frequency']:.2f} trades/day")

    print("\napp.py script finished.")
