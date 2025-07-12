import pandas as pd
from imblearn.over_sampling import SMOTE

def get_label_distribution(df, label_column='pivot_label'):
    """
    Computes and logs the distribution of labels in the dataframe.
    """
    label_counts = df[label_column].value_counts()
    print("Label Distribution:")
    print(label_counts)
    
    if 1 in label_counts and 2 in label_counts and 0 in label_counts:
        imbalance_ratio_high = label_counts[0] / label_counts[1]
        imbalance_ratio_low = label_counts[0] / label_counts[2]
        print(f"Imbalance Ratio (None/High): {imbalance_ratio_high:.2f}")
        print(f"Imbalance Ratio (None/Low): {imbalance_ratio_low:.2f}")
    else:
        print("Could not compute imbalance ratio due to missing labels.")

def augment_positive_examples(df, n_left=5, n_right=5, atr_distance_factor=0.5, min_bar_gap=5):
    """
    Augments positive examples by relaxing pivot-detection parameters.
    """
    from app import generate_candidate_pivots, prune_and_label_pivots
    
    print("Augmenting positive examples with relaxed parameters...")
    augmented_df = generate_candidate_pivots(df.copy(), n_left=n_left, n_right=n_right)
    augmented_df = prune_and_label_pivots(augmented_df, 'atr_14', atr_distance_factor=atr_distance_factor, min_bar_gap=min_bar_gap)
    
    return augmented_df

def apply_smote(X, y):
    """
    Applies SMOTE to handle class imbalance.
    """
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled

def add_momentum_features(df):
    """
    Adds momentum features (delta-features) for atr, rsi, and ema slopes.
    """
    df['atr_delta'] = df['atr_14'].diff(periods=5)
    df['rsi_delta'] = df['rsi_14'].diff(periods=5)
    df['ema_slope_delta'] = df['ema20_ema50_norm_atr'].diff(periods=5)
    return df

def encode_contextual_indicators(df):
    """
    Encodes contextual indicators like session times and market volatility.
    """
    df['session'] = df['timestamp'].dt.hour.apply(
        lambda x: 'asian' if 23 <= x or x < 8 else ('london' if 8 <= x < 16 else 'ny')
    )
    df['volatility_regime'] = (df['atr_14'] > df['atr_14'].quantile(0.75)).astype(int)
    return df

def mark_categorical_features(df):
    """
    Marks session and regime labels as categorical for CatBoost.
    """
    df['session'] = df['session'].astype('category')
    df['volatility_regime'] = df['volatility_regime'].astype('category')
    return df

def get_tuned_hyperparameters():
    """
    Returns a dictionary of tuned hyperparameters for the CatBoost model.
    """
    return {
        'learning_rate': 0.03,
        'depth': 5,
        'l2_leaf_reg': 0.3,
        'iterations': 150,
        'class_weights': {0: 1, 1: 3, 2: 3}
    }

def plot_probability_distribution(y_true, y_pred_proba):
    """
    Plots the distribution of P_Swing_Score for each class.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame({'true_label': y_true, 'p_swing_score': y_pred_proba[:, 1]})
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='p_swing_score', hue='true_label', multiple='stack', bins=50)
    plt.title('Probability Distribution of P_Swing_Score')
    plt.show()

def get_dynamic_threshold(y_true, y_pred_proba):
    """
    Determines a dynamic threshold from the ROC or precision-recall curve.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve
    import numpy as np
    from sklearn.preprocessing import label_binarize

    # Binarize y_true for class 1 vs the rest
    y_true_binarized = label_binarize(y_true, classes=[0, 1, 2])[:, 1]

    fpr, tpr, thresholds_roc = roc_curve(y_true_binarized, y_pred_proba[:, 1])
    precision, recall, thresholds_prc = precision_recall_curve(y_true_binarized, y_pred_proba[:, 1])
    
    # Find the threshold that gives the best balance between TPR and FPR
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds_roc[optimal_idx]
    
    return optimal_threshold

def fallback_rule(df, p_high_threshold=0.3, p_low_threshold=0.3):
    """
    Implements a fallback rule for extreme swing indicator values.
    """
    # Example fallback rule: if ATR change is in the 95th percentile,
    # and both P_High and P_Low are below the threshold, trigger a manual check.
    atr_change_threshold = df['atr_14'].pct_change().quantile(0.95)
    
    manual_check_indices = df[
        (df['atr_14'].pct_change() > atr_change_threshold) &
        (df['P_High'] < p_high_threshold) &
        (df['P_Low'] < p_low_threshold)
    ].index
    
    return manual_check_indices
