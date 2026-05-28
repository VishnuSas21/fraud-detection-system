# ============================================================
# FRAUD DETECTION — XGBoost Model + Feature Importance
# Fraud Detection System | Vishnu Saseendran
# ============================================================
# Purpose:
#   1. Load transaction data from MySQL
#   2. Engineer fraud signals as features
#   3. Train XGBoost with class imbalance handling
#   4. Compare binary flag features vs raw/engineered features
#   5. Extract feature importance to validate SQL weights
#   6. Threshold analysis — precision/recall trade-off
#   7. Hybrid decision engine performance
# ============================================================

import pandas as pd
import numpy as np
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, confusion_matrix)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ─────────────────────────────────────────────
def load_data():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',          # update with your MySQL password
        database='fraud_db'
    )
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()

    df['txn_time'] = pd.to_datetime(df['txn_time'])
    print(f"Loaded: {len(df):,} rows | Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    return df


# ── 2. Feature Engineering ───────────────────────────────────
def engineer_features(df):

    # Daily velocity per user
    daily_vel = df.groupby(
        ['user_id', df['txn_time'].dt.date]
    )['txn_id'].transform('count')

    # Amount vs user's own average
    user_avg = df.groupby('user_id')['amount'].transform('mean')

    features = pd.DataFrame({
        # Raw continuous features — better for XGBoost than binary flags
        'amount':              df['amount'],
        'failed_logins':       df['failed_logins'],
        'account_age':         df['account_age'],
        'daily_velocity':      daily_vel,
        'amt_vs_avg_ratio':    (df['amount'] / user_avg.replace(0, np.nan)).fillna(1),
        'txn_hour':            df['txn_time'].dt.hour,

        # Binary categorical features
        'kyc_verified':        df['kyc_verified'],
        'is_foreign':          (df['country'] != df['txn_location']).astype(int),
        'flag_cnp':            (df['txn_type'] == 'CNP').astype(int),
        'flag_risky_merchant': df['merchant_cat'].isin(['Crypto','Gaming']).astype(int),
        'flag_high_risk_seg':  (df['risk_segment'] == 'High').astype(int),
    })

    print(f"Features engineered: {list(features.columns)}")
    print(f"\nFeature statistics:")
    print(features.describe().round(2))
    return features


# ── 3. Train XGBoost Model ───────────────────────────────────
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class imbalance ratio
    # Tells XGBoost: missing 1 fraud = same cost as 63 wrong legitimate txns
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nClass imbalance ratio: {fraud_ratio:.1f}x")
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Fraud in train: {y_train.sum()} | Fraud in test: {y_test.sum()}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,            # captures multi-signal fraud combinations
        learning_rate=0.05,     # slower learning = more accurate
        scale_pos_weight=fraud_ratio,   # handles class imbalance
        random_state=42,
        eval_metric='aucpr',    # optimise for precision-recall (better for fraud)
        subsample=0.8,          # prevents overfitting
        colsample_bytree=0.8,   # prevents overfitting
        base_score=0.5          # required for SHAP compatibility
    )

    model.fit(X_train, y_train)
    print("\nModel trained successfully")
    return model, X_test, y_test


# ── 4. Evaluate Model ────────────────────────────────────────
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*55)
    print("MODEL PERFORMANCE")
    print("="*55)
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob):.4f}")
    print(f"AUC-PR:   {average_precision_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (Legit approved):  {cm[0][0]:,}")
    print(f"  False Positives (Legit blocked):    {cm[0][1]:,}")
    print(f"  False Negatives (Fraud missed):     {cm[1][0]:,}")
    print(f"  True Positives  (Fraud caught):     {cm[1][1]:,}")

    print("\nNote: Near-perfect scores reflect synthetic data leakage.")
    print("Fraud labels were generated from same features used in training.")
    print("Real-world models achieve 0.75-0.90 AUC-ROC on production data.")

    return y_prob


# ── 5. Feature Importance → SQL Weight Validation ───────────
def feature_importance_analysis(model, feature_names):

    importance = pd.DataFrame({
        'feature':    feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Scale to SQL weights (5-35 range)
    min_w, max_w = 5, 35
    active = importance[importance['importance'] > 0].copy()
    active['sql_weight'] = (
        min_w +
        (active['importance'] - active['importance'].min()) /
        (active['importance'].max() - active['importance'].min()) *
        (max_w - min_w)
    ).round(0).astype(int)

    importance = importance.merge(
        active[['feature', 'sql_weight']], on='feature', how='left'
    ).fillna(0)
    importance['sql_weight'] = importance['sql_weight'].astype(int)

    print("\n" + "="*55)
    print("FEATURE IMPORTANCE → SQL WEIGHT MAPPING")
    print("="*55)
    print(importance.to_string(index=False))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'], importance['importance'], color='steelblue')
    plt.xlabel('XGBoost Importance Score')
    plt.title('Feature Importance — Fraud Detection Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved: feature_importance.png")

    return importance


# ── 6. Threshold Analysis ────────────────────────────────────
def threshold_analysis(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:, 1]
    total_fraud = y_test.sum()

    print("\n" + "="*70)
    print("THRESHOLD ANALYSIS — Precision / Recall Trade-off")
    print("="*70)
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>8} {'FP Rate':>10} {'Caught':>8} {'Missed':>8}")
    print("-"*70)

    results = []
    for t in [i/100 for i in range(10, 91, 5)]:
        y_pred_t = (y_prob >= t).astype(int)
        flagged = y_pred_t.sum()
        if flagged == 0:
            continue

        tp = ((y_pred_t == 1) & (y_test == 1)).sum()
        fp = ((y_pred_t == 1) & (y_test == 0)).sum()
        fn = total_fraud - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / total_fraud if total_fraud > 0 else 0
        f1        = (2 * precision * recall /
                     (precision + recall)) if (precision + recall) > 0 else 0
        fp_rate   = fp / flagged * 100

        results.append({
            'threshold': t, 'precision': precision,
            'recall': recall, 'f1': f1, 'fp_rate': fp_rate,
            'caught': tp, 'missed': fn
        })

        print(f"{t:>10.2f} {precision:>9.1%} {recall:>9.1%} "
              f"{f1:>8.3f} {fp_rate:>9.1f}% {tp:>8} {fn:>8}")

    best = max(results, key=lambda x: x['f1'])
    print(f"\nBest F1 threshold: {best['threshold']:.2f} "
          f"(F1={best['f1']:.3f}, Recall={best['recall']:.1%})")

    return results


# ── Main ─────────────────────────────────────────────────────
if __name__ == '__main__':

    # Load data
    df = load_data()

    # Engineer features
    X = engineer_features(df)
    y = df['is_fraud']

    # Train model
    model, X_test, y_test = train_model(X, y)

    # Evaluate
    y_prob = evaluate_model(model, X_test, y_test)

    # Feature importance → SQL weights
    importance = feature_importance_analysis(model, list(X.columns))

    # Threshold analysis
    threshold_results = threshold_analysis(model, X_test, y_test)

    print("\n" + "="*55)
    print("PROJECT COMPLETE")
    print("="*55)
    print("Files generated:")
    print("  feature_importance.png — XGBoost importance chart")
    print("\nNext step: Run sql/04_hybrid_decision_engine.sql")
    print("to see the final rule engine performance.")
