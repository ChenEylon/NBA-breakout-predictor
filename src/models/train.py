"""
Phase 3: Model Training & Evaluation
Trains XGBoost, Random Forest, and LightGBM on breakout prediction.
Optimizes decision threshold for precision. Saves best model.

Usage:
    python src/models/train.py
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

from src.features.engineer import load_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = [str(y) + "-" + str(y + 1)[2:] for y in range(2010, 2025)]
VAL_SEASON = "2025-26"  # predict 2026-27 breakouts using 2025-26 as latest features

RANDOM_STATE = 42


def prepare_data(df: pd.DataFrame, feature_cols: list[str]):
    """Subset and impute features; returns (X, y, available_cols)."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    y = df["breakout"].copy()
    X = X.fillna(X.median(numeric_only=True))
    return X, y, available


def best_precision_threshold(y_true, y_prob, min_recall=0.15):
    """Find threshold maximizing precision subject to recall >= min_recall."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    valid = rec[:-1] >= min_recall
    if not valid.any():
        return 0.5
    best_idx = np.argmax(prec[:-1][valid])
    return float(thresholds[valid][best_idx])


def evaluate_at_threshold(y_true, y_prob, threshold):
    """Compute precision/recall/F1/AUC at a fixed decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "threshold": threshold,
    }


def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier with scale_pos_weight to handle class imbalance."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=neg / pos,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    """Train a LightGBM classifier with is_unbalance=True for imbalanced labels."""
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        is_unbalance=True,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def run():
    """Train all three models, evaluate on the held-out validation season, and save artifacts."""
    df = load_features()
    feat_cols = get_feature_columns()

    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    val_df = df[df["season"] == VAL_SEASON]

    X_train, y_train, available_feats = prepare_data(train_df, feat_cols)
    X_val, y_val, _ = prepare_data(val_df, feat_cols)

    log.info(f"Train: {len(X_train)} rows | Val: {len(X_val)} rows | Features: {len(available_feats)}")
    log.info(f"Train breakout rate: {y_train.mean():.3f} | Val: {y_val.mean():.3f}")

    model_fns = {
        "xgboost": train_xgboost,
        "random_forest": train_random_forest,
        "lightgbm": train_lightgbm,
    }

    results = {}
    trained_models = {}

    for name, fn in model_fns.items():
        log.info(f"Training {name}...")
        model = fn(X_train, y_train)
        trained_models[name] = model

        val_prob = model.predict_proba(X_val)[:, 1]

        # CV oof probabilities for threshold tuning on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        train_prob_oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold = best_precision_threshold(y_train, train_prob_oof)

        metrics = evaluate_at_threshold(y_val, val_prob, threshold)
        results[name] = metrics
        log.info(f"{name}: precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
                 f"f1={metrics['f1']:.3f} auc={metrics['auc_roc']:.3f} threshold={threshold:.3f}")

    # Pick best model by precision
    best_name = max(results, key=lambda n: results[n]["precision"])
    log.info(f"\nBest model: {best_name} (precision={results[best_name]['precision']:.3f})")

    best_model = trained_models[best_name]
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    (MODELS_DIR / "best_model_name.txt").write_text(best_name)
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(available_feats))

    # Save all metrics
    metrics_out = {"models": results, "best": best_name, "val_season": VAL_SEASON}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    # Save all models for dashboard comparison
    for name, model in trained_models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    log.info(f"Saved to {MODELS_DIR}")

    # Show top 20 breakout candidates for current season (using best model on 2024-25)
    current_df = df[df["season"] == VAL_SEASON].copy()
    X_current, _, _ = prepare_data(current_df, feat_cols)
    probs = trained_models[best_name].predict_proba(X_current)[:, 1]
    current_df = current_df.copy()
    current_df["breakout_prob"] = probs
    top20 = current_df.sort_values("breakout_prob", ascending=False).head(20)
    print("\n=== Top 20 Breakout Candidates (2025 → 2026 season) ===")
    print(top20[["name", "team_abbr", "age", "pts", "ts_pct", "pie", "breakout_prob", "breakout"]].to_string())


if __name__ == "__main__":
    run()
