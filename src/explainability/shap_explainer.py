"""
Phase 4a: SHAP Explainability
Computes SHAP values for the best model and extracts per-player feature importance.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.features.engineer import load_features, get_feature_columns
from config import CURRENT_SEASON, TOP_N_CANDIDATES, TOP_N_SHAP_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parents[2] / "models"


def load_best_model():
    """Load the saved best model and its feature column list."""
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    feature_cols = json.loads((MODELS_DIR / "feature_columns.json").read_text())
    return model, feature_cols


def prepare_current_season(feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter features to the current season and align columns to the trained feature set."""
    df = load_features()
    current = df[df["season"] == CURRENT_SEASON].copy()
    available = [c for c in feature_cols if c in current.columns]
    X = current[available].fillna(current[available].median(numeric_only=True))
    return current, X


def compute_shap(model, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values using TreeExplainer; returns the positive-class array."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # For binary classifiers some libraries return list of [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values


def get_top_shap_features(shap_row: np.ndarray, feature_names: list[str], n: int = TOP_N_SHAP_FEATURES) -> list[dict]:
    """Return the top-n features sorted by absolute SHAP value with direction labels."""
    pairs = sorted(zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)[:n]
    return [
        {
            "feature": name,
            "shap_value": float(val),
            "direction": "positive" if val > 0 else "negative",
        }
        for name, val in pairs
    ]


def build_player_context(row: pd.Series, breakout_prob: float, shap_features: list[dict]) -> dict:
    """Assemble a player context dict combining stats, prediction, and SHAP drivers."""
    return {
        "name": row["name"],
        "team": row.get("team_abbr", "N/A"),
        "age": row.get("age"),
        "season": CURRENT_SEASON,
        "pts": row.get("pts"),
        "ts_pct": row.get("ts_pct"),
        "usg_pct": row.get("usg_pct"),
        "pie": row.get("pie"),
        "years_in_league": int(row.get("years_in_league", 0)),
        "team_changed": bool(row.get("team_changed", 0)),
        "breakout_prob": round(float(breakout_prob), 4),
        "top_shap_features": shap_features,
    }


def run() -> list[dict]:
    """Return SHAP-enriched player context dicts for the top-N breakout candidates."""
    model, feature_cols = load_best_model()
    current_df, X = prepare_current_season(feature_cols)

    probs = model.predict_proba(X)[:, 1]
    current_df = current_df.copy()
    current_df["breakout_prob"] = probs

    top20_df = current_df.sort_values("breakout_prob", ascending=False).head(TOP_N_CANDIDATES)
    top20_idx = top20_df.index

    shap_values = compute_shap(model, X)
    X_indexed = X.reset_index(drop=True)
    current_df_reset = current_df.reset_index(drop=True)
    top20_positions = [current_df.index.get_loc(i) for i in top20_idx]

    candidates = []
    for pos in top20_positions:
        row = current_df.iloc[pos]
        prob = float(probs[pos])
        shap_row = shap_values[pos]
        shap_feats = get_top_shap_features(shap_row, list(X.columns))
        ctx = build_player_context(row, prob, shap_feats)
        candidates.append(ctx)

    log.info(f"Built SHAP context for {len(candidates)} candidates")
    return candidates


if __name__ == "__main__":
    candidates = run()
    for c in candidates[:3]:
        print(f"\n{c['name']} ({c['team']}) — breakout prob: {c['breakout_prob']:.1%}")
        for feat in c["top_shap_features"][:4]:
            print(f"  {feat['feature']:30s}  SHAP={feat['shap_value']:+.4f}  ({feat['direction']})")
