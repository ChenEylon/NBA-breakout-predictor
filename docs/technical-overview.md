# NBA Breakout Player Predictor — Technical Overview

## Problem Statement

Every NBA offseason, teams and analysts attempt to identify which young players are poised for a significant statistical leap. This project builds a data-driven system to answer that question: given a player's current and historical performance, how likely are they to have a breakout season next year?

A **breakout** is defined as any of the following improvements versus the prior season (with minimum playing time of ≥40 GP and ≥20 MPG in both seasons):
- Points per game increases ≥20% AND the player scores ≥12 PPG in the breakout year
- PIE (Player Impact Estimate) increases ≥20%
- True Shooting % increases ≥20%

---

## Pipeline Architecture

The project is structured as a 5-phase pipeline:

### Phase 1 — Data Acquisition (`src/data/`)
- Fetches 15 seasons (2010-11 through 2025-26) of NBA per-game and advanced stats via the official `nba_api`
- Stores data in a local SQLite database (`data/nba.db`)
- Implements JSON caching and exponential-backoff retry logic to handle rate limiting
- Collects both **base stats** (PTS, AST, REB, STL, BLK, TOV, FG%, 3P%, FT%) and **advanced stats** (TS%, USG%, PIE, EFG%, Net/Off/Def Rating, AST%, REB%)

### Phase 2 — Feature Engineering (`src/features/engineer.py`)
- Merges base and advanced stats on `(player_id, season)`
- Computes **lag features**: prior season values for all 16 tracked stats
- Computes **YoY deltas and % changes** to capture momentum and growth trajectory
- Adds **demographic context**: age, years in league, team change indicator
- Applies qualifying filters to remove players with insufficient playing time
- Labels each row with the binary `breakout` target (no data leakage — only prior-season stats used as features)
- Output: ~60+ features saved to `data/features.parquet`

### Phase 3 — Model Training (`src/models/train.py`)
- Trains on seasons 2010-11 through 2024-25; validates on 2025-26 (held out)
- Three models trained and compared:
  - **XGBoost**: scale_pos_weight to handle class imbalance (~14% breakout rate)
  - **Random Forest**: balanced class weights, 300 trees
  - **LightGBM**: is_unbalance=True, selected as best model
- Decision threshold optimized per model using cross-validated OOF probabilities, maximizing precision subject to recall ≥ 15%
- All artifacts saved: `.joblib` models, `metrics.json`, `feature_columns.json`

### Phase 4 — SHAP Explainability (`src/explainability/shap_explainer.py`)
- Uses `shap.TreeExplainer` to compute feature-level contributions for every player
- For the top 20 breakout candidates, extracts the 8 most influential SHAP features
- Each feature is tagged with direction (positive/negative) and magnitude
- This surfaces *why* a player was flagged — e.g., "prior-season PIE and usage rate are the top positive drivers"

### Phase 5 — Streamlit Dashboard (`app/dashboard.py`)
Three interactive views:
1. **Top 20 Candidates**: Horizontal bar chart + full stats table, filterable by min probability and age range
2. **Player Deep Dive**: Per-player radar chart (current vs prior season stats, normalized), SHAP waterfall chart of top feature drivers
3. **Model Performance**: Metrics comparison table, grouped bar chart (precision/recall/AUC-ROC), precision-recall curves for all three models

---

## Model Results

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| **LightGBM ⭐ (selected)** | **0.563** | 0.243 | 0.340 | 0.655 |
| Random Forest | 0.550 | 0.297 | 0.386 | 0.717 |
| XGBoost | 0.500 | 0.162 | 0.245 | 0.677 |

**Selection rationale:** Precision was prioritized over recall. In a scouting context, surfacing a short list of high-confidence candidates is more actionable than a long list with many false positives. LightGBM's 56% precision means that roughly 1 in 2 flagged players genuinely breaks out — meaningfully better than the 14% base rate.

---

## Key Design Decisions

**No data leakage:** Features are strictly prior-season statistics. The breakout label is computed from current-season stats, which are never included as model inputs.

**Threshold optimization:** Rather than using the default 0.5 threshold, each model's optimal cutoff is found on cross-validated out-of-fold training predictions, then applied to the held-out validation season.

**Precision over recall:** For a recommendation tool, false positives (incorrectly flagged players) are more damaging than false negatives (missed breakouts). The model is calibrated accordingly.

**Caching at every stage:** Raw NBA API responses are cached as JSON, features are stored in Parquet, and models are serialized with joblib. The full pipeline only needs to run once; the dashboard reads from pre-built artifacts.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Data ingestion | `nba_api`, `pandas`, `SQLite3` |
| Feature engineering | `pandas`, `numpy` |
| Modeling | `scikit-learn`, `xgboost`, `lightgbm`, `joblib` |
| Explainability | `shap` (TreeExplainer) |
| Dashboard | `streamlit`, `plotly` |
| Language | Python 3.10+ |

---

## Limitations & Future Work

- **Model performance** reflects the inherent difficulty of predicting breakouts from box scores alone. Incorporating injury history, coaching changes, contract year status, and player tracking data (SportVU/Second Spectrum) would likely improve precision significantly.
- **Position data** is not currently used. `LeagueDashPlayerStats` doesn't return position; an enrichment step using `CommonAllPlayers` would allow position-stratified analysis and position-specific breakout definitions.
- **Hyperparameter tuning** is currently hand-set. Optuna-based Bayesian optimization is a natural next step.
- **Temporal cross-validation** (walk-forward validation across multiple held-out seasons) would give a more robust performance estimate than a single held-out season.
- **Feature expansion**: playoff minutes, preseason performance, contract year incentives, age-curve modeling (comparing players at the same career stage rather than same age).
