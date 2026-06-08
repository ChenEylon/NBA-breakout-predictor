# NBA Breakout Player Predictor

**[🚀 Live Demo → nba-breakout-predictor.streamlit.app](https://nba-breakout-predictor.streamlit.app)**

End-to-end ML pipeline that identifies NBA players poised for a breakout season — using 15 years of historical data, ensemble models, and SHAP explainability.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)

---

## Overview

This project answers one question: **which NBA players are most likely to have a breakout season next year?**

The pipeline:
1. Pulls 15+ seasons of per-game and advanced stats from the official NBA Stats API
2. Engineers 60+ features capturing player trajectory (lag stats, year-over-year deltas, team changes, usage trends)
3. Trains and compares three ensemble models with class-imbalance handling and threshold optimization
4. Explains every prediction with SHAP values showing which features drove each player's score
5. Presents everything in an interactive Streamlit dashboard

---

## Architecture

```
NBA Stats API
     │
     ▼
 SQLite DB
(nba.db)
     │
     ▼
Feature Engineering
(60+ features: lag stats, YoY deltas, efficiency metrics)
     │
     ▼
Model Training
  ┌─────────┐  ┌───────────────┐  ┌──────────┐
  │ XGBoost │  │ Random Forest │  │ LightGBM │  ← best
  └─────────┘  └───────────────┘  └──────────┘
     │
     ▼
SHAP Explainability
     │
     ▼
Streamlit Dashboard
  • Top 20 Candidates  • Player Deep Dive  • Model Performance
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Data** | `nba_api`, `pandas`, `SQLite` |
| **Features** | `pandas`, `numpy` |
| **Modeling** | `scikit-learn`, `xgboost`, `lightgbm`, `joblib` |
| **Explainability** | `shap` (TreeExplainer) |
| **Dashboard** | `streamlit`, `plotly` |

---

## Model Results

Evaluated on the 2025-26 held-out validation season. Selection criterion: highest precision (minimizes false positives).

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| **LightGBM ⭐** | **0.563** | 0.243 | 0.340 | 0.655 |
| Random Forest | 0.550 | 0.297 | 0.386 | 0.717 |
| XGBoost | 0.500 | 0.162 | 0.245 | 0.677 |

Precision was prioritized over recall — a scouting tool should surface confident candidates rather than an exhaustive list. All three models beat the naive baseline (~14% breakout rate in the training data).

---

## Breakout Definition

A player is labeled a breakout if **any** of the following holds versus their prior season, subject to minimum playing time (≥40 GP, ≥20 MPG in both seasons):

- **Points:** ≥20% scoring increase AND ≥12 PPG in the breakout year
- **PIE:** Player Impact Estimate increases ≥20%
- **TS%:** True Shooting % increases ≥20%

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch 15 seasons of NBA data (caches to data/raw/, stores in data/nba.db)
python src/data/fetch.py

# 3. Engineer features and label breakout seasons
python src/features/engineer.py

# 4. Train models and evaluate on held-out 2025-26 season
python src/models/train.py

# 5. Launch the dashboard
streamlit run app/dashboard.py
```

> **Note:** Steps 2–4 take ~10 minutes on first run due to NBA API rate limiting. Subsequent runs use the local cache.

---

## Project Structure

```
NBA_star_finder/
├── app/
│   └── dashboard.py              # Streamlit dashboard — 3 interactive views
├── src/
│   ├── data/
│   │   ├── fetch.py              # NBA API ingestion with caching & retry logic
│   │   └── database.py           # SQLite schema and query helpers
│   ├── features/
│   │   └── engineer.py           # Lag features, YoY deltas, breakout labeling
│   ├── models/
│   │   └── train.py              # XGBoost / RF / LightGBM training & evaluation
│   └── explainability/
│       └── shap_explainer.py     # SHAP TreeExplainer — top drivers per player
├── models/                       # Saved .joblib model artifacts + metrics.json
├── data/                         # nba.db, features.parquet, raw JSON cache
├── reports/                      # Per-player scouting report JSON
├── config.py                     # Shared constants and thresholds
└── requirements.txt
```

---

## Limitations & Future Work

- **Model performance:** 56% precision reflects the inherent difficulty of predicting breakouts from box scores alone. Incorporating injury history, coaching changes, contract year incentives, and player tracking data would likely improve results significantly.
- **Position data:** `LeagueDashPlayerStats` does not return position; a future enrichment step using `CommonAllPlayers` would allow position-stratified analysis.
- **Hyperparameter tuning:** Model configs are currently hand-set; Optuna-based tuning is a natural next step.
- **Feature expansion:** Playoff minutes, preseason performance, and age curve modeling (comparing players at the same career stage) are strong candidate additions.
