"""
Phase 2: Feature Engineering & Target Labeling
Merges base + advanced stats, computes lag features, YoY deltas,
and labels breakout seasons.

Usage:
    python src/features/engineer.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.database import load_season_stats, load_advanced_stats
from config import MIN_GP, MIN_MPG, BREAKOUT_DELTA_PCT, BREAKOUT_MIN_PTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURES_PATH = Path(__file__).parents[2] / "data" / "features.parquet"

MIN_MIN = MIN_MPG  # local alias used in qualifying filter below

PTS_BREAKOUT_INCREASE = BREAKOUT_DELTA_PCT
PTS_BREAKOUT_MIN = BREAKOUT_MIN_PTS
PER_BREAKOUT_INCREASE = BREAKOUT_DELTA_PCT
TS_BREAKOUT_INCREASE = BREAKOUT_DELTA_PCT

LAG_STATS = ["pts", "ast", "reb", "stl", "blk", "tov", "fg_pct", "fg3_pct", "ft_pct",
             "min", "ts_pct", "usg_pct", "pie", "efg_pct", "net_rating", "off_rating"]


def season_to_year(season: str) -> int:
    """'2019-20' → 2019 (the year the season started)."""
    return int(season.split("-")[0])


def build_features() -> pd.DataFrame:
    """Merge base + advanced stats, engineer lag/delta features, and label breakout seasons."""
    base = load_season_stats()
    adv = load_advanced_stats()

    df = base.merge(adv, on=["player_id", "season"], how="left")

    df["season_year"] = df["season"].apply(season_to_year)
    df = df.sort_values(["player_id", "season_year"]).reset_index(drop=True)

    # --- Lag features (previous season values) ---
    for stat in LAG_STATS:
        if stat in df.columns:
            df[f"prev_{stat}"] = df.groupby("player_id")[stat].shift(1)
            df[f"delta_{stat}"] = df[stat] - df[f"prev_{stat}"]
            df[f"pct_change_{stat}"] = df[f"delta_{stat}"] / (df[f"prev_{stat}"].abs() + 1e-6)

    # --- Lag team to detect team change ---
    df["prev_team_id"] = df.groupby("player_id")["team_id"].shift(1)
    df["team_changed"] = (df["team_id"] != df["prev_team_id"]).astype(int)
    df.loc[df["prev_team_id"].isna(), "team_changed"] = 0

    # --- Years in league (proxy: season_year - first season_year for player) ---
    df["first_season_year"] = df.groupby("player_id")["season_year"].transform("min")
    df["years_in_league"] = df["season_year"] - df["first_season_year"]

    # --- Position one-hot (basic: G, F, C from nba_api) ---
    # nba_api doesn't return position in LeagueDashPlayerStats — skip for now
    # Position can be added from CommonAllPlayers in a future enrichment step

    # --- Apply qualifying filters ---
    prev_gp = df.groupby("player_id")["gp"].shift(1)
    prev_min = df.groupby("player_id")["min"].shift(1)
    qualifies = (
        (df["gp"] >= MIN_GP) &
        (df["min"] >= MIN_MIN) &
        (prev_gp >= MIN_GP) &
        (prev_min >= MIN_MIN)
    )
    df = df[qualifies].copy()

    # --- Target: Breakout ---
    pts_breakout = (
        (df["pct_change_pts"] >= PTS_BREAKOUT_INCREASE) &
        (df["pts"] >= PTS_BREAKOUT_MIN)
    )
    pie_breakout = df.get("pct_change_pie", pd.Series(False, index=df.index)) >= PER_BREAKOUT_INCREASE
    ts_breakout = df.get("pct_change_ts_pct", pd.Series(False, index=df.index)) >= TS_BREAKOUT_INCREASE

    df["breakout"] = (pts_breakout | pie_breakout | ts_breakout).astype(int)

    log.info(f"Feature set: {len(df)} rows, {df.columns.tolist()}")
    log.info(f"Breakout rate: {df['breakout'].mean():.3f} ({df['breakout'].sum()} breakout seasons)")

    return df


def get_feature_columns() -> list[str]:
    """Return the feature column names used for model training (prior-season stats only — no leakage)."""
    # Only use PRIOR season stats + demographics — never current season.
    # Current stats would leak breakout labels since breakout is computed from current stats.
    lag_feats = [f"prev_{s}" for s in LAG_STATS]
    context_feats = ["age", "years_in_league", "team_changed"]
    return lag_feats + context_feats


def save_features(df: pd.DataFrame):
    """Persist the feature DataFrame to Parquet."""
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)
    log.info(f"Saved features to {FEATURES_PATH}")


def load_features() -> pd.DataFrame:
    """Load the pre-built feature DataFrame from Parquet."""
    return pd.read_parquet(FEATURES_PATH)


if __name__ == "__main__":
    df = build_features()
    save_features(df)

    print("\n=== Feature Sample (2024-25 season, top pts scorers) ===")
    sample = df[df["season"] == "2024-25"].sort_values("pts", ascending=False).head(10)
    print(sample[["name", "pts", "prev_pts", "pct_change_pts", "ts_pct", "pie", "breakout"]].to_string())

    print(f"\nBreakout rate by season:")
    print(df.groupby("season")["breakout"].agg(["sum", "mean"]).rename(
        columns={"sum": "breakouts", "mean": "rate"}
    ).to_string())
