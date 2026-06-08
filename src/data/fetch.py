"""
Phase 1: Data Acquisition
Fetches 15 seasons of NBA stats (2010-11 through 2024-25) from nba_api,
caches raw JSON to data/raw/, and stores results in SQLite.

Usage:
    python src/data/fetch.py
"""

import json
import time
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    CommonAllPlayers,
)
from nba_api.stats.library.parameters import SeasonTypeAllStar

from src.data.database import init_db, upsert_players, upsert_season_stats, upsert_advanced_stats
from config import SEASONS, RATE_LIMIT_SLEEP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _sleep():
    time.sleep(RATE_LIMIT_SLEEP)


def _cache_path(name: str) -> Path:
    return RAW_DIR / f"{name}.json"


def _load_cache(name: str):
    """Return cached JSON data for key, or None if not cached."""
    p = _cache_path(name)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_cache(name: str, data):
    """Write data to the JSON cache."""
    _cache_path(name).write_text(json.dumps(data))


def fetch_with_retry(fn, retries=5, backoff=2.0):
    """Call fn() with exponential backoff on failure."""
    delay = RATE_LIMIT_SLEEP
    for attempt in range(retries):
        try:
            result = fn()
            _sleep()
            return result
        except Exception as e:
            if attempt == retries - 1:
                raise
            log.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= backoff


def fetch_all_players() -> pd.DataFrame:
    """Fetch the full NBA player roster from CommonAllPlayers, with caching."""
    cache_key = "all_players"
    cached = _load_cache(cache_key)
    if cached:
        log.info("players: loaded from cache")
        return pd.DataFrame(cached)

    log.info("Fetching all players from CommonAllPlayers...")
    data = fetch_with_retry(
        lambda: CommonAllPlayers(is_only_current_season=0, league_id="00", season="2024-25")
        .get_data_frames()[0]
    )

    cols = {
        "PERSON_ID": "player_id",
        "DISPLAY_FIRST_LAST": "name",
        "TEAM_ABBREVIATION": "team_abbr",
        "PLAYERCODE": "player_code",
    }
    df = data.rename(columns=cols)[["player_id", "name"]].drop_duplicates()
    _save_cache(cache_key, df.to_dict(orient="records"))
    log.info(f"Fetched {len(df)} players")
    return df


def fetch_season_base(season: str) -> pd.DataFrame:
    """Fetch per-game base stats for all players in a season, with caching."""
    cache_key = f"base_{season}"
    cached = _load_cache(cache_key)
    if cached:
        return pd.DataFrame(cached)

    log.info(f"Fetching base stats: {season}")
    data = fetch_with_retry(
        lambda: LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
        ).get_data_frames()[0]
    )

    col_map = {
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "name",
        "TEAM_ID": "team_id",
        "TEAM_ABBREVIATION": "team_abbr",
        "AGE": "age",
        "GP": "gp",
        "MIN": "min",
        "PTS": "pts",
        "AST": "ast",
        "REB": "reb",
        "STL": "stl",
        "BLK": "blk",
        "TOV": "tov",
        "FG_PCT": "fg_pct",
        "FG3_PCT": "fg3_pct",
        "FT_PCT": "ft_pct",
    }
    keep = list(col_map.keys())
    df = data[keep].rename(columns=col_map)
    df["season"] = season

    _save_cache(cache_key, df.to_dict(orient="records"))
    return df


def fetch_season_advanced(season: str) -> pd.DataFrame:
    """Fetch per-game advanced stats (TS%, PIE, usage, ratings) for a season, with caching."""
    cache_key = f"advanced_{season}"
    cached = _load_cache(cache_key)
    if cached:
        return pd.DataFrame(cached)

    log.info(f"Fetching advanced stats: {season}")
    data = fetch_with_retry(
        lambda: LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
        ).get_data_frames()[0]
    )

    col_map = {
        "PLAYER_ID": "player_id",
        "TS_PCT": "ts_pct",
        "USG_PCT": "usg_pct",
        "PIE": "pie",
        "EFG_PCT": "efg_pct",
        "E_OFF_RATING": "off_rating",
        "E_DEF_RATING": "def_rating",
        "E_NET_RATING": "net_rating",
        "AST_PCT": "ast_pct",
        "OREB_PCT": "oreb_pct",
        "DREB_PCT": "dreb_pct",
        "REB_PCT": "reb_pct",
    }

    available = {k: v for k, v in col_map.items() if k in data.columns}
    df = data[list(available.keys())].rename(columns=available)
    df["season"] = season

    _save_cache(cache_key, df.to_dict(orient="records"))
    return df


def run():
    """Run the full data ingestion pipeline: players + all seasons → SQLite."""
    init_db()

    players_df = fetch_all_players()
    upsert_players(players_df)

    all_base = []
    all_advanced = []
    failed = []

    for season in tqdm(SEASONS, desc="Fetching seasons"):
        try:
            base = fetch_season_base(season)
            all_base.append(base)
        except Exception as e:
            log.warning(f"Skipping base stats for {season}: {e}")
            failed.append(f"base:{season}")

        try:
            adv = fetch_season_advanced(season)
            all_advanced.append(adv)
        except Exception as e:
            log.warning(f"Skipping advanced stats for {season}: {e}")
            failed.append(f"advanced:{season}")

    if failed:
        log.warning(f"Failed fetches (re-run to retry): {failed}")

    base_df = pd.concat(all_base, ignore_index=True)
    adv_df = pd.concat(all_advanced, ignore_index=True)

    log.info(f"Upserting {len(base_df)} base rows, {len(adv_df)} advanced rows...")
    upsert_season_stats(base_df)
    upsert_advanced_stats(adv_df)
    log.info("Phase 1 complete. Database populated.")


if __name__ == "__main__":
    run()
