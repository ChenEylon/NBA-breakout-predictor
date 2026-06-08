"""SQLite schema definition and query helpers for the NBA stats database."""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parents[2] / "data" / "nba.db"


def get_connection() -> sqlite3.Connection:
    """Return a connection to the local SQLite database, creating the data dir if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create all tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS players (
                player_id   INTEGER PRIMARY KEY,
                name        TEXT NOT NULL,
                position    TEXT,
                dob         TEXT
            );

            CREATE TABLE IF NOT EXISTS season_stats (
                player_id   INTEGER,
                season      TEXT,
                team_id     INTEGER,
                team_abbr   TEXT,
                age         REAL,
                gp          INTEGER,
                min         REAL,
                pts         REAL,
                ast         REAL,
                reb         REAL,
                stl         REAL,
                blk         REAL,
                tov         REAL,
                fg_pct      REAL,
                fg3_pct     REAL,
                ft_pct      REAL,
                PRIMARY KEY (player_id, season)
            );

            CREATE TABLE IF NOT EXISTS advanced_stats (
                player_id   INTEGER,
                season      TEXT,
                ts_pct      REAL,
                usg_pct     REAL,
                pie         REAL,
                efg_pct     REAL,
                off_rating  REAL,
                def_rating  REAL,
                net_rating  REAL,
                ast_pct     REAL,
                oreb_pct    REAL,
                dreb_pct    REAL,
                reb_pct     REAL,
                PRIMARY KEY (player_id, season)
            );
        """)


def upsert_players(df: pd.DataFrame):
    """Replace the players table with the provided DataFrame."""
    with get_connection() as conn:
        df.to_sql("players", conn, if_exists="replace", index=False,
                  method="multi", chunksize=500)


def upsert_season_stats(df: pd.DataFrame):
    """Replace the season_stats table with the provided DataFrame."""
    with get_connection() as conn:
        df.to_sql("season_stats", conn, if_exists="replace", index=False,
                  method="multi", chunksize=500)


def upsert_advanced_stats(df: pd.DataFrame):
    """Replace the advanced_stats table with the provided DataFrame."""
    with get_connection() as conn:
        df.to_sql("advanced_stats", conn, if_exists="replace", index=False,
                  method="multi", chunksize=500)


def load_season_stats() -> pd.DataFrame:
    """Load all rows from season_stats as a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM season_stats", conn)


def load_advanced_stats() -> pd.DataFrame:
    """Load all rows from advanced_stats as a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM advanced_stats", conn)


def load_players() -> pd.DataFrame:
    """Load all rows from the players table as a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM players", conn)
