"""Shared constants and thresholds for the NBA Breakout Predictor pipeline."""

SEASONS = [
    "2010-11", "2011-12", "2012-13", "2013-14", "2014-15",
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    "2025-26",
]

CURRENT_SEASON = "2025-26"

# Qualifying thresholds — both the current and prior season must meet these
MIN_GP = 40
MIN_MPG = 20.0

# Breakout label thresholds
BREAKOUT_DELTA_PCT = 0.20   # minimum % improvement to count as a breakout
BREAKOUT_MIN_PTS = 12.0     # player must also score ≥12 PPG in the breakout year

TOP_N_CANDIDATES = 20
TOP_N_SHAP_FEATURES = 8

RATE_LIMIT_SLEEP = 0.7  # seconds between NBA API calls
