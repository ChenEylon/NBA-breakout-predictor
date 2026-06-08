"""
Phase 4b: LLM Scouting Report Generation
Uses Claude API to generate per-player scouting reports based on SHAP features.
Reports are cached to reports/{player_id}.json.

Usage:
    python src/explainability/llm_reporter.py
"""

import json
import logging
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.explainability.shap_explainer import run as get_candidates

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parents[2] / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "claude-sonnet-4-6"
RATE_LIMIT_SLEEP = 0.5  # seconds between API calls


def _report_path(player_id: int) -> Path:
    """Return the cache file path for a player's scouting report."""
    return REPORTS_DIR / f"{player_id}.json"


def _build_prompt(ctx: dict) -> str:
    name = ctx["name"]
    team = ctx.get("team", "N/A")
    age = ctx.get("age", "N/A")
    years = ctx.get("years_in_league", 0)
    prob = ctx["breakout_prob"]
    pts = ctx.get("pts", 0) or 0
    ts = ctx.get("ts_pct", 0) or 0
    usg = ctx.get("usg_pct", 0) or 0
    pie = ctx.get("pie", 0) or 0
    changed_teams = ctx.get("team_changed", False)

    shap_lines = []
    for feat in ctx.get("top_shap_features", [])[:8]:
        direction = "↑" if feat["direction"] == "positive" else "↓"
        shap_lines.append(
            f"  {direction} {feat['feature'].replace('prev_', 'Prior-season ')}: SHAP={feat['shap_value']:+.3f}"
        )
    shap_text = "\n".join(shap_lines)

    return f"""You are an expert NBA scout and data analyst. Write a 3-paragraph scouting report for {name}.

PLAYER PROFILE:
- Name: {name}
- Team: {team}
- Age: {age}
- Years in league: {years}
- Recently changed teams: {changed_teams}

2025-26 SEASON STATS:
- Points per game: {pts:.1f}
- True shooting %: {ts:.1%}
- Usage rate: {usg:.1%}
- PIE (Player Impact Estimate): {pie:.3f}

MODEL PREDICTION:
- Breakout probability for 2026-27: {prob:.1%}

TOP SHAP DRIVERS (features most influencing this prediction):
{shap_text}

Write a professional 3-paragraph scouting report:
- Paragraph 1: Player profile and what makes them a breakout candidate (reference their age, role, and key stats)
- Paragraph 2: Deep dive on the top SHAP drivers — explain what each feature tells us about their development trajectory
- Paragraph 3: Risks, ceiling/floor, and specific scenarios under which the breakout happens (or doesn't)

Be specific, cite the actual numbers, and write like a professional scout — not a chatbot. Keep each paragraph to 3-4 sentences."""


def generate_report(ctx: dict, client: anthropic.Anthropic) -> str:
    """Call the Claude API and return the generated scouting report text."""
    prompt = _build_prompt(ctx)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def load_or_generate(ctx: dict, client: anthropic.Anthropic, force: bool = False) -> dict | None:
    """Load a cached report for a player, or generate one via the Claude API.

    Returns None if generation fails, so callers can skip gracefully.
    """
    player_id = ctx.get("player_id") or hash(ctx["name"])
    path = _report_path(player_id)

    if path.exists() and not force:
        existing = json.loads(path.read_text())
        log.info(f"Loaded cached report: {ctx['name']}")
        return existing

    log.info(f"Generating report for {ctx['name']} ({ctx['breakout_prob']:.1%})...")
    try:
        report_text = generate_report(ctx, client)
    except Exception as e:
        log.warning(f"Failed to generate report for {ctx['name']}: {e}")
        return None
    time.sleep(RATE_LIMIT_SLEEP)

    record = {
        "player_id": player_id,
        "name": ctx["name"],
        "team": ctx.get("team"),
        "age": ctx.get("age"),
        "breakout_prob": ctx["breakout_prob"],
        "pts": ctx.get("pts"),
        "ts_pct": ctx.get("ts_pct"),
        "usg_pct": ctx.get("usg_pct"),
        "pie": ctx.get("pie"),
        "years_in_league": ctx.get("years_in_league"),
        "team_changed": ctx.get("team_changed"),
        "top_shap_features": ctx.get("top_shap_features", []),
        "report": report_text,
    }
    path.write_text(json.dumps(record, indent=2))
    return record


def run(force: bool = False) -> list[dict]:
    """Generate or load scouting reports for all top breakout candidates."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Create a .env file with:\n  ANTHROPIC_API_KEY=your-key-here"
        )

    client = anthropic.Anthropic(api_key=api_key)
    candidates = get_candidates()

    # Attach a stable player_id from the player name hash (SHAP runner doesn't expose DB id)
    for i, ctx in enumerate(candidates):
        if "player_id" not in ctx:
            ctx["player_id"] = abs(hash(ctx["name"])) % (10**9)

    reports = []
    failed = []
    for ctx in candidates:
        record = load_or_generate(ctx, client, force=force)
        if record is not None:
            reports.append(record)
        else:
            failed.append(ctx["name"])

    if failed:
        log.warning(f"Failed to generate reports for: {failed}")
    log.info(f"Generated/loaded {len(reports)} scouting reports → {REPORTS_DIR}/")
    return reports


if __name__ == "__main__":
    reports = run()
    for r in reports[:3]:
        print(f"\n{'=' * 60}")
        print(f"{r['name']} ({r['team']}) — {r['breakout_prob']:.1%} breakout prob")
        print(f"{'=' * 60}")
        print(r["report"])
