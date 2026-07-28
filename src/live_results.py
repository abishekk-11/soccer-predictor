"""Small, resilient live-result cache for the in-season form display.

The trained forecast data remains a reproducible completed-season archive.
This module is intentionally separate: it refreshes only completed current
season league matches so the application can show current form without a
retrain or a manual data refresh after every matchday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any

import pandas as pd
import requests

from src.league_config import SEASON, LeagueConfig


ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
LIVE_SEASON_START_YEAR = int(SEASON.split("/")[0])
LIVE_SEASON_LABEL = f"{LIVE_SEASON_START_YEAR}-{str(LIVE_SEASON_START_YEAR + 1)[-2:]}"
LIVE_RESULTS_TTL = timedelta(minutes=15)
REQUEST_TIMEOUT = 20

RESULT_COLUMNS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season"]

# The Premier League fixture file intentionally uses the concise names shown
# in the application. ESPN returns the longer official names for these clubs.
TEAM_ALIASES = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "Sheffield Wednesday": "Sheffield Weds",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}


@dataclass(frozen=True)
class LiveResults:
    """Completed current-season results and their retrieval information."""

    matches: pd.DataFrame
    fetched_at: datetime
    source_url: str
    refresh_error: str | None = None


_cache: dict[str, LiveResults] = {}
_cache_lock = Lock()


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def _event_season_year(event: dict[str, Any]) -> int | None:
    try:
        return int(event.get("season", {}).get("year"))
    except (TypeError, ValueError):
        return None


def _event_date(event: dict[str, Any]) -> pd.Timestamp:
    timestamp = pd.Timestamp(event["date"])
    return timestamp.tz_convert(None) if timestamp.tzinfo else timestamp


def _completed_result(event: dict[str, Any], config: LeagueConfig) -> dict[str, Any] | None:
    competitions = event.get("competitions") or []
    if not competitions or not competitions[0].get("status", {}).get("type", {}).get("completed"):
        return None
    competitors = {
        item.get("homeAway"): item
        for item in competitions[0].get("competitors", [])
    }
    home, away = competitors.get("home"), competitors.get("away")
    if not home or not away:
        return None
    try:
        home_goals = int(float(home["score"]))
        away_goals = int(float(away["score"]))
    except (KeyError, TypeError, ValueError):
        return None
    home_name = TEAM_ALIASES.get(str(home["team"].get("displayName", "")), home["team"].get("displayName", ""))
    away_name = TEAM_ALIASES.get(str(away["team"].get("displayName", "")), away["team"].get("displayName", ""))
    return {
        "Div": config.name,
        "Date": _event_date(event).date().isoformat(),
        "HomeTeam": home_name,
        "AwayTeam": away_name,
        "FTHG": home_goals,
        "FTAG": away_goals,
        "FTR": "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D",
        "Season": LIVE_SEASON_LABEL,
    }


def _download_live_results(config: LeagueConfig) -> pd.DataFrame:
    events: dict[str, dict[str, Any]] = {}
    for calendar_year in (LIVE_SEASON_START_YEAR, LIVE_SEASON_START_YEAR + 1):
        response = requests.get(
            ESPN_SCOREBOARD.format(league=config.espn_league),
            params={"limit": 1000, "dates": calendar_year},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        for event in response.json().get("events", []):
            if _event_season_year(event) == LIVE_SEASON_START_YEAR:
                events[str(event["id"])] = event
    rows = [row for event in events.values() if (row := _completed_result(event, config))]
    if not rows:
        return _empty_results()
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).drop_duplicates(
        ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    ).sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def live_results_for(config: LeagueConfig, now: datetime | None = None) -> LiveResults:
    """Return current 2026/27 completed results, refreshing at most every 15 minutes.

    If the live source is temporarily unavailable, the last successful snapshot
    remains usable. Before the season starts, the valid empty snapshot lets the
    form layer transparently fall back to last season's final matches.
    """
    retrieved_at = now or datetime.now(timezone.utc)
    with _cache_lock:
        cached = _cache.get(config.key)
        if cached and retrieved_at - cached.fetched_at < LIVE_RESULTS_TTL:
            return cached
        source_url = ESPN_SCOREBOARD.format(league=config.espn_league)
        try:
            snapshot = LiveResults(
                matches=_download_live_results(config),
                fetched_at=retrieved_at,
                source_url=source_url,
            )
        except requests.RequestException as error:
            if cached:
                snapshot = LiveResults(
                    matches=cached.matches,
                    fetched_at=retrieved_at,
                    source_url=cached.source_url,
                    refresh_error=str(error),
                )
            else:
                snapshot = LiveResults(
                    matches=_empty_results(),
                    fetched_at=retrieved_at,
                    source_url=source_url,
                    refresh_error=str(error),
                )
        _cache[config.key] = snapshot
        return snapshot
