"""Live squad refreshes for the five-league player-prediction layer.

The deployed application should not keep presenting a player for a club after
that player has moved.  This module checks the public ESPN roster feed whenever
a selected team has not been refreshed recently.  Completed player statistics
remain the reproducible 2016/17–2025/26 archive; only squad membership,
position, and an available portrait are refreshed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any
import re
import unicodedata

import pandas as pd
import requests

from src.league_config import SEASON, LeagueConfig
from src.live_results import TEAM_ALIASES


ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
ESPN_ROSTER = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams/{team_id}/roster"
LIVE_ROSTER_TTL = timedelta(hours=1)
DIRECTORY_TTL = timedelta(hours=6)
REQUEST_TIMEOUT = 25
MIN_LIVE_ROSTER_SIZE = 15
LIVE_SEASON_YEAR = int(SEASON.split("/")[0])

ROSTER_COLUMNS = ["team", "player_code", "player_name", "position", "photo_url", "roster_source"]


@dataclass(frozen=True)
class LiveRoster:
    """A current team squad or a transparent fallback to the bundled roster."""

    players: pd.DataFrame
    fetched_at: datetime
    source_url: str
    live: bool
    refresh_error: str | None = None


@dataclass(frozen=True)
class TeamDirectory:
    """The ESPN team IDs needed to query current club rosters."""

    team_ids: dict[str, str]
    fetched_at: datetime


_roster_cache: dict[tuple[str, str], LiveRoster] = {}
_directory_cache: dict[str, TeamDirectory] = {}
_cache_lock = Lock()


def _name_key(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", normalized.lower())


def _canonical_team(value: object) -> str:
    name = str(value or "").strip()
    return TEAM_ALIASES.get(name, name)


def _empty_roster() -> pd.DataFrame:
    return pd.DataFrame(columns=ROSTER_COLUMNS)


def _fixture_team_ids(fixtures: pd.DataFrame) -> dict[str, str]:
    """Read ESPN IDs embedded in the four non-Premier-League fixture files."""
    team_ids: dict[str, str] = {}
    for venue in ("Home", "Away"):
        name_column, id_column = f"{venue}Team", f"{venue}TeamId"
        if id_column not in fixtures.columns:
            continue
        for row in fixtures[[name_column, id_column]].dropna().drop_duplicates().itertuples(index=False):
            team_ids[_canonical_team(row[0])] = str(row[1])
    return team_ids


def _scoreboard_team_ids(config: LeagueConfig, unresolved_teams: set[str]) -> dict[str, str]:
    """Find IDs absent from a fixture file, including promoted Premier League clubs."""
    if not unresolved_teams:
        return {}
    league_slugs = [config.espn_league]
    if config.espn_league == "eng.1":
        league_slugs.append("eng.2")

    team_ids: dict[str, str] = {}
    for league_slug in league_slugs:
        for calendar_year in (LIVE_SEASON_YEAR, LIVE_SEASON_YEAR + 1):
            response = requests.get(
                ESPN_SCOREBOARD.format(league=league_slug),
                params={"limit": 1000, "dates": calendar_year},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            for event in response.json().get("events", []):
                competitions = event.get("competitions") or []
                competitors = competitions[0].get("competitors", []) if competitions else []
                for competitor in competitors:
                    team = competitor.get("team", {})
                    name = _canonical_team(team.get("displayName"))
                    if name in unresolved_teams:
                        team_ids[name] = str(team.get("id"))
        if unresolved_teams.issubset(team_ids):
            break
    return {team: team_id for team, team_id in team_ids.items() if team_id and team_id != "None"}


def _team_directory(config: LeagueConfig, fixtures: pd.DataFrame, now: datetime) -> TeamDirectory:
    with _cache_lock:
        cached = _directory_cache.get(config.key)
        if cached and now - cached.fetched_at < DIRECTORY_TTL:
            return cached

    team_ids = _fixture_team_ids(fixtures)
    fixture_teams = {
        _canonical_team(team)
        for column in ("HomeTeam", "AwayTeam")
        for team in fixtures[column].dropna().astype(str)
    }
    missing = fixture_teams.difference(team_ids)
    if missing:
        try:
            team_ids.update(_scoreboard_team_ids(config, missing))
        except requests.RequestException:
            # The local roster remains available if the ID directory is
            # temporarily unreachable; a later request will try again.
            pass
    snapshot = TeamDirectory(team_ids=team_ids, fetched_at=now)
    with _cache_lock:
        _directory_cache[config.key] = snapshot
    return snapshot


def _roster_response(config: LeagueConfig, team_id: str) -> tuple[dict[str, Any], str]:
    source_url = ESPN_ROSTER.format(league=config.espn_league, team_id=team_id)
    response = requests.get(source_url, params={"season": LIVE_SEASON_YEAR}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json(), source_url


def _merge_cached_portraits(live_roster: pd.DataFrame, bundled_roster: pd.DataFrame) -> pd.DataFrame:
    """Keep an existing usable portrait when a live feed omits its headshot."""
    result = live_roster.copy()
    if bundled_roster.empty:
        return result
    by_name = {
        _name_key(row.player_name): row.photo_url
        for row in bundled_roster.itertuples(index=False)
        if pd.notna(row.photo_url) and str(row.photo_url).strip()
    }
    missing = result.photo_url.isna() | result.photo_url.eq("")
    result.loc[missing, "photo_url"] = result.loc[missing, "player_name"].map(
        lambda name: by_name.get(_name_key(name))
    )
    return result


def _live_players(response: dict[str, Any], team: str, bundled_roster: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | None]] = []
    for athlete in response.get("athletes", []):
        player_id = str(athlete.get("id") or "").strip()
        player_name = str(athlete.get("displayName") or "").strip()
        if not player_id or not player_name:
            continue
        headshot = athlete.get("headshot") or {}
        position = athlete.get("position") or {}
        rows.append({
            "team": team,
            "player_code": player_id,
            "player_name": player_name,
            "position": str(position.get("displayName") or "Midfielder"),
            "photo_url": headshot.get("href"),
            "roster_source": "ESPN live roster",
        })
    roster = pd.DataFrame(rows, columns=ROSTER_COLUMNS).drop_duplicates("player_code")
    return _merge_cached_portraits(roster, bundled_roster)


def live_roster_for(
    config: LeagueConfig,
    fixtures: pd.DataFrame,
    team: str,
    bundled_roster: pd.DataFrame,
    now: datetime | None = None,
) -> LiveRoster:
    """Return the latest reliable squad for a selected team.

    A roster is replaced only when the live feed has a credible squad-sized
    response.  Otherwise the app retains its bundled roster rather than
    dropping valid players during a temporary provider outage.
    """
    retrieved_at = now or datetime.now(timezone.utc)
    cache_key = (config.key, team)
    with _cache_lock:
        cached = _roster_cache.get(cache_key)
        if cached and retrieved_at - cached.fetched_at < LIVE_ROSTER_TTL:
            return cached

    fallback = bundled_roster[bundled_roster.team == team].copy()
    directory = _team_directory(config, fixtures, retrieved_at)
    team_id = directory.team_ids.get(_canonical_team(team))
    source_url = ESPN_ROSTER.format(league=config.espn_league, team_id=team_id or "")
    if not team_id:
        snapshot = LiveRoster(
            players=fallback,
            fetched_at=retrieved_at,
            source_url=source_url,
            live=False,
            refresh_error="Current team ID was unavailable from the live schedule feed.",
        )
    else:
        try:
            response, source_url = _roster_response(config, team_id)
            live_players = _live_players(response, team, fallback)
            if len(live_players) < MIN_LIVE_ROSTER_SIZE:
                snapshot = LiveRoster(
                    players=fallback,
                    fetched_at=retrieved_at,
                    source_url=source_url,
                    live=False,
                    refresh_error="The live provider returned an incomplete squad.",
                )
            else:
                snapshot = LiveRoster(
                    players=live_players,
                    fetched_at=retrieved_at,
                    source_url=source_url,
                    live=True,
                )
        except requests.RequestException as error:
            snapshot = LiveRoster(
                players=fallback,
                fetched_at=retrieved_at,
                source_url=source_url,
                live=False,
                refresh_error=str(error),
            )
    with _cache_lock:
        _roster_cache[cache_key] = snapshot
    return snapshot
