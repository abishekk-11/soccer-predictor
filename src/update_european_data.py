"""Refresh data for La Liga, Bundesliga, Ligue 1 and Serie A.

The official league calendars in :mod:`src.league_config` remain the visible
fixture references. ESPN's public scoreboard and roster feeds supply a
consistent machine-readable schedule, completed result, current headshots,
and ten completed seasons of player goals, assists, starts and appearances.
Keeping the transformation here makes the forecast data reproducible without
depending on four different rendered website layouts.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from src.league_config import COMPLETED_SEASON, DATA_DIR, LEAGUES, LeagueConfig


ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
ESPN_ROSTER = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams/{team_id}/roster"
SECOND_TIER_BY_TOP_FLIGHT = {
    "esp.1": "esp.2",
    "ger.1": "ger.2",
    "fra.1": "fra.2",
    "ita.1": "ita.2",
}
FIRST_HISTORY_YEAR = 2016
LAST_COMPLETED_YEAR = 2025
UPCOMING_YEAR = 2026
REQUEST_TIMEOUT = 90
MAX_ROSTER_WORKERS = 8
MIN_LIVE_ROSTER_SIZE = 15


def _get_json(url: str, **params: object) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _event_season_year(event: dict[str, Any]) -> int | None:
    try:
        return int(event.get("season", {}).get("year"))
    except (TypeError, ValueError):
        return None


def _event_date(event: dict[str, Any]) -> pd.Timestamp:
    value = pd.Timestamp(event["date"])
    return value.tz_convert(None) if value.tzinfo else value


def _competitors(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    competitions = event.get("competitions", [])
    if not competitions:
        return None
    by_venue = {item.get("homeAway"): item for item in competitions[0].get("competitors", [])}
    if not by_venue.get("home") or not by_venue.get("away"):
        return None
    return by_venue["home"], by_venue["away"]


def _is_completed(event: dict[str, Any]) -> bool:
    competitions = event.get("competitions", [])
    return bool(competitions and competitions[0].get("status", {}).get("type", {}).get("completed"))


def _result_row(event: dict[str, Any], league_name: str) -> dict[str, Any] | None:
    competitors = _competitors(event)
    if not competitors or not _is_completed(event):
        return None
    home, away = competitors
    try:
        home_goals, away_goals = int(float(home["score"])), int(float(away["score"]))
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "Div": league_name,
        "Date": _event_date(event).date().isoformat(),
        "HomeTeam": home["team"]["displayName"],
        "AwayTeam": away["team"]["displayName"],
        "FTHG": home_goals,
        "FTAG": away_goals,
        "FTR": "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D",
        "Season": f"{_event_season_year(event)}-{str(_event_season_year(event) + 1)[-2:]}",
    }


def _fixture_row(event: dict[str, Any]) -> dict[str, Any] | None:
    competitors = _competitors(event)
    if not competitors:
        return None
    home, away = competitors
    return {
        "Date": _event_date(event).date().isoformat(),
        "HomeTeam": home["team"]["displayName"],
        "AwayTeam": away["team"]["displayName"],
        "HomeTeamId": str(home["team"]["id"]),
        "AwayTeamId": str(away["team"]["id"]),
        "EventId": str(event["id"]),
    }


def _calendar_events(config: LeagueConfig, calendar_year: int) -> list[dict[str, Any]]:
    payload = _get_json(
        ESPN_SCOREBOARD.format(league=config.espn_league),
        limit=1000,
        dates=calendar_year,
    )
    return list(payload.get("events", []))


def _season_events(config: LeagueConfig, season_year: int) -> list[dict[str, Any]]:
    """Collect a European season spanning its two calendar years."""
    events: dict[str, dict[str, Any]] = {}
    for calendar_year in (season_year, season_year + 1):
        for event in _calendar_events(config, calendar_year):
            if _event_season_year(event) == season_year:
                events[str(event["id"])] = event
    return list(events.values())


def _stat_value(athlete: dict[str, Any], statistic: str) -> float:
    categories = athlete.get("statistics", {}).get("splits", {}).get("categories", [])
    for category in categories:
        for item in category.get("stats", []):
            if item.get("name") == statistic:
                try:
                    return float(item.get("value", 0))
                except (TypeError, ValueError):
                    return 0.0
    return 0.0


def _roster_response(
    config: LeagueConfig,
    team_id: str,
    season_year: int,
    league_slug: str | None = None,
) -> dict[str, Any]:
    return _get_json(
        ESPN_ROSTER.format(league=league_slug or config.espn_league, team_id=team_id),
        season=season_year,
    )


def _team_ids(events: list[dict[str, Any]]) -> set[str]:
    """Return the distinct club IDs represented by a season's events."""
    return {
        str(competitor["team"]["id"])
        for event in events
        for competitor in (_competitors(event) or ())
    }


def _season_name(season_year: int) -> str:
    return f"{season_year}-{str(season_year + 1)[-2:]}"


def _completed_player_history(
    config: LeagueConfig,
    events_by_season: dict[int, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Collect goals, assists and availability across the past ten seasons.

    The roster endpoint is scoped to a club and season, so a player moving
    within a league can appear more than once. Those stints are aggregated to
    one player-season record, giving the scorer/assist ranking a complete
    competition history rather than only the latest club's portion.
    """
    requests_to_make = [
        (season_year, team_id)
        for season_year, events in events_by_season.items()
        for team_id in _team_ids(events)
    ]
    responses: list[tuple[int, str, dict[str, Any]]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_ROSTER_WORKERS) as executor:
        futures = {
            executor.submit(_roster_response, config, team_id, season_year): (season_year, team_id)
            for season_year, team_id in requests_to_make
        }
        for future in as_completed(futures):
            season_year, team_id = futures[future]
            try:
                responses.append((season_year, team_id, future.result()))
            except requests.RequestException as error:
                failures.append(f"{season_year}/{team_id}: {error}")
    if failures:
        detail = "; ".join(failures[:3])
        raise RuntimeError(f"Could not download {len(failures)} {config.name} roster archives ({detail})")

    player_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for season_year, _, response in responses:
        season = _season_name(season_year)
        for athlete in response.get("athletes", []):
            player_id = str(athlete["id"])
            key = (season, player_id)
            appearances = _stat_value(athlete, "appearances")
            row = player_rows.setdefault(key, {
                "season": season,
                "player_code": player_id,
                "player_name": athlete.get("displayName", "Unknown player"),
                # The public feed has no dependable historic minutes or xG/xA
                # for every competition, so those values remain explicitly
                # unavailable rather than being inferred.
                "minutes": 0,
                "goals": 0.0,
                "assists": 0.0,
                "appearances": 0.0,
                "starts": 0.0,
                "xg": 0.0,
                "xa": 0.0,
            })
            row["goals"] += _stat_value(athlete, "totalGoals")
            row["assists"] += _stat_value(athlete, "goalAssists")
            row["appearances"] += appearances
            row["starts"] += max(0, appearances - _stat_value(athlete, "subIns"))

    players = pd.DataFrame(player_rows.values())
    if players.empty:
        raise ValueError(f"No completed player records received for {config.name}")
    return players.sort_values(["season", "player_name", "player_code"])


def _player_archives(
    config: LeagueConfig,
    events_by_season: dict[int, list[dict[str, Any]]],
    fixtures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build portfolios and complete current squads with safe carryovers.

    Some preseason roster feeds are incomplete while transfer windows are open.
    Merge their available entries with the completed 2025/26 squad for the same
    club; newly promoted clubs use their second-tier squad instead. This gives
    every fixture a usable roster without pretending the carryovers are a
    confirmed 2026/27 selection.
    """
    players = _completed_player_history(config, events_by_season)

    current_teams = {
        str(row.HomeTeamId): row.HomeTeam for row in fixtures[["HomeTeamId", "HomeTeam"]].drop_duplicates().itertuples(index=False)
    }
    current_teams.update({
        str(row.AwayTeamId): row.AwayTeam for row in fixtures[["AwayTeamId", "AwayTeam"]].drop_duplicates().itertuples(index=False)
    })
    completed_top_flight_team_ids = _team_ids(events_by_season[LAST_COMPLETED_YEAR])
    roster_rows: list[dict[str, Any]] = []
    promoted_player_rows: list[dict[str, Any]] = []
    for team_id, team_name in current_teams.items():
        current_response = _roster_response(config, team_id, UPCOMING_YEAR)
        current_athletes = current_response.get("athletes", [])
        is_promoted = team_id not in completed_top_flight_team_ids
        fallback_league = SECOND_TIER_BY_TOP_FLIGHT.get(config.espn_league) if is_promoted else config.espn_league
        try:
            fallback_response = _roster_response(config, team_id, LAST_COMPLETED_YEAR, fallback_league)
            fallback_athletes = fallback_response.get("athletes", [])
        except requests.RequestException:
            fallback_athletes = []
        current_ids = {str(athlete["id"]) for athlete in current_athletes}
        # A complete current feed is authoritative: retaining last season's
        # omitted names would reintroduce players who have transferred away.
        # Carryovers are only a temporary safety net for an incomplete feed.
        combined_athletes = [(athlete, "2026/27 roster feed") for athlete in current_athletes]
        if len(current_ids) < MIN_LIVE_ROSTER_SIZE:
            combined_athletes += [
                (athlete, "2025/26 squad carryover")
                for athlete in fallback_athletes if str(athlete["id"]) not in current_ids
            ]
        for athlete, roster_source in combined_athletes:
            position = athlete.get("position", {}).get("displayName", "Midfielder")
            roster_rows.append({
                "team": team_name,
                "player_code": str(athlete["id"]),
                "player_name": athlete.get("displayName", "Unknown player"),
                "position": position,
                "photo_url": None,
                "roster_source": roster_source,
            })
        # The 2025/26 lower-division record is the only completed competitive
        # season available for newly promoted clubs. Retain it so their player
        # signals are informed by evidence rather than just a position prior.
        if is_promoted:
            for athlete in fallback_athletes:
                appearances = _stat_value(athlete, "appearances")
                promoted_player_rows.append({
                    "season": COMPLETED_SEASON,
                    "player_code": str(athlete["id"]),
                    "player_name": athlete.get("displayName", "Unknown player"),
                    "minutes": 0,
                    "goals": _stat_value(athlete, "totalGoals"),
                    "assists": _stat_value(athlete, "goalAssists"),
                    "appearances": appearances,
                    "starts": max(0, appearances - _stat_value(athlete, "subIns")),
                    "xg": 0.0,
                    "xa": 0.0,
                })

    roster = pd.DataFrame(roster_rows).drop_duplicates(["team", "player_code"])
    if roster.empty:
        raise ValueError(f"No 2026/27 roster records received for {config.name}")
    # When a transferred player appears in a live 2026/27 feed for one club,
    # do not retain their prior club's 2025/26 carryover entry.
    live_player_ids = set(roster.loc[roster.roster_source == "2026/27 roster feed", "player_code"])
    roster = roster[
        ~(
            roster.roster_source.eq("2025/26 squad carryover")
            & roster.player_code.isin(live_player_ids)
        )
    ].copy()
    players = pd.concat([players, pd.DataFrame(promoted_player_rows)], ignore_index=True)
    numeric_columns = ["minutes", "goals", "assists", "appearances", "starts", "xg", "xa"]
    players = players.groupby(["season", "player_code", "player_name"], as_index=False)[numeric_columns].sum()
    team_sizes = roster.groupby("team").size()
    sparse_teams = team_sizes[team_sizes < 15]
    if not sparse_teams.empty:
        names = ", ".join(sparse_teams.index)
        raise ValueError(f"Incomplete roster data for {config.name}: {names}")
    return players.sort_values(["season", "player_name", "player_code"]), roster.sort_values(["team", "position", "player_name"])


def refresh_league(config: LeagueConfig) -> dict[str, int | str]:
    """Download, validate and store a non-Premier-League data bundle."""
    history_rows: list[dict[str, Any]] = []
    events_by_season: dict[int, list[dict[str, Any]]] = {}
    for season_year in range(FIRST_HISTORY_YEAR, LAST_COMPLETED_YEAR + 1):
        events = _season_events(config, season_year)
        events_by_season[season_year] = events
        rows = [row for event in events if (row := _result_row(event, config.name))]
        if not rows:
            raise ValueError(f"No completed {config.name} data found for {season_year}/{season_year + 1}")
        history_rows.extend(rows)

    matches = pd.DataFrame(history_rows).drop_duplicates(
        ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season"]
    ).sort_values(["Date", "HomeTeam", "AwayTeam"])
    upcoming_events = _season_events(config, UPCOMING_YEAR)
    fixtures = pd.DataFrame([row for event in upcoming_events if (row := _fixture_row(event))]).drop_duplicates("EventId")
    fixtures = fixtures.sort_values(["Date", "HomeTeam", "AwayTeam"])
    expected_fixtures = config.expected_teams * (config.expected_teams - 1)
    if len(fixtures) != expected_fixtures:
        raise ValueError(f"Expected {expected_fixtures} {config.name} fixtures, received {len(fixtures)}")
    if len(set(fixtures.HomeTeam).union(fixtures.AwayTeam)) != config.expected_teams:
        raise ValueError(f"Expected {config.expected_teams} {config.name} clubs in the fixture list")

    players, roster = _player_archives(config, events_by_season, fixtures)
    DATA_DIR.mkdir(exist_ok=True)
    matches.to_csv(config.matches_path, index=False)
    fixtures.to_csv(config.fixtures_path, index=False)
    players.to_csv(config.players_path, index=False)
    roster.to_csv(config.rosters_path, index=False)
    return {
        "name": config.name,
        "results_rows": int(len(matches)),
        "results_through": str(matches.Date.max()),
        "fixtures_rows": int(len(fixtures)),
        "roster_rows": int(len(roster)),
        "player_rows": int(len(players)),
        "player_seasons": int(players.season.nunique()),
    }


def _write_manifest(summaries: dict[str, dict[str, int | str]]) -> None:
    """Record a complete or per-league refresh without dropping prior runs."""
    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous = manifest.get("european_leagues", {})
    previous_summaries = {
        key: value for key, value in previous.items()
        if key in LEAGUES and key != "premier_league" and isinstance(value, dict)
    }
    manifest["european_leagues"] = {
        **previous_summaries,
        **summaries,
        "season": "2026-27",
        "completed_player_season": COMPLETED_SEASON,
        "player_history_window": f"{FIRST_HISTORY_YEAR}/{str(FIRST_HISTORY_YEAR + 1)[-2:]}–{LAST_COMPLETED_YEAR}/{str(LAST_COMPLETED_YEAR + 1)[-2:]}",
        "fixture_references": {
            key: config.official_fixture_url
            for key, config in LEAGUES.items() if key != "premier_league"
        },
        "machine_readable_source": "https://site.api.espn.com/",
    }
    manifest["updated_on"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def local_summary(config: LeagueConfig) -> dict[str, int | str]:
    """Return manifest metrics for an already refreshed local league bundle."""
    matches = pd.read_csv(config.matches_path, parse_dates=["Date"])
    fixtures = pd.read_csv(config.fixtures_path)
    players = pd.read_csv(config.players_path)
    roster = pd.read_csv(config.rosters_path)
    return {
        "name": config.name,
        "results_rows": int(len(matches)),
        "results_through": str(matches.Date.max().date()),
        "fixtures_rows": int(len(fixtures)),
        "roster_rows": int(len(roster)),
        "player_rows": int(len(players)),
        "player_seasons": int(players.season.nunique()),
    }


def refresh_all() -> dict[str, dict[str, int | str]]:
    """Refresh the four added top-flight competitions."""
    summaries = {
        key: refresh_league(config)
        for key, config in LEAGUES.items()
        if key != "premier_league"
    }
    _write_manifest(summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh European league data and player portfolios")
    parser.add_argument(
        "--league",
        choices=["all", *(key for key in LEAGUES if key != "premier_league")],
        default="all",
        help="Refresh one league or all four (default).",
    )
    args = parser.parse_args()
    if args.league == "all":
        result = refresh_all()
    else:
        result = {args.league: refresh_league(LEAGUES[args.league])}
        _write_manifest(result)
    print(json.dumps(result, indent=2))
