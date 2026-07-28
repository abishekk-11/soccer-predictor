"""Resolve working player portraits for the five current-season rosters.

The public roster feed used for the four added leagues supplies athlete IDs but
its generated CDN portrait URLs are not consistently available. This updater
replaces those failed URLs with a cached public thumbnail lookup. It preserves
the existing Premier League portraits, and unresolved players deliberately fall
back to the initials avatar in the interface.
"""

from __future__ import annotations

import argparse
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.league_config import DATA_DIR, LEAGUES


PORTRAIT_SEARCH_URL = "https://www.thesportsdb.com/api/v1/json/3/searchplayers.php"
TEAM_SEARCH_URL = "https://www.thesportsdb.com/api/v1/json/3/searchteams.php"
TEAM_PLAYERS_URL = "https://www.thesportsdb.com/api/v1/json/3/lookup_all_players.php"
WIKIMEDIA_QUERY_URL = "https://en.wikipedia.org/w/api.php"
WIKIDATA_QUERY_URL = "https://query.wikidata.org/sparql"
CACHE_PATH = DATA_DIR / "player_portrait_cache.csv"
REQUEST_TIMEOUT = 20
MAX_WORKERS = 10
# The public service has a modest request limit. Two concurrent teams keep the
# refresh dependable while still avoiding the old one-request-per-player path.
TEAM_LOOKUP_WORKERS = 2
WIKIMEDIA_BATCH_SIZE = 20
WIKIDATA_BATCH_SIZE = 75
BROKEN_ESPN_PREFIX = "https://a.espncdn.com/i/headshots/soccer/players/full/"


def _name_key(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", normalized.lower())


def _find_portrait(player_name: str) -> str | None:
    response = requests.get(PORTRAIT_SEARCH_URL, params={"p": player_name}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    candidates = response.json().get("player") or []
    exact_matches = [
        candidate for candidate in candidates
        if _name_key(candidate.get("strPlayer")) == _name_key(player_name)
    ]
    for candidate in [*exact_matches, *candidates]:
        # A thumbnail is a conventional head-and-shoulders portrait. A cutout
        # is still preferable to an invented image where no thumbnail exists.
        portrait = candidate.get("strThumb") or candidate.get("strCutout")
        if portrait and str(portrait).startswith("https://"):
            return str(portrait)
    return None


def _sportsdb_json(url: str, params: dict[str, str]) -> dict:
    """Read a SportsDB payload with a small, respectful rate-limit retry."""
    response = None
    for attempt in range(3):
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        retry_after = response.headers.get("Retry-After")
        delay = min(float(retry_after) if retry_after else 2 ** (attempt + 1), 8.0)
        time.sleep(delay)
    if response is not None:
        response.raise_for_status()
    raise requests.RequestException("Portrait service did not return a response")


def _team_roster_portraits(team_name: str) -> dict[str, str]:
    """Fetch the public thumbnail roster for one club, when the feed has it."""
    candidates = _sportsdb_json(TEAM_SEARCH_URL, {"t": team_name}).get("teams") or []
    team_key = _name_key(team_name)
    soccer_candidates = [
        candidate for candidate in candidates
        if str(candidate.get("strSport", "")).lower() == "soccer"
    ]
    exact = [
        candidate for candidate in soccer_candidates
        if _name_key(candidate.get("strTeam")) == team_key
        or team_key in {
            _name_key(alias) for alias in str(candidate.get("strTeamAlternate") or "").split(",")
        }
    ]
    selected = (exact or soccer_candidates or candidates)
    if not selected or not selected[0].get("idTeam"):
        return {}
    response = _sportsdb_json(TEAM_PLAYERS_URL, {"id": str(selected[0]["idTeam"])})
    return {
        _name_key(player.get("strPlayer")): str(player.get("strThumb") or player.get("strCutout"))
        for player in response.get("player") or []
        if _name_key(player.get("strPlayer"))
        and str(player.get("strThumb") or player.get("strCutout") or "").startswith("https://")
    }


def _bulk_team_portraits(team_names: list[str]) -> tuple[dict[str, str], int]:
    """Look up team rosters concurrently without issuing one call per player."""
    resolved: dict[str, str] = {}
    failures = 0
    with ThreadPoolExecutor(max_workers=TEAM_LOOKUP_WORKERS) as executor:
        futures = {executor.submit(_team_roster_portraits, team): team for team in team_names}
        for future in as_completed(futures):
            try:
                resolved.update(future.result())
            except (requests.RequestException, ValueError):
                failures += 1
    return resolved, failures


def _sparql_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"@en'


def _wikidata_portraits(name_by_key: dict[str, str]) -> dict[str, str]:
    """Find exact English-name Commons images in compact Wikidata batches."""
    resolved: dict[str, str] = {}
    player_names = list(name_by_key.values())
    for start in range(0, len(player_names), WIKIDATA_BATCH_SIZE):
        batch = player_names[start:start + WIKIDATA_BATCH_SIZE]
        query = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?name ?image WHERE {
              VALUES ?name { %s }
              ?person rdfs:label ?name ; wdt:P18 ?image .
            }
        """ % " ".join(_sparql_string(name) for name in batch)
        try:
            response = requests.get(
                WIKIDATA_QUERY_URL,
                params={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json", "User-Agent": "soccer-predictor/2.1 (local data refresh)"},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            continue
        for binding in payload.get("results", {}).get("bindings", []):
            name = binding.get("name", {}).get("value")
            image = binding.get("image", {}).get("value")
            key = _name_key(name)
            if key in name_by_key and image:
                resolved[key] = str(image).replace("http://", "https://", 1)
        if start + WIKIDATA_BATCH_SIZE < len(player_names):
            time.sleep(1.0)
    return resolved


def _wikimedia_portraits(name_by_key: dict[str, str]) -> dict[str, str]:
    """Resolve exact player-page thumbnails in compact Wikimedia batches."""
    resolved: dict[str, str] = {}
    player_names = list(name_by_key.values())
    for start in range(0, len(player_names), WIKIMEDIA_BATCH_SIZE):
        batch = player_names[start:start + WIKIMEDIA_BATCH_SIZE]
        response = None
        for attempt in range(3):
            response = requests.get(
                WIKIMEDIA_QUERY_URL,
                params={
                    "action": "query",
                    "titles": "|".join(batch),
                    "prop": "pageimages",
                    "pithumbsize": 240,
                    "format": "json",
                },
                headers={"User-Agent": "soccer-predictor/2.1 (local data refresh)"},
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After")
            delay = min(float(retry_after) if retry_after else 2 ** (attempt + 1), 8.0)
            time.sleep(delay)
        if response is None or not response.ok:
            continue
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            title = str(page.get("title", ""))
            thumbnail = page.get("thumbnail", {}).get("source")
            key = _name_key(title)
            if key in name_by_key and thumbnail and str(thumbnail).startswith("https://"):
                resolved[key] = str(thumbnail)
        # Keep the local data refresh polite and avoid a burst of API calls.
        if start + WIKIMEDIA_BATCH_SIZE < len(player_names):
            time.sleep(0.7)
    return resolved


def _read_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    cache = pd.read_csv(CACHE_PATH)
    return {
        str(row.name_key): str(row.photo_url)
        for row in cache.itertuples(index=False)
        if pd.notna(row.photo_url) and str(row.photo_url).startswith("https://")
    }


def _write_cache(cache: dict[str, str]) -> None:
    rows = [
        {"name_key": key, "photo_url": value, "updated_on": datetime.now(timezone.utc).isoformat()}
        for key, value in sorted(cache.items())
    ]
    pd.DataFrame(rows, columns=["name_key", "photo_url", "updated_on"]).to_csv(CACHE_PATH, index=False)


def resolve_league_portraits(
    league_key: str,
    max_workers: int = MAX_WORKERS,
    individual_limit: int = 60,
    wiki_limit: int | None = None,
    team_lookup: bool = True,
) -> dict[str, int | str]:
    """Replace unavailable player images in one configured roster file."""
    config = LEAGUES[league_key]
    roster = pd.read_csv(config.rosters_path)
    # A fresh roster refresh may contain only empty portrait cells; preserve an
    # object column so cached URL strings can be restored safely.
    roster["photo_url"] = roster["photo_url"].astype("object")
    cache = _read_cache()
    roster["name_key"] = roster.player_name.map(_name_key)
    broken = roster.photo_url.fillna("").str.startswith(BROKEN_ESPN_PREFIX)
    missing = roster.photo_url.isna() | roster.photo_url.eq("") | broken
    # Team roster lookups are much more efficient than a player-by-player
    # search and often include portraits for players without Wikipedia pages.
    unresolved_names = roster.loc[missing & ~roster.name_key.isin(cache)]
    if team_lookup:
        team_resolved, team_failures = _bulk_team_portraits(
            sorted(unresolved_names.team.dropna().astype(str).unique())
        )
    else:
        team_resolved, team_failures = {}, 0
    cache.update(team_resolved)
    unresolved = roster.loc[missing & ~roster.name_key.isin(cache)].copy()
    unresolved["position_rank"] = unresolved.position.map(
        {"Forward": 0, "Midfielder": 1, "Defender": 2, "Goalkeeper": 3}
    ).fillna(4)
    unresolved = unresolved.sort_values(["position_rank", "player_name"]).drop_duplicates("name_key")
    if wiki_limit is not None:
        unresolved = unresolved.head(wiki_limit)
    name_by_key = {
        row.name_key: row.player_name
        for row in unresolved.itertuples(index=False)
    }
    wikidata_resolved = _wikidata_portraits(name_by_key)
    cache.update(wikidata_resolved)
    name_by_key = {
        key: name for key, name in name_by_key.items()
        if key not in cache
    }
    wiki_resolved = _wikimedia_portraits(name_by_key)
    cache.update(wiki_resolved)
    # Wikimedia gives broad, quick exact-name coverage. The slower player
    # search then focuses on attacking players first, where a fixture headshot
    # is most valuable, instead of issuing thousands of one-by-one requests.
    lookup_order = [key for key in unresolved.name_key if key not in cache][:individual_limit]
    resolved: dict[str, str] = {}
    failures = team_failures
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_find_portrait, name_by_key[name_key]): name_key
            for name_key in lookup_order
        }
        for future in as_completed(futures):
            name_key = futures[future]
            try:
                portrait = future.result()
            except requests.RequestException:
                failures += 1
                continue
            if portrait:
                resolved[name_key] = portrait
    cache.update(resolved)
    _write_cache(cache)

    portrait_by_key = {**cache}
    roster.loc[missing, "photo_url"] = roster.loc[missing, "name_key"].map(portrait_by_key)
    # Do not keep a known broken URL around: the UI will use initials for the
    # remaining unresolved players without making a failing image request.
    roster.loc[roster.photo_url.fillna("").str.startswith(BROKEN_ESPN_PREFIX), "photo_url"] = pd.NA
    roster["portrait_source"] = roster.photo_url.map(
        lambda value: "TheSportsDB" if isinstance(value, str) and "thesportsdb.com" in value else "Wikimedia Commons" if isinstance(value, str) and "wikimedia.org" in value else "Premier League" if pd.notna(value) else "Unavailable"
    )
    roster.drop(columns="name_key").to_csv(config.rosters_path, index=False)
    available = roster.photo_url.notna().sum()
    return {
        "league": config.name,
        "players": int(len(roster)),
        "portraits_available": int(available),
        "portraits_unavailable": int(len(roster) - available),
        "newly_resolved": int(len(team_resolved) + len(wikidata_resolved) + len(wiki_resolved) + len(resolved)),
        "lookup_failures": int(failures),
    }


def resolve_all_portraits() -> dict[str, dict[str, int | str]]:
    """Restore usable public portraits after a full five-league roster refresh."""
    return {
        key: resolve_league_portraits(key)
        for key in LEAGUES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve player portraits for current rosters")
    parser.add_argument("--league", choices=["all", *LEAGUES], default="all")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--individual-limit", type=int, default=60)
    parser.add_argument("--wiki-limit", type=int, default=None)
    parser.add_argument("--skip-team-source", action="store_true")
    args = parser.parse_args()
    keys = LEAGUES if args.league == "all" else [args.league]
    summaries = {
        key: resolve_league_portraits(
            key,
            args.max_workers,
            args.individual_limit,
            args.wiki_limit,
            team_lookup=not args.skip_team_source,
        )
        for key in keys
    }
    print(summaries)


if __name__ == "__main__":
    main()
