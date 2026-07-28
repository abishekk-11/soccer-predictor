"""Build the local player archive used by the player-intelligence UI.

No 2026/27 performance is imported.  The 2026/27 roster list is used solely
to identify the clubs' players and their official portrait identifiers; stats
are restricted to the ten completed seasons ending in 2025/26.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FPL_ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv"
ROSTER_BASE = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/2026-2027/{name}.csv"
SEASONS = [f"{year}-{str(year + 1)[-2:]}" for year in range(2016, 2026)]

TEAM_ALIASES = {
    "Bournemouth": "Bournemouth",
    "Brighton": "Brighton",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Man Utd": "Man United",
    "Nott'm Forest": "Nott'm Forest",
    "Spurs": "Tottenham",
}


def get_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def number_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0)


def player_seasons() -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for season in SEASONS:
        raw = get_csv(FPL_ARCHIVE.format(season=season))
        player_name = raw["first_name"].fillna("").str.strip() + " " + raw["second_name"].fillna("").str.strip()
        stats = pd.DataFrame({
            "season": season,
            "player_code": raw["code"].astype("Int64").astype(str),
            "player_name": player_name.str.replace(r"\s+", " ", regex=True).str.strip(),
            "goals": number_column(raw, "goals_scored"),
            "assists": number_column(raw, "assists"),
            "minutes": number_column(raw, "minutes"),
            "starts": number_column(raw, "starts"),
            "xg": number_column(raw, "expected_goals"),
            "xa": number_column(raw, "expected_assists"),
        })
        records.append(stats)
    return pd.concat(records, ignore_index=True)


def roster() -> pd.DataFrame:
    players = get_csv(ROSTER_BASE.format(name="players"))
    teams = get_csv(ROSTER_BASE.format(name="teams"))
    merged = players.merge(teams[["code", "name"]], left_on="team_code", right_on="code", how="left")
    result = pd.DataFrame({
        "team": merged["name"].map(lambda value: TEAM_ALIASES.get(str(value), str(value))),
        "player_code": merged["player_code"].astype("Int64").astype(str),
        "player_name": (
            merged["first_name"].fillna("").str.strip() + " " + merged["second_name"].fillna("").str.strip()
        ).str.replace(r"\s+", " ", regex=True).str.strip(),
        "position": merged["position"],
        "photo_url": merged["player_code"].map(
            lambda code: f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{int(code)}.png"
        ),
    })
    return result.sort_values(["team", "position", "player_name"]).reset_index(drop=True)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    seasons = player_seasons()
    team_roster = roster()
    seasons.to_csv(DATA / "player_season_stats.csv", index=False)
    team_roster.to_csv(DATA / "player_rosters.csv", index=False)
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["player_data"] = {
        "seasons": SEASONS,
        "latest_performance_season": "2025-26",
        "player_season_rows": int(len(seasons)),
        "roster_rows": int(len(team_roster)),
        "sources": {
            "historical_stats": "https://github.com/vaastav/Fantasy-Premier-League",
            "rosters_and_portraits": "https://github.com/olbauday/FPL-Core-Insights",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["player_data"], indent=2))


if __name__ == "__main__":
    main()
