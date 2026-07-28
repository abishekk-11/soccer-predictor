"""Download, validate, and store the data needed for the 2026/27 predictor."""

from __future__ import annotations

import json
import re
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.update_player_data import main as update_player_data
from src.update_european_data import refresh_all as refresh_european_leagues
from src.update_player_portraits import resolve_all_portraits


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281/{season}/{division}.csv"
FIXTURE_URL = "https://www.premierleague.com/en/news/4675097/all-380-fixtures-for-202627-premier-league-season/"

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
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
    "Tottenham Hotspur": "Tottenham",
}


def canonical_team(value: str) -> str:
    return TEAM_ALIASES.get(str(value).strip(), str(value).strip())


def season_label(code: str) -> str:
    return f"20{code[:2]}-{code[2:]}"


def download_csv(code: str, division: str) -> pd.DataFrame:
    url = FOOTBALL_DATA.format(season=code, division=division)
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    frame = pd.read_csv(BytesIO(response.content), encoding="utf-8-sig")
    if len(frame) == 0:
        raise ValueError(f"No rows returned from {url}")
    return frame


def normalise(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    columns = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing {sorted(missing)}")
    result = frame[columns].copy()
    result["Date"] = pd.to_datetime(result["Date"], dayfirst=True, errors="coerce")
    result["HomeTeam"] = result["HomeTeam"].map(canonical_team)
    result["AwayTeam"] = result["AwayTeam"].map(canonical_team)
    result["FTHG"] = pd.to_numeric(result["FTHG"], errors="coerce")
    result["FTAG"] = pd.to_numeric(result["FTAG"], errors="coerce")
    result["FTR"] = result["FTR"].astype(str).str.strip()
    result["Season"] = season
    result = result.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    result = result[result.FTR.isin(["H", "D", "A"])].copy()
    result[["FTHG", "FTAG"]] = result[["FTHG", "FTAG"]].astype(int)
    return result


def historical_premier_league() -> pd.DataFrame:
    existing = pd.read_csv(DATA / "matches.csv", encoding="utf-8-sig")
    old = existing[existing["Season"].astype(str).str[:4].astype(int) < 2018].copy()
    old["Date"] = pd.to_datetime(old["Date"], dayfirst=True, errors="coerce")
    old["HomeTeam"] = old["HomeTeam"].map(canonical_team)
    old["AwayTeam"] = old["AwayTeam"].map(canonical_team)
    old = old[["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season"]]
    downloaded = [normalise(download_csv(code, "E0"), season_label(code)) for code in (
        "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"
    )]
    all_matches = pd.concat([old, *downloaded], ignore_index=True)
    all_matches = all_matches.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season"]
    ).sort_values(["Date", "HomeTeam", "AwayTeam"])
    expected_rows = sum(380 for _ in downloaded)
    if sum(len(frame) for frame in downloaded) != expected_rows:
        raise ValueError("One or more completed Premier League seasons does not contain 380 matches")
    return all_matches.reset_index(drop=True)


def parse_fixtures(html: str) -> pd.DataFrame:
    text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
    fixture_rows: list[dict[str, object]] = []
    current_date: date | None = None
    previous_month: int | None = None
    days = {
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    }
    months = {
        name: number for number, name in enumerate(
            ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1
        )
    }
    date_pattern = re.compile(
        r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) (\d{1,2}) "
        r"(January|February|March|April|May|June|July|August|September|October|November|December)(?: (20\d{2}))?$"
    )
    fixture_pattern = re.compile(r"^(?:\d{1,2}:\d{2} )?(.+?) v (.+?)(?: \([^)]*\))?\*?$")

    for line in text.splitlines():
        line = line.strip().rstrip("*").strip()
        matched_date = date_pattern.match(line)
        if matched_date:
            _, day_text, month_name, explicit_year = matched_date.groups()
            month = months[month_name]
            if explicit_year:
                year = int(explicit_year)
            elif current_date is None:
                raise ValueError("Fixture page begins without a full date")
            else:
                year = current_date.year + int(previous_month is not None and month < previous_month)
            current_date = date(year, month, int(day_text))
            previous_month = month
            continue

        matched_fixture = fixture_pattern.match(line)
        if not matched_fixture or current_date is None:
            continue
        home, away = matched_fixture.groups()
        home, away = canonical_team(home), canonical_team(away)
        # This page has fixture-looking explanatory prose, so accept only the
        # official 2026/27 members.
        season_teams = {
            "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea",
            "Coventry", "Crystal Palace", "Everton", "Fulham", "Hull", "Ipswich", "Leeds",
            "Liverpool", "Man City", "Man United", "Newcastle", "Nott'm Forest", "Sunderland", "Tottenham",
        }
        if home in season_teams and away in season_teams:
            fixture_rows.append({"Date": current_date.isoformat(), "HomeTeam": home, "AwayTeam": away})

    fixtures = pd.DataFrame(fixture_rows).drop_duplicates().sort_values(["Date", "HomeTeam"])
    if len(fixtures) != 380:
        raise ValueError(f"Expected 380 official fixtures, parsed {len(fixtures)}")
    return fixtures.reset_index(drop=True)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    premier_league = historical_premier_league()
    championship = normalise(download_csv("2526", "E1"), "2025-26")
    fixture_response = requests.get(FIXTURE_URL, timeout=45)
    fixture_response.raise_for_status()
    fixtures = parse_fixtures(fixture_response.text)

    premier_league.to_csv(DATA / "matches.csv", index=False, date_format="%Y-%m-%d")
    premier_league[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].to_csv(
        DATA / "clean_matches.csv", index=False
    )
    championship.to_csv(DATA / "championship_2025_26.csv", index=False, date_format="%Y-%m-%d")
    fixtures.to_csv(DATA / "fixtures_2026_27.csv", index=False)

    manifest = {
        "updated_on": pd.Timestamp.utcnow().isoformat(),
        "premier_league_rows": int(len(premier_league)),
        "premier_league_data_through": str(premier_league.Date.max().date()),
        "championship_2025_26_rows": int(len(championship)),
        "fixtures_2026_27_rows": int(len(fixtures)),
        "sources": {
            "results": "https://www.football-data.co.uk/englandm.php",
            "fixtures": FIXTURE_URL,
        },
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Keep roster identities and the completed-player-season archive in sync
    # whenever the normal data-refresh command is run.
    update_player_data()
    manifest["european_leagues"] = refresh_european_leagues()
    # The European roster feed often supplies unusable CDN image paths. Put
    # cached, working public portraits back after rebuilding the rosters.
    manifest["player_portraits"] = resolve_all_portraits()
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
