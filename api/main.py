"""Five-league API for pre-season 2026/27 fixture forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.league_config import COMPLETED_SEASON, LEAGUES, LeagueConfig, get_league
from src.live_results import LIVE_RESULTS_TTL, LIVE_SEASON_LABEL, live_results_for
from src.modeling import FEATURE_COLUMNS, FeatureState, completed_state
from src.player_data import PlayerDataStore


DEFAULT_LEAGUE = "premier_league"
FORM_WINDOW = 5
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
PREMIER_LEAGUE_PROMOTED_TEAMS = {"Coventry", "Hull", "Ipswich"}
PREMIER_LEAGUE_RESULTS_URL = "https://www.premierleague.com/en/matches/premier-league/2025-26/"

app = FastAPI(title="European Soccer Predictor", version="2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchInput(BaseModel):
    league: str = DEFAULT_LEAGUE
    home_team: str
    away_team: str
    fixture_date: date | None = None


@dataclass
class LeagueRuntime:
    config: LeagueConfig
    model_artifact: dict
    matches: pd.DataFrame
    fixtures: pd.DataFrame
    state: FeatureState
    player_store: PlayerDataStore
    teams: list[str]
    data_through: str
    default_fixture_date: date


def _load_state(config: LeagueConfig, matches: pd.DataFrame) -> FeatureState:
    state = completed_state(matches)
    # The existing Premier League workflow has a separate lower-division data
    # source for its newly promoted clubs.  The other leagues deliberately
    # retain neutral priors for clubs without top-flight history.
    if config.key == DEFAULT_LEAGUE:
        championship_path = config.matches_path.parent / "championship_2025_26.csv"
        if championship_path.exists():
            championship = pd.read_csv(championship_path, parse_dates=["Date"])
            for club in PREMIER_LEAGUE_PROMOTED_TEAMS:
                state.replace_with_promoted_history(club, championship)
    return state


def _load_runtime(config: LeagueConfig) -> LeagueRuntime:
    needed = [config.model_path, config.matches_path, config.fixtures_path, config.players_path, config.rosters_path]
    missing = [str(path.name) for path in needed if not path.exists()]
    if missing:
        raise RuntimeError(
            f"{config.name} is not ready: missing {', '.join(missing)}. "
            "Run `python -m src.train --league all` after refreshing the data."
        )
    matches = pd.read_csv(config.matches_path, parse_dates=["Date"])
    fixtures = pd.read_csv(config.fixtures_path, parse_dates=["Date"])
    teams = sorted(set(fixtures.HomeTeam).union(fixtures.AwayTeam))
    if len(teams) != config.expected_teams:
        raise RuntimeError(f"Expected {config.expected_teams} {config.name} teams, found {len(teams)}")
    cross_league_player_archives = (
        [league.players_path for key, league in LEAGUES.items() if key != DEFAULT_LEAGUE]
        if config.key != DEFAULT_LEAGUE else None
    )
    return LeagueRuntime(
        config=config,
        model_artifact=joblib.load(config.model_path),
        matches=matches,
        fixtures=fixtures.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True),
        state=_load_state(config, matches),
        player_store=PlayerDataStore(
            seasons_path=config.players_path,
            roster_path=config.rosters_path,
            availability_by_appearances=config.key != DEFAULT_LEAGUE,
            additional_season_paths=cross_league_player_archives,
        ),
        teams=teams,
        data_through=matches.Date.max().date().isoformat(),
        default_fixture_date=fixtures.Date.min().date(),
    )


# Player portfolios are substantially larger than the match model. Load a
# league on demand and cache it so opening the app is fast while a later league
# switch still has its full roster data ready after the first request.
RUNTIMES: dict[str, LeagueRuntime] = {}


def runtime_for(league_key: str) -> LeagueRuntime:
    try:
        config = get_league(league_key)
    except (KeyError, ValueError) as error:
        available = ", ".join(LEAGUES)
        raise HTTPException(status_code=404, detail=f"Unknown league '{league_key}'. Choose one of: {available}.") from error
    if league_key not in RUNTIMES:
        RUNTIMES[league_key] = _load_runtime(config)
    return RUNTIMES[league_key]


def _require_teams(runtime: LeagueRuntime, home: str, away: str) -> None:
    if home not in runtime.teams or away not in runtime.teams:
        raise HTTPException(status_code=400, detail=f"Both clubs must be in the 2026/27 {runtime.config.name}.")
    if home == away:
        raise HTTPException(status_code=400, detail="Home and away teams must be different.")


def recent_team_stats(runtime: LeagueRuntime, team: str, last_n: int = FORM_WINDOW) -> dict:
    """Return the latest five league results, refreshed during the season."""
    competition = runtime.config.name
    source_url = runtime.config.results_url
    source_matches = runtime.matches[runtime.matches.Season == COMPLETED_SEASON]
    if runtime.config.key == DEFAULT_LEAGUE and team in PREMIER_LEAGUE_PROMOTED_TEAMS:
        championship_path = runtime.config.matches_path.parent / "championship_2025_26.csv"
        source_matches = pd.read_csv(championship_path, parse_dates=["Date"])
        competition = "EFL Championship"
        source_url = "https://www.efl.com/competitions/championship/"

    # Current-season completed matches are pulled at request time and cached
    # briefly. Combine them with the prior completed season so the opening
    # rounds still have a meaningful five-match window, then naturally age the
    # old season out as 2026/27 results accumulate.
    live_results = live_results_for(runtime.config)
    all_matches = pd.concat([source_matches, live_results.matches], ignore_index=True)
    all_matches["Date"] = pd.to_datetime(all_matches["Date"], errors="coerce")
    all_matches = all_matches.dropna(subset=["Date"])
    games = all_matches[
        (all_matches.HomeTeam == team) | (all_matches.AwayTeam == team)
    ].sort_values(["Date", "HomeTeam", "AwayTeam"]).tail(last_n)
    live_games = live_results.matches[
        (live_results.matches.HomeTeam == team) | (live_results.matches.AwayTeam == team)
    ]
    is_live = not live_games.empty
    if is_live:
        competition = runtime.config.name
        source_url = live_results.source_url
    if games.empty:
        return {
            "form": [], "win_rate": 0.0, "goals_scored": 0.0,
            "goals_conceded": 0.0, "matches_used": 0, "matches": [],
            "competition": competition, "source_season": COMPLETED_SEASON,
            "source_url": source_url, "data_through": None,
            "live": False, "refresh_minutes": int(LIVE_RESULTS_TTL.total_seconds() // 60),
        }

    form: list[str] = []
    fixture_details: list[dict] = []
    for _, game in games.iterrows():
        is_home = game.HomeTeam == team
        opponent = game.AwayTeam if is_home else game.HomeTeam
        goals_for = int(game.FTHG if is_home else game.FTAG)
        goals_against = int(game.FTAG if is_home else game.FTHG)
        result = "W" if goals_for > goals_against else "D" if goals_for == goals_against else "L"
        form.append(result)
        fixture_details.append({
            "date": game.Date.date().isoformat(),
            "opponent": opponent,
            "venue": "Home" if is_home else "Away",
            "goals_for": goals_for,
            "goals_against": goals_against,
            "result": result,
        })
    return {
        "form": form,
        "win_rate": form.count("W") / len(form),
        "goals_scored": sum(match["goals_for"] for match in fixture_details),
        "goals_conceded": sum(match["goals_against"] for match in fixture_details),
        "matches_used": len(fixture_details),
        "matches": fixture_details,
        "competition": competition,
        "source_season": f"{LIVE_SEASON_LABEL} · live" if is_live else COMPLETED_SEASON,
        "source_url": source_url,
        "data_through": games.Date.max().date().isoformat(),
        "live": is_live,
        "refresh_minutes": int(LIVE_RESULTS_TTL.total_seconds() // 60),
    }


def build_features(runtime: LeagueRuntime, home: str, away: str, fixture_date: date) -> pd.DataFrame:
    row = runtime.state.preview(home, away, pd.Timestamp(fixture_date))
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def probabilities_for(runtime: LeagueRuntime, features: pd.DataFrame) -> dict[str, float]:
    artifact = runtime.model_artifact
    classes = np.asarray(artifact["classes"])
    logistic = artifact["logistic_model"].predict_proba(features)[0]
    boosted = artifact["boosted_model"].predict_proba(features)[0]

    def align(raw: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
        values = np.zeros(len(classes))
        for raw_index, label in enumerate(model_classes):
            values[np.where(classes == label)[0][0]] = raw[raw_index]
        return values

    logistic = align(logistic, artifact["logistic_model"].classes_)
    boosted = align(boosted, artifact["boosted_model"].classes_)
    probability_vector = (1 - artifact["boosted_weight"]) * logistic + artifact["boosted_weight"] * boosted
    return {label: float(probability_vector[index]) for index, label in enumerate(classes)}


def prediction_payload(runtime: LeagueRuntime, home: str, away: str, fixture_date: date) -> dict:
    probabilities = probabilities_for(runtime, build_features(runtime, home, away, fixture_date))
    predicted_code = max(probabilities, key=probabilities.get)
    predicted_label = {"H": "Home Team Wins", "D": "Draw", "A": "Away Team Wins"}[predicted_code]
    ordered = {"home": probabilities["H"], "draw": probabilities["D"], "away": probabilities["A"]}
    return {
        "league": runtime.config.key,
        "league_name": runtime.config.name,
        "home_team": home,
        "away_team": away,
        "fixture_date": fixture_date.isoformat(),
        "prediction": predicted_label,
        "confidence": round(max(ordered.values()), 3),
        "probabilities": {key: round(value, 3) for key, value in ordered.items()},
        "fair_odds": {key: round(1 / value, 2) for key, value in ordered.items()},
        "form_window": FORM_WINDOW,
        "data_through": runtime.data_through,
    }


@app.get("/health")
def health_check():
    """Lightweight health check used by the hosting provider."""
    return {"status": "ok", "version": app.version}


@app.get("/leagues")
def get_leagues():
    return {
        "season": "2026/27",
        "leagues": [
            {
                "key": config.key,
                "name": config.name,
                "team_count": config.expected_teams,
                "fixtures_loaded": config.expected_teams * (config.expected_teams - 1),
                "official_fixture_url": config.official_fixture_url,
            }
            for config in LEAGUES.values()
        ],
    }


@app.get("/model-info")
def model_info(league: str = DEFAULT_LEAGUE):
    runtime = runtime_for(league)
    metrics = runtime.model_artifact["metrics"]
    return {
        "league": runtime.config.key,
        "league_name": runtime.config.name,
        "season": "2026/27",
        "teams": runtime.teams,
        "team_count": len(runtime.teams),
        "data_through": runtime.data_through,
        "fixtures_loaded": int(len(runtime.fixtures)),
        "official_fixture_url": runtime.config.official_fixture_url,
        "validation": metrics["selected_blend"],
        "validation_method": metrics["evaluation_method"],
        "promoted_club_handling": (
            "Coventry, Hull, and Ipswich use 2025/26 Championship form with a conservative division adjustment."
            if runtime.config.key == DEFAULT_LEAGUE
            else "Clubs without 2025/26 top-flight history use neutral pre-season model priors."
        ),
    }


@app.get("/teams")
def get_teams(league: str = DEFAULT_LEAGUE):
    runtime = runtime_for(league)
    return {"league": runtime.config.key, "league_name": runtime.config.name, "teams": runtime.teams, "season": "2026/27"}


@app.get("/fixtures")
def get_fixtures(league: str = DEFAULT_LEAGUE):
    runtime = runtime_for(league)
    return {
        "league": runtime.config.key,
        "league_name": runtime.config.name,
        "season": "2026/27",
        "official_fixture_url": runtime.config.official_fixture_url,
        "fixtures": [
            {"date": row.Date.date().isoformat(), "home_team": row.HomeTeam, "away_team": row.AwayTeam}
            for _, row in runtime.fixtures.iterrows()
        ],
    }


@app.get("/form/{team}")
def team_form(team: str, league: str = DEFAULT_LEAGUE):
    runtime = runtime_for(league)
    if team not in runtime.teams:
        raise HTTPException(status_code=404, detail=f"Team is not in the 2026/27 {runtime.config.name}.")
    stats = recent_team_stats(runtime, team)
    return {
        "league": runtime.config.key,
        "league_name": runtime.config.name,
        "team": team,
        "form": stats["form"],
        "goals_scored": stats["goals_scored"],
        "goals_conceded": stats["goals_conceded"],
        "matches_used": stats["matches_used"],
        "matches": stats["matches"],
        "source_competition": stats["competition"],
        "source_season": stats["source_season"],
        "source_url": stats["source_url"],
        "data_through": stats["data_through"],
        "live": stats["live"],
        "refresh_minutes": stats["refresh_minutes"],
        "current_season": LIVE_SEASON_LABEL,
    }


@app.get("/match-players")
def match_players(
    home_team: str,
    away_team: str,
    league: str = DEFAULT_LEAGUE,
    fixture_date: date | None = None,
):
    """Return the selected match's rosters and score/assist signals."""
    runtime = runtime_for(league)
    _require_teams(runtime, home_team, away_team)
    selected_date = fixture_date or runtime.default_fixture_date
    forecast = prediction_payload(runtime, home_team, away_team, selected_date)
    payload = runtime.player_store.matchup(home_team, away_team, forecast["probabilities"])
    return {
        "league": runtime.config.key,
        "league_name": runtime.config.name,
        "home_team": home_team,
        "away_team": away_team,
        "fixture_date": selected_date.isoformat(),
        "forecast_probabilities": forecast["probabilities"],
        "source_url": runtime.config.results_url,
        "player_stat_basis": runtime.player_store.stat_basis,
        **payload,
    }


@app.post("/predict")
def predict(match: MatchInput):
    runtime = runtime_for(match.league)
    _require_teams(runtime, match.home_team, match.away_team)
    fixture_date = match.fixture_date or runtime.default_fixture_date
    return prediction_payload(runtime, match.home_team, match.away_team, fixture_date)


# Keep the dashboard and its API on one origin in production.  Mounting this
# last preserves the explicit API and documentation routes above.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
