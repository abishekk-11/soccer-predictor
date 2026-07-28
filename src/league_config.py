"""Shared configuration for the five supported European leagues."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SEASON = "2026/27"
COMPLETED_SEASON = "2025-26"


@dataclass(frozen=True)
class LeagueConfig:
    """Stable IDs and local artefact names for one competition."""

    key: str
    name: str
    espn_league: str
    official_fixture_url: str
    results_url: str
    matches_file: str
    fixtures_file: str
    players_file: str
    rosters_file: str
    model_file: str
    metrics_file: str
    expected_teams: int

    @property
    def matches_path(self) -> Path:
        return DATA_DIR / self.matches_file

    @property
    def fixtures_path(self) -> Path:
        return DATA_DIR / self.fixtures_file

    @property
    def players_path(self) -> Path:
        return DATA_DIR / self.players_file

    @property
    def rosters_path(self) -> Path:
        return DATA_DIR / self.rosters_file

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / self.model_file

    @property
    def metrics_path(self) -> Path:
        return MODELS_DIR / self.metrics_file


LEAGUES: dict[str, LeagueConfig] = {
    "premier_league": LeagueConfig(
        key="premier_league",
        name="Premier League",
        espn_league="eng.1",
        official_fixture_url="https://www.premierleague.com/en/fixtures",
        results_url="https://www.premierleague.com/en/matches/premier-league/2025-26/",
        matches_file="matches.csv",
        fixtures_file="fixtures_2026_27.csv",
        players_file="player_season_stats.csv",
        rosters_file="player_rosters.csv",
        model_file="premier_league_predictor.joblib",
        metrics_file="model_metrics.json",
        expected_teams=20,
    ),
    "la_liga": LeagueConfig(
        key="la_liga",
        name="La Liga",
        espn_league="esp.1",
        official_fixture_url="https://www.laliga.com/en-GB/laliga-easports/calendar",
        results_url="https://www.laliga.com/en-GB/laliga-easports/calendar",
        matches_file="matches_la_liga.csv",
        fixtures_file="fixtures_la_liga_2026_27.csv",
        players_file="player_season_stats_la_liga.csv",
        rosters_file="player_rosters_la_liga.csv",
        model_file="la_liga_predictor.joblib",
        metrics_file="la_liga_metrics.json",
        expected_teams=20,
    ),
    "bundesliga": LeagueConfig(
        key="bundesliga",
        name="Bundesliga",
        espn_league="ger.1",
        official_fixture_url="https://www.bundesliga.com/en/bundesliga/matchday/2026-2027/1",
        results_url="https://www.bundesliga.com/en/bundesliga/matchday",
        matches_file="matches_bundesliga.csv",
        fixtures_file="fixtures_bundesliga_2026_27.csv",
        players_file="player_season_stats_bundesliga.csv",
        rosters_file="player_rosters_bundesliga.csv",
        model_file="bundesliga_predictor.joblib",
        metrics_file="bundesliga_metrics.json",
        expected_teams=18,
    ),
    "ligue_1": LeagueConfig(
        key="ligue_1",
        name="Ligue 1",
        espn_league="fra.1",
        official_fixture_url="https://ligue1.com/en/calendar/ligue1",
        results_url="https://ligue1.com/en/calendar/ligue1",
        matches_file="matches_ligue_1.csv",
        fixtures_file="fixtures_ligue_1_2026_27.csv",
        players_file="player_season_stats_ligue_1.csv",
        rosters_file="player_rosters_ligue_1.csv",
        model_file="ligue_1_predictor.joblib",
        metrics_file="ligue_1_metrics.json",
        expected_teams=18,
    ),
    "serie_a": LeagueConfig(
        key="serie_a",
        name="Serie A",
        espn_league="ita.1",
        official_fixture_url="https://en.legaseriea.it/serie-a/fixtures-results",
        results_url="https://en.legaseriea.it/serie-a/fixtures-results",
        matches_file="matches_serie_a.csv",
        fixtures_file="fixtures_serie_a_2026_27.csv",
        players_file="player_season_stats_serie_a.csv",
        rosters_file="player_rosters_serie_a.csv",
        model_file="serie_a_predictor.joblib",
        metrics_file="serie_a_metrics.json",
        expected_teams=20,
    ),
}


def get_league(key: str) -> LeagueConfig:
    """Return a configured league or raise a useful error for API callers."""
    try:
        return LEAGUES[key]
    except KeyError as error:
        available = ", ".join(LEAGUES)
        raise ValueError(f"Unknown league '{key}'. Choose one of: {available}.") from error
