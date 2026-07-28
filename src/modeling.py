"""Leakage-safe, pre-match feature construction for football predictions.

Every feature is calculated from matches that had finished before the fixture
being scored.  Matches on the same calendar day are scored as a batch before
any of that day's results are added to the state.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_difference",
    "home_form_ppg_5",
    "away_form_ppg_5",
    "home_form_gf_5",
    "away_form_gf_5",
    "home_form_ga_5",
    "away_form_ga_5",
    "home_home_ppg_5",
    "away_away_ppg_5",
    "home_season_ppg",
    "away_season_ppg",
    "home_season_goal_difference",
    "away_season_goal_difference",
    "home_games_in_sample",
    "away_games_in_sample",
    "home_rest_days",
    "away_rest_days",
]

DEFAULT_ELO = 1500.0
HOME_ADVANTAGE_ELO = 55.0
ELO_K = 24.0


@dataclass
class TeamState:
    elo: float = DEFAULT_ELO
    recent: deque = field(default_factory=lambda: deque(maxlen=10))
    home_recent: deque = field(default_factory=lambda: deque(maxlen=5))
    away_recent: deque = field(default_factory=lambda: deque(maxlen=5))
    season: list = field(default_factory=list)
    season_name: str | None = None
    last_played: pd.Timestamp | None = None


def _mean(values: Iterable[float], default: float) -> float:
    values = list(values)
    return float(np.mean(values)) if values else default


def _stats(games: Iterable[tuple[int, int, int]], default_ppg: float = 1.35) -> tuple[float, float, float]:
    games = list(games)
    if not games:
        return default_ppg, 1.35, 1.35
    return (
        _mean((game[2] for game in games), default_ppg),
        _mean((game[0] for game in games), 1.35),
        _mean((game[1] for game in games), 1.35),
    )


class FeatureState:
    """A mutable record of team strength and completed-match form."""

    def __init__(self) -> None:
        self.teams: defaultdict[str, TeamState] = defaultdict(TeamState)

    def _season_stats(self, team: TeamState) -> tuple[float, float]:
        if not team.season:
            return 1.35, 0.0
        ppg, gf, ga = _stats(team.season)
        return ppg, gf - ga

    @staticmethod
    def _rest_days(team: TeamState, fixture_date: pd.Timestamp) -> float:
        if team.last_played is None:
            return 7.0
        days = (fixture_date.normalize() - team.last_played.normalize()).days
        return float(np.clip(days, 2, 21))

    def preview(self, home: str, away: str, fixture_date: pd.Timestamp | date) -> dict[str, float]:
        """Return the model row that was knowable before a fixture."""
        fixture_date = pd.Timestamp(fixture_date)
        home_state = self.teams[home]
        away_state = self.teams[away]
        home_ppg, home_gf, home_ga = _stats(home_state.recent)
        away_ppg, away_gf, away_ga = _stats(away_state.recent)
        home_venue_ppg, _, _ = _stats(home_state.home_recent)
        away_venue_ppg, _, _ = _stats(away_state.away_recent)
        home_season_ppg, home_season_gd = self._season_stats(home_state)
        away_season_ppg, away_season_gd = self._season_stats(away_state)

        return {
            "home_elo": home_state.elo,
            "away_elo": away_state.elo,
            "elo_difference": home_state.elo - away_state.elo,
            "home_form_ppg_5": home_ppg,
            "away_form_ppg_5": away_ppg,
            "home_form_gf_5": home_gf,
            "away_form_gf_5": away_gf,
            "home_form_ga_5": home_ga,
            "away_form_ga_5": away_ga,
            "home_home_ppg_5": home_venue_ppg,
            "away_away_ppg_5": away_venue_ppg,
            "home_season_ppg": home_season_ppg,
            "away_season_ppg": away_season_ppg,
            "home_season_goal_difference": home_season_gd,
            "away_season_goal_difference": away_season_gd,
            "home_games_in_sample": min(len(home_state.recent), 10),
            "away_games_in_sample": min(len(away_state.recent), 10),
            "home_rest_days": self._rest_days(home_state, fixture_date),
            "away_rest_days": self._rest_days(away_state, fixture_date),
        }

    def update(self, row: pd.Series) -> None:
        """Add a completed match to the state after its features were made."""
        home = str(row.HomeTeam)
        away = str(row.AwayTeam)
        fixture_date = pd.Timestamp(row.Date)
        season = str(row.Season)
        home_state = self.teams[home]
        away_state = self.teams[away]

        for team in (home_state, away_state):
            if team.season_name != season:
                team.season.clear()
                team.season_name = season

        home_goals = int(row.FTHG)
        away_goals = int(row.FTAG)
        home_points = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
        away_points = 3 if away_goals > home_goals else 1 if home_goals == away_goals else 0

        expected_home = 1 / (1 + 10 ** (-((home_state.elo + HOME_ADVANTAGE_ELO - away_state.elo) / 400)))
        actual_home = 1.0 if home_points == 3 else 0.5 if home_points == 1 else 0.0
        goal_multiplier = np.log(abs(home_goals - away_goals) + 1) * 2.2 / (
            2.2 + 0.001 * abs(home_state.elo - away_state.elo)
        )
        home_change = ELO_K * max(goal_multiplier, 1.0) * (actual_home - expected_home)
        home_state.elo += home_change
        away_state.elo -= home_change

        home_game = (home_goals, away_goals, home_points)
        away_game = (away_goals, home_goals, away_points)
        home_state.recent.append(home_game)
        home_state.home_recent.append(home_game)
        home_state.season.append(home_game)
        home_state.last_played = fixture_date
        away_state.recent.append(away_game)
        away_state.away_recent.append(away_game)
        away_state.season.append(away_game)
        away_state.last_played = fixture_date

    def replace_with_promoted_history(self, team_name: str, lower_league_matches: pd.DataFrame) -> None:
        """Seed a promoted team's recent form from its latest Championship season.

        The score uses a conservative Premier-League adjustment so that strong
        Championship results do not get treated as directly equivalent to top
        flight results.  The historical Premier League ratings of other clubs
        are deliberately left untouched.
        """
        club_matches = lower_league_matches[
            (lower_league_matches.HomeTeam == team_name)
            | (lower_league_matches.AwayTeam == team_name)
        ].sort_values("Date")
        if club_matches.empty:
            return

        lower_state = FeatureState()
        for _, match in club_matches.iterrows():
            lower_state.update(match)
        source = lower_state.teams[team_name]
        target = self.teams[team_name]
        ppg, _, _ = _stats(source.season)
        # A one-division promotion penalty is built in; it is intentionally
        # modest so that results remain probability estimates, not certainties.
        target.elo = float(np.clip(DEFAULT_ELO + 100 * (ppg - 1.35) - 90, 1380, 1510))
        target.recent = deque(source.recent, maxlen=10)
        target.home_recent = deque(source.home_recent, maxlen=5)
        target.away_recent = deque(source.away_recent, maxlen=5)
        target.season = list(source.season)
        target.season_name = source.season_name
        target.last_played = source.last_played


def build_feature_table(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create one no-leakage feature row and target for each completed match."""
    required = {"Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ordered = matches.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"], errors="raise")
    ordered = ordered.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    state = FeatureState()
    features: list[dict[str, float]] = []
    targets: list[str] = []
    feature_dates: list[pd.Timestamp] = []

    # No same-day results are available to another fixture's prediction.
    for fixture_date, day_matches in ordered.groupby("Date", sort=True):
        day_rows = []
        for _, row in day_matches.iterrows():
            day_rows.append((state.preview(row.HomeTeam, row.AwayTeam, fixture_date), row))
        for feature_row, row in day_rows:
            features.append(feature_row)
            targets.append(str(row.FTR))
            feature_dates.append(fixture_date)
            state.update(row)

    X = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    X.index = pd.DatetimeIndex(feature_dates, name="Date")
    return X, pd.Series(targets, index=X.index, name="FTR")


def completed_state(matches: pd.DataFrame) -> FeatureState:
    """Replay completed results and return the current pre-match state."""
    state = FeatureState()
    ordered = matches.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"], errors="raise")
    ordered = ordered.sort_values(["Date", "HomeTeam", "AwayTeam"])
    for _, row in ordered.iterrows():
        state.update(row)
    return state
