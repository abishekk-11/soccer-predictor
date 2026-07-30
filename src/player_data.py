"""Player-stat aggregation and pre-match scoring/assist signals.

The match model deliberately uses only team-level, pre-match data.  This
module is a separate, explanatory player layer: it ranks each selected club's
roster using completed player seasons through 2025/26.  It is not a confirmed
line-up or a betting model.
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PLAYER_SEASONS_PATH = ROOT / "data" / "player_season_stats.csv"
ROSTER_PATH = ROOT / "data" / "player_rosters.csv"
LATEST_SEASON = "2025-26"
FIRST_SEASON = "2016-17"


def _name_key(value: object) -> str:
    """Create a conservative comparison key for display-name fallbacks."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", normalized.lower())


def _number(value: object) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else 0.0


POSITION_PRIORS = {
    "Forward": {"goal": 0.34, "assist": 0.12},
    "Midfielder": {"goal": 0.15, "assist": 0.18},
    "Defender": {"goal": 0.05, "assist": 0.07},
    "Goalkeeper": {"goal": 0.002, "assist": 0.01},
}


class PlayerDataStore:
    """Read player archives and produce transparent, shrinkage-based ranks.

    Goals and assists use the whole supplied ten-season window.  Expected
    goals and expected assists are folded in only for seasons whose source
    actually provides those fields, so older zero-filled archive rows do not
    incorrectly look like zero chance creation.
    """

    def __init__(
        self,
        seasons_path: Path = PLAYER_SEASONS_PATH,
        roster_path: Path = ROSTER_PATH,
        latest_season: str | None = None,
        availability_by_appearances: bool = False,
        additional_season_paths: Iterable[Path] | None = None,
    ):
        season_paths = [Path(seasons_path), *(Path(path) for path in (additional_season_paths or []))]
        unique_season_paths = list(dict.fromkeys(season_paths))
        if not all(path.exists() for path in unique_season_paths) or not roster_path.exists():
            missing = [str(path.name) for path in (*unique_season_paths, roster_path) if not path.exists()]
            raise FileNotFoundError(
                f"Missing player data: {', '.join(missing)}. Run python -m src.update_player_data."
            )
        # A player who transferred into one of the four added leagues still
        # deserves credit for completed top-five-league output elsewhere. The
        # combined archive prevents those current roster entries looking blank
        # merely because their prior club played in another supported league.
        season_frames = [pd.read_csv(path) for path in unique_season_paths]
        self.seasons = pd.concat(season_frames, ignore_index=True)
        self.roster = pd.read_csv(roster_path)
        self.seasons["player_code"] = self.seasons["player_code"].astype(str)
        self.roster["player_code"] = self.roster["player_code"].astype(str)
        # The Premier League archive supplies minutes, while ESPN's public
        # roster feed supplies appearances.  Both are useful availability
        # signals; retain the distinction so the UI never presents an
        # invented minutes total as an official statistic.
        if "appearances" not in self.seasons:
            self.seasons["appearances"] = pd.to_numeric(self.seasons.get("minutes", 0), errors="coerce").fillna(0) / 70
        numeric_columns = ["goals", "assists", "minutes", "appearances", "starts", "xg", "xa"]
        for column in numeric_columns:
            self.seasons[column] = pd.to_numeric(self.seasons.get(column, 0), errors="coerce").fillna(0)
        # Players can change league midway through a season. Their completed
        # stints are additive, while duplicated provider rows collapse safely.
        self.seasons = self.seasons.groupby(
            ["season", "player_code", "player_name"], as_index=False
        )[numeric_columns].sum()
        self.seasons["name_key"] = self.seasons["player_name"].map(_name_key)
        self.roster["name_key"] = self.roster["player_name"].map(_name_key)
        # Indexed tables avoid constructing thousands of tiny DataFrames when
        # the API starts. Fixture requests then pull only the selected squads.
        self.seasons_by_code = self.seasons.set_index("player_code", drop=False).sort_index()
        self.seasons_by_name = self.seasons.set_index("name_key", drop=False).sort_index()
        self.player_codes = set(self.seasons_by_code.index)
        self.player_names = set(self.seasons_by_name.index)
        self.season_order = {
            season: index for index, season in enumerate(
                sorted(self.seasons["season"].dropna().unique())
            )
        }
        self.latest_season = latest_season or LATEST_SEASON
        self.availability_by_appearances = availability_by_appearances
        # ESPN's public European-league archive exposes appearances but not a
        # reliable minutes total.  Surface that distinction to consumers so a
        # missing value is never presented as official minutes played.
        self.stat_basis = "appearances" if availability_by_appearances else "minutes"
        self.latest_index = self.season_order.get(self.latest_season, max(self.season_order.values(), default=0))
        self.first_season = min(self.season_order, default=FIRST_SEASON)
        self.expected_stat_seasons = set(
            self.seasons.groupby("season")[["xg", "xa"]]
            .sum()
            .query("xg > 0 or xa > 0")
            .index
            .astype(str)
        )

    @staticmethod
    def _position(value: object) -> str:
        position = str(value or "").strip().title()
        return position if position in POSITION_PRIORS else "Midfielder"

    def _player_history(self, player_code: str, name_key: str) -> pd.DataFrame:
        if str(player_code) in self.player_codes:
            return self.seasons_by_code.loc[[str(player_code)]].copy()
        if name_key and name_key in self.player_names:
            return self.seasons_by_name.loc[[name_key]].copy()
        return pd.DataFrame()

    def _signals(self, history: pd.DataFrame, position: str) -> dict[str, float | int | str | None]:
        """Calculate empirical rates with recency weighting and minutes shrinkage."""
        prior = POSITION_PRIORS[position]
        if history.empty:
            return {
                "latest_goals": None, "latest_assists": None, "latest_minutes": None,
                "latest_appearances": None,
                "latest_starts": None, "latest_xg": None, "latest_xa": None,
                "career_goals": 0, "career_assists": 0, "career_minutes": 0,
                "career_appearances": 0,
                "seasons_played": 0, "last_active_season": None,
                "goal_rate_per90": prior["goal"], "assist_rate_per90": prior["assist"],
                "availability": 0.0, "has_history": False,
            }

        history = history.copy()
        history["season_index"] = history["season"].map(self.season_order).fillna(0)
        history["recency"] = 0.82 ** (self.latest_index - history["season_index"])
        rate_minutes = history["appearances"] * 70 if self.availability_by_appearances else history["minutes"]
        weighted_minutes = float((rate_minutes * history["recency"]).sum())
        weighted_goals = float((history["goals"] * history["recency"]).sum())
        weighted_assists = float((history["assists"] * history["recency"]).sum())
        # A 0.45-season positional prior dampens tiny samples without masking a
        # sustained scoring or creative record.
        prior_minutes = 0.45 * 900
        observed_goal_rate = 90 * (
            weighted_goals + prior["goal"] / 90 * prior_minutes
        ) / (weighted_minutes + prior_minutes)
        observed_assist_rate = 90 * (
            weighted_assists + prior["assist"] / 90 * prior_minutes
        ) / (weighted_minutes + prior_minutes)

        # The FPL archive starts carrying xG/xA partway through this ten-year
        # window.  Blend it with realised output only where it exists.  The
        # blend is capped so a short, recent xG sample cannot erase a player's
        # broader completed-season history.
        expected_history = history[history["season"].isin(self.expected_stat_seasons)]
        expected_rate_minutes = expected_history["appearances"] * 70 if self.availability_by_appearances else expected_history["minutes"]
        expected_minutes = float((expected_rate_minutes * expected_history["recency"]).sum())
        if expected_minutes:
            expected_goal_rate = 90 * float(
                (expected_history["xg"] * expected_history["recency"]).sum()
            ) / expected_minutes
            expected_assist_rate = 90 * float(
                (expected_history["xa"] * expected_history["recency"]).sum()
            ) / expected_minutes
            expected_weight = min(0.35, 0.35 * expected_minutes / 1_800)
            goal_rate = (1 - expected_weight) * observed_goal_rate + expected_weight * expected_goal_rate
            assist_rate = (1 - expected_weight) * observed_assist_rate + expected_weight * expected_assist_rate
        else:
            goal_rate = observed_goal_rate
            assist_rate = observed_assist_rate

        latest = history[history.season == self.latest_season]
        if latest.empty:
            latest_row = None
            latest_minutes = 0.0
        else:
            latest_row = latest.iloc[-1]
            latest_minutes = _number(latest_row.minutes)
        latest_appearances = 0.0 if latest_row is None else _number(latest_row.appearances)

        career_minutes = int(round(history.minutes.sum()))
        career_appearances = int(round(history.appearances.sum()))
        last_active = history[history.appearances > 0] if self.availability_by_appearances else history[history.minutes > 0]
        last_active_season = None if last_active.empty else str(last_active.sort_values("season_index").iloc[-1].season)
        # This is a role/availability proxy, not a prediction of a starting XI.
        availability = min(1.0, latest_appearances / 22) if self.availability_by_appearances else min(1.0, latest_minutes / 1_800)
        # A tiny, old sample should not turn a player into an apparent favourite.
        # A player with a meaningful recent PL record but no 2025/26 minutes is
        # retained at a low confidence level rather than treated as a starter.
        low_latest_sample = latest_appearances < 3 if self.availability_by_appearances else latest_minutes < 180
        meaningful_career = career_appearances >= 10 if self.availability_by_appearances else career_minutes >= 900
        if low_latest_sample:
            availability = 0.10 if meaningful_career and last_active_season in {"2024-25", "2025-26"} else 0.0

        latest_has_expected_stats = (
            latest_row is not None
            and str(latest_row.season) in self.expected_stat_seasons
        )
        return {
            "latest_goals": None if latest_row is None else int(_number(latest_row.goals)),
            "latest_assists": None if latest_row is None else int(_number(latest_row.assists)),
            "latest_minutes": None if latest_row is None or self.availability_by_appearances else int(_number(latest_row.minutes)),
            "latest_appearances": None if latest_row is None else int(latest_appearances),
            "latest_starts": None if latest_row is None else int(_number(latest_row.starts)),
            # The European historical feed does not publish reliable xG/xA.
            # Leave those cells unavailable rather than displaying 0.0.
            "latest_xg": round(_number(latest_row.xg), 2) if latest_has_expected_stats else None,
            "latest_xa": round(_number(latest_row.xa), 2) if latest_has_expected_stats else None,
            "career_goals": int(round(history.goals.sum())),
            "career_assists": int(round(history.assists.sum())),
            "career_minutes": career_minutes if not self.availability_by_appearances else None,
            "career_appearances": career_appearances,
            "seasons_played": int((history.appearances > 0).sum()) if self.availability_by_appearances else int((history.minutes > 0).sum()),
            "last_active_season": last_active_season,
            "goal_rate_per90": round(goal_rate, 3),
            "assist_rate_per90": round(assist_rate, 3),
            "availability": round(availability, 3),
            "has_history": bool(career_appearances > 0) if self.availability_by_appearances else bool(career_minutes > 0),
        }

    def team_players(self, team: str, roster_override: pd.DataFrame | None = None) -> list[dict]:
        roster_source = self.roster if roster_override is None else roster_override
        roster = roster_source[roster_source.team == team].copy()
        if roster.empty:
            return []
        if "name_key" not in roster:
            roster["name_key"] = roster.player_name.map(_name_key)
        players: list[dict] = []
        for row in roster.sort_values(["position", "player_name"]).itertuples(index=False):
            position = self._position(row.position)
            metrics = self._signals(self._player_history(row.player_code, row.name_key), position)
            players.append({
                "player_code": str(row.player_code),
                "name": str(row.player_name),
                "position": position,
                "photo_url": str(row.photo_url) if pd.notna(row.photo_url) else None,
                **metrics,
            })
        return players

    @staticmethod
    def _apply_match_likelihoods(players: list[dict], team_goal_expectation: float) -> list[dict]:
        if not players:
            return players
        scorer_signals = []
        assister_signals = []
        for player in players:
            position = player["position"]
            position_weight = {"Forward": 1.0, "Midfielder": 0.9, "Defender": 0.58, "Goalkeeper": 0.02}[position]
            scorer_signals.append(player["goal_rate_per90"] * player["availability"] * position_weight)
            assister_signals.append(player["assist_rate_per90"] * player["availability"] * position_weight)
        # Leave some of the team chance unallocated.  It represents unconfirmed
        # starters, new signings and players without a suitable PL sample.
        scorer_total = sum(scorer_signals) + 0.75
        assister_total = sum(assister_signals) + 0.75
        for player, scorer_signal, assister_signal in zip(players, scorer_signals, assister_signals):
            scorer_share = scorer_signal / scorer_total if scorer_total else 0.0
            assister_share = assister_signal / assister_total if assister_total else 0.0
            # Poisson-style conversion expresses a clearly labelled 1+ action
            # chance. It conditions only on historical role and team attack
            # expectation, not injuries or a confirmed line-up.
            player["score_likelihood"] = round(1 - math.exp(-team_goal_expectation * scorer_share), 3)
            player["assist_likelihood"] = round(1 - math.exp(-(team_goal_expectation * 0.82) * assister_share), 3)
        return players

    @staticmethod
    def _fixture_pick(players: list[dict], likelihood_key: str) -> dict | None:
        """Expose the leading fixture signal without claiming a line-up."""
        eligible = [
            player for player in players
            if (
                player["has_history"]
                and player["availability"] > 0
                and player["latest_appearances"] is not None
                and player["latest_appearances"] >= 3
                and player[likelihood_key] > 0
            )
        ]
        if not eligible:
            return None
        player = max(eligible, key=lambda item: (item[likelihood_key], item["availability"]))
        return {
            "player_code": player["player_code"],
            "name": player["name"],
            "position": player["position"],
            "photo_url": player["photo_url"],
            "likelihood": player[likelihood_key],
        }

    def matchup(
        self,
        home: str,
        away: str,
        probabilities: dict[str, float],
        roster_overrides: dict[str, pd.DataFrame] | None = None,
    ) -> dict:
        roster_overrides = roster_overrides or {}
        home_players = self.team_players(home, roster_overrides.get(home))
        away_players = self.team_players(away, roster_overrides.get(away))
        home_pressure = float(np.clip(1.1 + 0.75 * (probabilities["home"] - probabilities["away"]), 0.5, 1.9))
        away_pressure = float(np.clip(0.95 + 0.75 * (probabilities["away"] - probabilities["home"]), 0.4, 1.75))
        self._apply_match_likelihoods(home_players, home_pressure)
        self._apply_match_likelihoods(away_players, away_pressure)
        for players in (home_players, away_players):
            players.sort(key=lambda player: (player["score_likelihood"], player["assist_likelihood"]), reverse=True)
        return {
            "home": {"team": home, "players": home_players, "projected_team_goals": round(home_pressure, 2)},
            "away": {"team": away, "players": away_players, "projected_team_goals": round(away_pressure, 2)},
            "fixture_predictions": {
                "home": {
                    "likely_scorer": self._fixture_pick(home_players, "score_likelihood"),
                    "likely_assister": self._fixture_pick(home_players, "assist_likelihood"),
                },
                "away": {
                    "likely_scorer": self._fixture_pick(away_players, "score_likelihood"),
                    "likely_assister": self._fixture_pick(away_players, "assist_likelihood"),
                },
            },
            "latest_season": self.latest_season,
            "historical_window": f"{self.first_season} to {self.latest_season}",
            "methodology": (
                "Scores blend recency-weighted goals and assists per 90, available xG/xA, "
                "a position prior, recent minutes as a role proxy, and the team match forecast. "
                "They are not confirmed line-ups."
            ),
        }
