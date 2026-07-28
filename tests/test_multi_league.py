"""Regression coverage for the four added European competitions."""

from __future__ import annotations

import unittest

import pandas as pd

from src.league_config import LEAGUES
from src.player_data import PlayerDataStore


class MultiLeagueDataTests(unittest.TestCase):
    def test_every_configured_league_has_a_complete_fixture_list(self) -> None:
        for config in LEAGUES.values():
            with self.subTest(league=config.key):
                fixtures = pd.read_csv(config.fixtures_path)
                teams = set(fixtures.HomeTeam).union(fixtures.AwayTeam)
                self.assertEqual(len(fixtures), config.expected_teams * (config.expected_teams - 1))
                self.assertEqual(len(teams), config.expected_teams)

    def test_added_leagues_have_all_ten_completed_player_seasons(self) -> None:
        expected_seasons = {f"{year}-{str(year + 1)[-2:]}" for year in range(2016, 2026)}
        for key in ("la_liga", "bundesliga", "ligue_1", "serie_a"):
            with self.subTest(league=key):
                seasons = set(pd.read_csv(LEAGUES[key].players_path).season)
                self.assertEqual(seasons, expected_seasons)

    def test_added_league_player_data_uses_appearance_basis(self) -> None:
        config = LEAGUES["serie_a"]
        store = PlayerDataStore(
            seasons_path=config.players_path,
            roster_path=config.rosters_path,
            availability_by_appearances=True,
        )
        players = store.team_players("Genoa")
        self.assertEqual(store.stat_basis, "appearances")
        self.assertGreater(len(players), 10)
        self.assertTrue(all("latest_appearances" in player for player in players))

    def test_added_league_rosters_do_not_contain_sparse_teams(self) -> None:
        for key in ("la_liga", "bundesliga", "ligue_1", "serie_a"):
            with self.subTest(league=key):
                roster = pd.read_csv(LEAGUES[key].rosters_path)
                self.assertGreaterEqual(roster.groupby("team").size().min(), 15)

    def test_added_league_teams_have_fixture_scorer_and_assist_signals(self) -> None:
        additional_paths = [
            config.players_path
            for key, config in LEAGUES.items()
            if key != "premier_league"
        ]
        for key in ("la_liga", "bundesliga", "ligue_1", "serie_a"):
            with self.subTest(league=key):
                config = LEAGUES[key]
                store = PlayerDataStore(
                    seasons_path=config.players_path,
                    roster_path=config.rosters_path,
                    availability_by_appearances=True,
                    additional_season_paths=additional_paths,
                )
                fixtures = pd.read_csv(config.fixtures_path)
                teams = sorted(set(fixtures.HomeTeam).union(fixtures.AwayTeam))
                for team in teams:
                    matchup = store.matchup(team, team, {"home": 0.4, "draw": 0.2, "away": 0.4})
                    prediction = matchup["fixture_predictions"]["home"]
                    self.assertIsNotNone(prediction["likely_scorer"], team)
                    self.assertIsNotNone(prediction["likely_assister"], team)

    def test_known_ligue_1_fixture_targets_have_working_portrait_sources(self) -> None:
        roster = pd.read_csv(LEAGUES["ligue_1"].rosters_path)
        targets = roster[roster.player_name.isin(["Ousmane Dembélé", "Pablo Pagis", "Moses Simon"])]
        self.assertEqual(set(targets.player_name), {"Ousmane Dembélé", "Pablo Pagis", "Moses Simon"})
        self.assertTrue(targets.photo_url.notna().all())
        self.assertTrue(targets.photo_url.str.startswith("https://").all())


if __name__ == "__main__":
    unittest.main()
