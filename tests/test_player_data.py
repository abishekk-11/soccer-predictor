"""Regression coverage for the player-intelligence data layer."""

from __future__ import annotations

import unittest

from src.player_data import LATEST_SEASON, PlayerDataStore


class PlayerDataStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = PlayerDataStore()

    def test_selected_rosters_return_all_players_with_portrait_urls(self) -> None:
        players = self.store.team_players("Arsenal")

        self.assertGreaterEqual(len(players), 20)
        self.assertTrue(all(player["photo_url"] for player in players))
        self.assertTrue(all("premierleague.com" in player["photo_url"] for player in players))

    def test_matchup_limits_performance_data_to_2025_26_or_earlier(self) -> None:
        payload = self.store.matchup(
            "Arsenal", "Coventry", {"home": 0.50, "draw": 0.25, "away": 0.25}
        )

        self.assertEqual(payload["latest_season"], LATEST_SEASON)
        self.assertEqual(len(payload["home"]["players"]), 28)
        self.assertEqual(len(payload["away"]["players"]), 28)
        self.assertTrue(all(0 <= player["score_likelihood"] <= 1 for player in payload["home"]["players"]))
        self.assertTrue(all(0 <= player["assist_likelihood"] <= 1 for player in payload["away"]["players"]))

    def test_score_rank_is_descending(self) -> None:
        payload = self.store.matchup(
            "Liverpool", "Man City", {"home": 0.42, "draw": 0.24, "away": 0.34}
        )
        score_likelihoods = [player["score_likelihood"] for player in payload["home"]["players"]]

        self.assertEqual(score_likelihoods, sorted(score_likelihoods, reverse=True))

    def test_matchup_exposes_likely_scorer_and_assister_for_each_team(self) -> None:
        payload = self.store.matchup(
            "Liverpool", "Man City", {"home": 0.42, "draw": 0.24, "away": 0.34}
        )

        for venue in ("home", "away"):
            prediction = payload["fixture_predictions"][venue]
            self.assertIsNotNone(prediction["likely_scorer"])
            self.assertIsNotNone(prediction["likely_assister"])
            self.assertGreater(prediction["likely_scorer"]["likelihood"], 0)
            self.assertGreater(prediction["likely_assister"]["likelihood"], 0)


if __name__ == "__main__":
    unittest.main()
