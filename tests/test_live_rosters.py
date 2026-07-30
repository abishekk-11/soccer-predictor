from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from src import live_rosters
from src.league_config import LEAGUES


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def _athlete(index: int, name: str) -> dict:
    return {
        "id": str(1000 + index),
        "displayName": name,
        "position": {"displayName": "Midfielder"},
        "headshot": None,
    }


class LiveRosterTests(TestCase):
    def setUp(self) -> None:
        live_rosters._roster_cache.clear()
        live_rosters._directory_cache.clear()
        self.config = LEAGUES["la_liga"]
        self.fixtures = pd.DataFrame([
            {"HomeTeam": "Example FC", "AwayTeam": "Other FC", "HomeTeamId": "96", "AwayTeamId": "97"},
        ])
        self.bundled = pd.DataFrame([
            {
                "team": "Example FC",
                "player_code": "old-1",
                "player_name": "Current Player 1",
                "position": "Midfielder",
                "photo_url": "https://images.example/current-player.jpg",
                "roster_source": "bundled",
            },
            {
                "team": "Example FC",
                "player_code": "departed-9",
                "player_name": "Departed Player",
                "position": "Forward",
                "photo_url": None,
                "roster_source": "bundled",
            },
        ])
        self.now = datetime(2026, 7, 30, tzinfo=timezone.utc)

    def test_complete_live_squad_replaces_stale_bundled_membership(self) -> None:
        payload = {"athletes": [_athlete(index, f"Current Player {index}") for index in range(1, 17)]}
        with patch("src.live_rosters.requests.get", return_value=_Response(payload)) as get:
            snapshot = live_rosters.live_roster_for(
                self.config, self.fixtures, "Example FC", self.bundled, now=self.now
            )
            cached = live_rosters.live_roster_for(
                self.config, self.fixtures, "Example FC", self.bundled,
                now=self.now + timedelta(minutes=30),
            )

        self.assertTrue(snapshot.live)
        self.assertEqual(len(snapshot.players), 16)
        self.assertNotIn("Departed Player", set(snapshot.players.player_name))
        self.assertEqual(
            snapshot.players.loc[snapshot.players.player_name == "Current Player 1", "photo_url"].iloc[0],
            "https://images.example/current-player.jpg",
        )
        self.assertIs(cached, snapshot)
        self.assertEqual(get.call_count, 1)

    def test_incomplete_live_response_keeps_the_bundled_squad(self) -> None:
        payload = {"athletes": [_athlete(index, f"Current Player {index}") for index in range(1, 4)]}
        with patch("src.live_rosters.requests.get", return_value=_Response(payload)):
            snapshot = live_rosters.live_roster_for(
                self.config, self.fixtures, "Example FC", self.bundled, now=self.now
            )

        self.assertFalse(snapshot.live)
        self.assertEqual(set(snapshot.players.player_name), {"Current Player 1", "Departed Player"})
        self.assertIn("incomplete squad", snapshot.refresh_error or "")
