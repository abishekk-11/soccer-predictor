"""Coverage for the automatic current-season form layer."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import pandas as pd

from api.main import recent_team_stats
from src.league_config import COMPLETED_SEASON, LEAGUES
from src.live_results import LIVE_SEASON_LABEL, LiveResults, RESULT_COLUMNS


def _match(date: str, home: str, away: str, home_goals: int, away_goals: int, season: str) -> dict:
    return {
        "Div": "La Liga",
        "Date": date,
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": home_goals,
        "FTAG": away_goals,
        "FTR": "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D",
        "Season": season,
    }


class LiveFormTests(unittest.TestCase):
    def setUp(self) -> None:
        historical = pd.DataFrame([
            _match("2025-05-01", "Example FC", "Rivals 1", 2, 0, COMPLETED_SEASON),
            _match("2025-05-08", "Rivals 2", "Example FC", 1, 1, COMPLETED_SEASON),
            _match("2025-05-15", "Example FC", "Rivals 3", 1, 0, COMPLETED_SEASON),
            _match("2025-05-22", "Rivals 4", "Example FC", 2, 1, COMPLETED_SEASON),
        ])
        historical["Date"] = pd.to_datetime(historical["Date"])
        self.runtime = SimpleNamespace(config=LEAGUES["la_liga"], matches=historical)

    def test_current_season_results_are_included_in_the_last_five(self) -> None:
        live_matches = pd.DataFrame([
            _match("2026-08-16", "Example FC", "Rivals 5", 3, 1, LIVE_SEASON_LABEL),
            _match("2026-08-23", "Rivals 6", "Example FC", 2, 2, LIVE_SEASON_LABEL),
        ], columns=RESULT_COLUMNS)
        live = LiveResults(
            matches=live_matches,
            fetched_at=datetime.now(timezone.utc),
            source_url="https://site.api.espn.com/live-test",
        )
        with patch("api.main.live_results_for", return_value=live):
            stats = recent_team_stats(self.runtime, "Example FC")

        self.assertTrue(stats["live"])
        self.assertEqual(stats["source_season"], f"{LIVE_SEASON_LABEL} · live")
        self.assertEqual(stats["data_through"], "2026-08-23")
        self.assertEqual(len(stats["matches"]), 5)
        self.assertEqual(stats["matches"][-1]["date"], "2026-08-23")
        self.assertEqual(stats["matches"][-1]["result"], "D")

    def test_preseason_keeps_the_completed_season_fallback(self) -> None:
        empty_live = LiveResults(
            matches=pd.DataFrame(columns=RESULT_COLUMNS),
            fetched_at=datetime.now(timezone.utc),
            source_url="https://site.api.espn.com/live-test",
        )
        with patch("api.main.live_results_for", return_value=empty_live):
            stats = recent_team_stats(self.runtime, "Example FC")

        self.assertFalse(stats["live"])
        self.assertEqual(stats["source_season"], COMPLETED_SEASON)
        self.assertEqual(stats["data_through"], "2025-05-22")


if __name__ == "__main__":
    unittest.main()
