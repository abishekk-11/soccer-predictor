# European Soccer Predictor — 2026/27

Pitch IQ forecasts selected 2026/27 fixtures in five major European leagues:
Premier League, La Liga, Bundesliga, Ligue 1, and Serie A. Select a league,
then an official fixture, to see win/draw/loss probabilities, completed-season
form, live in-season recent form, and pre-line-up goal and assist targets for
both squads.

## How it works

- Each league trains its own copy of the same leakage-safe algorithm: Elo,
  recent form, home/away form, in-season performance, and rest days feed a
  calibrated Logistic Regression + Histogram Gradient Boosting blend.
- Training uses only information available before a match. Fixtures on the same
  day are feature-built together, preventing same-day result leakage.
- Results cover 2016/17 through the completed **2025/26** season for La Liga,
  Bundesliga, Ligue 1, and Serie A; the existing Premier League history remains
  available from 1993/94.
- Player portfolios pair current 2026/27 headshots, when a public portrait is
  available, with completed 2016/17–2025/26 league goals, assists, starts, and
  appearances. Fixture-specific likely scorer
  and assister signals blend that ten-season history, a position prior,
  availability, and the selected match forecast. They are explanatory signals,
  not predicted starting XIs. The non-Premier-League source exposes appearances
  rather than dependable historic minutes, and the UI labels them accurately.
- The Recent form panel checks the completed 2026/27 league results for all
  five leagues every 15 minutes while the API is running. Before matchday one,
  it shows the latest 2025/26 form; early in the season it combines the new
  results with the end of the prior season until five current league matches
  are available.

## Run it

```bash
./.venv/bin/python -m src.train --league all
./.venv/bin/uvicorn api.main:app --reload
```

Open `frontend/index.html` while the local API is running. The top selectors
let you switch between all five leagues and their full published fixture lists.

To refresh data before retraining:

```bash
./.venv/bin/python -m src.update_data
./.venv/bin/python -m src.train --league all
```

The refresh restores cached public player portraits after rebuilding each
roster. It leaves the initials avatar in place when no reliable headshot can
be found.

## API examples

```bash
curl http://127.0.0.1:8000/leagues
curl 'http://127.0.0.1:8000/fixtures?league=la_liga'
curl 'http://127.0.0.1:8000/match-players?league=serie_a&home_team=Genoa&away_team=Napoli&fixture_date=2026-08-22'
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"league":"bundesliga","home_team":"Bayern Munich","away_team":"VfB Stuttgart","fixture_date":"2026-08-28"}'
```

## Sources

- Fixture references: [Premier League](https://www.premierleague.com/en/fixtures), [La Liga](https://www.laliga.com/en-GB/laliga-easports/calendar), [Bundesliga](https://www.bundesliga.com/en/bundesliga/matchday/2026-2027/1), [Ligue 1](https://ligue1.com/en/calendar/ligue1), [Serie A](https://en.legaseriea.it/serie-a/fixtures-results)
- Reproducible non-Premier-League schedule, results, and roster refresh: [ESPN public soccer feeds](https://site.api.espn.com/)
- Premier League historic results: [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php)
- Player portraits: [TheSportsDB](https://www.thesportsdb.com/) and [Wikimedia Commons](https://commons.wikimedia.org/), cached locally; unavailable portraits use initials.

## Important limitations

The figures are model estimates—not bookmaker prices, betting advice, or a
guarantee. Pre-season projections cannot account for late transfers, injuries,
line-ups, manager changes, or future results. Recent form updates automatically;
refresh and retrain after each match round to update the model and player data.
