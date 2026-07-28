# Pitch IQ — Five-League Soccer Predictor

Pitch IQ is a full-stack soccer analytics application that forecasts upcoming fixtures across Europe's five largest domestic leagues: the Premier League, La Liga, Bundesliga, Ligue 1, and Serie A.

Built as a portfolio project for data and software engineering roles, it combines historical match data, reproducible machine-learning workflows, a FastAPI backend, and an interactive browser interface. Users can select a fixture and explore match-outcome probabilities, recent team form, and pre-line-up goal and assist signals for each squad.

> Forecasts are analytical estimates, not betting advice or guarantees.

## Live demo

Try the deployed application at **[pitch-iq-soccer-predictor.onrender.com](https://pitch-iq-soccer-predictor.onrender.com)**.

The demo runs on Render's free tier. After 15 minutes without visitors, the first request can take about a minute while the service wakes up; subsequent requests are much faster.

## What the project does

- Covers five leagues and their published 2026/27 fixtures.
- Produces home-win, draw, and away-win probabilities for a selected match.
- Shows each team's last five completed league fixtures, goals scored, and goals conceded.
- Updates the recent-form panel from completed 2026/27 results while the season is running. Before matchday one, it falls back to 2025/26; early-season form is supplemented with the end of the prior season until five current matches are available.
- Builds player portfolios with current roster information, available headshots, position, and ten seasons of completed league goals, assists, starts, and appearances.
- Ranks likely scorers and assisters for both teams using player history, position, availability, and the selected fixture forecast.

## Tech stack

| Area | Technologies |
| --- | --- |
| Backend & data pipeline | Python, FastAPI, Uvicorn, Pandas, NumPy, Requests, Beautiful Soup |
| Machine learning | scikit-learn, Logistic Regression, Histogram Gradient Boosting, Joblib |
| Frontend | HTML, CSS, vanilla JavaScript, Chart.js |
| Data sources | ESPN public soccer feeds, Football-Data.co.uk, official league fixture pages, TheSportsDB, and Wikimedia Commons |

## How the match model is built

Each league has its own trained model artifact, so the same modeling approach can adapt to the scoring patterns and competitive balance of that competition rather than pooling all leagues together.

1. **Historical data** — The pipeline uses completed league matches. La Liga, Bundesliga, Ligue 1, and Serie A use data from 2016/17 through 2025/26; Premier League history is available from 1993/94 onward.
2. **Leakage-safe feature engineering** — Every fixture is represented only by information that would have been known before kickoff. Matches on the same date are transformed as a group before their results are added to the team state.
3. **Features** — The model considers Elo rating and rating difference, recent results and goals, home/away form, in-season points per game and goal difference, rest days, and the amount of current-season evidence available.
4. **Models** — A regularized Logistic Regression model and a Histogram Gradient Boosting classifier each predict the three possible outcomes: home win, draw, or away win.
5. **Validation and blending** — A chronological holdout period beginning in August 2023 is used to evaluate candidate blends. The final blend is selected by log loss, then both components are refit on all completed data. More recent training fixtures receive higher weight using a four-year half-life.
6. **Player contribution signals** — Goal and assist rankings are explanatory, pre-line-up signals. They blend long-term player production, position priors, current availability, and the selected fixture's expected scoring environment. They are not predicted starting lineups.

## Project structure

```text
api/            FastAPI routes and prediction endpoints
src/            Data refresh, feature engineering, training, and live-form logic
frontend/       Single-page interface, charts, and styling
data/           Cached match, fixture, roster, and player data
models/         Per-league trained model artifacts
tests/          Automated checks
```

## Run locally

### 1. Get the project

```bash
git clone https://github.com/abishekk-11/soccer-predictor.git
cd soccer-predictor
```

### 2. Create and activate a virtual environment

Python 3.11 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Start the API

Keep this terminal open:

```bash
python -m uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`, with interactive documentation at `http://127.0.0.1:8000/docs`.

### 5. Start the frontend

Open a second terminal in the project folder, activate the same virtual environment if needed, then run:

```bash
python -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500` in your browser. Select a league and fixture to view the prediction.

## Refresh data and retrain

The repository includes trained artifacts, so these commands are optional for first launch. Run them after you refresh the underlying data or want to reproduce the training pipeline.

```bash
python -m src.update_data
python -m src.train --league all
```

To run the automated checks:

```bash
python -m unittest discover -s tests -v
```

## API examples

```bash
curl http://127.0.0.1:8000/leagues
curl 'http://127.0.0.1:8000/fixtures?league=la_liga'
curl 'http://127.0.0.1:8000/match-players?league=serie_a&home_team=Genoa&away_team=Napoli&fixture_date=2026-08-22'
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"league":"bundesliga","home_team":"Bayern Munich","away_team":"VfB Stuttgart","fixture_date":"2026-08-28"}'
```

## Data sources and attribution

- Fixture references: [Premier League](https://www.premierleague.com/en/fixtures), [La Liga](https://www.laliga.com/en-GB/laliga-easports/calendar), [Bundesliga](https://www.bundesliga.com/en/bundesliga/matchday/2026-2027/1), [Ligue 1](https://ligue1.com/en/calendar/ligue1), and [Serie A](https://en.legaseriea.it/serie-a/fixtures-results)
- Reproducible non-Premier-League schedule, results, and roster refresh: [ESPN public soccer feeds](https://site.api.espn.com/)
- Premier League historic results: [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php)
- Player portraits: [TheSportsDB](https://www.thesportsdb.com/) and [Wikimedia Commons](https://commons.wikimedia.org/). Portraits are cached locally; the application uses initials when no reliable image is available.

## Limitations and responsible use

The project does not know future injuries, transfers, tactical changes, manager changes, confirmed lineups, or future results. The recent-form display refreshes during the season, but the underlying match and player models must be refreshed and retrained to incorporate newly completed rounds. Use the results as a demonstration of data engineering and applied machine learning—not as a substitute for professional analysis or betting advice.
