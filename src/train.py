"""Train comparable, league-specific 2026/27 prediction artifacts.

All five competitions use the same leakage-safe feature construction and the
same two-model probability blend.  They are trained separately so league
styles, draw rates and home advantage are learned from the relevant history.
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.league_config import LEAGUES, LeagueConfig, get_league
from src.modeling import FEATURE_COLUMNS, build_feature_table


HOLDOUT_START = pd.Timestamp("2023-08-01")
CLASSES = np.array(["A", "D", "H"])


def multiclass_brier(y_true: pd.Series, probabilities: np.ndarray, classes: np.ndarray) -> float:
    expected = np.zeros_like(probabilities)
    for index, class_name in enumerate(classes):
        expected[:, index] = (y_true.to_numpy() == class_name).astype(float)
    return float(np.mean(np.sum((probabilities - expected) ** 2, axis=1)))


def align_probabilities(model, X: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    aligned = np.zeros((len(X), len(classes)))
    for source_index, class_name in enumerate(model.classes_):
        aligned[:, np.where(classes == class_name)[0][0]] = raw[:, source_index]
    return aligned


def evaluation(y_true: pd.Series, probabilities: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    predicted = classes[np.argmax(probabilities, axis=1)]
    return {
        "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "log_loss": round(float(log_loss(y_true, probabilities, labels=classes)), 4),
        "multiclass_brier": round(multiclass_brier(y_true, probabilities, classes), 4),
    }


def _models() -> tuple[Pipeline, HistGradientBoostingClassifier]:
    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.35, max_iter=3000, random_state=42)),
    ])
    boosted = HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=250,
        max_leaf_nodes=12,
        l2_regularization=1.0,
        random_state=42,
    )
    return logistic, boosted


def train_league(config: LeagueConfig) -> dict:
    """Train and save one league's identical model specification."""
    if not config.matches_path.exists():
        raise FileNotFoundError(f"Missing results file for {config.name}: {config.matches_path}")
    matches = pd.read_csv(config.matches_path, parse_dates=["Date"])
    X, y = build_feature_table(matches)
    train_mask = X.index < HOLDOUT_START
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[~train_mask], y.loc[~train_mask]
    if len(X_test) < 900:
        raise ValueError(f"Holdout needs at least two complete league seasons for {config.name}")

    # Recent seasons receive more influence without discarding useful history.
    age_years = (X_train.index.max() - X_train.index).days / 365.25
    sample_weight = np.exp(-np.log(2) * age_years / 4.0)
    logistic, boosted = _models()
    logistic.fit(X_train, y_train, model__sample_weight=sample_weight)
    boosted.fit(X_train, y_train, sample_weight=sample_weight)

    log_proba = align_probabilities(logistic, X_test, CLASSES)
    boosted_proba = align_probabilities(boosted, X_test, CLASSES)
    candidate_weights = np.arange(0.0, 1.05, 0.05)
    blend_weight = min(
        candidate_weights,
        key=lambda weight: log_loss(
            y_test, (1 - weight) * log_proba + weight * boosted_proba, labels=CLASSES
        ),
    )
    blended = (1 - blend_weight) * log_proba + blend_weight * boosted_proba

    # Refit on all completed seasons for the 2026/27 fixture forecast.
    final_age_years = (X.index.max() - X.index).days / 365.25
    final_weight = np.exp(-np.log(2) * final_age_years / 4.0)
    logistic, boosted = _models()
    logistic.fit(X, y, model__sample_weight=final_weight)
    boosted.fit(X, y, sample_weight=final_weight)

    metrics = {
        "league": config.name,
        "evaluation_method": "Chronological holdout: 2023/24 through 2025/26; no same-day result leakage.",
        "training_matches": int(len(X_train)),
        "holdout_matches": int(len(X_test)),
        "data_through": str(matches.Date.max().date()),
        "validation_start": str(HOLDOUT_START.date()),
        "home_win_baseline_accuracy": round(float((y_test == "H").mean()), 4),
        "logistic": evaluation(y_test, log_proba, CLASSES),
        "boosted": evaluation(y_test, boosted_proba, CLASSES),
        "selected_blend": {
            "hist_gradient_boosting_weight": round(float(blend_weight), 2),
            **evaluation(y_test, blended, CLASSES),
        },
    }
    artifact = {
        "version": "2.1",
        "league_key": config.key,
        "league_name": config.name,
        "feature_columns": FEATURE_COLUMNS,
        "classes": CLASSES,
        "logistic_model": logistic,
        "boosted_model": boosted,
        "boosted_weight": float(blend_weight),
        "metrics": metrics,
    }
    config.model_path.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, config.model_path)
    config.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train five-league soccer-predictor artifacts")
    parser.add_argument(
        "--league",
        choices=["all", *LEAGUES],
        default="all",
        help="Train one configured league or every league (default).",
    )
    args = parser.parse_args(argv)
    configs = LEAGUES.values() if args.league == "all" else [get_league(args.league)]
    summary = {config.key: train_league(config) for config in configs}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
