"""
Win-probability model.

Target  : home_team_won (1 if final_home_score > final_away_score else 0).
Features: state-level features + score differential + season.

Two estimators trained:
    - Logistic regression baseline
    - HistGradientBoosting primary
Then we wrap the primary in a sigmoid/isotonic CalibratedClassifierCV.

Train/test split is BY SEASON: oldest seasons train, last N seasons test.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .build_states import add_state_columns
from .config import (
    CALIBRATION_PLOT, MAX_WP_TRAINING_ROWS, SEED, TEST_SEASON_COUNT,
    WP_BASELINE_PATH, WP_EVAL_CSV, WP_FEATURES_PATH, WP_MODEL_PATH,
)

# Features used by both models. Categorical columns are listed separately.
NUMERIC_FEATURES: List[str] = [
    "inning",
    "is_top_inning",
    "batting_team_is_home",
    "home_score",
    "away_score",
    "score_diff_home",
    "outs_before",
    "runner_1b",
    "runner_2b",
    "runner_3b",
    "balls",
    "strikes",
    "season",
]
CATEGORICAL_FEATURES: List[str] = [
    "base_state",
    "count_state",
]
ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_wp_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add state columns + label, drop bad rows."""
    df = add_state_columns(df)
    df = df.dropna(subset=["final_home_score", "final_away_score"]).copy()
    # Drop ties (rare) — we don't have a 3-class model.
    df = df[df["final_home_score"] != df["final_away_score"]].copy()
    df["home_team_won"] = (df["final_home_score"] > df["final_away_score"]).astype(int)
    # Cap rows for memory safety.
    if len(df) > MAX_WP_TRAINING_ROWS:
        df = df.sample(MAX_WP_TRAINING_ROWS, random_state=SEED).reset_index(drop=True)
    return df


def split_train_test_by_season(
    df: pd.DataFrame, test_season_count: int = TEST_SEASON_COUNT
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df["season"].unique())
    if len(seasons) <= test_season_count:
        # Not enough seasons — use a small holdout fraction instead.
        cut = int(0.8 * len(df))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
    test_seasons = set(seasons[-test_season_count:])
    train = df[~df["season"].isin(test_seasons)].copy()
    test = df[df["season"].isin(test_seasons)].copy()
    return train, test


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def train_logistic_baseline(train: pd.DataFrame) -> Pipeline:
    pipe = Pipeline([
        ("pre", _build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None, random_state=SEED)),
    ])
    pipe.fit(train[ALL_FEATURES], train["home_team_won"])
    return pipe


def train_gradient_boosting_model(train: pd.DataFrame) -> Pipeline:
    """HistGradientBoosting handles categoricals natively, but we keep it inside
    the same ColumnTransformer pattern for consistency."""
    pipe = Pipeline([
        ("pre", _build_preprocessor()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
        )),
    ])
    pipe.fit(train[ALL_FEATURES], train["home_team_won"])
    return pipe


def calibrate_model(model: Pipeline, X_cal: pd.DataFrame, y_cal: pd.Series) -> CalibratedClassifierCV:
    """Wrap an already-fit model with isotonic calibration on a held-out set."""
    cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, label: str) -> Dict:
    proba = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, proba)
    ll = log_loss(y_test, np.clip(proba, 1e-6, 1 - 1e-6))
    return {"model": label, "brier": brier, "log_loss": ll, "n_test": len(y_test)}


def plot_calibration(models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for name, m in models.items():
        proba = m.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=15, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Observed P(home win)")
    ax.set_title("WP model calibration (test seasons)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_model_artifacts(
    primary_calibrated,
    baseline,
    feature_list: List[str],
    eval_rows: List[Dict],
) -> None:
    joblib.dump(primary_calibrated, WP_MODEL_PATH)
    joblib.dump(baseline, WP_BASELINE_PATH)
    WP_FEATURES_PATH.write_text(json.dumps({
        "features": feature_list,
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    }, indent=2))
    pd.DataFrame(eval_rows).to_csv(WP_EVAL_CSV, index=False)


def predict_home_wp(model, state_features: Dict) -> float:
    """Single-row prediction helper used by the dashboard."""
    X = pd.DataFrame([{c: state_features.get(c) for c in ALL_FEATURES}])
    return float(model.predict_proba(X)[0, 1])
