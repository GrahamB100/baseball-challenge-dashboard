"""
Step 3: Train win-probability models, calibrate, evaluate, save artifacts.

Outputs:
    artifacts/wp_model.joblib            (calibrated GB; PRIMARY)
    artifacts/wp_logreg_baseline.joblib  (logistic-regression baseline)
    artifacts/wp_features.json
    outputs/wp_evaluation.csv
    outputs/calibration_plot.png

Run:
    python scripts/03_train_win_probability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import CALIBRATION_PLOT, CLEAN_PARQUET                    # noqa: E402
from src.train_win_probability import (                                   # noqa: E402
    ALL_FEATURES, calibrate_model, evaluate_model, plot_calibration,
    prepare_wp_training_data, save_model_artifacts,
    split_train_test_by_season, train_gradient_boosting_model,
    train_logistic_baseline,
)


def main() -> None:
    print(f"Loading {CLEAN_PARQUET} ...")
    df = pd.read_parquet(CLEAN_PARQUET)

    print("Preparing training data ...")
    df = prepare_wp_training_data(df)
    print(f"  {len(df):,} rows; {df['season'].nunique()} seasons; "
          f"label balance home_won={df['home_team_won'].mean():.3f}")

    train, test = split_train_test_by_season(df)
    print(f"  train rows = {len(train):,}, test rows = {len(test):,}")

    # Carve a calibration slice off the END of the training set
    # (more recent than rest of train, to mimic deployment conditions).
    cal_size = max(50_000, int(0.1 * len(train)))
    cal = train.sample(cal_size, random_state=0)
    train_fit = train.drop(index=cal.index)

    print("Training logistic regression baseline ...")
    logreg = train_logistic_baseline(train_fit)

    print("Training HistGradientBoosting primary ...")
    gb = train_gradient_boosting_model(train_fit)

    print("Calibrating GB on held-out slice (isotonic) ...")
    gb_cal = calibrate_model(gb, cal[ALL_FEATURES], cal["home_team_won"])

    print("Evaluating on test seasons ...")
    rows = [
        evaluate_model(logreg, test[ALL_FEATURES], test["home_team_won"], "logreg_baseline"),
        evaluate_model(gb,     test[ALL_FEATURES], test["home_team_won"], "gradient_boosting_uncal"),
        evaluate_model(gb_cal, test[ALL_FEATURES], test["home_team_won"], "gradient_boosting_calibrated"),
    ]
    for r in rows:
        print(f"  {r['model']:>30s}  Brier={r['brier']:.4f}  LogLoss={r['log_loss']:.4f}  n={r['n_test']:,}")

    print("Plotting calibration ...")
    plot_calibration(
        {"logreg": logreg, "gb_uncal": gb, "gb_calibrated": gb_cal},
        test[ALL_FEATURES],
        test["home_team_won"],
        CALIBRATION_PLOT,
    )

    print("Saving artifacts ...")
    save_model_artifacts(
        primary_calibrated=gb_cal,
        baseline=logreg,
        feature_list=ALL_FEATURES,
        eval_rows=rows,
    )
    print("Done.")


if __name__ == "__main__":
    main()
