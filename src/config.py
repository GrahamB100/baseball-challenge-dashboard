"""
Project paths, constants, and tunable knobs.

All other modules import from here so that path changes are localized.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_RAW_DIR: Path = PROJECT_ROOT / "data_raw"
RETROSHEET_DIR: Path = DATA_RAW_DIR / "retrosheet"

DATA_CLEAN_DIR: Path = PROJECT_ROOT / "data_clean"
CLEAN_PARQUET: Path = DATA_CLEAN_DIR / "retrosheet_clean.parquet"

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
RE_24_PATH: Path = ARTIFACTS_DIR / "re_24_state.parquet"
RE_COUNT_PATH: Path = ARTIFACTS_DIR / "re_count_state.parquet"
WP_MODEL_PATH: Path = ARTIFACTS_DIR / "wp_model.joblib"
WP_BASELINE_PATH: Path = ARTIFACTS_DIR / "wp_logreg_baseline.joblib"
WP_FEATURES_PATH: Path = ARTIFACTS_DIR / "wp_features.json"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
WP_EVAL_CSV: Path = OUTPUTS_DIR / "wp_evaluation.csv"
CALIBRATION_PLOT: Path = OUTPUTS_DIR / "calibration_plot.png"

# ---------------------------------------------------------------------------
# Modeling knobs
# ---------------------------------------------------------------------------
# Cap rows used for WP training so we never blow up RAM.
MAX_WP_TRAINING_ROWS: int = 1_500_000

# How many of the most-recent seasons go into the test set.
TEST_SEASON_COUNT: int = 3

# Random seed everywhere.
SEED: int = 42

# ---------------------------------------------------------------------------
# Game-state vocabularies
# ---------------------------------------------------------------------------
BASE_STATES: tuple[str, ...] = (
    "000", "100", "010", "001", "110", "101", "011", "111",
)

OUTS_VALUES: tuple[int, ...] = (0, 1, 2)

COUNT_STATES: tuple[str, ...] = (
    "0-0", "1-0", "2-0", "3-0",
    "0-1", "1-1", "2-1", "3-1",
    "0-2", "1-2", "2-2", "3-2",
)

# Challenge-value interpretation thresholds (decimal WP units).
CHALLENGE_THRESHOLDS = {
    "negligible": 0.005,
    "small": 0.02,
    "moderate": 0.06,
    "high_impact": 0.15,
}

# Ensure dirs exist on import — cheap.
for _p in (DATA_RAW_DIR, RETROSHEET_DIR, DATA_CLEAN_DIR, ARTIFACTS_DIR, OUTPUTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)
