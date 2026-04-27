"""
Tiny IO + formatting helpers shared by training scripts and the Streamlit app.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd

from .config import (
    RE_24_PATH, RE_COUNT_PATH, WP_FEATURES_PATH, WP_MODEL_PATH,
)


def artifacts_exist() -> bool:
    """True iff every required artifact is on disk."""
    return all(p.exists() for p in (RE_24_PATH, WP_MODEL_PATH, WP_FEATURES_PATH))


def load_re_tables() -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    re_count = pd.read_parquet(RE_COUNT_PATH) if RE_COUNT_PATH.exists() else None
    re_24 = pd.read_parquet(RE_24_PATH)
    return re_count, re_24


def load_wp_model():
    return joblib.load(WP_MODEL_PATH)


def load_wp_feature_list() -> dict:
    return json.loads(WP_FEATURES_PATH.read_text())


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def fmt_pct(p: float, digits: int = 1) -> str:
    """0.532 -> '53.2%'."""
    return f"{100*p:.{digits}f}%"


def fmt_pp(p: float, digits: int = 2) -> str:
    """0.0734 -> '+7.34 pp' (percentage points)."""
    sign = "+" if p >= 0 else ""
    return f"{sign}{100*p:.{digits}f} pp"


def fmt_runs(r: float, digits: int = 3) -> str:
    sign = "+" if r >= 0 else ""
    return f"{sign}{r:.{digits}f} runs"
