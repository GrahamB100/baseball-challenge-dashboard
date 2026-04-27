"""
Game-state feature builders. Pure functions over a DataFrame of plate
appearances (or a single user-provided dict for the dashboard).
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import BASE_STATES, COUNT_STATES, OUTS_VALUES


def create_base_state(r1: int, r2: int, r3: int) -> str:
    """Return base-state string '<1B><2B><3B>', e.g. '101' = 1B & 3B."""
    return f"{int(bool(r1))}{int(bool(r2))}{int(bool(r3))}"


def create_count_state(balls: int, strikes: int) -> str:
    """Return count string like '2-1'."""
    b = max(0, min(int(balls), 3))
    s = max(0, min(int(strikes), 2))
    return f"{b}-{s}"


def create_re_state(base_state: str, outs: int, count_state: str | None = None) -> str:
    """Compose RE state key. Count is optional (24-state fallback if None)."""
    base_out = f"{base_state}_{int(outs)}"
    if count_state is None:
        return base_out
    return f"{base_out}_{count_state}"


def create_score_diff(home_score: int, away_score: int) -> int:
    """home - away score. Positive = home leading."""
    return int(home_score) - int(away_score)


def create_batting_team_home_flag(is_top_inning: int) -> int:
    """If it's the top of the inning the AWAY team is batting; bottom = home."""
    return 0 if int(is_top_inning) == 1 else 1


def add_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized: add base_state, count_state, re_state_24, re_state_count,
    score_diff_home, batting_team_is_home columns."""
    out = df.copy()
    out["base_state"] = (
        out["runner_1b"].astype(int).astype(str)
        + out["runner_2b"].astype(int).astype(str)
        + out["runner_3b"].astype(int).astype(str)
    )
    out["count_state"] = (
        out["balls"].clip(0, 3).astype(int).astype(str)
        + "-"
        + out["strikes"].clip(0, 2).astype(int).astype(str)
    )
    out["re_state_24"] = out["base_state"] + "_" + out["outs_before"].astype(int).astype(str)
    out["re_state_count"] = out["re_state_24"] + "_" + out["count_state"]
    out["score_diff_home"] = out["home_score"].astype(int) - out["away_score"].astype(int)
    out["batting_team_is_home"] = (out["is_top_inning"].astype(int) == 0).astype(int)
    return out


def create_model_features(state: Dict) -> Dict:
    """
    Build the feature dict expected by the trained WP model from a single
    game-state dict. Used by the Streamlit app at inference time.
    """
    base_state = create_base_state(
        state.get("runner_1b", 0),
        state.get("runner_2b", 0),
        state.get("runner_3b", 0),
    )
    count_state = create_count_state(state.get("balls", 0), state.get("strikes", 0))
    is_top = int(state["is_top_inning"])
    return {
        "inning": int(state["inning"]),
        "is_top_inning": is_top,
        "batting_team_is_home": 0 if is_top == 1 else 1,
        "home_score": int(state["home_score"]),
        "away_score": int(state["away_score"]),
        "score_diff_home": int(state["home_score"]) - int(state["away_score"]),
        "outs_before": int(state["outs_before"]),
        "runner_1b": int(bool(state.get("runner_1b", 0))),
        "runner_2b": int(bool(state.get("runner_2b", 0))),
        "runner_3b": int(bool(state.get("runner_3b", 0))),
        "balls": int(state.get("balls", 0)),
        "strikes": int(state.get("strikes", 0)),
        "base_state": base_state,
        "count_state": count_state,
        "season": int(state.get("season", 2024)),
    }


def validate_game_state_input(state: Dict) -> None:
    """Raise a friendly ValueError if the user gave us nonsense."""
    if state["inning"] < 1 or state["inning"] > 30:
        raise ValueError(f"inning must be 1..30, got {state['inning']}")
    if state["is_top_inning"] not in (0, 1):
        raise ValueError("is_top_inning must be 0 or 1")
    if state["outs_before"] not in OUTS_VALUES:
        raise ValueError(f"outs must be in {OUTS_VALUES}")
    if state["home_score"] < 0 or state["away_score"] < 0:
        raise ValueError("scores must be >= 0")
    if state.get("balls", 0) not in (0, 1, 2, 3):
        raise ValueError("balls must be 0..3")
    if state.get("strikes", 0) not in (0, 1, 2):
        raise ValueError("strikes must be 0..2")
    base = create_base_state(
        state.get("runner_1b", 0),
        state.get("runner_2b", 0),
        state.get("runner_3b", 0),
    )
    if base not in BASE_STATES:
        raise ValueError(f"base state {base} not recognized")
