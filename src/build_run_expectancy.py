"""
Run expectancy table construction and lookup.

Definitions
-----------
For each plate appearance row, define:

    runs_remaining_in_half_inning =
        total_runs_scored_by_batting_team_in_that_half_inning
        - runs_scored_before_current_state

Then group by RE state (base_state + outs, optionally + count) and take the
mean of runs_remaining_in_half_inning. That's the standard Tango/THT formulation.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .build_states import add_state_columns, create_base_state, create_count_state, create_re_state


def calculate_runs_remaining_by_half_inning(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `runs_remaining` column = runs scored by batting team from this
    PA through end of the half inning."""
    out = df.copy()
    half_id = (
        out["game_id"].astype(str)
        + "_"
        + out["inning"].astype(str)
        + "_"
        + out["is_top_inning"].astype(int).astype(str)
    )
    out["_half_inning_id"] = half_id

    # Runs scored in each half inning (sum of runs_scored over rows in that half).
    half_total = out.groupby("_half_inning_id")["runs_scored"].transform("sum")
    # Cumulative runs *before* current row inside each half inning.
    cum_before = out.groupby("_half_inning_id")["runs_scored"].cumsum() - out["runs_scored"]
    out["runs_remaining"] = half_total - cum_before
    return out.drop(columns=["_half_inning_id"])


def build_run_expectancy_table(df: pd.DataFrame, count_aware: bool = False) -> pd.DataFrame:
    """Aggregate mean runs_remaining by state."""
    needed = "re_state_count" if count_aware else "re_state_24"
    if needed not in df.columns:
        df = add_state_columns(df)

    grp = df.groupby(needed)["runs_remaining"]
    table = grp.agg(["mean", "count"]).reset_index()
    table.columns = ["re_state", "run_expectancy", "n"]
    return table


def build_fallback_24_state_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper — same as build_run_expectancy_table(count_aware=False)."""
    return build_run_expectancy_table(df, count_aware=False)


def lookup_run_expectancy(
    state_dict: dict,
    re_count_table: Optional[pd.DataFrame],
    re_24_table: pd.DataFrame,
) -> float:
    """
    Look up RE for a state dict. Tries:
      1) full count-specific table  (if provided & key found)
      2) 24-state base-out table    (always present)
      3) league-average fallback (warn)
    """
    base = create_base_state(
        state_dict.get("runner_1b", 0),
        state_dict.get("runner_2b", 0),
        state_dict.get("runner_3b", 0),
    )
    outs = int(state_dict.get("outs_before", 0))
    count = create_count_state(state_dict.get("balls", 0), state_dict.get("strikes", 0))
    key_full = create_re_state(base, outs, count)
    key_24 = create_re_state(base, outs)

    if re_count_table is not None and not re_count_table.empty:
        hit = re_count_table.loc[re_count_table["re_state"] == key_full, "run_expectancy"]
        if len(hit):
            return float(hit.iloc[0])

    hit = re_24_table.loc[re_24_table["re_state"] == key_24, "run_expectancy"]
    if len(hit):
        return float(hit.iloc[0])

    warnings.warn(f"RE state '{key_24}' not found; using global mean fallback.")
    return float(re_24_table["run_expectancy"].mean())
