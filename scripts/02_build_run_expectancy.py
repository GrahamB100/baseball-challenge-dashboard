"""
Step 2: Build run-expectancy tables from cleaned data.

Outputs:
    artifacts/re_24_state.parquet       (8 base states x 3 outs = 24 rows)
    artifacts/re_count_state.parquet    (24 x 12 counts = 288 rows; PA-start
                                         counts collapse to 0-0, so this file
                                         is a broadcast of the 24-state values
                                         until pitch-level counts are tracked)

Run:
    python scripts/02_build_run_expectancy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.build_run_expectancy import (                                   # noqa: E402
    build_fallback_24_state_table, calculate_runs_remaining_by_half_inning,
)
from src.build_states import add_state_columns                           # noqa: E402
from src.config import (                                                  # noqa: E402
    BASE_STATES, CLEAN_PARQUET, COUNT_STATES, OUTS_VALUES, RE_24_PATH,
    RE_COUNT_PATH,
)
from src.model_utils import save_parquet                                  # noqa: E402


def main() -> None:
    print(f"Loading {CLEAN_PARQUET} ...")
    df = pd.read_parquet(CLEAN_PARQUET)
    if "re_state_24" not in df.columns:
        df = add_state_columns(df)

    print("Computing runs_remaining per half inning ...")
    df = calculate_runs_remaining_by_half_inning(df)

    print("Building 24-state RE table ...")
    re24 = build_fallback_24_state_table(df)
    save_parquet(re24, RE_24_PATH)
    print(re24.sort_values("re_state").to_string(index=False))

    # Count-state RE: at PA start the count is 0-0 in our parser, so we
    # broadcast the 24-state values across all 12 counts to provide a usable
    # table while keeping the API in place. Once pitch-level counts are
    # tracked this script can be re-run without code changes.
    print("Building count-aware RE table (broadcast PA-start RE across 12 counts) ...")
    rows = []
    for _, r in re24.iterrows():
        base, outs = r["re_state"].split("_")
        for c in COUNT_STATES:
            rows.append({
                "re_state": f"{base}_{outs}_{c}",
                "run_expectancy": r["run_expectancy"],
                "n": r["n"],
            })
    re_count = pd.DataFrame(rows)
    save_parquet(re_count, RE_COUNT_PATH)
    print(f"Wrote {RE_24_PATH.name} ({len(re24)} rows) and "
          f"{RE_COUNT_PATH.name} ({len(re_count)} rows).")


if __name__ == "__main__":
    main()
