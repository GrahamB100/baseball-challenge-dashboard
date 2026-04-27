"""
Step 1: Parse raw Retrosheet event files (or a pre-parsed CSV) and write
        data_clean/retrosheet_clean.parquet with one row per plate appearance.

Run from the project root:
    python scripts/01_build_clean_data.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `src` importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_states import add_state_columns                # noqa: E402
from src.config import CLEAN_PARQUET, RETROSHEET_DIR          # noqa: E402
from src.load_retrosheet import load_parsed_csv, load_retrosheet_events  # noqa: E402
from src.model_utils import save_parquet                      # noqa: E402


def main() -> None:
    t0 = time.time()

    # Optional: limit to specific seasons via env var (comma-separated or range).
    # Examples:  SEASONS=2020,2021,2022   or   SEASONS=2010-2024
    import os
    seasons_env = os.environ.get("SEASONS")
    seasons = None
    if seasons_env:
        if "-" in seasons_env:
            a, b = seasons_env.split("-")
            seasons = list(range(int(a), int(b) + 1))
        else:
            seasons = [int(s) for s in seasons_env.split(",")]
        print(f"SEASONS filter active: {seasons}")

    # If a pre-parsed CSV is dropped at data_raw/retrosheet/parsed.csv, prefer it.
    parsed_csv = RETROSHEET_DIR / "parsed.csv"
    if parsed_csv.exists():
        print(f"Loading pre-parsed CSV: {parsed_csv}")
        df = load_parsed_csv(parsed_csv)
    else:
        print(f"Parsing raw Retrosheet event files under {RETROSHEET_DIR} ...")
        df = load_retrosheet_events(RETROSHEET_DIR, seasons=seasons, progress=True)

    print(f"Parsed {len(df):,} plate appearances across "
          f"{df['game_id'].nunique():,} games and "
          f"{df['season'].nunique()} seasons.")

    # Add state columns once so downstream scripts get them for free.
    df = add_state_columns(df)

    save_parquet(df, CLEAN_PARQUET)
    print(f"Wrote {CLEAN_PARQUET}  ({CLEAN_PARQUET.stat().st_size/1e6:.1f} MB)")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
