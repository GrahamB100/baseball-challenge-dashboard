# Recovery Plan (token-budget protection)

If this conversation runs out of tokens, this document tells the next agent (or Graham)
exactly how to finish the project. Files are produced in this order; if any step is
incomplete, restart from that step.

## 0. Project location
`<repo>/CLAUDE OUTPUTS/baseball-challenge-dashboard/`

Folder layout (already created):
```
baseball-challenge-dashboard/
  README.md
  requirements.txt
  PLAN.md                      <- this file
  app/streamlit_app.py
  src/
    __init__.py
    config.py
    load_retrosheet.py
    build_states.py
    build_run_expectancy.py
    train_win_probability.py
    challenge_value.py
    model_utils.py
  scripts/
    01_build_clean_data.py
    02_build_run_expectancy.py
    03_train_win_probability.py
  data_raw/retrosheet/{1990s,2000s,2010s,2020s}/   <- *.EVA / *.EVN
  data_clean/                  <- retrosheet_clean.parquet
  artifacts/                   <- re_*.parquet, wp_model.joblib, wp_features.json
  outputs/                     <- wp_evaluation.csv, calibration_plot.png
```

## 1. Build order
1. `pip install -r requirements.txt`
2. Unzip Retrosheet event files (`*eve.zip`) into `data_raw/retrosheet/<decade>/`
3. `python scripts/01_build_clean_data.py`   -> `data_clean/retrosheet_clean.parquet`
4. `python scripts/02_build_run_expectancy.py` -> `artifacts/re_24_state.parquet`, `artifacts/re_count_state.parquet`
5. `python scripts/03_train_win_probability.py` -> `artifacts/wp_model.joblib`, `artifacts/wp_features.json`, `outputs/wp_evaluation.csv`, `outputs/calibration_plot.png`
6. `streamlit run app/streamlit_app.py`

## 2. If you hit token cap before finishing
Each `src/` module is self-contained. Open it, finish any TODO blocks (search for `TODO`),
then run the scripts in order. The Streamlit app gracefully degrades if artifacts are missing
(it shows "Run scripts first" message).

## 3. Hard limits we enforced to avoid runaway compute
- Event parsing: streams files line-by-line, no full-file load.
- WP training: subsamples to MAX_WP_TRAINING_ROWS (default 1.5M plays) — set in
  `src/config.py`. On smaller machines, lower this.
- HistGradientBoosting: max_iter=300, early_stopping enabled.
- Train/test split: chronological by season — last 3 seasons in test set.

## 4. Known scope cuts vs. ideal
- Pitch-level count states are NOT fully tracked. RE table is built at PA level
  (count = 0-0 at PA start). Count-aware RE is produced as a "broadcast" of the
  24-state table. A future upgrade can walk pitch sequences (the field is in `pitches`
  on each `play` record) to fill real per-count states.
- WP model uses count as a feature with PA-start values. The model still learns
  base-out-score-inning shape correctly; count adds marginal signal.
- Postseason games are excluded by default (Retrosheet event files include them
  with games dated in October/November; we filter by regular-season game IDs).
- 2020 ghost-runner extra innings: kept as-is. Discussed in README limitations.
