# MLB Replay Challenge Value Dashboard

A Streamlit app that quantifies the value of an MLB replay challenge using:

- A **24-state run expectancy** table built from Retrosheet play-by-play data.
- A **calibrated gradient-boosted win probability** model trained on the same data.
- A clean rule for converting the pre/post game-state into a single
  `Challenge_Value` number from the perspective of the challenging team.

## What the app does

You enter the game state **before** and **after** the challenge:

- Inning, top/bottom
- Score (home / away)
- Batting team (home / away) and challenging team (batting / fielding)
- Outs, runners on base, ball-strike count

The dashboard returns:

- `RE_pre`, `RE_post`, `RE_delta = RE_post - RE_pre`
- `WP_pre`, `WP_post` (from batting-team perspective)
- `WPA_batting = WP_post - WP_pre`
- `Challenge_Value`:
  - if challenging team is batting team: `Challenge_Value = WPA_batting`
  - if challenging team is fielding team: `Challenge_Value = -1 * WPA_batting`
- A plain-English interpretation bucket (`negligible`, `small`, `moderate`,
  `high-impact`, `game-changing`).

## Project layout

```
baseball-challenge-dashboard/
  README.md
  requirements.txt
  PLAN.md
  app/streamlit_app.py
  src/
    config.py            paths + constants
    load_retrosheet.py   raw .EVA/.EVN parser + pre-parsed CSV loader
    build_states.py      base_state / count_state / re_state helpers
    build_run_expectancy.py
    train_win_probability.py
    challenge_value.py   WPA / challenge-value math + interpretation
    model_utils.py       artifact IO + display formatting
  scripts/
    01_build_clean_data.py
    02_build_run_expectancy.py
    03_train_win_probability.py
  data_raw/retrosheet/   <- drop your unzipped .EVA / .EVN files here
  data_clean/            retrosheet_clean.parquet
  artifacts/             RE tables + WP model + feature manifest
  outputs/               evaluation CSV + calibration plot
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Where to put Retrosheet files

Put unzipped Retrosheet event files (`.EVA` / `.EVN`) under
`data_raw/retrosheet/`. Subfolders are fine — the parser walks the tree
recursively. Filenames must start with the 4-digit season, e.g.
`2024TBA.EVA`.

You can also drop a pre-parsed CSV at `data_raw/retrosheet/parsed.csv`
(for example exported from Chadwick `cwevent`). The script will detect
it and skip raw parsing.

## Run the pipeline

```bash
python scripts/01_build_clean_data.py
python scripts/02_build_run_expectancy.py
python scripts/03_train_win_probability.py
```

Then launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## How RE and WPA are calculated

**Run expectancy** for state `s = (base_state, outs)`:

```
RE(s) = mean over all PAs in state s of (
    runs_scored_by_batting_team_in_that_half_inning
    - runs_scored_before_this_PA
)
```

This is the standard Tango formulation. We aggregate over all
regular-season PAs in the loaded seasons.

**Win probability**: a `HistGradientBoostingClassifier` predicts
`P(home_team_wins | state)`. Features include inning, half-inning,
score, outs, base state, count, season. The classifier is wrapped in
`CalibratedClassifierCV(method="isotonic")` on a held-out slice of the
training data so probabilities are well-behaved out of the box.

To get win probability **from the batting team's perspective**:

```
WP_batting = WP_home          if batting team is home
            = 1 - WP_home     if batting team is away
```

## Assumptions and limitations

- Retrosheet schemas vary depending on parsing/export method. The
  `load_retrosheet.py` module either:
  - parses raw `.EVA`/`.EVN` files with a hand-rolled Python event
    parser, or
  - reads a pre-parsed CSV and maps common Retrosheet column names to
    a standard schema. Missing required columns raise a clear error.
- **Ball-strike count is captured at the END of each plate appearance**
  in the raw event files; pitch-by-pitch counts require walking the
  `pitches` field of each play. This MVP does not do that walk yet, so
  the `count_state` is treated as `0-0` at the start of each PA. The
  count-aware RE table is therefore a broadcast of the 24-state values.
  This is documented and the lookup function falls back gracefully.
- This MVP **does not** automatically detect challenges from raw event
  files — it expects the user to enter pre/post state in the dashboard.
  Building a challenge auto-detector would require parsing the
  `info,replay,*` and `com,...` records that Retrosheet sometimes
  includes, plus comparing pre/post `play` lines.
- The WP model is **historical**. It doesn't know about specific teams
  on the field, who's pitching/hitting, weather, etc.
- **Calibration > raw accuracy.** We track Brier score and log-loss
  primarily and include a calibration plot in `outputs/`.
- **Postseason** games are excluded by default (filtered by month in
  the game_id).
- **2020 / extra-inning ghost runner.** Retrosheet records the runner
  starting on 2B as a normal runner on 2B at the start of the half
  inning, so the model sees these states naturally.

## Token-budget recovery

If you ran out of tokens mid-build, see `PLAN.md` — it describes the
exact build order and which file to resume from.
