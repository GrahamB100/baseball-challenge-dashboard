"""
Streamlit dashboard: Replay Challenge Value.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd                                          # noqa: E402
import plotly.graph_objects as go                            # noqa: E402
import streamlit as st                                       # noqa: E402

from src.build_run_expectancy import lookup_run_expectancy   # noqa: E402
from src.build_states import (                                # noqa: E402
    create_model_features, validate_game_state_input,
)
from src.challenge_value import (                             # noqa: E402
    calculate_challenge_value, calculate_wp_batting,
    generate_interpretation_text,
)
from src.model_utils import (                                 # noqa: E402
    artifacts_exist, fmt_pct, fmt_pp, fmt_runs, load_re_tables,
    load_wp_feature_list, load_wp_model,
)
from src.train_win_probability import predict_home_wp         # noqa: E402

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="MLB Replay Challenge Value",
    page_icon=":baseball:",
    layout="wide",
)

# ----------------------------------------------------------------------
# Artifact loading (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def _load_all():
    re_count, re_24 = load_re_tables()
    model = load_wp_model()
    features_meta = load_wp_feature_list()
    return re_count, re_24, model, features_meta


# ----------------------------------------------------------------------
# Sidebar — pre/post challenge inputs
# ----------------------------------------------------------------------
BASE_OPTIONS = {
    "Bases empty (000)": (0, 0, 0),
    "Runner on 1B (100)": (1, 0, 0),
    "Runner on 2B (010)": (0, 1, 0),
    "Runner on 3B (001)": (0, 0, 1),
    "Runners on 1B & 2B (110)": (1, 1, 0),
    "Runners on 1B & 3B (101)": (1, 0, 1),
    "Runners on 2B & 3B (011)": (0, 1, 1),
    "Bases loaded (111)": (1, 1, 1),
}


def _state_inputs(prefix: str, defaults: dict) -> dict:
    st.markdown(f"### {prefix.title()}-challenge state")
    col_a, col_b = st.columns(2)
    with col_a:
        inning = st.number_input(
            "Inning", 1, 20, value=defaults["inning"], key=f"{prefix}_inning"
        )
        outs = st.selectbox(
            "Outs", [0, 1, 2], index=defaults["outs"], key=f"{prefix}_outs"
        )
        balls = st.selectbox(
            "Balls", [0, 1, 2, 3], index=defaults["balls"], key=f"{prefix}_balls"
        )
    with col_b:
        is_top = st.radio(
            "Half inning", ["Top", "Bottom"],
            index=defaults["is_top"], horizontal=True, key=f"{prefix}_half",
        )
        runners_label = st.selectbox(
            "Runners on base", list(BASE_OPTIONS.keys()),
            index=defaults["runners_idx"], key=f"{prefix}_runners",
        )
        strikes = st.selectbox(
            "Strikes", [0, 1, 2], index=defaults["strikes"], key=f"{prefix}_strikes"
        )

    home_score = st.number_input(
        "Home score", 0, 50, value=defaults["home_score"], key=f"{prefix}_hs"
    )
    away_score = st.number_input(
        "Away score", 0, 50, value=defaults["away_score"], key=f"{prefix}_as"
    )
    r1, r2, r3 = BASE_OPTIONS[runners_label]
    return {
        "inning": int(inning),
        "is_top_inning": 1 if is_top == "Top" else 0,
        "home_score": int(home_score),
        "away_score": int(away_score),
        "outs_before": int(outs),
        "runner_1b": r1,
        "runner_2b": r2,
        "runner_3b": r3,
        "balls": int(balls),
        "strikes": int(strikes),
        "season": 2024,
    }


with st.sidebar:
    st.header("Game-state inputs")
    st.caption("Enter the state of the game **before** and **after** the replay challenge.")

    batting_team_label = st.radio(
        "Batting team", ["Home", "Away"], index=1, horizontal=True
    )
    batting_team_is_home = batting_team_label == "Home"

    challenging_team_label = st.radio(
        "Challenging team",
        ["Batting team", "Fielding team"],
        index=1,
        horizontal=True,
    )
    challenging_team_is_batting = challenging_team_label == "Batting team"

    st.divider()
    pre = _state_inputs(
        "pre",
        defaults={
            "inning": 7, "outs": 1, "balls": 1, "strikes": 1,
            "is_top": 0, "runners_idx": 1,
            "home_score": 2, "away_score": 1,
        },
    )
    st.divider()
    post = _state_inputs(
        "post",
        defaults={
            "inning": 7, "outs": 2, "balls": 1, "strikes": 1,
            "is_top": 0, "runners_idx": 0,
            "home_score": 2, "away_score": 1,
        },
    )

# ----------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------
st.title(":baseball: Replay Challenge Value Dashboard")
st.caption(
    "Compare run expectancy and win probability before vs. after a replay "
    "challenge to quantify the call's value."
)

if not artifacts_exist():
    st.error(
        "Model / RE artifacts are missing. Run the data pipeline first:\n\n"
        "```\n"
        "python scripts/01_build_clean_data.py\n"
        "python scripts/02_build_run_expectancy.py\n"
        "python scripts/03_train_win_probability.py\n"
        "```"
    )
    st.stop()

re_count, re_24, model, features_meta = _load_all()

# Validate user input
try:
    validate_game_state_input(pre)
    validate_game_state_input(post)
except ValueError as e:
    st.error(f"Input error: {e}")
    st.stop()

# RE
re_pre = lookup_run_expectancy(pre, re_count, re_24)
re_post = lookup_run_expectancy(post, re_count, re_24)
re_delta = re_post - re_pre

# WP — always predict P(home win), then convert to batting-team WP
try:
    wp_home_pre = predict_home_wp(model, create_model_features(pre))
    wp_home_post = predict_home_wp(model, create_model_features(post))
except Exception as e:
    st.error(f"WP model prediction failed: {e}")
    st.stop()

wp_bat_pre = calculate_wp_batting(wp_home_pre, batting_team_is_home)
wp_bat_post = calculate_wp_batting(wp_home_post, batting_team_is_home)
cv = calculate_challenge_value(wp_bat_pre, wp_bat_post, challenging_team_is_batting)

# ----- Metrics -----
st.subheader("Headline metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Pre RE", f"{re_pre:.3f}")
m2.metric("Post RE", f"{re_post:.3f}", fmt_runs(re_delta))
m3.metric("Pre WP (batting team)", fmt_pct(wp_bat_pre))
m4.metric("Post WP (batting team)", fmt_pct(wp_bat_post), fmt_pp(cv["wpa_batting"]))

challenger_label = "challenging team"
challenger_label = "batting team" if challenging_team_is_batting else "fielding team"
st.subheader(f"Challenge value (from the {challenger_label}'s perspective)")
big1, big2 = st.columns(2)
big1.metric("WPA (batting-team perspective)", fmt_pp(cv["wpa_batting"]))
big2.metric("Challenge Value", fmt_pp(cv["challenge_value"]))

# ----- Charts -----
chart_l, chart_r = st.columns(2)
with chart_l:
    fig = go.Figure(data=[
        go.Bar(name="RE", x=["Pre", "Post"], y=[re_pre, re_post],
               marker_color=["#94a3b8", "#1d4ed8"], text=[f"{re_pre:.3f}", f"{re_post:.3f}"],
               textposition="outside"),
    ])
    fig.update_layout(title="Run Expectancy", yaxis_title="Expected runs",
                      showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)
with chart_r:
    fig = go.Figure(data=[
        go.Bar(name="WP", x=["Pre", "Post"],
               y=[wp_bat_pre*100, wp_bat_post*100],
               marker_color=["#94a3b8", "#16a34a"],
               text=[f"{wp_bat_pre*100:.1f}%", f"{wp_bat_post*100:.1f}%"],
               textposition="outside"),
    ])
    fig.update_layout(title="Win Probability (batting team)",
                      yaxis_title="WP (%)", yaxis_range=[0, 100],
                      showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

# ----- Interpretation -----
st.subheader("Interpretation")
st.info(generate_interpretation_text(
    challenge_value=cv["challenge_value"],
    re_delta=re_delta,
    challenging_team_label=challenger_label,
))

# ----- Notes / limitations -----
with st.expander("Model notes & limitations"):
    st.markdown(
        """
- **WP model:** HistGradientBoosting, isotonic-calibrated, trained on
  Retrosheet plate-appearance states. Train/test split is by season
  (last 3 seasons in the holdout). See `outputs/wp_evaluation.csv` and
  `outputs/calibration_plot.png` for diagnostics.
- **RE table:** classic 24-state base/out table (Tango formulation).
  Pitch-level count states are not currently distinguishable in our
  parser; the count-aware table broadcasts the 24-state values across
  all 12 counts as a fallback.
- **Postseason** is excluded from training data.
- **2020 season** is included; the universal-DH and 7-inning-doubleheader
  rules don't materially affect base/out RE estimates.
- **Extra-inning ghost runners (2020+)**: the model sees these as a
  runner on 2B in the top of the inning. Reasonable but imperfect.
- The model is **historical**. It does not condition on the specific
  teams playing, the score state of the bullpens, weather, etc.
"""
    )
