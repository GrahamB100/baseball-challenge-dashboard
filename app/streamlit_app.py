"""
Streamlit dashboard: Replay Challenge Value.

Visual style: Baseball Savant-inspired — navy header, white stat cards with
small uppercase labels and large bold values, red/navy charts.

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
# Page config + Savant palette
# ----------------------------------------------------------------------
NAVY = "#041E42"
RED = "#D50000"
LIGHT_GRAY = "#F4F4F4"
BORDER = "#D8D8D8"
MUTED = "#6B7280"

st.set_page_config(
    page_title="Replay Challenge Value",
    page_icon=":baseball:",
    layout="wide",
)

# ----------------------------------------------------------------------
# Custom CSS — Baseball Savant aesthetic
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <style>
      /* Page background and base font */
      .stApp {{
        background: #FFFFFF;
      }}
      html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
        color: #111827;
      }}
      /* Hide default Streamlit chrome that fights the look */
      header[data-testid="stHeader"] {{ background: transparent; }}
      .block-container {{ padding-top: 1.2rem; max-width: 1280px; }}

      /* --- Savant-style header bar --- */
      .savant-header {{
        background: {NAVY};
        color: #FFFFFF;
        padding: 18px 24px;
        margin: 0 0 18px 0;
        border-bottom: 4px solid {RED};
      }}
      .savant-header h1 {{
        margin: 0;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #FFFFFF;
      }}
      .savant-header p {{
        margin: 4px 0 0 0;
        font-size: 13px;
        color: #C9D1DB;
        letter-spacing: 0.02em;
      }}

      /* --- Section header (bold uppercase, red underline) --- */
      .savant-section {{
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {NAVY};
        border-bottom: 2px solid {RED};
        padding-bottom: 6px;
        margin: 26px 0 14px 0;
      }}

      /* --- Stat card --- */
      .savant-card {{
        background: #FFFFFF;
        border: 1px solid {BORDER};
        border-top: 3px solid {NAVY};
        padding: 14px 16px 16px 16px;
        height: 100%;
      }}
      .savant-card.accent {{ border-top-color: {RED}; }}
      .savant-label {{
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {MUTED};
        margin-bottom: 4px;
      }}
      .savant-value {{
        font-size: 32px;
        font-weight: 800;
        color: {NAVY};
        line-height: 1.05;
        font-variant-numeric: tabular-nums;
      }}
      .savant-value.accent {{ color: {RED}; }}
      .savant-delta {{
        font-size: 12px;
        font-weight: 600;
        margin-top: 6px;
        color: {MUTED};
      }}
      .savant-delta.pos {{ color: #15803D; }}
      .savant-delta.neg {{ color: {RED}; }}

      /* Interpretation box */
      .savant-interp {{
        background: {LIGHT_GRAY};
        border-left: 4px solid {NAVY};
        padding: 14px 18px;
        font-size: 14px;
        color: #1F2937;
        line-height: 1.55;
      }}

      /* Sidebar look */
      [data-testid="stSidebar"] > div:first-child {{
        background: {LIGHT_GRAY};
        border-right: 1px solid {BORDER};
      }}
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {{
        color: {NAVY};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 13px;
      }}

      /* Force ALL sidebar widget labels and text to be readable */
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] .stRadio label,
      [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p,
      [data-testid="stSidebar"] .stSelectbox label,
      [data-testid="stSidebar"] .stNumberInput label,
      [data-testid="stSidebar"] [data-baseweb="radio"] {{
        color: {NAVY} !important;
        font-weight: 600 !important;
        font-size: 12.5px !important;
        letter-spacing: 0.04em;
        opacity: 1 !important;
      }}

      /* Caption / helper text */
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: {MUTED} !important;
        font-size: 12px !important;
      }}

      /* Number input + selectbox value text — dark, readable on white field */
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] [data-baseweb="input"] input {{
        color: #111827 !important;
        background: #FFFFFF !important;
      }}

      /* Radio option text */
      [data-testid="stSidebar"] div[role="radiogroup"] label > div:last-child,
      [data-testid="stSidebar"] div[role="radiogroup"] label p {{
        color: {NAVY} !important;
        font-weight: 500 !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
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
# Card helpers (HTML so we can style precisely)
# ----------------------------------------------------------------------
def stat_card(label: str, value: str, delta: str | None = None,
              delta_kind: str = "neutral", accent: bool = False) -> str:
    """Returns the HTML for a Savant-style stat card."""
    delta_class = {"pos": "pos", "neg": "neg", "neutral": ""}[delta_kind]
    delta_html = f'<div class="savant-delta {delta_class}">{delta}</div>' if delta else ""
    accent_cls = "accent" if accent else ""
    return f"""
        <div class="savant-card {accent_cls}">
            <div class="savant-label">{label}</div>
            <div class="savant-value {accent_cls}">{value}</div>
            {delta_html}
        </div>
    """


def section(title: str) -> None:
    st.markdown(f'<div class="savant-section">{title}</div>', unsafe_allow_html=True)


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
    st.markdown(f"#### {prefix.upper()}-CHALLENGE STATE")
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
    st.markdown("### GAME-STATE INPUTS")
    st.caption("Set the state of the game **before** and **after** the replay challenge.")

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
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="savant-header">
        <h1>Replay Challenge Value</h1>
        <p>Run expectancy + calibrated win probability for any challenge state &mdash; Retrosheet</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Artifact / input checks
# ----------------------------------------------------------------------
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

try:
    validate_game_state_input(pre)
    validate_game_state_input(post)
except ValueError as e:
    st.error(f"Input error: {e}")
    st.stop()


# ----------------------------------------------------------------------
# Calculations
# ----------------------------------------------------------------------
re_pre = lookup_run_expectancy(pre, re_count, re_24)
re_post = lookup_run_expectancy(post, re_count, re_24)
re_delta = re_post - re_pre

try:
    wp_home_pre = predict_home_wp(model, create_model_features(pre))
    wp_home_post = predict_home_wp(model, create_model_features(post))
except Exception as e:
    st.error(f"WP model prediction failed: {e}")
    st.stop()

wp_bat_pre = calculate_wp_batting(wp_home_pre, batting_team_is_home)
wp_bat_post = calculate_wp_batting(wp_home_post, batting_team_is_home)
cv = calculate_challenge_value(wp_bat_pre, wp_bat_post, challenging_team_is_batting)
challenger_label = "batting team" if challenging_team_is_batting else "fielding team"


# ----------------------------------------------------------------------
# Headline metrics row
# ----------------------------------------------------------------------
section("Run Expectancy")
re_col_1, re_col_2, re_col_3 = st.columns(3)
re_col_1.markdown(stat_card("RE PRE", f"{re_pre:.3f}", "expected runs"), unsafe_allow_html=True)
re_col_2.markdown(stat_card("RE POST", f"{re_post:.3f}", "expected runs"), unsafe_allow_html=True)
re_delta_kind = "pos" if re_delta > 0 else ("neg" if re_delta < 0 else "neutral")
re_col_3.markdown(
    stat_card("RE DELTA", fmt_runs(re_delta), "post − pre", delta_kind=re_delta_kind, accent=True),
    unsafe_allow_html=True,
)

section("Win Probability — Batting Team Perspective")
wp_col_1, wp_col_2, wp_col_3 = st.columns(3)
wp_col_1.markdown(stat_card("WP PRE", fmt_pct(wp_bat_pre)), unsafe_allow_html=True)
wp_col_2.markdown(stat_card("WP POST", fmt_pct(wp_bat_post)), unsafe_allow_html=True)
wpa_kind = "pos" if cv["wpa_batting"] > 0 else ("neg" if cv["wpa_batting"] < 0 else "neutral")
wp_col_3.markdown(
    stat_card("WPA (BATTING)", fmt_pp(cv["wpa_batting"]), "post − pre",
              delta_kind=wpa_kind, accent=True),
    unsafe_allow_html=True,
)

section(f"Challenge Value — {challenger_label.title()} Perspective")
cv_kind = "pos" if cv["challenge_value"] > 0 else ("neg" if cv["challenge_value"] < 0 else "neutral")
cv_col_1, cv_col_2, cv_col_3 = st.columns([2, 1, 1])
cv_col_1.markdown(
    stat_card("CHALLENGE VALUE", fmt_pp(cv["challenge_value"]),
              "from the challenging team's perspective",
              delta_kind=cv_kind, accent=True),
    unsafe_allow_html=True,
)
cv_col_2.markdown(stat_card("WP (HOME)", fmt_pct(wp_home_pre) + " → " + fmt_pct(wp_home_post)), unsafe_allow_html=True)
cv_col_3.markdown(stat_card("RE SHIFT", fmt_runs(re_delta)), unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
section("Visual Comparison")

def _savant_layout(fig: go.Figure, title: str) -> go.Figure:
    axis_label = dict(color=NAVY, size=13, family="Arial")
    axis_title = dict(color=NAVY, size=12, family="Arial Black")
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left",
                   font=dict(size=14, color=NAVY, family="Arial Black")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=50, b=50),
        height=340,
        showlegend=False,
        font=dict(family="Arial", color=NAVY, size=12),
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor=NAVY,
        linewidth=1.5,
        ticks="outside",
        tickcolor=NAVY,
        tickfont=axis_label,
        title_font=axis_title,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=True,
        zerolinecolor=NAVY,
        zerolinewidth=1,
        tickfont=axis_label,
        title_font=axis_title,
        showline=True,
        linecolor=NAVY,
        linewidth=1.5,
    )
    return fig

chart_l, chart_r = st.columns(2)

with chart_l:
    fig = go.Figure(data=[
        go.Bar(
            x=["PRE", "POST"], y=[re_pre, re_post],
            marker_color=[NAVY, RED],
            text=[f"<b>{re_pre:.3f}</b>", f"<b>{re_post:.3f}</b>"],
            textposition="outside",
            textfont=dict(color=NAVY, size=14),
            width=[0.55, 0.55],
        ),
    ])
    fig = _savant_layout(fig, "RUN EXPECTANCY")
    fig.update_yaxes(title_text="Expected runs")
    st.plotly_chart(fig, use_container_width=True)

with chart_r:
    fig = go.Figure(data=[
        go.Bar(
            x=["PRE", "POST"], y=[wp_bat_pre*100, wp_bat_post*100],
            marker_color=[NAVY, RED],
            text=[f"<b>{wp_bat_pre*100:.1f}%</b>", f"<b>{wp_bat_post*100:.1f}%</b>"],
            textposition="outside",
            textfont=dict(color=NAVY, size=14),
            width=[0.55, 0.55],
        ),
    ])
    fig = _savant_layout(fig, "WIN PROBABILITY (BATTING TEAM)")
    fig.update_yaxes(title_text="WP (%)", range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# Interpretation
# ----------------------------------------------------------------------
section("Interpretation")
interp = generate_interpretation_text(
    challenge_value=cv["challenge_value"],
    re_delta=re_delta,
    challenging_team_label=challenger_label,
)
# Render interpretation HTML-safe — strip the markdown bold from the helper
# and apply our own styling.
st.markdown(f'<div class="savant-interp">{interp}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Notes / limitations
# ----------------------------------------------------------------------
with st.expander("MODEL NOTES & LIMITATIONS"):
    st.markdown(
        """
- **WP model:** HistGradientBoosting, isotonic-calibrated, trained on
  Retrosheet plate-appearance states. Train/test split is by
  season (last 3 seasons in the holdout). See `outputs/wp_evaluation.csv`
  and `outputs/calibration_plot.png` for diagnostics.
- **RE table:** classic 24-state base/out table (Tango formulation).
  Pitch-level count states are not currently distinguishable in our
  parser; the count-aware table broadcasts the 24-state values across all
  12 counts as a fallback.
- **Postseason** is excluded from training data.
- **Extra-inning ghost runners (2020+)**: the model sees these as a
  runner on 2B in the top of the inning. Reasonable but imperfect.
- The model is **historical**. It does not condition on the specific
  teams playing, the score state of the bullpens, weather, etc.
        """
    )

st.markdown(
    f"""
    <div style='text-align: center; color: {MUTED}; font-size: 11px;
                margin-top: 32px; padding-top: 16px;
                border-top: 1px solid {BORDER}; letter-spacing: 0.04em;'>
        DATA: RETROSHEET &nbsp;|&nbsp; MODEL: HISTGRADIENTBOOSTING (CALIBRATED)
        &nbsp;|&nbsp; STYLE INSPIRED BY BASEBALL SAVANT
    </div>
    """,
    unsafe_allow_html=True,
)
