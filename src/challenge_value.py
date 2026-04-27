"""
Challenge value: combine WP & RE deltas into a single, interpretable number
from the perspective of the team that asked for the challenge.
"""
from __future__ import annotations

from typing import Dict

from .config import CHALLENGE_THRESHOLDS


def calculate_wp_batting(wp_home: float, batting_team_is_home: bool) -> float:
    """Convert P(home win) to P(batting-team win)."""
    return float(wp_home) if batting_team_is_home else 1.0 - float(wp_home)


def calculate_challenge_value(
    wp_batting_pre: float,
    wp_batting_post: float,
    challenging_team_is_batting: bool,
) -> Dict[str, float]:
    wpa_batting = wp_batting_post - wp_batting_pre
    if challenging_team_is_batting:
        challenge_value = wpa_batting
    else:
        challenge_value = -1.0 * wpa_batting
    return {
        "wpa_batting": wpa_batting,
        "challenge_value": challenge_value,
    }


def _bucket(abs_cv: float) -> str:
    if abs_cv < CHALLENGE_THRESHOLDS["negligible"]:
        return "negligible"
    if abs_cv < CHALLENGE_THRESHOLDS["small"]:
        return "small"
    if abs_cv < CHALLENGE_THRESHOLDS["moderate"]:
        return "moderate"
    if abs_cv < CHALLENGE_THRESHOLDS["high_impact"]:
        return "high-impact"
    return "game-changing"


def generate_interpretation_text(
    challenge_value: float,
    re_delta: float,
    challenging_team_label: str,
) -> str:
    """Plain-English summary, ready for the dashboard."""
    abs_cv = abs(challenge_value)
    bucket = _bucket(abs_cv)
    direction = "gained" if challenge_value > 0 else ("lost" if challenge_value < 0 else "broke even on")
    cv_pp = challenge_value * 100.0  # percentage points

    head = (
        f"This was a **{bucket}** challenge for the {challenging_team_label}. "
        f"They {direction} {abs(cv_pp):.2f} percentage points of win probability."
    )

    if bucket == "negligible":
        body = (
            "The change in win probability is essentially noise — the call did "
            "not meaningfully shift the game's outcome distribution."
        )
    elif bucket == "small":
        body = (
            "A modest shift. Useful in aggregate over a season, but unlikely "
            "to be remembered as a turning point."
        )
    elif bucket == "moderate":
        body = (
            "A real swing. The kind of leverage that justifies burning a "
            "challenge in close games."
        )
    elif bucket == "high-impact":
        body = (
            "A high-leverage moment — the challenge altered win probability by "
            "more than 6 points. Calls of this size show up clearly in WPA "
            "leaderboards."
        )
    else:
        body = (
            "Game-changing. The replay decision moved win probability by more "
            "than 15 points — comparable to a late-inning home run or a key "
            "double play. This is the upper tail of challenge leverage."
        )

    re_line = f"Run-expectancy change: {re_delta:+.3f} expected runs."

    return f"{head}\n\n{body}\n\n{re_line}"
