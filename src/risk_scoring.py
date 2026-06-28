from __future__ import annotations

from typing import Any
import pandas as pd


RISK_WEIGHTS = {
    "night_activity": 1,
    "rapid_transactions": 2,
    "cross_border_change": 3,
    "post_account_change_activity": 3,
}


def map_risk_level(score: int) -> str:
    """
    Map numeric risk score to a qualitative label.
    """
    if score <= 2:
        return "low"
    if score <= 5:
        return "moderate"
    if score <= 8:
        return "elevated"
    return "high"


def score_account_risk(anomaly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score accounts using heuristic anomaly weights.

    Note:
        This model is heuristic and intended for triage,
        not definitive attribution.
    """
    if anomaly_df.empty:
        return pd.DataFrame(columns=["account_id", "risk_score", "risk_level", "triggered_indicators"])

    working_df = anomaly_df.copy()
    working_df["weight"] = working_df["anomaly_type"].map(RISK_WEIGHTS).fillna(0)

    grouped = (
        working_df.groupby("account_id")
        .agg(
            risk_score=("weight", "sum"),
            triggered_indicators=("anomaly_type", lambda x: sorted(set(x))),
        )
        .reset_index()
    )

    grouped["risk_level"] = grouped["risk_score"].apply(map_risk_level)

    return grouped[["account_id", "risk_score", "risk_level", "triggered_indicators"]].sort_values(
        ["risk_score", "account_id"], ascending=[False, True]
    )