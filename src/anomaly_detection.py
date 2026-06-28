from __future__ import annotations

from typing import Any, cast
import pandas as pd


FINDING_COLUMNS = [
    "account_id",
    "transaction_id",
    "anomaly_type",
    "indicator_strength",
    "observation",
]


def _empty_findings() -> pd.DataFrame:
    return pd.DataFrame(columns=FINDING_COLUMNS)


def _prepare_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()
    working_df["timestamp"] = pd.to_datetime(working_df["timestamp"], errors="coerce")
    return working_df


def flag_night_activity(
    df: pd.DataFrame,
    start_hour: int = 0,
    end_hour: int = 5,
) -> pd.DataFrame:
    working_df = _prepare_timestamps(df)
    working_df = working_df.dropna(subset=["timestamp"]).copy()
    working_df["hour"] = working_df["timestamp"].dt.hour

    flagged = working_df[
        (working_df["hour"] >= start_hour) & (working_df["hour"] <= end_hour)
    ].copy()

    if flagged.empty:
        return _empty_findings()

    flagged["anomaly_type"] = "night_activity"
    flagged["indicator_strength"] = "low"
    flagged["observation"] = flagged["timestamp"].apply(
        lambda ts: f"Transaction observed at {ts}"
    )

    return flagged[FINDING_COLUMNS]


def flag_rapid_transactions(
    df: pd.DataFrame,
    window_minutes: int = 15,
    min_count: int = 3,
) -> pd.DataFrame:
    working_df = _prepare_timestamps(df)
    working_df = working_df.dropna(subset=["timestamp"]).copy()
    working_df = working_df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

    findings: list[dict[str, Any]] = []

    for account_id, group in working_df.groupby("account_id"):
        group = group.reset_index(drop=True)

        for i in range(len(group)):
            start_time = cast(pd.Timestamp, group["timestamp"].iloc[i])

            end_time = start_time + pd.Timedelta(minutes=window_minutes)

            window_slice = group[
                (group["timestamp"] >= start_time)
                & (group["timestamp"] <= end_time)
            ]

            if len(window_slice) >= min_count:
                findings.append(
                    {
                        "account_id": account_id,
                        "transaction_id": str(group["transaction_id"].iloc[i]),
                        "anomaly_type": "rapid_transactions",
                        "indicator_strength": "medium",
                        "observation": (
                            f"{len(window_slice)} transactions within "
                            f"{window_minutes} minutes"
                        ),
                    }
                )
                break

    if not findings:
        return _empty_findings()

    return pd.DataFrame(findings)[FINDING_COLUMNS]


def flag_cross_border_change(df: pd.DataFrame) -> pd.DataFrame:
    working_df = _prepare_timestamps(df)
    working_df = working_df.sort_values(["account_id", "timestamp"])

    findings: list[dict[str, Any]] = []

    for account_id, group in working_df.groupby("account_id"):
        unique_countries = (
            group["country"].dropna().astype(str).str.strip().unique()
        )

        if len(unique_countries) > 1:
            findings.append(
                {
                    "account_id": account_id,
                    "transaction_id": str(group["transaction_id"].iloc[0]),
                    "anomaly_type": "cross_border_change",
                    "indicator_strength": "high",
                    "observation": (
                        f"Country pattern shifted across "
                        f"{len(unique_countries)} locations: "
                        f"{', '.join(unique_countries)}"
                    ),
                }
            )

    if not findings:
        return _empty_findings()

    return pd.DataFrame(findings)[FINDING_COLUMNS]


def flag_post_account_change_activity(df: pd.DataFrame) -> pd.DataFrame:
    working_df = _prepare_timestamps(df)

    suspicious_changes = {
        "email_changed",
        "password_reset",
        "region_changed",
        "device_changed",
    }

    flagged = working_df[
        working_df["account_status_change"].isin(suspicious_changes)
    ].copy()

    if flagged.empty:
        return _empty_findings()

    flagged["anomaly_type"] = "post_account_change_activity"
    flagged["indicator_strength"] = "high"
    flagged["observation"] = flagged["account_status_change"].apply(
        lambda event: f"Transaction linked to account event: {event}"
    )

    return flagged[FINDING_COLUMNS]


def run_all_anomaly_checks(df: pd.DataFrame) -> pd.DataFrame:
    results = [
        flag_night_activity(df),
        flag_rapid_transactions(df),
        flag_cross_border_change(df),
        flag_post_account_change_activity(df),
    ]

    combined = pd.concat(results, ignore_index=True)

    if combined.empty:
        return _empty_findings()

    combined = combined.drop_duplicates()
    combined = combined.sort_values(
        ["account_id", "anomaly_type"]
    ).reset_index(drop=True)

    return combined