from __future__ import annotations

from typing import Any
import pandas as pd


SUSPICIOUS_ACCOUNT_EVENTS = {
    "email_changed",
    "password_reset",
    "region_changed",
    "device_changed",
}


def prepare_timeline_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare transaction data for timeline reconstruction.

    Steps:
    - convert timestamp column to datetime
    - sort chronologically
    - remove rows with invalid timestamps
    """
    working_df = df.copy()
    working_df["timestamp"] = pd.to_datetime(working_df["timestamp"], errors="coerce")
    working_df = working_df.dropna(subset=["timestamp"])
    working_df = working_df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

    return working_df


def extract_timeline_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract timeline-relevant events from the prepared dataframe.

    Events include:
    - account status changes
    - night activity
    - high-value transactions
    - pending transactions
    """
    working_df = prepare_timeline_data(df)
    findings: list[dict[str, Any]] = []

    for _, row in working_df.iterrows():
        timestamp = row["timestamp"]
        account_id = row["account_id"]
        transaction_id = row["transaction_id"]
        amount = row["amount"]
        country = row["country"]
        status = row["status"]
        channel = row["channel"]
        account_change = row["account_status_change"]

        if account_change in SUSPICIOUS_ACCOUNT_EVENTS:
            findings.append(
                {
                    "account_id": account_id,
                    "timestamp": timestamp,
                    "event_type": "account_change",
                    "transaction_id": transaction_id,
                    "event_summary": f"Account event detected: {account_change}",
                }
            )

        if timestamp.hour >= 0 and timestamp.hour <= 5:
            findings.append(
                {
                    "account_id": account_id,
                    "timestamp": timestamp,
                    "event_type": "night_activity",
                    "transaction_id": transaction_id,
                    "event_summary": f"Night transaction via {channel} from {country}",
                }
            )

        if pd.notna(amount) and float(amount) >= 10000:
            findings.append(
                {
                    "account_id": account_id,
                    "timestamp": timestamp,
                    "event_type": "high_value_transaction",
                    "transaction_id": transaction_id,
                    "event_summary": f"High-value transaction observed: {amount}",
                }
            )

        if status == "pending":
            findings.append(
                {
                    "account_id": account_id,
                    "timestamp": timestamp,
                    "event_type": "pending_activity",
                    "transaction_id": transaction_id,
                    "event_summary": "Transaction remained in pending state",
                }
            )

    return pd.DataFrame(findings).sort_values(["account_id", "timestamp"]).reset_index(drop=True)


def build_account_timeline(df: pd.DataFrame, account_id: str) -> list[str]:
    """
    Build a readable timeline for a single account.

    Returns:
        A list of formatted timeline entries.
    """
    events_df = extract_timeline_events(df)
    account_events = events_df[events_df["account_id"] == account_id].copy()

    if account_events.empty:
        return [f"No timeline events identified for account {account_id}."]

    timeline_entries = []
    for _, row in account_events.iterrows():
        entry = f"[{row['timestamp']}] {row['event_type']} - {row['event_summary']}"
        timeline_entries.append(entry)

    return timeline_entries


def build_full_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a full structured timeline dataframe for all accounts.
    """
    return extract_timeline_events(df)