from __future__ import annotations

import pandas as pd


def summarize_counterparties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize counterparty diversity and transaction volume by account.
    """
    working_df = df.copy()

    if working_df.empty:
        return pd.DataFrame(
            columns=["account_id", "unique_counterparties", "total_transactions"]
        )

    summary = (
        working_df.groupby("account_id")
        .agg(
            unique_counterparties=("counterparty", "nunique"),
            total_transactions=("transaction_id", "count"),
        )
        .reset_index()
        .sort_values(
            ["unique_counterparties", "total_transactions"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    return summary


def find_top_counterparties(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Identify the most frequently recurring counterparties.
    """
    working_df = df.copy()

    if working_df.empty:
        return pd.DataFrame(columns=["counterparty", "transaction_count"])

    summary = (
        working_df.groupby("counterparty")
        .agg(transaction_count=("transaction_id", "count"))
        .reset_index()
        .sort_values("transaction_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return summary


def summarize_country_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize country-level transaction exposure by account.
    """
    working_df = df.copy()

    if working_df.empty:
        return pd.DataFrame(
            columns=["account_id", "unique_countries", "total_transactions"]
        )

    summary = (
        working_df.groupby("account_id")
        .agg(
            unique_countries=("country", "nunique"),
            total_transactions=("transaction_id", "count"),
        )
        .reset_index()
        .sort_values(
            ["unique_countries", "total_transactions"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    return summary


def build_account_counterparty_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple account-to-counterparty transaction matrix.
    """
    working_df = df.copy()

    if working_df.empty:
        return pd.DataFrame()

    matrix = pd.pivot_table(
        working_df,
        index="account_id",
        columns="counterparty",
        values="transaction_id",
        aggfunc="count",
        fill_value=0,
    )

    return matrix


def detect_concentrated_counterparty_risk(
    df: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """
    Flag accounts whose transaction activity is overly concentrated on a single counterparty.

    threshold=0.5 means one counterparty accounts for >= 50% of transactions.
    """
    working_df = df.copy()

    expected_columns = [
        "account_id",
        "dominant_counterparty",
        "dominant_share",
        "total_transactions",
    ]

    if working_df.empty:
        return pd.DataFrame(columns=expected_columns)

    findings = []

    for account_id, group in working_df.groupby("account_id"):
        total = len(group)
        if total == 0:
            continue

        counts = group["counterparty"].value_counts(dropna=False)

        if counts.empty:
            continue

        dominant_counterparty = counts.index[0]
        dominant_count = counts.iloc[0]
        dominant_share = dominant_count / total

        if dominant_share >= threshold:
            findings.append(
                {
                    "account_id": account_id,
                    "dominant_counterparty": dominant_counterparty,
                    "dominant_share": round(float(dominant_share), 3),
                    "total_transactions": int(total),
                }
            )

    if not findings:
        return pd.DataFrame(columns=expected_columns)

    result = pd.DataFrame(findings, columns=expected_columns)

    return (
        result.sort_values(
            ["dominant_share", "total_transactions"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def build_relationship_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Run all current relationship mapping summaries.
    """
    return {
        "counterparty_summary": summarize_counterparties(df),
        "top_counterparties": find_top_counterparties(df),
        "country_exposure": summarize_country_exposure(df),
        "account_counterparty_matrix": build_account_counterparty_matrix(df),
        "concentration_risk": detect_concentrated_counterparty_risk(df),
    }