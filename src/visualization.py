from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _prepare_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure timestamp column is datetime and remove invalid rows.
    """
    working_df = df.copy()
    working_df["timestamp"] = pd.to_datetime(working_df["timestamp"], errors="coerce")
    working_df = working_df.dropna(subset=["timestamp"])
    return working_df


def _finalize_plot(ax, save_path: str | None = None) -> None:
    """
    Apply layout fixes and optionally save figure.
    """
    ax.figure.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def _apply_date_axis(ax, freq: str = "day") -> None:
    """
    Format date axis depending on aggregation frequency.
    """
    if freq == "day":
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    elif freq == "week":
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    elif freq == "month":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))


def plot_transaction_distribution(
    df: pd.DataFrame,
    bins: int = 50,
    save_path: str | None = None,
) -> None:
    """
    Plot transaction amount distribution.
    """
    working_df = df.copy()
    amounts = pd.to_numeric(working_df["amount"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(amounts, bins=bins)

    ax.set_title("Transaction Amount Distribution", fontsize=14)
    ax.set_xlabel("Amount", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.grid(True, alpha=0.3)

    _finalize_plot(ax, save_path)


def plot_transactions_by_country(
    df: pd.DataFrame,
    top_n: int = 10,
    save_path: str | None = None,
) -> None:
    """
    Plot top countries by transaction count.
    """
    working_df = df.copy()
    country_counts = working_df["country"].astype(str).value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    country_counts.plot(kind="bar", ax=ax)

    ax.set_title("Top Countries by Transaction Volume", fontsize=14)
    ax.set_xlabel("Country", fontsize=11)
    ax.set_ylabel("Transaction Count", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    _finalize_plot(ax, save_path)


def plot_counterparty_concentration(
    df: pd.DataFrame,
    top_n: int = 10,
    save_path: str | None = None,
) -> None:
    """
    Plot most frequent counterparties.
    """
    working_df = df.copy()
    counterparty_counts = working_df["counterparty"].astype(str).value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    counterparty_counts.plot(kind="bar", ax=ax)

    ax.set_title("Top Counterparties by Frequency", fontsize=14)
    ax.set_xlabel("Counterparty", fontsize=11)
    ax.set_ylabel("Transaction Count", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    _finalize_plot(ax, save_path)


def plot_account_activity(
    df: pd.DataFrame,
    account_id: str,
    save_path: str | None = None,
) -> None:
    """
    Plot transaction amounts over time for a specific account.
    """
    working_df = _prepare_timestamp_column(df)
    working_df["amount"] = pd.to_numeric(working_df["amount"], errors="coerce")

    account_df = working_df[working_df["account_id"] == account_id].copy()
    account_df = account_df.sort_values("timestamp")

    if account_df.empty:
        print(f"No data found for account {account_id}.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(account_df["timestamp"], account_df["amount"], marker="o", linewidth=1.5)

    ax.set_title(f"Transaction Activity for {account_id}", fontsize=14)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Amount", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    _apply_date_axis(ax, "day")
    _finalize_plot(ax, save_path)


def plot_inflow_outflow_by_period(
    df: pd.DataFrame,
    period: str = "W",
    save_path: str | None = None,
) -> None:
    """
    Plot inbound vs outbound totals aggregated by period.
    Use period='D', 'W', or 'M'.
    """
    working_df = _prepare_timestamp_column(df)
    working_df["amount"] = pd.to_numeric(working_df["amount"], errors="coerce")
    working_df["transaction_type"] = working_df["transaction_type"].astype(str).str.lower()

    summary = (
        working_df.groupby(
            [pd.Grouper(key="timestamp", freq=period), "transaction_type"]
        )["amount"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    summary.plot(ax=ax, marker="o", linewidth=2)

    ax.set_title(f"Inflow vs Outflow by Period ({period})", fontsize=14)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Total Amount", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Transaction Type")

    if period == "D":
        _apply_date_axis(ax, "day")
    elif period == "W":
        _apply_date_axis(ax, "week")
    elif period == "M":
        _apply_date_axis(ax, "month")
    else:
        _apply_date_axis(ax, "day")

    _finalize_plot(ax, save_path)


def plot_daily_net_flow(
    df: pd.DataFrame,
    period: str = "W",
    save_path: str | None = None,
) -> None:
    """
    Plot net flow = inbound - outbound aggregated by period.
    """
    working_df = _prepare_timestamp_column(df)
    working_df["amount"] = pd.to_numeric(working_df["amount"], errors="coerce")
    working_df["transaction_type"] = working_df["transaction_type"].astype(str).str.lower()

    inbound = (
        working_df[working_df["transaction_type"] == "inbound"]
        .groupby(pd.Grouper(key="timestamp", freq=period))["amount"]
        .sum()
    )

    outbound = (
        working_df[working_df["transaction_type"] == "outbound"]
        .groupby(pd.Grouper(key="timestamp", freq=period))["amount"]
        .sum()
    )

    net_flow = inbound.subtract(outbound, fill_value=0).sort_index()

    fig, ax = plt.subplots(figsize=(16, 7))
    net_flow.plot(ax=ax, marker="o", linewidth=2)

    ax.set_title(f"Net Flow by Period ({period})", fontsize=14)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Net Amount", fontsize=11)
    ax.axhline(0, linewidth=1)
    ax.grid(True, alpha=0.3)

    if period == "D":
        _apply_date_axis(ax, "day")
    elif period == "W":
        _apply_date_axis(ax, "week")
    elif period == "M":
        _apply_date_axis(ax, "month")
    else:
        _apply_date_axis(ax, "day")

    _finalize_plot(ax, save_path)


def plot_pending_activity_by_period(
    df: pd.DataFrame,
    period: str = "W",
    save_path: str | None = None,
) -> None:
    """
    Plot pending transaction counts aggregated by period.
    """
    working_df = _prepare_timestamp_column(df)
    working_df["status"] = working_df["status"].astype(str).str.lower().str.strip()

    pending_df = working_df[working_df["status"] == "pending"].copy()

    counts = (
        pending_df.groupby(pd.Grouper(key="timestamp", freq=period))["transaction_id"]
        .count()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(16, 7))

    if counts.empty:
        ax.text(0.5, 0.5, "No pending transactions found.", ha="center", va="center", fontsize=12)
        ax.set_title(f"Pending Activity by Period ({period})", fontsize=14)
        ax.set_axis_off()
        _finalize_plot(ax, save_path)
        return

    counts.plot(kind="bar", ax=ax)

    ax.set_title(f"Pending Activity by Period ({period})", fontsize=14)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Pending Transaction Count", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    _finalize_plot(ax, save_path)


def plot_transaction_volume_by_hour(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """
    Plot transaction counts by hour of day.
    """
    working_df = _prepare_timestamp_column(df)
    working_df["hour"] = working_df["timestamp"].dt.hour

    hourly_counts = (
        working_df.groupby("hour")["transaction_id"]
        .count()
        .reindex(range(24), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    hourly_counts.plot(kind="bar", ax=ax)

    ax.set_title("Transaction Volume by Hour", fontsize=14)
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Transaction Count", fontsize=11)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", alpha=0.3)

    _finalize_plot(ax, save_path)


def plot_account_status_changes(
    df: pd.DataFrame,
    account_id: str,
    save_path: str | None = None,
) -> None:
    """
    Plot status change event counts for a specific account.
    """
    working_df = _prepare_timestamp_column(df)
    working_df["account_status_change"] = (
        working_df["account_status_change"].astype(str).str.lower().str.strip()
    )

    account_df = working_df[working_df["account_id"] == account_id].copy()
    account_df = account_df[account_df["account_status_change"] != "none"]

    fig, ax = plt.subplots(figsize=(12, 6))

    if account_df.empty:
        ax.text(
            0.5,
            0.5,
            f"No account status change events found for {account_id}.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title(f"Account Status Change Events for {account_id}", fontsize=14)
        ax.set_axis_off()
        _finalize_plot(ax, save_path)
        return

    counts = (
        account_df.groupby("account_status_change")["transaction_id"]
        .count()
        .sort_values(ascending=False)
    )

    counts.plot(kind="bar", ax=ax)

    ax.set_title(f"Account Status Change Events for {account_id}", fontsize=14)
    ax.set_xlabel("Account Status Change", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    _finalize_plot(ax, save_path)


def plot_account_inflow_outflow(
    df: pd.DataFrame,
    account_id: str,
    period: str = "W",
    save_path: str | None = None,
) -> None:
    """
    Plot inbound vs outbound amounts over time for a selected account.
    """
    working_df = _prepare_timestamp_column(df)
    working_df["amount"] = pd.to_numeric(working_df["amount"], errors="coerce")
    working_df["transaction_type"] = working_df["transaction_type"].astype(str).str.lower()

    account_df = working_df[working_df["account_id"] == account_id].copy()

    fig, ax = plt.subplots(figsize=(16, 7))

    if account_df.empty:
        ax.text(
            0.5,
            0.5,
            f"No data found for account {account_id}.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title(f"Inflow vs Outflow for {account_id}", fontsize=14)
        ax.set_axis_off()
        _finalize_plot(ax, save_path)
        return

    summary = (
        account_df.groupby(
            [pd.Grouper(key="timestamp", freq=period), "transaction_type"]
        )["amount"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    summary.plot(ax=ax, marker="o", linewidth=2)

    ax.set_title(f"Inflow vs Outflow for {account_id} ({period})", fontsize=14)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Amount", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Transaction Type")

    if period == "D":
        _apply_date_axis(ax, "day")
    elif period == "W":
        _apply_date_axis(ax, "week")
    elif period == "M":
        _apply_date_axis(ax, "month")
    else:
        _apply_date_axis(ax, "day")

    _finalize_plot(ax, save_path)