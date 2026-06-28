from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any


# =========================
# CONFIG
# =========================

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 1, 1)
OUTPUT_PATH = Path("data/raw/synthetic_transactions.csv")
RANDOM_SEED = 42

COUNTRIES = ["TR", "AE", "DE", "FR", "SG", "US"]
HIGH_RISK_COUNTRIES = {"AE", "SG"}
CHANNELS = ["web", "mobile", "atm"]
CURRENCIES = ["USD", "EUR"]


# =========================
# ACCOUNT PROFILE
# =========================

@dataclass(frozen=True)
class AccountProfile:
    account_id: str
    baseline_country: str
    currency: str
    normal_daily_range: Tuple[int, int]
    normal_amount_range: Tuple[int, int]
    preferred_channels: Tuple[str, ...]
    anomaly_probability: float


ACCOUNT_PROFILES: Dict[str, AccountProfile] = {
    "ACC-1001": AccountProfile("ACC-1001", "TR", "USD", (2, 6), (800, 6000), ("web", "mobile"), 0.18),
    "ACC-2044": AccountProfile("ACC-2044", "TR", "USD", (1, 4), (100, 1500), ("atm", "web"), 0.04),
    "ACC-3320": AccountProfile("ACC-3320", "FR", "EUR", (1, 3), (1500, 5500), ("web",), 0.03),
    "ACC-7788": AccountProfile("ACC-7788", "DE", "EUR", (0, 2), (200, 2200), ("web", "mobile"), 0.05),
    "ACC-9001": AccountProfile("ACC-9001", "US", "USD", (2, 8), (500, 9000), ("web", "mobile", "atm"), 0.10),
}


# =========================
# HELPERS
# =========================

def random_ip() -> str:
    return ".".join(str(random.randint(1, 255)) for _ in range(4))


def random_device() -> str:
    return f"DEV-{random.randint(1000, 9999)}"


def random_counterparty() -> str:
    return f"CTR-{random.randint(10, 999)}"


def random_session_id() -> str:
    return f"SES-{random.randint(1, 99999)}"


def is_cross_border(country: str, baseline: str) -> bool:
    return country != baseline


def weighted_status(score: int) -> str:
    if score >= 90:
        return random.choice(["blocked", "completed"])
    if score >= 75:
        return random.choice(["completed", "pending"])
    return "completed"


def compute_risk_score(
    amount: int,
    hour: int,
    change_flag: str,
    country: str,
    baseline_country: str,
    is_burst: bool,
    txn_count: int,
) -> int:
    score = 10

    if amount > 10000:
        score += 25
    elif amount > 5000:
        score += 10

    if hour < 6 or hour == 23:
        score += 15

    if change_flag != "none":
        score += 20

    if is_cross_border(country, baseline_country):
        score += 10

    if country in HIGH_RISK_COUNTRIES:
        score += 10

    if is_burst:
        score += 20

    if txn_count >= 4:
        score += 10

    return min(score, 99)


# =========================
# CORE BUILD
# =========================

def build_row(
    txn_id: int,
    account: AccountProfile,
    ts: datetime,
    amount: int,
    tx_type: str,
    country: str,
    channel: str,
    device: str,
    ip: str,
    change_flag: str,
    geo_velocity: int,
    is_burst: bool,
    txn_count: int,
    notes: str,
) -> List[Any]:

    score = compute_risk_score(
        amount,
        ts.hour,
        change_flag,
        country,
        account.baseline_country,
        is_burst,
        txn_count,
    )

    return [
        f"TXN-{txn_id:06d}",
        account.account_id,
        ts.strftime("%Y-%m-%d %H:%M:%S"),
        amount,
        account.currency,
        tx_type,
        random_counterparty(),
        country,
        channel,
        device,
        ip,
        weighted_status(score),
        change_flag,
        random_session_id(),
        geo_velocity,
        score,
        "true" if score >= 75 else "false",
        notes,
    ]


# =========================
# NORMAL DATA
# =========================

def generate_normal(account: AccountProfile, day: datetime, txn_id: int):
    rows = []
    count = random.randint(*account.normal_daily_range)

    for _ in range(count):
        hour = random.randint(7, 22)
        ts = day.replace(hour=hour, minute=random.randint(0,59), second=random.randint(0,59))

        amount = random.randint(*account.normal_amount_range)
        country = account.baseline_country if random.random() < 0.9 else random.choice(COUNTRIES)

        rows.append(build_row(
            txn_id,
            account,
            ts,
            amount,
            random.choice(["inbound","outbound"]),
            country,
            random.choice(account.preferred_channels),
            random_device(),
            random_ip(),
            "none",
            random.randint(0, 200),
            False,
            1,
            "baseline behavior"
        ))

        txn_id += 1

    return rows, txn_id


# =========================
# ANOMALY BURST
# =========================

def generate_burst(account: AccountProfile, day: datetime, txn_id: int):
    rows = []
    burst_size = random.randint(3, 7)

    base_time = day.replace(hour=random.choice([0,1,2,3,4,5,23]), minute=0, second=0)
    device = random_device()
    ip = random_ip()
    country = random.choice([c for c in COUNTRIES if c != account.baseline_country])
    change_flag = random.choice(["password_reset","device_changed","email_changed"])

    for i in range(burst_size):
        ts = base_time + timedelta(minutes=i * random.randint(1,3))

        rows.append(build_row(
            txn_id,
            account,
            ts,
            random.randint(7000, 20000),
            "outbound",
            country,
            random.choice(["web","mobile"]),
            device,
            ip,
            change_flag if i == 0 else "none",
            random.randint(800, 1500),
            True,
            burst_size,
            "anomalous burst"
        ))

        txn_id += 1

    return rows, txn_id


# =========================
# GENERATOR
# =========================

def generate_dataset():
    random.seed(RANDOM_SEED)

    rows = []
    current = START_DATE
    txn_id = 1

    while current < END_DATE:
        for account in ACCOUNT_PROFILES.values():

            normal_rows, txn_id = generate_normal(account, current, txn_id)
            rows.extend(normal_rows)

            if random.random() < account.anomaly_probability:
                anomaly_rows, txn_id = generate_burst(account, current, txn_id)
                rows.extend(anomaly_rows)

        current += timedelta(days=1)

    rows.sort(key=lambda x: x[2])
    return rows


# =========================
# WRITE CSV (FIXED)
# =========================

def write_csv(rows):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "transaction_id","account_id","timestamp","amount","currency",
            "transaction_type","counterparty","country","channel","device_id",
            "ip_address","status","account_status_change","session_id",
            "geo_velocity_km_h","risk_score","is_flagged","notes"
        ])

        for row in rows:
            writer.writerow([str(x) for x in row])  #  TYPE FIX


# =========================
# MAIN
# =========================

def main():
    rows = generate_dataset()
    write_csv(rows)

    print(f"Dataset created: {OUTPUT_PATH}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()