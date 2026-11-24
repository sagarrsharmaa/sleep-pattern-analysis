"""Shared loading and feature-engineering helpers used by the analysis scripts and the dashboard."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "sleep_dataset.csv"

AGE_BINS = [17, 29, 44, 65]
AGE_LABELS = ["18-29", "30-44", "45-65"]

# Columns fed to the machine-learning models
MODEL_FEATURES = [
    "bedtime_hr",
    "bedtime_dev_hr",
    "time_in_bed_hr",
    "sleep_latency_min",
    "total_sleep_hr",
    "sleep_dev_hr",
    "sleep_efficiency_pct",
    "awakenings",
    "waso_min",
    "deep_sleep_pct",
    "rem_sleep_pct",
    "avg_heart_rate_bpm",
    "min_spo2_pct",
    "breathing_events_per_hr",
    "caffeine_mg",
    "screen_time_min",
    "daily_steps",
    "stress_level",
    "age",
    "bmi",
    "is_weekend",
]

PATTERN_ORDER = [
    "normal",
    "insomnia",
    "sleep_deprivation",
    "irregular_schedule",
    "apnea_risk",
    "oversleeping",
]


def bedtime_to_hours(clock: pd.Series) -> pd.Series:
    """'23:30' -> 23.5, '01:15' -> 25.25 (times before 15:00 are treated as after midnight)."""
    parts = clock.str.split(":", expand=True).astype(int)
    hours = parts[0] + parts[1] / 60
    return hours.where(hours >= 15, hours + 24)


def load_dataset(path: Path | str = DATASET) -> pd.DataFrame:
    df = pd.read_csv(path)
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric bedtime, personal-baseline deviations and an age group."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["bedtime_hr"] = bedtime_to_hours(df["bedtime"])
    # Personal baseline: each participant's own median. Uses no labels.
    person = df.groupby("participant_id")
    df["bedtime_dev_hr"] = df["bedtime_hr"] - person["bedtime_hr"].transform("median")
    df["sleep_dev_hr"] = df["total_sleep_hr"] - person["total_sleep_hr"].transform("median")
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    return df
