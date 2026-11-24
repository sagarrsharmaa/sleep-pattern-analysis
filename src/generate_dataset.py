"""
Generate the synthetic sleep dataset used in this project.

Design
------
* 5,000 nightly records = 100 participants x 50 consecutive nights.
* Every participant gets a stable profile: age, gender, BMI, chronotype
  (habitual bedtime), personal sleep need, resting heart rate, lifestyle
  habits and a few risk tendencies (insomnia-prone, apnea-prone,
  irregular-schedule-prone).
* Every night is first assigned a *pattern* - normal or one of five
  abnormal patterns - and the night's measurements are then drawn from that
  pattern with realistic noise. The generating pattern is stored as the
  ground-truth label (abnormality_type / abnormality_flag). Because the
  measurements are noisy, simple threshold rules do NOT recover the labels
  perfectly, which lets us evaluate detection methods honestly.

Run:  python src/generate_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_PARTICIPANTS = 100
N_NIGHTS = 50
START_DATE = "2024-01-01"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_CSV = DATA_DIR / "sleep_dataset.csv"
OUT_DICT = DATA_DIR / "data_dictionary.csv"

PATTERNS = [
    "normal",
    "insomnia",
    "sleep_deprivation",
    "irregular_schedule",
    "apnea_risk",
    "oversleeping",
]

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. Participant profiles
# ---------------------------------------------------------------------------
def make_profiles() -> pd.DataFrame:
    n = N_PARTICIPANTS
    age = rng.integers(18, 66, n)
    bmi = np.clip(rng.normal(25, 4, n), 17, 40)
    profiles = pd.DataFrame(
        {
            "participant_id": [f"P{i:03d}" for i in range(1, n + 1)],
            "age": age,
            "gender": rng.choice(["Male", "Female"], n),
            "bmi": bmi.round(1),
            # habitual bedtime in hours; values above 24 mean after midnight
            "chronotype": rng.normal(23.25, 0.75, n),
            "sleep_need": np.clip(rng.normal(7.4, 0.5, n), 6.3, 8.8),
            "base_hr": np.clip(rng.normal(60, 6, n), 46, 78),
            "caffeine_habit": rng.choice([0, 80, 160, 240], n, p=[0.2, 0.35, 0.3, 0.15]),
            "screen_habit": rng.choice([15, 45, 90], n, p=[0.3, 0.45, 0.25]),
            "stress_base": rng.uniform(2.5, 6.5, n),
            "steps_base": rng.normal(7500, 2000, n),
        }
    )
    # risk tendencies - apnea is far more likely for older, higher-BMI people
    apnea_p = np.where((profiles.bmi >= 29) & (profiles.age >= 38), 0.6, 0.03)
    profiles["apnea_prone"] = rng.random(n) < apnea_p
    profiles["insomnia_prone"] = rng.random(n) < 0.10
    profiles["irregular_prone"] = rng.random(n) < 0.10
    return profiles


# ---------------------------------------------------------------------------
# 2. Pick the pattern for one night
# ---------------------------------------------------------------------------
def pick_pattern(p: pd.Series, weekend: bool) -> str:
    probs = {
        "insomnia": 0.28 if p.insomnia_prone else 0.02,
        "apnea_risk": 0.45 if p.apnea_prone else 0.003,
        "irregular_schedule": (0.20 if p.irregular_prone else 0.015) * (1.6 if weekend else 1.0),
        "sleep_deprivation": 0.015 if weekend else 0.035,
        "oversleeping": 0.03 if weekend else 0.006,
    }
    r = rng.random()
    cumulative = 0.0
    for name, prob in probs.items():
        cumulative += prob
        if r < cumulative:
            return name
    return "normal"


# ---------------------------------------------------------------------------
# 3. Generate the measurements for one night
# ---------------------------------------------------------------------------
def make_night(p: pd.Series, date: pd.Timestamp, pattern: str) -> dict:
    weekend = date.day_of_week in (4, 5)  # Friday / Saturday nights

    # lifestyle for the day
    stress = rng.normal(p.stress_base, 1.3)
    caffeine = max(0.0, rng.normal(p.caffeine_habit, 40))
    screen = max(0.0, rng.normal(p.screen_habit, 20) + (30 if weekend else 0))
    steps = np.clip(rng.normal(p.steps_base - (1500 if weekend else 0), 2000), 1500, 20000)
    if pattern == "insomnia":
        stress += 2.0
    elif pattern == "sleep_deprivation":
        stress += 1.5
    stress = int(np.clip(round(stress), 1, 10))

    # baseline ("normal") night - deliberately noisy, real sleep varies a lot
    bed = p.chronotype + (0.6 if weekend else 0.0) + rng.normal(0, 0.45)
    mean_latency = 6 + 0.02 * caffeine + 0.05 * screen + 1.0 * stress
    latency = mean_latency * rng.lognormal(0, 0.45)
    awakenings = rng.poisson(0.9 + 0.035 * (p.age - 18))
    waso = awakenings * rng.uniform(3, 9) + abs(rng.normal(0, 6))
    total_sleep = rng.normal(p.sleep_need + (0.4 if weekend else 0.0), 0.55)
    deep = rng.normal(22 - 0.15 * (p.age - 18), 3.5)
    rem = rng.normal(22, 3)
    hr = p.base_hr + rng.normal(0, 3) + 0.01 * caffeine
    spo2 = np.clip(rng.normal(94.4, 1.4), 88.0, 98.5)
    breathing = rng.gamma(2.0, 1.8)

    # pattern-specific changes; severity 0.4 (mild) .. 1.0 (severe), so mild
    # abnormal nights overlap with normal ones - just like real data
    sev = rng.uniform(0.4, 1.0)
    if pattern == "insomnia":
        latency = latency + sev * rng.uniform(15, 75)
        awakenings = awakenings + rng.poisson(1 + 3 * sev)
        waso = waso + sev * rng.uniform(15, 70)
        total_sleep -= sev * rng.uniform(0.5, 2.5)
        deep -= 4 * sev
        hr += 4 * sev
    elif pattern == "sleep_deprivation":
        bed += sev * rng.uniform(1.0, 3.0)
        total_sleep = p.sleep_need - sev * rng.uniform(1.5, 4.0) + rng.normal(0, 0.3)
        latency *= 0.6
        waso *= 0.7
        rem -= 4 * sev
        deep += 2 * sev
    elif pattern == "irregular_schedule":
        direction = 1 if rng.random() < 0.75 else -1
        bed += direction * (1.0 + sev * rng.uniform(1.0, 3.0))
        total_sleep = rng.normal(p.sleep_need - 0.7 * sev, 0.9)
        latency += 10 * sev
    elif pattern == "apnea_risk":
        breathing = breathing + sev * rng.uniform(8, 40)
        spo2 = spo2 - sev * rng.uniform(2, 10)
        awakenings = awakenings + rng.poisson(1 + 2.5 * sev)
        waso = waso + sev * rng.uniform(5, 35)
        total_sleep -= sev * rng.uniform(0, 1.2)
        deep -= 6 * sev
        hr += 5 * sev
    elif pattern == "oversleeping":
        total_sleep = p.sleep_need + 1.0 + sev * rng.uniform(1.0, 3.5)

    # keep values physiologically plausible and internally consistent
    total_sleep = float(np.clip(total_sleep, 2.5, 13.0))
    latency = float(np.clip(latency, 1, 150))
    waso = float(np.clip(waso, 0, 240))
    deep = float(np.clip(deep, 4, 32))
    rem = float(np.clip(rem, 10, 30))
    light = 100 - deep - rem
    time_in_bed = total_sleep + latency / 60 + waso / 60
    efficiency = total_sleep / time_in_bed * 100
    wake = bed + time_in_bed

    return {
        "participant_id": p.participant_id,
        "date": date.strftime("%Y-%m-%d"),
        "day_of_week": date.day_name(),
        "is_weekend": int(weekend),
        "age": int(p.age),
        "gender": p.gender,
        "bmi": p.bmi,
        "bedtime": to_clock(bed),
        "wake_time": to_clock(wake),
        "time_in_bed_hr": round(time_in_bed, 2),
        "sleep_latency_min": round(latency, 1),
        "total_sleep_hr": round(total_sleep, 2),
        "sleep_efficiency_pct": round(efficiency, 1),
        "awakenings": int(awakenings),
        "waso_min": round(waso, 1),
        "deep_sleep_pct": round(deep, 1),
        "rem_sleep_pct": round(rem, 1),
        "light_sleep_pct": round(light, 1),
        "avg_heart_rate_bpm": round(float(hr), 1),
        "min_spo2_pct": round(float(spo2), 1),
        "breathing_events_per_hr": round(float(breathing), 1),
        "caffeine_mg": int(round(caffeine)),
        "screen_time_min": int(round(screen)),
        "daily_steps": int(round(steps)),
        "stress_level": stress,
        "abnormality_type": pattern,
        "abnormality_flag": int(pattern != "normal"),
    }


def to_clock(hours: float) -> str:
    """Convert hours (may exceed 24) to an HH:MM clock string."""
    total_min = int(round(hours * 60)) % (24 * 60)
    return f"{total_min // 60:02d}:{total_min % 60:02d}"


# ---------------------------------------------------------------------------
# 4. Data dictionary
# ---------------------------------------------------------------------------
DICTIONARY = [
    ("record_id", "string", "Unique ID of the nightly record (S00001-S05000)."),
    ("participant_id", "string", "Anonymous participant ID (P001-P100). No names or personal identifiers are stored."),
    ("date", "string date", "Calendar date on which the night started (YYYY-MM-DD)."),
    ("day_of_week", "string/category", "Day name of the night's start date."),
    ("is_weekend", "integer (0/1)", "1 for Friday and Saturday nights, otherwise 0."),
    ("age", "integer", "Participant age in years (18-65)."),
    ("gender", "string/category", "Participant gender (Male/Female)."),
    ("bmi", "float", "Body-mass index (kg/m^2)."),
    ("bedtime", "string time", "Clock time the participant went to bed (HH:MM, 24-hour)."),
    ("wake_time", "string time", "Clock time the participant got out of bed (HH:MM, 24-hour)."),
    ("time_in_bed_hr", "float", "Hours from bedtime to wake time = total sleep + latency + WASO."),
    ("sleep_latency_min", "float", "Minutes taken to fall asleep after going to bed."),
    ("total_sleep_hr", "float", "Total hours actually asleep."),
    ("sleep_efficiency_pct", "float", "Total sleep / time in bed x 100."),
    ("awakenings", "integer", "Number of awakenings after falling asleep."),
    ("waso_min", "float", "Wake After Sleep Onset: minutes awake after first falling asleep."),
    ("deep_sleep_pct", "float", "Share of total sleep spent in deep (N3) sleep."),
    ("rem_sleep_pct", "float", "Share of total sleep spent in REM sleep."),
    ("light_sleep_pct", "float", "Share of total sleep spent in light (N1/N2) sleep; deep + REM + light = 100."),
    ("avg_heart_rate_bpm", "float", "Average heart rate during sleep (beats per minute)."),
    ("min_spo2_pct", "float", "Lowest blood-oxygen saturation recorded during the night (%)."),
    ("breathing_events_per_hr", "float", "Breathing disturbances per hour of sleep (apnea-hypopnea style index)."),
    ("caffeine_mg", "integer", "Caffeine consumed during the day (mg)."),
    ("screen_time_min", "integer", "Screen use in the two hours before bed (minutes)."),
    ("daily_steps", "integer", "Steps walked during the day."),
    ("stress_level", "integer", "Self-reported stress for the day (1 = low, 10 = high)."),
    ("abnormality_type", "string/category", "Ground-truth sleep pattern: normal, insomnia, sleep_deprivation, irregular_schedule, apnea_risk or oversleeping."),
    ("abnormality_flag", "integer (0/1)", "1 if abnormality_type is not 'normal', otherwise 0."),
]


def build_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, dtype, desc in DICTIONARY:
        samples = df[col].iloc[[0, 1, 2]].tolist()
        rows.append(
            {
                "column_name": col,
                "data_type": dtype,
                "sample_value_1": samples[0],
                "sample_value_2": samples[1],
                "sample_value_3": samples[2],
                "nullable": "No",
                "description": desc,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    profiles = make_profiles()
    dates = pd.date_range(START_DATE, periods=N_NIGHTS, freq="D")

    records = []
    for _, p in profiles.iterrows():
        for date in dates:
            pattern = pick_pattern(p, weekend=date.day_of_week in (4, 5))
            records.append(make_night(p, date, pattern))

    df = pd.DataFrame(records).sort_values(["date", "participant_id"]).reset_index(drop=True)
    df.insert(0, "record_id", [f"S{i:05d}" for i in range(1, len(df) + 1)])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    build_dictionary(df).to_csv(OUT_DICT, index=False)

    print(f"Saved {len(df):,} rows x {df.shape[1]} columns -> {OUT_CSV.relative_to(ROOT)}")
    print(f"Abnormal nights: {df.abnormality_flag.sum()} ({df.abnormality_flag.mean():.1%})")
    print(df.abnormality_type.value_counts().to_string())


if __name__ == "__main__":
    main()
