"""
Sleep Pattern Analysis and Abnormality Detection - interactive dashboard.

Run locally:  streamlit run app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from detection import binary_metrics, isolation_forest_flags, make_classifier, rule_based_screen  # noqa: E402
from features import AGE_LABELS, DATASET, MODEL_FEATURES, PATTERN_ORDER, add_features  # noqa: E402

PRIMARY = "#512A7A"
PATTERN_COLORS = {
    "normal": "#8C8C8C",
    "insomnia": "#512A7A",
    "sleep_deprivation": "#2A9D8F",
    "irregular_schedule": "#E9A23B",
    "apnea_risk": "#D1495B",
    "oversleeping": "#4A7FC1",
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="Sleep Pattern Analysis", page_icon="🌙", layout="wide")


# ---------------------------------------------------------------------------
# Data and models (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load(file) -> pd.DataFrame:
    return add_features(pd.read_csv(file))


@st.cache_data
def run_detection(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rule_prediction"] = rule_based_screen(out)
    out["if_flag"], out["anomaly_score"] = isolation_forest_flags(out)
    return out


@st.cache_resource
def train_model(df: pd.DataFrame):
    return make_classifier().fit(df[MODEL_FEATURES], df["abnormality_type"])


def clock(h: float) -> str:
    m = int(round(h * 60)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🌙 Controls")
upload = st.sidebar.file_uploader("Upload a sleep dataset (CSV)", type="csv",
                                  help="Must follow the 28-column schema in data/data_dictionary.csv")
data = run_detection(load(upload if upload else DATASET))

st.sidebar.subheader("Filters")
gender = st.sidebar.multiselect("Gender", sorted(data.gender.unique()), default=sorted(data.gender.unique()))
ages = st.sidebar.multiselect("Age group", AGE_LABELS, default=AGE_LABELS)
patterns = st.sidebar.multiselect("Sleep pattern", PATTERN_ORDER, default=PATTERN_ORDER)
d_min, d_max = data.date.min().date(), data.date.max().date()
dates = st.sidebar.date_input("Date range", (d_min, d_max), min_value=d_min, max_value=d_max)
if isinstance(dates, tuple) and len(dates) == 2:
    start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
else:
    start, end = pd.Timestamp(d_min), pd.Timestamp(d_max)

df = data[
    data.gender.isin(gender)
    & data.age_group.isin(ages)
    & data.abnormality_type.isin(patterns)
    & data.date.between(start, end)
]
st.sidebar.caption(f"{len(df):,} of {len(data):,} nights selected")

# ---------------------------------------------------------------------------
# Header and KPIs
# ---------------------------------------------------------------------------
st.title("🌙 Sleep Pattern Analysis & Abnormality Detection")
st.caption(f"Nightly sleep records · {data.participant_id.nunique()} participants · "
           f"{data.date.min():%d %b %Y} – {data.date.max():%d %b %Y} · Author: Sagar Sharma")

if df.empty:
    st.warning("No nights match the current filters.")
    st.stop()

k = st.columns(6)
k[0].metric("Nights", f"{len(df):,}")
k[1].metric("Participants", df.participant_id.nunique())
k[2].metric("Avg sleep", f"{df.total_sleep_hr.mean():.2f} h")
k[3].metric("Avg efficiency", f"{df.sleep_efficiency_pct.mean():.1f}%")
k[4].metric("Abnormal nights", int(df.abnormality_flag.sum()))
k[5].metric("Abnormal rate", f"{df.abnormality_flag.mean():.1%}")

tab_over, tab_abn, tab_person, tab_check, tab_data = st.tabs(
    ["📈 Overview", "🚨 Abnormalities", "👤 Participant view", "🧪 Check a night", "🗂 Data"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_over:
    c1, c2 = st.columns(2)
    fig = px.histogram(df, x="total_sleep_hr", nbins=45, color_discrete_sequence=[PRIMARY],
                       title="Total sleep per night", labels={"total_sleep_hr": "Total sleep (h)"})
    fig.add_vrect(x0=7, x1=9, fillcolor="#2A9D8F", opacity=0.12, line_width=0, annotation_text="7–9 h recommended")
    c1.plotly_chart(fig)

    fig = px.histogram(df, x="sleep_efficiency_pct", nbins=45, color_discrete_sequence=["#2A9D8F"],
                       title="Sleep efficiency", labels={"sleep_efficiency_pct": "Sleep efficiency (%)"})
    fig.add_vline(x=85, line_dash="dash", line_color="#D1495B", annotation_text="85%")
    c2.plotly_chart(fig)

    c1, c2 = st.columns(2)
    bt = df.assign(night=df.is_weekend.map({0: "Weekday", 1: "Weekend (Fri/Sat)"}))
    fig = px.histogram(bt, x="bedtime_hr", color="night", nbins=50, barmode="overlay", histnorm="percent",
                       title="Bedtime – weekday vs weekend", color_discrete_sequence=[PRIMARY, "#E9A23B"],
                       labels={"bedtime_hr": "Bedtime", "night": ""})
    ticks = list(range(19, 31))
    fig.update_xaxes(tickvals=ticks, ticktext=[clock(t) for t in ticks])
    c1.plotly_chart(fig)

    stages = (df.groupby("age_group", observed=True)[["deep_sleep_pct", "rem_sleep_pct", "light_sleep_pct"]]
              .mean().reset_index().melt(id_vars="age_group", var_name="stage", value_name="share"))
    stages["stage"] = stages.stage.map({"deep_sleep_pct": "Deep", "rem_sleep_pct": "REM", "light_sleep_pct": "Light"})
    fig = px.bar(stages, x="age_group", y="share", color="stage", text_auto=".1f",
                 title="Sleep stages by age group (%)", color_discrete_sequence=[PRIMARY, "#2A9D8F", "#E9A23B"],
                 labels={"age_group": "Age group", "share": "Share of sleep (%)", "stage": ""})
    c2.plotly_chart(fig)

    day = df.groupby("day_of_week").agg(sleep=("total_sleep_hr", "mean"),
                                        abnormal=("abnormality_flag", "mean")).reindex(DAYS).reset_index()
    fig = go.Figure()
    fig.add_bar(x=day.day_of_week, y=day.sleep, name="Avg sleep (h)", marker_color=PRIMARY)
    fig.add_scatter(x=day.day_of_week, y=day.abnormal * 100, name="Abnormal nights (%)", yaxis="y2",
                    mode="lines+markers", line_color="#D1495B")
    fig.update_layout(title="Sleep and abnormal-night rate by day of week",
                      yaxis=dict(title="Avg sleep (h)", range=[day.sleep.min() - 0.6, day.sleep.max() + 0.3]),
                      yaxis2=dict(title="Abnormal (%)", overlaying="y", side="right", rangemode="tozero"),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig)

# ---------------------------------------------------------------------------
# Abnormalities
# ---------------------------------------------------------------------------
with tab_abn:
    c1, c2 = st.columns(2)
    counts = df.abnormality_type.value_counts().reindex(PATTERN_ORDER).dropna().reset_index()
    counts.columns = ["pattern", "nights"]
    fig = px.bar(counts, x="pattern", y="nights", color="pattern", text="nights",
                 color_discrete_map=PATTERN_COLORS, title="Nights by sleep pattern")
    fig.update_layout(showlegend=False)
    c1.plotly_chart(fig)

    fig = px.scatter(df, x="breathing_events_per_hr", y="min_spo2_pct", color="abnormality_type",
                     color_discrete_map=PATTERN_COLORS, opacity=0.6, title="Blood oxygen vs breathing disturbances",
                     labels={"breathing_events_per_hr": "Breathing events / h", "min_spo2_pct": "Min SpO2 (%)",
                             "abnormality_type": "Pattern"})
    fig.add_vline(x=15, line_dash="dash", line_color=PRIMARY)
    fig.add_hline(y=90, line_dash="dash", line_color=PRIMARY)
    c2.plotly_chart(fig)

    st.subheader("Detection methods on the selected nights")
    truth = df.abnormality_flag
    res = pd.DataFrame({
        "Rule-based screening": binary_metrics(truth, (df.rule_prediction != "normal").astype(int)),
        "Isolation Forest (unsupervised)": binary_metrics(truth, df.if_flag),
    }).T[["accuracy", "precision", "recall", "f1"]]
    st.dataframe(res.style.format("{:.3f}"))
    st.caption("The Random Forest is evaluated on unseen participants in the EDA report "
               "(test-set F1 ≈ 0.88). Here it powers the 'Check a night' tab.")

    st.subheader("Most unusual nights (Isolation Forest score)")
    cols = ["record_id", "participant_id", "date", "total_sleep_hr", "sleep_latency_min", "sleep_efficiency_pct",
            "min_spo2_pct", "breathing_events_per_hr", "abnormality_type", "rule_prediction", "anomaly_score"]
    st.dataframe(df.nlargest(15, "anomaly_score")[cols], hide_index=True)

# ---------------------------------------------------------------------------
# Participant view
# ---------------------------------------------------------------------------
with tab_person:
    pid = st.selectbox("Participant", sorted(df.participant_id.unique()))
    p = df[df.participant_id == pid].sort_values("date")
    info = p.iloc[0]
    c = st.columns(5)
    c[0].metric("Age / gender", f"{info.age} · {info.gender}")
    c[1].metric("Usual bedtime", clock(p.bedtime_hr.median()))
    c[2].metric("Avg sleep", f"{p.total_sleep_hr.mean():.2f} h")
    c[3].metric("Avg efficiency", f"{p.sleep_efficiency_pct.mean():.1f}%")
    c[4].metric("Abnormal nights", f"{int(p.abnormality_flag.sum())} / {len(p)}")

    fig = px.line(p, x="date", y="total_sleep_hr", markers=True, title=f"{pid} – nightly sleep",
                  labels={"total_sleep_hr": "Total sleep (h)", "date": ""}, color_discrete_sequence=["#8C8C8C"])
    ab = p[p.abnormality_flag == 1]
    for pattern, g in ab.groupby("abnormality_type"):
        fig.add_scatter(x=g.date, y=g.total_sleep_hr, mode="markers", name=pattern,
                        marker=dict(size=12, color=PATTERN_COLORS[pattern], symbol="diamond"))
    fig.add_hrect(y0=7, y1=9, fillcolor="#2A9D8F", opacity=0.08, line_width=0)
    st.plotly_chart(fig)

    fig = px.line(p, x="date", y="bedtime_hr", markers=True, title=f"{pid} – bedtime",
                  labels={"bedtime_hr": "Bedtime", "date": ""}, color_discrete_sequence=[PRIMARY])
    lo, hi = int(p.bedtime_hr.min()), int(p.bedtime_hr.max()) + 2
    fig.update_yaxes(tickvals=list(range(lo, hi)), ticktext=[clock(t) for t in range(lo, hi)])
    st.plotly_chart(fig)

    if not ab.empty:
        st.markdown("**Abnormal nights**")
        st.dataframe(ab[["date", "bedtime", "wake_time", "total_sleep_hr", "sleep_latency_min",
                         "sleep_efficiency_pct", "min_spo2_pct", "abnormality_type", "rule_prediction"]],
                     hide_index=True)

# ---------------------------------------------------------------------------
# Check a night
# ---------------------------------------------------------------------------
with tab_check:
    st.markdown("Enter one night's measurements. The night is checked by the **rule-based screen** and the "
                "**Random Forest model** (trained on the full dataset).")
    with st.form("night"):
        c1, c2, c3 = st.columns(3)
        bed = c1.time_input("Bedtime", value=pd.Timestamp("2024-01-01 23:15").time())
        usual_bed = c1.time_input("Usual bedtime", value=pd.Timestamp("2024-01-01 23:15").time())
        latency = c1.number_input("Minutes to fall asleep", 1.0, 150.0, 15.0)
        total = c1.number_input("Total sleep (h)", 2.5, 13.0, 7.3, step=0.1)
        usual_sleep = c1.number_input("Usual sleep (h)", 4.0, 11.0, 7.4, step=0.1)
        awak = c2.number_input("Awakenings", 0, 20, 1)
        waso = c2.number_input("Minutes awake after falling asleep (WASO)", 0.0, 240.0, 10.0)
        deep = c2.number_input("Deep sleep %", 4.0, 32.0, 19.0)
        rem = c2.number_input("REM sleep %", 10.0, 30.0, 22.0)
        hr = c2.number_input("Avg heart rate (bpm)", 40.0, 100.0, 61.0)
        spo2 = c3.number_input("Minimum SpO2 %", 75.0, 100.0, 94.5)
        breath = c3.number_input("Breathing events / h", 0.0, 80.0, 3.0)
        caffeine = c3.number_input("Caffeine (mg)", 0, 600, 100)
        screen = c3.number_input("Screen time before bed (min)", 0, 300, 45)
        steps = c3.number_input("Daily steps", 0, 30000, 7000)
        c4, c5, c6 = st.columns(3)
        stress = c4.slider("Stress level", 1, 10, 4)
        age = c5.number_input("Age", 18, 90, 25)
        bmi = c6.number_input("BMI", 15.0, 45.0, 23.0)
        weekend = st.checkbox("Weekend night (Fri/Sat)")
        submitted = st.form_submit_button("Analyse night", type="primary")

    if submitted:
        def to_hr(t):
            h = t.hour + t.minute / 60
            return h if h >= 15 else h + 24

        tib = total + latency / 60 + waso / 60
        night = pd.DataFrame([{
            "bedtime_hr": to_hr(bed), "bedtime_dev_hr": to_hr(bed) - to_hr(usual_bed), "time_in_bed_hr": tib,
            "sleep_latency_min": latency, "total_sleep_hr": total, "sleep_dev_hr": total - usual_sleep,
            "sleep_efficiency_pct": total / tib * 100, "awakenings": awak, "waso_min": waso,
            "deep_sleep_pct": deep, "rem_sleep_pct": rem, "avg_heart_rate_bpm": hr, "min_spo2_pct": spo2,
            "breathing_events_per_hr": breath, "caffeine_mg": caffeine, "screen_time_min": screen,
            "daily_steps": steps, "stress_level": stress, "age": age, "bmi": bmi, "is_weekend": int(weekend),
        }])
        rule = rule_based_screen(night).iloc[0]
        model = train_model(data)
        probs = pd.Series(model.predict_proba(night[MODEL_FEATURES])[0], index=model.classes_).sort_values(ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Sleep efficiency", f"{night.sleep_efficiency_pct.iloc[0]:.1f}%")
        c2.metric("Rule-based screen", rule.replace("_", " "))
        c3.metric("Model prediction", probs.index[0].replace("_", " "), f"{probs.iloc[0]:.0%} confidence",
                  delta_color="off")
        prob_df = probs.rename_axis("pattern").reset_index(name="probability")
        fig = px.bar(prob_df, x="pattern", y="probability", color="pattern", color_discrete_map=PATTERN_COLORS,
                     labels={"pattern": "Pattern", "probability": "Probability"}, title="Model probabilities")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig)
        st.info("Educational tool built on synthetic data - not a medical diagnosis.")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
with tab_data:
    raw_cols = [c for c in pd.read_csv(DATASET, nrows=0).columns]
    st.dataframe(df[raw_cols], hide_index=True)
    st.download_button("Download filtered data (CSV)", df[raw_cols].to_csv(index=False).encode(),
                       "sleep_data_filtered.csv", "text/csv")
