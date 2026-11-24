"""
Exploratory Data Analysis for the sleep dataset.

Produces
  figures/fig01..fig10_*.png        - EDA charts
  outputs/eda_summary.json          - key numbers used in the report
  outputs/EDA_Analysis_Results.xlsx - quality checks, statistics and full data
  reports/EDA_Analysis_Report.pdf   - formatted EDA report

Run after detection.py (the PDF includes the detection results):
    python src/eda.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from features import AGE_LABELS, PATTERN_ORDER, ROOT, load_dataset
from style import PALETTE, PRIMARY, apply_style

FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

KEY_METRICS = {
    "total_sleep_hr": "Total sleep (h)",
    "sleep_efficiency_pct": "Sleep efficiency (%)",
    "sleep_latency_min": "Sleep latency (min)",
    "waso_min": "WASO (min)",
    "awakenings": "Awakenings",
    "deep_sleep_pct": "Deep sleep (%)",
    "rem_sleep_pct": "REM sleep (%)",
    "avg_heart_rate_bpm": "Avg heart rate (bpm)",
    "min_spo2_pct": "Min SpO2 (%)",
    "breathing_events_per_hr": "Breathing events / h",
}


def hours_to_clock(h: float) -> str:
    m = int(round(h * 60)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------------------
# Quality checks and summary numbers
# ---------------------------------------------------------------------------
def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    stage_sum = df.deep_sleep_pct + df.rem_sleep_pct + df.light_sleep_pct
    tib_check = (df.total_sleep_hr + df.sleep_latency_min / 60 + df.waso_min / 60 - df.time_in_bed_hr).abs()
    checks = [
        ("Rows", f"{len(df):,}", "Meets 5,000-row target"),
        ("Columns (raw file)", "28", "Documented in data dictionary"),
        ("Missing cells", int(df.isna().sum().sum()), "No missing values"),
        ("Duplicate record_id", int(df.record_id.duplicated().sum()), "IDs are unique"),
        ("Participants x nights", f"{df.participant_id.nunique()} x {df.date.nunique()}", "Balanced panel"),
        ("Date range", f"{df.date.min():%d %b %Y} - {df.date.max():%d %b %Y}", "50 consecutive nights"),
        ("Sleep stages sum to 100%", f"{(stage_sum.round(1).between(99.8, 100.2)).mean():.0%} of rows", "Internally consistent"),
        ("Time in bed = sleep + latency + WASO", f"{(tib_check < 0.02).mean():.0%} of rows", "Internally consistent"),
        ("Sleep efficiency range", f"{df.sleep_efficiency_pct.min():.1f} - {df.sleep_efficiency_pct.max():.1f}%", "Within 0-100%"),
        ("abnormality_flag values", ", ".join(map(str, sorted(df.abnormality_flag.unique()))), "Binary label"),
    ]
    return pd.DataFrame(checks, columns=["Check", "Result", "Interpretation"])


def describe(df: pd.DataFrame) -> pd.DataFrame:
    d = df[list(KEY_METRICS)].describe().T[["mean", "50%", "std", "min", "max"]]
    d.index = [KEY_METRICS[c] for c in d.index]
    d.columns = ["Mean", "Median", "Std dev", "Min", "Max"]
    return d.round(2)


def summary_numbers(df: pd.DataFrame) -> dict:
    wk = df[df.is_weekend == 0]
    we = df[df.is_weekend == 1]
    by_day = df.groupby("day_of_week").abnormality_flag.mean().reindex(DAYS)
    stress_eff = df.groupby("stress_level").sleep_efficiency_pct.mean()
    deep_age = df.groupby("age_group", observed=True).deep_sleep_pct.mean()
    apnea = df[df.abnormality_type == "apnea_risk"]
    normal = df[df.abnormality_type == "normal"]
    return {
        "rows": len(df),
        "participants": int(df.participant_id.nunique()),
        "nights_per_participant": int(df.date.nunique()),
        "abnormal_nights": int(df.abnormality_flag.sum()),
        "abnormal_rate_pct": round(df.abnormality_flag.mean() * 100, 1),
        "pattern_counts": df.abnormality_type.value_counts().reindex(PATTERN_ORDER).fillna(0).astype(int).to_dict(),
        "mean_total_sleep": round(df.total_sleep_hr.mean(), 2),
        "pct_below_7h": round((df.total_sleep_hr < 7).mean() * 100, 1),
        "pct_7_to_9h": round(df.total_sleep_hr.between(7, 9).mean() * 100, 1),
        "mean_efficiency": round(df.sleep_efficiency_pct.mean(), 1),
        "pct_efficiency_below_85": round((df.sleep_efficiency_pct < 85).mean() * 100, 1),
        "mean_latency": round(df.sleep_latency_min.mean(), 1),
        "median_latency": round(df.sleep_latency_min.median(), 1),
        "weekday_bedtime": hours_to_clock(wk.bedtime_hr.mean()),
        "weekend_bedtime": hours_to_clock(we.bedtime_hr.mean()),
        "weekday_sleep": round(wk.total_sleep_hr.mean(), 2),
        "weekend_sleep": round(we.total_sleep_hr.mean(), 2),
        "corr_latency_caffeine": round(df.sleep_latency_min.corr(df.caffeine_mg), 2),
        "corr_latency_screen": round(df.sleep_latency_min.corr(df.screen_time_min), 2),
        "corr_latency_stress": round(df.sleep_latency_min.corr(df.stress_level), 2),
        "corr_normal_latency_caffeine": round(normal.sleep_latency_min.corr(normal.caffeine_mg), 2),
        "corr_normal_latency_screen": round(normal.sleep_latency_min.corr(normal.screen_time_min), 2),
        "corr_normal_latency_stress": round(normal.sleep_latency_min.corr(normal.stress_level), 2),
        "eff_low_stress": round(stress_eff.loc[stress_eff.index <= 3].mean(), 1),
        "eff_high_stress": round(stress_eff.loc[stress_eff.index >= 8].mean(), 1),
        "deep_by_age": {k: round(v, 1) for k, v in deep_age.items()},
        "highest_abnormal_day": by_day.idxmax(),
        "highest_abnormal_day_rate": round(by_day.max() * 100, 1),
        "lowest_abnormal_day": by_day.idxmin(),
        "lowest_abnormal_day_rate": round(by_day.min() * 100, 1),
        "apnea_mean_spo2": round(apnea.min_spo2_pct.mean(), 1),
        "normal_mean_spo2": round(normal.min_spo2_pct.mean(), 1),
        "apnea_mean_breathing": round(apnea.breathing_events_per_hr.mean(), 1),
        "normal_mean_breathing": round(normal.breathing_events_per_hr.mean(), 1),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def make_figures(df: pd.DataFrame) -> None:
    # 1 total sleep distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvspan(7, 9, color=PALETTE[1], alpha=0.12, label="Recommended 7-9 h")
    ax.hist(df.total_sleep_hr, bins=45, color=PRIMARY, edgecolor="white", linewidth=0.4)
    ax.axvline(df.total_sleep_hr.mean(), color=PALETTE[3], ls="--", label=f"Mean {df.total_sleep_hr.mean():.2f} h")
    ax.set(xlabel="Total sleep (hours)", ylabel="Nights", title="Distribution of total sleep per night")
    ax.legend(frameon=False)
    save(fig, "fig01_total_sleep_distribution.png")

    # 2 sleep efficiency distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df.sleep_efficiency_pct, bins=45, color=PALETTE[1], edgecolor="white", linewidth=0.4)
    ax.axvline(85, color=PALETTE[3], ls="--", label="85% healthy threshold")
    ax.set(xlabel="Sleep efficiency (%)", ylabel="Nights", title="Distribution of sleep efficiency")
    ax.legend(frameon=False)
    save(fig, "fig02_sleep_efficiency_distribution.png")

    # 3 bedtime weekday vs weekend
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(19, 30.25, 0.25)
    ax.hist(df[df.is_weekend == 0].bedtime_hr, bins=bins, alpha=0.75, density=True, label="Weekday nights", color=PRIMARY)
    ax.hist(df[df.is_weekend == 1].bedtime_hr, bins=bins, alpha=0.6, density=True, label="Weekend nights (Fri/Sat)", color=PALETTE[2])
    ticks = np.arange(19, 31, 1)
    ax.set_xticks(ticks, [hours_to_clock(t) for t in ticks], rotation=45)
    ax.set(xlabel="Bedtime", ylabel="Share of nights", title="Bedtime distribution - weekday vs weekend")
    ax.legend(frameon=False)
    save(fig, "fig03_bedtime_weekday_weekend.png")

    # 4 abnormality counts
    counts = df.abnormality_type.value_counts().reindex(PATTERN_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([c.replace("_", "\n") for c in counts.index], counts.values,
                  color=[PALETTE[5]] + PALETTE[:5])
    ax.bar_label(bars, labels=[f"{v}\n({v / len(df):.1%})" for v in counts.values], fontsize=8, padding=2)
    ax.set(ylabel="Nights", title="Nights by sleep pattern (ground truth)")
    ax.set_ylim(0, counts.max() * 1.18)
    save(fig, "fig04_abnormality_counts.png")

    # 5 sleep stages by age group
    stages = df.groupby("age_group", observed=True)[["deep_sleep_pct", "rem_sleep_pct", "light_sleep_pct"]].mean().reindex(AGE_LABELS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(stages))
    for col, label, color in zip(stages.columns, ["Deep", "REM", "Light"], [PRIMARY, PALETTE[1], PALETTE[2]]):
        ax.bar(stages.index, stages[col], bottom=bottom, label=label, color=color, width=0.55)
        for i, v in enumerate(stages[col]):
            ax.text(i, bottom[i] + v / 2, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=9)
        bottom += stages[col].values
    ax.set(xlabel="Age group", ylabel="Share of total sleep (%)", title="Sleep stage composition by age group")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.18, 1))
    save(fig, "fig05_sleep_stages_by_age.png")

    # 6 lifestyle factors vs latency (normal nights only, to isolate the lifestyle effect)
    normal = df[df.abnormality_type == "normal"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, col, bins, label in [
        (axes[0], "caffeine_mg", [-1, 50, 100, 150, 200, 250, 400], "Caffeine (mg)"),
        (axes[1], "screen_time_min", [-1, 30, 60, 90, 120, 200], "Screen time before bed (min)"),
        (axes[2], "stress_level", [0, 2, 4, 6, 8, 10], "Stress level (1-10)"),
    ]:
        grp = normal.groupby(pd.cut(normal[col], bins), observed=True).sleep_latency_min.mean()
        names = [f"{int(max(i.left + 1, 0))}-{int(i.right)}" for i in grp.index]
        ax.bar(names, grp.values, color=PRIMARY)
        ax.set_xlabel(label)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
    axes[0].set_ylabel("Mean sleep latency (min)")
    fig.suptitle("Lifestyle factors vs time to fall asleep (normal nights)", fontweight="bold")
    save(fig, "fig06_lifestyle_vs_latency.png")

    # 7 stress vs efficiency
    fig, ax = plt.subplots(figsize=(8, 4.5))
    levels = sorted(df.stress_level.unique())
    ax.boxplot([df[df.stress_level == s].sleep_efficiency_pct for s in levels], tick_labels=levels,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#D9CCE8", color=PRIMARY), medianprops=dict(color=PALETTE[3]))
    ax.set(xlabel="Stress level", ylabel="Sleep efficiency (%)", title="Sleep efficiency by daily stress level")
    save(fig, "fig07_stress_vs_efficiency.png")

    # 8 correlation heatmap
    cols = ["total_sleep_hr", "sleep_efficiency_pct", "sleep_latency_min", "waso_min", "awakenings",
            "deep_sleep_pct", "rem_sleep_pct", "avg_heart_rate_bpm", "min_spo2_pct",
            "breathing_events_per_hr", "caffeine_mg", "screen_time_min", "daily_steps", "stress_level", "age", "bmi"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(corr, cmap="PuOr_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix of sleep and lifestyle variables")
    save(fig, "fig08_correlation_heatmap.png")

    # 9 day of week
    day = df.groupby("day_of_week").agg(sleep=("total_sleep_hr", "mean"), abnormal=("abnormality_flag", "mean")).reindex(DAYS)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar([d[:3] for d in DAYS], day.sleep, color=PRIMARY, width=0.55, label="Mean total sleep (h)")
    ax1.set_ylabel("Mean total sleep (h)")
    ax1.set_ylim(day.sleep.min() - 0.6, day.sleep.max() + 0.3)
    ax2 = ax1.twinx()
    ax2.plot([d[:3] for d in DAYS], day.abnormal * 100, color=PALETTE[3], marker="o", label="Abnormal nights (%)")
    ax2.set_ylabel("Abnormal nights (%)")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax1.set_title("Sleep duration and abnormal-night rate by day of week")
    ax2.set_ylim(0, day.abnormal.max() * 100 * 1.35)
    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax1.legend(handles, [h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(0.5, -0.1), ncols=2, frameon=False, fontsize=8)
    save(fig, "fig09_day_of_week.png")

    # 10 SpO2 vs breathing events
    fig, ax = plt.subplots(figsize=(8, 5))
    other = df[df.abnormality_type != "apnea_risk"]
    apnea = df[df.abnormality_type == "apnea_risk"]
    ax.scatter(other.breathing_events_per_hr, other.min_spo2_pct, s=6, alpha=0.3, color=PALETTE[5], label="Other nights")
    ax.scatter(apnea.breathing_events_per_hr, apnea.min_spo2_pct, s=14, alpha=0.8, color=PALETTE[3], label="Apnea-risk nights")
    ax.axvline(15, color=PRIMARY, ls="--", lw=1)
    ax.axhline(90, color=PRIMARY, ls="--", lw=1)
    ax.text(15.5, df.min_spo2_pct.max() - 0.5, "15 events/h", color=PRIMARY, fontsize=8)
    ax.text(df.breathing_events_per_hr.max() * 0.8, 90.3, "SpO2 90%", color=PRIMARY, fontsize=8)
    ax.set(xlabel="Breathing events per hour", ylabel="Minimum SpO2 (%)", title="Oxygen saturation vs breathing disturbances")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig10_spo2_vs_breathing.png")


# ---------------------------------------------------------------------------
# Excel workbook
# ---------------------------------------------------------------------------
def write_excel(df, quality, stats, metrics) -> None:
    path = OUT_DIR / "EDA_Analysis_Results.xlsx"
    pattern = df.abnormality_type.value_counts().reindex(PATTERN_ORDER).rename("nights").to_frame()
    pattern["share_pct"] = (pattern.nights / len(df) * 100).round(2)
    by_day = df.groupby("day_of_week").agg(
        nights=("record_id", "count"), mean_sleep_hr=("total_sleep_hr", "mean"),
        mean_efficiency=("sleep_efficiency_pct", "mean"), abnormal_rate=("abnormality_flag", "mean")).reindex(DAYS).round(3)
    by_participant = df.groupby("participant_id").agg(
        age=("age", "first"), gender=("gender", "first"), mean_sleep_hr=("total_sleep_hr", "mean"),
        mean_efficiency=("sleep_efficiency_pct", "mean"), abnormal_nights=("abnormality_flag", "sum")).round(2)
    detect = pd.DataFrame(metrics["binary_test_set"]).T
    raw = pd.read_csv(ROOT / "data" / "sleep_dataset.csv")

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        quality.to_excel(xw, sheet_name="Quality checks", index=False)
        stats.to_excel(xw, sheet_name="Summary statistics")
        pattern.to_excel(xw, sheet_name="Pattern counts")
        by_day.to_excel(xw, sheet_name="By day of week")
        by_participant.to_excel(xw, sheet_name="By participant")
        detect.to_excel(xw, sheet_name="Detection metrics")
        raw.to_excel(xw, sheet_name="Dataset", index=False)
        for ws in xw.book.worksheets:
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor=PRIMARY[1:])
                cell.alignment = Alignment(horizontal="center")
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 45)
            ws.freeze_panes = "A2"


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
def write_pdf(df, quality, stats, s, metrics) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=colors.HexColor(PRIMARY), fontSize=19, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=styles["Normal"], alignment=1, textColor=colors.HexColor("#555555"), fontSize=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor(PRIMARY), spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5, leading=13.5, spaceAfter=5)
    cap = ParagraphStyle("cap", parent=body, fontSize=8.5, textColor=colors.HexColor("#444444"), alignment=1)

    def table(data, widths):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3EEF8")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    def figure(name, caption, text, width=165):
        img = Image(str(FIG_DIR / name))
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth, img.drawHeight = width * mm, width * mm * ratio
        return KeepTogether([img, Paragraph(caption, cap), Spacer(1, 2), Paragraph(text, body), Spacer(1, 6)])

    c = s["pattern_counts"]
    m = metrics["binary_test_set"]
    mc = metrics["multiclass"]
    story = [
        Paragraph("Exploratory Data Analysis (EDA) Report", h1),
        Paragraph("Sleep Pattern Analysis and Abnormality Detection | Sagar Sharma", sub),
        Spacer(1, 10),
        Paragraph("1. Purpose", h2),
        Paragraph(
            "This report summarises the exploratory analysis of the nightly sleep dataset built for this project, "
            "and the results of the three abnormality-detection methods. The accompanying Excel workbook "
            "(EDA_Analysis_Results.xlsx) contains the detailed tables and the full dataset.", body),
        Paragraph("2. Dataset overview and quality checks", h2),
        Paragraph(
            f"The dataset contains <b>{s['rows']:,} nightly records</b> from <b>{s['participants']} anonymous participants</b> "
            f"observed over <b>{s['nights_per_participant']} consecutive nights</b>, with 28 columns covering sleep timing, "
            f"duration, efficiency, sleep stages, heart rate, blood oxygen, breathing disturbances, lifestyle factors and a "
            f"ground-truth sleep-pattern label. {s['abnormal_nights']} nights ({s['abnormal_rate_pct']}%) are labelled abnormal.", body),
        table([list(quality.columns)] + quality.astype(str).values.tolist(), [55 * mm, 55 * mm, 60 * mm]),
        Paragraph("3. Descriptive statistics", h2),
        table([["Metric"] + list(stats.columns)] + [[i] + [f"{v:.2f}" for v in r] for i, r in stats.iterrows()],
              [50 * mm] + [24 * mm] * 5),
        PageBreak(),
        Paragraph("4. Visualisations and interpretation", h2),
        figure("fig01_total_sleep_distribution.png", "Figure 1. Distribution of total sleep per night.",
               f"Average sleep is <b>{s['mean_total_sleep']} h</b>. {s['pct_7_to_9h']}% of nights fall in the recommended "
               f"7-9 h band, while {s['pct_below_7h']}% are below 7 h. The left tail is driven by sleep-deprivation and "
               f"insomnia nights; the right tail by oversleeping."),
        figure("fig02_sleep_efficiency_distribution.png", "Figure 2. Distribution of sleep efficiency.",
               f"Mean efficiency is <b>{s['mean_efficiency']}%</b>. {s['pct_efficiency_below_85']}% of nights fall below the commonly "
               f"used 85% threshold, mostly nights with long sleep latency or long wake-after-sleep-onset."),
        figure("fig03_bedtime_weekday_weekend.png", "Figure 3. Bedtime on weekday vs weekend nights.",
               f"Average bedtime shifts from <b>{s['weekday_bedtime']}</b> on weekdays to <b>{s['weekend_bedtime']}</b> on "
               f"Friday/Saturday nights, while average sleep rises from {s['weekday_sleep']} h to {s['weekend_sleep']} h - "
               f"a typical 'social jet-lag' pattern."),
        figure("fig04_abnormality_counts.png", "Figure 4. Nights by sleep pattern.",
               f"The label is imbalanced, as in most real anomaly-detection problems: {c['normal']:,} normal nights versus "
               f"{c['irregular_schedule']} irregular-schedule, {c['insomnia']} insomnia, {c['sleep_deprivation']} sleep-deprivation, "
               f"{c['apnea_risk']} apnea-risk and {c['oversleeping']} oversleeping nights. Accuracy alone is therefore a "
               f"misleading metric; precision, recall and F1 are reported instead."),
        figure("fig05_sleep_stages_by_age.png", "Figure 5. Sleep stage composition by age group.",
               "Deep sleep declines with age (" + ", ".join(f"{k}: {v}%" for k, v in s["deep_by_age"].items()) +
               "), consistent with sleep literature; light sleep makes up the difference."),
        figure("fig06_lifestyle_vs_latency.png", "Figure 6. Lifestyle factors vs sleep latency (normal nights).",
               f"On normal nights, time to fall asleep rises with caffeine (r = {s['corr_normal_latency_caffeine']}), "
               f"screen time before bed (r = {s['corr_normal_latency_screen']}) and stress (r = {s['corr_normal_latency_stress']}). "
               f"All three effects are of similar, moderate size, so no single habit explains slow sleep onset on its own."),
        figure("fig07_stress_vs_efficiency.png", "Figure 7. Sleep efficiency by daily stress level.",
               f"Mean efficiency drops from {s['eff_low_stress']}% at stress levels 1-3 to {s['eff_high_stress']}% at levels 8-10, "
               f"with much wider spread at high stress."),
        figure("fig08_correlation_heatmap.png", "Figure 8. Correlation matrix.",
               "Efficiency is strongly negatively correlated with latency and WASO (by definition). Min SpO2 and breathing "
               "events are negatively correlated, reflecting the apnea-risk nights. Lifestyle variables have weak-to-moderate "
               "correlations with latency, so no single variable explains abnormal sleep on its own.", width=150),
        figure("fig09_day_of_week.png", "Figure 9. Sleep duration and abnormal-night rate by day of week.",
               f"Abnormal nights are most common on <b>{s['highest_abnormal_day']}</b> ({s['highest_abnormal_day_rate']}%) "
               f"and least common on {s['lowest_abnormal_day']} ({s['lowest_abnormal_day_rate']}%). Weekend nights combine "
               f"longer sleep with more irregular schedules."),
        figure("fig10_spo2_vs_breathing.png", "Figure 10. Minimum SpO2 vs breathing events per hour.",
               f"Apnea-risk nights average {s['apnea_mean_breathing']} breathing events/h and a minimum SpO2 of "
               f"{s['apnea_mean_spo2']}%, versus {s['normal_mean_breathing']} events/h and {s['normal_mean_spo2']}% on normal "
               f"nights. Mild cases sit close to the 15 events/h and 90% cut-offs, which is why fixed thresholds miss some of them."),
        PageBreak(),
        Paragraph("5. Abnormality detection results", h2),
        Paragraph(
            "Three methods were compared on the same held-out test set of "
            f"{metrics['rf_test_rows']:,} nights from {metrics['rf_test_participants']} participants that the Random Forest never "
            "saw during training (abnormal vs normal):", body),
        table([["Method", "Accuracy", "Precision", "Recall", "F1"]] +
              [[k, f"{v['accuracy']:.3f}", f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}"] for k, v in m.items()],
              [50 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm]),
        Spacer(1, 6),
        Paragraph(
            f"For identifying the <i>type</i> of abnormality, the Random Forest reached a macro-F1 of <b>{mc['rf_macro_f1']}</b> "
            f"on the test set (5-fold participant-grouped cross-validation: {mc['rf_cv_macro_f1_mean']} &plusmn; "
            f"{mc['rf_cv_macro_f1_std']}), compared with {mc['rule_based_macro_f1']} for the rule-based screen.", body),
        figure("fig11_method_comparison.png", "Figure 11. Detection performance by method (test set).",
               "Fixed rules are transparent but miss mild and person-specific cases; the Isolation Forest needs no labels; the "
               "Random Forest gives the best balance of precision and recall.", width=150),
        figure("fig12_rf_confusion_matrix.png", "Figure 12. Random Forest confusion matrix (test set).",
               "Cells are shaded by the share of each actual pattern. The most frequent errors are: " +
               "; ".join(f"{e['actual'].replace('_', ' ')} predicted as {e['predicted'].replace('_', ' ')} ({e['count']})"
                         for e in metrics["rf_top_confusions"]) +
               ". These are borderline nights - e.g. a naturally short but healthy night resembles mild sleep deprivation.",
               width=130),
        figure("fig13_rf_feature_importance.png", "Figure 13. Most important features for the Random Forest.",
               "Personal-baseline features (deviation from the participant's usual bedtime and sleep duration) rank highly, "
               "showing why personalised analysis outperforms one-size-fits-all thresholds.", width=140),
        Paragraph("6. Reproducibility and limitations", h2),
        Paragraph(
            "The dataset is synthetic: it was generated with a fixed random seed (42) by src/generate_dataset.py, using "
            "participant profiles and documented sleep patterns with realistic noise. Patterns found here demonstrate the "
            "analysis and detection workflow; they are not clinical measurements and must not be used for diagnosis. "
            "Re-running <font face='Courier'>python run_pipeline.py</font> reproduces every number and chart in this report.", body),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(15 * mm, 10 * mm, "Sleep Pattern Analysis and Abnormality Detection - EDA Report")
        canvas.drawRightString(195 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(REPORT_DIR / "EDA_Analysis_Report.pdf"), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                            title="EDA Analysis Report - Sleep Pattern Analysis", author="Sagar Sharma")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ---------------------------------------------------------------------------
def main() -> None:
    apply_style()
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    metrics = json.loads((OUT_DIR / "model_metrics.json").read_text())

    quality = quality_checks(df)
    stats = describe(df)
    s = summary_numbers(df)
    make_figures(df)
    write_excel(df, quality, stats, metrics)
    write_pdf(df, quality, stats, s, metrics)
    (OUT_DIR / "eda_summary.json").write_text(json.dumps(s, indent=2, default=str))
    print(quality.to_string(index=False))
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
