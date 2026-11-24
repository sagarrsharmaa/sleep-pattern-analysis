"""
Abnormality detection: three approaches compared on the same data.

1. Rule-based screening   - transparent thresholds inspired by common sleep-medicine cut-offs.
2. Isolation Forest       - unsupervised anomaly detection (uses no labels).
3. Random Forest          - supervised classifier that predicts the abnormality type.
                            Evaluated on participants it has never seen (group split),
                            so it cannot memorise individual people.

Run:  python src/detection.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from features import MODEL_FEATURES, PATTERN_ORDER, ROOT, load_dataset
from style import PALETTE, PRIMARY, apply_style

FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
SEED = 42

# ---------------------------------------------------------------------------
# 1. Rule-based screening
# ---------------------------------------------------------------------------
RULES = {
    "apnea_risk": "breathing_events_per_hr >= 15 or min_spo2_pct < 90",
    "insomnia": "sleep_latency_min >= 30 or sleep_efficiency_pct < 80",
    "sleep_deprivation": "total_sleep_hr < 5",
    "oversleeping": "total_sleep_hr > 10",
    "irregular_schedule": "abs(bedtime_dev_hr) >= 2",
}


def rule_based_screen(df: pd.DataFrame) -> pd.Series:
    """Return the predicted pattern for every night. Rules are checked in priority order."""
    pred = pd.Series("normal", index=df.index)
    checks = {
        "apnea_risk": (df.breathing_events_per_hr >= 15) | (df.min_spo2_pct < 90),
        "insomnia": (df.sleep_latency_min >= 30) | (df.sleep_efficiency_pct < 80),
        "sleep_deprivation": df.total_sleep_hr < 5,
        "oversleeping": df.total_sleep_hr > 10,
        "irregular_schedule": df.bedtime_dev_hr.abs() >= 2,
    }
    # assign in reverse so the highest-priority rule wins
    for name in reversed(list(checks)):
        pred[checks[name]] = name
    return pred


# ---------------------------------------------------------------------------
# 2. Isolation Forest (unsupervised)
# ---------------------------------------------------------------------------
def isolation_forest_flags(df: pd.DataFrame, contamination: float = 0.13) -> tuple[pd.Series, pd.Series]:
    """Flag the most unusual nights. contamination = expected share of abnormal nights."""
    X = StandardScaler().fit_transform(df[MODEL_FEATURES])
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=SEED)
    flags = pd.Series((model.fit_predict(X) == -1).astype(int), index=df.index)
    scores = pd.Series(-model.score_samples(X), index=df.index)  # higher = more anomalous
    return flags, scores


# ---------------------------------------------------------------------------
# 3. Random Forest (supervised)
# ---------------------------------------------------------------------------
def make_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1
    )


def random_forest_eval(df: pd.DataFrame):
    X, y, groups = df[MODEL_FEATURES], df["abnormality_type"], df["participant_id"]

    # hold out 20% of participants as an unseen test set
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(split.split(X, y, groups))
    model = make_classifier().fit(X.iloc[train_idx], y.iloc[train_idx])
    y_test, y_pred = y.iloc[test_idx], model.predict(X.iloc[test_idx])

    # 5-fold cross-validation, again grouped by participant
    cv_scores = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_classifier().fit(X.iloc[tr], y.iloc[tr])
        cv_scores.append(f1_score(y.iloc[te], m.predict(X.iloc[te]), average="macro"))

    importances = pd.Series(model.feature_importances_, index=MODEL_FEATURES).sort_values(ascending=False)
    return model, y_test, pd.Series(y_pred, index=y_test.index), cv_scores, importances, len(set(groups.iloc[test_idx]))


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def binary_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def top_confusions(y_true: pd.Series, y_pred: pd.Series, n: int = 3) -> list:
    wrong = pd.DataFrame({"actual": y_true.values, "predicted": y_pred.values})
    wrong = wrong[wrong.actual != wrong.predicted]
    counts = wrong.value_counts().head(n)
    return [{"actual": a, "predicted": p, "count": int(c)} for (a, p), c in counts.items()]


def per_class_recall(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {c: round(float((y_pred[y_true == c] == c).mean()), 3) for c in PATTERN_ORDER if (y_true == c).any()}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_method_comparison(results: dict) -> None:
    methods = list(results)
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(metrics):
        vals = [results[k][m] for k in methods]
        bars = ax.bar(x + (i - 1) * 0.26, vals, 0.26, label=m.capitalize(), color=PALETTE[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (abnormal vs normal)")
    ax.set_title("Abnormality detection - method comparison")
    ax.legend(loc="upper left", ncols=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_method_comparison.png")
    plt.close(fig)


def plot_confusion(y_true, y_pred) -> None:
    labels = [c for c in PATTERN_ORDER if c in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_share = cm / cm.sum(axis=1, keepdims=True)  # shade by share of each actual class
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.imshow(row_share, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), [l.replace("_", "\n") for l in labels], fontsize=8)
    ax.set_yticks(range(len(labels)), [l.replace("_", " ") for l in labels], fontsize=8)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if row_share[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted pattern")
    ax.set_ylabel("Actual pattern")
    ax.set_title("Random Forest - confusion matrix (unseen participants)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_rf_confusion_matrix.png")
    plt.close(fig)


def plot_importance(importances: pd.Series) -> None:
    top = importances.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color=PRIMARY)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest - top 12 features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig13_rf_feature_importance.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    apply_style()
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    truth = df["abnormality_flag"]

    # Rules
    rule_pred = rule_based_screen(df)
    rule_flag = (rule_pred != "normal").astype(int)

    # Isolation Forest
    if_flag, if_score = isolation_forest_flags(df)

    # Random Forest
    rf_model, y_test, rf_pred, cv_scores, importances, n_test_people = random_forest_eval(df)
    rf_bin_true = (y_test != "normal").astype(int)
    rf_bin_pred = (rf_pred != "normal").astype(int)

    test_mask = df.index.isin(y_test.index)
    results = {
        "Rule-based": binary_metrics(truth, rule_flag),
        "Isolation Forest": binary_metrics(truth, if_flag),
        "Random Forest": binary_metrics(rf_bin_true, rf_bin_pred),
    }
    # same unseen test set for all three, for a like-for-like comparison
    results_test = {
        "Rule-based": binary_metrics(truth[test_mask], rule_flag[test_mask]),
        "Isolation Forest": binary_metrics(truth[test_mask], if_flag[test_mask]),
        "Random Forest": results["Random Forest"],
    }

    metrics = {
        "rules": RULES,
        "binary_full_dataset": {k: v for k, v in results.items() if k != "Random Forest"},
        "binary_test_set": results_test,
        "multiclass": {
            "rule_based_accuracy": round(accuracy_score(df.abnormality_type, rule_pred), 3),
            "rule_based_macro_f1": round(f1_score(df.abnormality_type, rule_pred, average="macro"), 3),
            "rf_accuracy": round(accuracy_score(y_test, rf_pred), 3),
            "rf_macro_f1": round(f1_score(y_test, rf_pred, average="macro"), 3),
            "rf_cv_macro_f1_mean": round(float(np.mean(cv_scores)), 3),
            "rf_cv_macro_f1_std": round(float(np.std(cv_scores)), 3),
        },
        "per_class_recall": {
            "rule_based": per_class_recall(df.abnormality_type, rule_pred),
            "random_forest": per_class_recall(y_test, rf_pred),
        },
        "rf_top_confusions": top_confusions(y_test, rf_pred),
        "rf_test_rows": int(len(y_test)),
        "rf_test_participants": int(n_test_people),
        "rf_top_features": {k: round(float(v), 4) for k, v in importances.head(10).items()},
        "isolation_forest_contamination": 0.13,
    }
    (OUT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    # per-night predictions for the dashboard / Excel workbook
    preds = df[["record_id", "participant_id", "date", "abnormality_type", "abnormality_flag"]].copy()
    preds["rule_prediction"] = rule_pred
    preds["isolation_forest_flag"] = if_flag
    preds["anomaly_score"] = if_score.round(4)
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)

    plot_method_comparison(results_test)
    plot_confusion(y_test, rf_pred)
    plot_importance(importances)

    print(json.dumps(metrics["binary_test_set"], indent=2))
    print(json.dumps(metrics["multiclass"], indent=2))
    print(json.dumps(metrics["per_class_recall"], indent=2))


if __name__ == "__main__":
    main()
