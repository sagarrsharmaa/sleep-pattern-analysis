# 🌙 Sleep Pattern Analysis and Abnormality Detection

Analyse nightly sleep data, uncover patterns (duration, timing, efficiency, sleep stages, lifestyle effects) and automatically detect abnormal nights such as **insomnia**, **sleep deprivation**, **irregular schedules**, **sleep-apnea risk** and **oversleeping**.

**Author:** Sagar Sharma · HCLTech Career Shaper Programme – III Year Industry Project (2025–26)

**Live dashboard:** https://sleep-pattern-analysis-sagar.streamlit.app

---

## What the project does

| Stage | What happens | Code |
|---|---|---|
| 1. Dataset | Builds a reproducible 5,000-night dataset (100 participants × 50 nights, 28 columns) | `src/generate_dataset.py` |
| 2. EDA | Data-quality checks, descriptive statistics and 10 charts | `src/eda.py` |
| 3. Detection | Compares rule-based screening, Isolation Forest and Random Forest | `src/detection.py` |
| 4. Reporting | Generates the EDA report (PDF) and an Excel workbook of results | `src/eda.py` |
| 5. Dashboard | Interactive Streamlit app with filters, participant drill-down and a "check a night" tool | `app.py` |

## Results (held-out test set: 1,000 nights from 20 unseen participants)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Rule-based screening | 0.667 | 0.694 | 0.680 |
| Isolation Forest (unsupervised) | 0.766 | 0.660 | 0.709 |
| **Random Forest (supervised)** | **0.847** | **0.924** | **0.884** |

Identifying the *type* of abnormality: Random Forest macro-F1 **0.884** (5-fold participant-grouped CV: 0.80 ± 0.06) vs 0.639 for fixed rules.
The most important features are **personal-baseline deviations** (tonight's sleep and bedtime compared with the person's own usual values), which is why a personalised model beats one-size-fits-all thresholds.

## Project structure

```
sleep-pattern-analysis/
├── app.py                    # Streamlit dashboard
├── run_pipeline.py           # runs dataset -> detection -> EDA in one go
├── requirements.txt
├── data/
│   ├── sleep_dataset.csv     # 5,000 nightly records
│   └── data_dictionary.csv   # column definitions
├── src/
│   ├── generate_dataset.py   # synthetic dataset generator (seeded)
│   ├── features.py           # loading + feature engineering
│   ├── detection.py          # rules, Isolation Forest, Random Forest + evaluation
│   ├── eda.py                # charts, Excel workbook, PDF report
│   └── style.py              # chart styling
├── figures/                  # all generated charts
├── outputs/                  # metrics (JSON), predictions, Excel workbook
└── reports/
    └── EDA_Analysis_Report.pdf
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py             # regenerate data, metrics, charts and the PDF report
streamlit run app.py               # open the dashboard at http://localhost:8501
```

## Dataset

The dataset is **synthetic** and fully reproducible (random seed 42). Each participant has a stable profile (age, BMI, chronotype, sleep need, habits, risk tendencies). Each night is generated either as a normal night or as one of five abnormal patterns, with a random severity so that mild cases overlap with normal nights, as they do in real data. The generating pattern is stored as the ground-truth label (`abnormality_type`, `abnormality_flag`). See `data/data_dictionary.csv` for all 28 columns.

> ⚠️ Educational project built on synthetic data. It is **not** a medical device and must not be used for diagnosis.

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Plotly · Streamlit · ReportLab · openpyxl
