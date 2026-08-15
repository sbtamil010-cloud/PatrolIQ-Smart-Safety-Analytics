# 🚔 PatrolIQ — Smart Safety Analytics Platform

**Domain:** Public Safety & Urban Analytics
**Developer:** Yogesh Kumar V
**Tech Stack:** Python · Streamlit · scikit-learn · MLflow · Plotly · Folium · UMAP · HDBSCAN

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Repository Structure](#repository-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Component Details](#component-details)
- [Streamlit Application](#streamlit-application)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Getting Started](#getting-started)
- [Known Gaps in This Snapshot](#known-gaps-in-this-snapshot)
- [Roadmap](#roadmap)

---

## Overview

PatrolIQ is a **safety intelligence platform** that applies unsupervised machine
learning to large-scale urban crime data. It is built around the Chicago crime
dataset and is designed to help analysts answer three operational questions:

- 🧭 *Where should patrol resources be concentrated tonight?*
- 🕒 *When do most incidents occur?*
- 🏘️ *Which areas are highest-risk?*

The project is organized as a standard data pipeline (clean → validate →
engineer features → analyze/cluster → track experiments) feeding a multi-page
Streamlit dashboard.

## Problem Statement

Urban police departments and city agencies must make patrol and resourcing
decisions against millions of historical incident records. PatrolIQ's target
workflow samples **500,000 recent crime records** from the Chicago Police
Department's open dataset (source: [Chicago Data Portal](https://data.cityofchicago.org/)),
cleans and enriches them, and surfaces geographic and temporal patterns through
clustering, dimensionality reduction, and interactive visualizations.

## Repository Structure

This reflects what is **actually present** in this repository snapshot (see
[Known Gaps](#known-gaps-in-this-snapshot) for what the pipeline expects but
does not yet include):

```
PatrolIQ-Smart-Safety-Analytics-Platform-main/
│
├── READ.md                          # Original project README (superseded by this file)
├── requirements.txt                 # Python dependencies
├── mlflow_server.ps1                # PowerShell helper to launch a local MLflow server
├── mlruns.db                        # SQLite-backed MLflow tracking store (populated)
│
├── deployment/
│   ├── streamlit_cloud_config.toml  # placeholder — currently empty
│   ├── github_actions.yml           # placeholder — currently empty
│   └── performance_test.py          # placeholder — currently empty
│
├── reports/
│   ├── data_validation_report.txt   # Output of the last validate_data.py run
│   ├── data_validation_summary.json # JSON form of the same report
│   ├── figures/                     # Empty — populated by eda_pipeline.py at run time
│   └── summaries/                   # Empty — populated by eda_pipeline.py at run time
│
└── src/
    ├── data_preprocessing/
    │   ├── clean_data.py            # Loads raw CSV, cleans, samples 500K rows
    │   └── validate_data.py         # Runs data-quality checks, writes reports/
    │
    ├── analysis/
    │   ├── eda_pipeline.py          # Generates EDA figures/summaries (crime types,
    │   │                            #   geographic scatter/heatmap, temporal trends,
    │   │                            #   arrest/domestic correlations)
    │   └── feature_engineering.py   # Builds the model-ready feature table
    │
    └── app/
        ├── Home.py                  # Streamlit entry point (also present as
        │                            #   emoji-named 🏠_Home.py for Streamlit's
        │                            #   sidebar icon convention)
        └── pages/
            ├── 01_🎯_Clustering_Analysis.py
            ├── 02_⏰_Temporal_Analysis.py
            ├── 03_🔬_Dimensionality_Reduction.py
            ├── 04_📊_EDA_Insights.py
            ├── 05_🗺️_Geographic_Heatmaps.py
            └── 06_📈_MLflow_Monitoring.py
```

## Pipeline Architecture

```
Chicago Crime Dataset (raw CSV, e.g. Crimes_-_2001_to_Present_*.csv)
            │
            ▼
   clean_data.py            → outlier clipping (IQR), datetime parsing,
                               time-stratified 500K-row sample
            │
            ▼
   validate_data.py         → missing values, duplicates, date/geo bounds,
                               crime-category and arrest integrity checks
                               → reports/data_validation_report.txt (+ .json)
            │
            ▼
   feature_engineering.py   → temporal features (season, weekend, time-of-day),
                               severity scoring, label encoding, a 12-cluster
                               KMeans geo pre-clustering, lat/lon binning &
                               normalization → model_ready_data.csv
            │
            ▼
   eda_pipeline.py           → crime-type distributions, geographic scatter/
                                heatmap, hourly/weekday/monthly/seasonal trends,
                                arrest & domestic-incident correlation
            │
            ▼
   Clustering & dimensionality-reduction analysis (surfaced in the Streamlit
   pages: K-Means / DBSCAN / Hierarchical geo clustering, temporal K-Means,
   PCA / t-SNE / UMAP)
            │
            ▼
   MLflow tracking (mlruns.db)  → params, metrics, run comparison
            │
            ▼
   Streamlit multi-page dashboard (src/app)
```

## Component Details

### `src/data_preprocessing/clean_data.py`
- Reads the raw Chicago crime CSV from `data/raw/`.
- Drops nulls and duplicate rows.
- Clips numeric outliers using the IQR method.
- Parses `Date`, derives `Year`, `Month`, `Day`, `Weekday`, `Hour`, `Minute`,
  and restricts records to `Year >= 2010`.
- Builds a **time-stratified sample**: one record per
  `(Year, Month, Day, Hour)` group first, then fills the remainder randomly
  up to 500,000 rows (fixed `random_state=42` for reproducibility).
- Writes `data/processed/sample_500000_rows.csv`.

### `src/data_preprocessing/validate_data.py`
- Loads two processed shards (`sample_250000_rows_01.csv` /
  `_02.csv`) and concatenates them.
- Checks: missing values, duplicate rows, dtype summary, future/invalid
  dates, year range, out-of-bound Chicago coordinates
  (lat 41.4–42.1, lon -87.95 to -87.5), crime-category validity against a
  fixed list of 31 Chicago Police primary types, and arrests missing a case
  number.
- Writes both a human-readable `.txt` report and a machine-readable `.json`
  summary to `reports/`.

### `src/analysis/feature_engineering.py`
- Derives `Season`, `IsWeekend`, and a 4-bucket `TimeOfDay`.
- Maps each of the 31 crime categories to a 0–5 **severity score** (5 =
  homicide / sexual assault / kidnapping / trafficking, down to 0 for
  non-criminal).
- Label-encodes crime type, location description, season, and time-of-day
  (encoders persisted with `joblib` to `models/`).
- Fits a **12-cluster KMeans** on raw lat/lon as a coarse geo feature
  (`GeoCluster`), independent of the geographic clustering analysis shown on
  the Clustering Analysis page.
- Bins latitude/longitude into 20 quantile buckets and min-max normalizes
  coordinates.
- Emits a 19-column model-ready feature table
  (`data/processed/model_ready_data.csv`).

### `src/analysis/eda_pipeline.py`
- Crime-type distribution (bar chart + interactive Plotly chart).
- Geographic scatter plot and a Folium heatmap (both on a 50K-row sample for
  performance).
- Weekday × hour heatmap, hourly line chart, monthly trend by year, seasonal
  bar chart.
- Arrest-rate and domestic-incident rate by crime type.
- Descriptive statistics and a 1,000-row UI drill-down sample.
- All figures/CSVs are written to `reports/figures/` and
  `reports/summaries/` respectively.

## Streamlit Application

A six-page dashboard built on top of the artifacts above:

| Page | Purpose |
|---|---|
| **Home** | Landing page, headline metrics, feature overview, quick-start guide |
| **🎯 Clustering Analysis** | K-Means (hotspots), DBSCAN (density zones), Hierarchical clustering, a risk-level heatmap, and a side-by-side algorithm comparison with a metric guide |
| **⏰ Temporal Analysis** | Identified temporal clusters, hourly/daily/weekly/monthly/seasonal patterns, day×hour and month×hour heatmaps, peak "danger time" analysis for violent crime |
| **🔬 Dimensionality Reduction** | PCA scree plot & 2D projections, feature-importance ranking and PC loadings, t-SNE and UMAP projections, a method comparison (PCA vs t-SNE vs UMAP) and 3D visualizations |
| **📊 EDA Insights** | Crime-type distributions, geographic patterns, temporal patterns, arrest/domestic correlation, and full statistical summaries |
| **🗺️ Geographic Heatmaps** | Crime density heatmap, location scatter, K-Means cluster centers with hotspot coordinates |
| **📈 MLflow Monitoring** | Experiment overview, per-model performance summaries, cross-experiment comparison, and a metrics dashboard reading directly from `mlruns.db` |

## MLflow Experiment Tracking

`mlruns.db` is a populated SQLite MLflow backend with **3 experiments and 6
finished runs**:

- `PatrolIQ_Clustering` — geographic K-Means/Hierarchical hotspot selection
  and temporal K-Means pattern discovery.
- `PatrolIQ_Dimensionality` — PCA variance/reconstruction tracking.
- `Default` — MLflow's built-in default experiment (unused).

Start a local tracking UI with:

```bash
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

(equivalently, run `mlflow_server.ps1` on Windows/PowerShell), then open
`http://localhost:5000`.

See [PROJECT_INSIGHTS.md](./PROJECT_INSIGHTS.md) for the actual logged
metrics and what they indicate about model quality.

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/PatrolIQ.git
cd PatrolIQ
pip install -r requirements.txt
```

### 2. Provide the raw data

`clean_data.py` expects a raw Chicago crime CSV at:

```
data/raw/Crimes_-_2001_to_Present_<date>.csv
```

Download it from the [Chicago Data Portal](https://data.cityofchicago.org/)
(the "Crimes - 2001 to Present" dataset) and place it in `data/raw/`.

### 3. Run the pipeline

```bash
python src/data_preprocessing/clean_data.py
python src/data_preprocessing/validate_data.py
python src/analysis/feature_engineering.py
python src/analysis/eda_pipeline.py
```

### 4. (Optional) Start MLflow tracking

```bash
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlartifacts
```

### 5. Launch the dashboard

```bash
streamlit run src/app/Home.py
```

## Known Gaps in This Snapshot

An honest audit of this repository turned up a few things worth flagging
before you rely on it as-is:

- **No `data/` directory.** Raw and processed CSVs referenced throughout the
  pipeline (`data/raw/...`, `data/processed/sample_*.csv`,
  `model_ready_data.csv`) are not included — expected, since a 500K-row
  dataset shouldn't live in git, but it means the pipeline can't be run
  end-to-end without first sourcing the raw data.
- **No `models/` directory.** `feature_engineering.py` writes label
  encoders and a geo-clustering model here via `joblib`; it doesn't exist
  yet in this snapshot.
- **The `src/models/` clustering modules referenced in the original README
  (`dimensionality_reduction.py`, `geo_clustering.py`,
  `temporal_clustering.py`, `utils.py`) are not present in `src/`.** The
  clustering and dimensionality-reduction logic that powers the Streamlit
  pages currently lives only inside the page files themselves
  (`src/app/pages/01_...py`, `03_...py`) rather than as reusable modules.
- **`reports/figures/` and `reports/summaries/` are empty.** They're
  populated only after `eda_pipeline.py` is run against real data.
- **The checked-in `reports/data_validation_report.txt` reflects a Git
  LFS pointer file, not the real dataset** — it reports "4 records, 1
  column," which is the signature of an unresolved LFS pointer rather than
  actual crime data. Re-run `validate_data.py` after pulling real data.
- **`deployment/streamlit_cloud_config.toml`, `deployment/github_actions.yml`,
  and `deployment/performance_test.py` are all empty placeholder files** —
  CI/CD and cloud deployment are not yet wired up despite the folder
  existing.
- **`validate_data.py` and `feature_engineering.py` expect two data shards**
  (`sample_250000_rows_01.csv` / `_02.csv`) while `clean_data.py` produces a
  single `sample_500000_rows.csv`. These naming conventions should be
  reconciled so the pipeline runs without manual file renaming/splitting.
- Both `Home.py` and the emoji-named `🏠_Home.py` exist with (near)
  identical content — pick one as the canonical Streamlit entry point to
  avoid drift.

## Roadmap

- [ ] Reconcile the sample-file naming between `clean_data.py` and the
      downstream scripts.
- [ ] Extract clustering/dimensionality-reduction logic out of the
      Streamlit pages into reusable `src/models/` modules, as originally
      documented.
- [ ] Fill in `deployment/` (Streamlit Cloud config + GitHub Actions CI).
- [ ] Add automated tests around `clean_data.py`'s sampling logic and
      `validate_data.py`'s checks.
- [ ] Commit small sample fixtures (not the full 500K dataset) so the
      pipeline can be smoke-tested without the full raw download.

---

## Business Use Cases

| Stakeholder | Value |
|---|---|
| **Police departments** | Optimize patrol routes and shift resourcing around identified hotspots and peak hours |
| **City administrations** | Justify budget/infrastructure decisions with data-backed urban safety analysis |
| **Analytics firms** | Package crime-intelligence dashboards or predictive-policing tooling as a service |
| **Emergency response** | Prioritize dispatch and multi-agency coordination around known risk windows |

## License & Data Source

Crime data sourced from the City of Chicago's public
[Crimes - 2001 to Present](https://data.cityofchicago.org/) dataset. Review
the portal's terms of use before redistributing any derived data.
