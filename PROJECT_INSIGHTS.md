# PatrolIQ — Project Insights

This document summarizes what can be **verified directly from this repository
snapshot** — the populated MLflow tracking store (`mlruns.db`) and the
pipeline source code — rather than assumed figures. Where the underlying
crime-analysis outputs (EDA figures/summaries) aren't present in this
snapshot, that's called out explicitly instead of invented.

---

## 1. Experiment Tracking Summary

`mlruns.db` contains **3 experiments** and **6 finished runs**, all logged
via MLflow's SQLite backend:

| Experiment | Runs | Focus |
|---|---|---|
| `PatrolIQ_Clustering` | 3 | Geographic hotspot clustering + temporal pattern clustering |
| `PatrolIQ_Dimensionality` | 3 | PCA-based dimensionality reduction |
| `Default` | 0 | MLflow's built-in default experiment (unused) |

## 2. Clustering Results

Across the three clustering runs, the logged parameters and metrics show an
iterative tuning process rather than a single fixed run:

| Run | Geo model | Geo k | Geo silhouette | Geo Davies–Bouldin | Temporal model | Temporal k | Temporal silhouette | Temporal Davies–Bouldin |
|---|---|---|---|---|---|---|---|---|
| `clustering_pipeline` | — | k range 3–8, selected **k=3** | 0.433 | — | K-Means | **k=6** | 0.278 | — |
| `clustering_pipeline_1762155556` | — | best **k=3** | 0.433 | **0.854** | K-Means | best **k=6** | 0.278 | **1.034** |
| `clustering_run_1762157164` (final) | **Hierarchical** (chosen) | — | 0.414 | — | **K-Means (k=5)** (chosen) | — | 0.266 | — |

**What this tells us:**

- **Geographic clustering silhouette scores land around 0.41–0.43.** That's
  a *moderate* separation — clusters are recognizable but not sharply
  distinct, which is typical for geospatial crime data where hotspots blur
  into surrounding areas rather than forming isolated islands.
- **The Davies–Bouldin score for geo clustering (0.85) is a genuinely good
  result** — lower is better, and sub-1.0 indicates reasonably compact,
  well-separated clusters despite the moderate silhouette score.
- **Temporal clustering is weaker** (silhouette ≈ 0.27–0.28, Davies–Bouldin
  ≈ 1.03). This is expected: hour/weekday/month features are cyclical and
  don't separate as cleanly under Euclidean-distance clustering as spatial
  coordinates do. If temporal segmentation is a priority, consider
  cyclical encoding (sin/cos transforms of hour/weekday/month) before
  clustering, or a density-based method that doesn't assume convex
  clusters.
- **The final run switched the chosen geo model from K-Means to
  Hierarchical clustering**, and the chosen k for geo clustering appears to
  have settled at a small value (k=3 in the earlier runs). A K-Means value
  of k=3 is low for "9 hotspots" as advertised elsewhere in the project —
  worth confirming which run's parameters correspond to the numbers shown
  in the live dashboard, since the Home page hardcodes "9 Geographic
  Hotspots" as a static figure rather than reading it from `mlruns.db`.
- **Runtime cost is non-trivial at scale:** the logged `geo_kmeans_runtime_sec`
  (≈175s) and `temporal_kmeans_runtime_sec` (≈142s) on the 500K-row sample
  suggest clustering re-runs are expensive enough to warrant caching
  results rather than recomputing on every dashboard load.

## 3. Dimensionality Reduction Results

All three dimensionality-reduction runs converge on the same PCA result:

- **`pca_explained_variance` ≈ 0.827 (82.7%)** — comfortably above the
  70% target stated in the project's deliverables table, using an
  `pca_variance_threshold` param of 0.8.
- **`pca_reconstruction_error` ≈ 0.173**, consistent with ~82.7% variance
  retained (error ≈ 1 − explained variance).
- Runs were performed on **19 numeric features** with `fast_mode=True`,
  suggesting a reduced/sampled run rather than the full 500K-row dataset —
  worth re-running with `fast_mode=False` before reporting these numbers as
  final production metrics.

## 4. Feature Engineering Design Choices

From `feature_engineering.py`, several deliberate design decisions are
worth calling out as insights into the modeling approach:

- **A 5-point severity scale** is hand-mapped from the 31 Chicago Police
  primary crime types (homicide/sexual assault/kidnapping/trafficking = 5,
  down to non-criminal = 0). This is a reasonable heuristic but is
  editorial rather than data-derived — it should be validated against a
  domain expert or a documented public-safety severity standard before use
  in any resourcing decision.
- **A 12-cluster KMeans geo pre-clustering step runs inside feature
  engineering itself**, separate from the geographic clustering shown on
  the dashboard's Clustering Analysis page. Having two independent geo
  clusterings (one embedded as a feature, one as the headline analysis) is
  easy to conflate — they answer different questions and use different k
  values, so dashboard copy should be explicit about which is which.
- **Label encoders and the geo-clustering model are persisted via
  `joblib`**, which means the feature engineering step is designed to be
  reusable at inference time (e.g., scoring new incidents) rather than a
  one-off batch transform — a good sign for productionizing this beyond a
  static dashboard.

## 5. Data Quality Process

`validate_data.py` implements a genuinely thorough validation pass:
missing values, duplicates, dtype summary, invalid/future dates, year
range, geographic bounds specific to Chicago (lat 41.4–42.1°N, lon
-87.95 to -87.5°W), crime-category validity against the official 31-type
taxonomy, and arrest records missing a case number.

**However, the checked-in report (`reports/data_validation_report.txt`)
was generated against a 4-record, 1-column file whose sole column is named
`version https://git-lfs.github.com/spec/v1`** — the unmistakable signature
of an unresolved Git LFS pointer file rather than the actual crime dataset.
In other words: **the validation report in this repo does not reflect a
real run against the 500K-row dataset.** Anyone relying on this repo should
re-run `validate_data.py` against the real processed data before trusting
its output.

## 6. Gaps Between Documentation and Code

A few mismatches surfaced while cross-checking the original README against
the actual source:

- The README documents `src/models/` (`dimensionality_reduction.py`,
  `geo_clustering.py`, `temporal_clustering.py`, `utils.py`) as a reusable
  modeling layer. **This directory doesn't exist** — the equivalent logic
  is currently embedded directly in the Streamlit page files, which makes
  it harder to unit test or reuse outside the dashboard.
- `clean_data.py` outputs a single `sample_500000_rows.csv`, but
  `validate_data.py` and `feature_engineering.py` both expect two
  pre-split shards (`sample_250000_rows_01.csv` / `_02.csv`). Either an
  intermediate splitting step is missing from the repo, or the sampling
  script was changed after the downstream scripts were written.
- The Home page's headline metrics ("9 Geographic Hotspots," "4 Temporal
  Patterns") are **hardcoded in the UI**, not pulled from `mlruns.db` —
  they will drift out of sync with whatever the latest experiment run
  actually found (per §2, the logged runs suggest k=3 geo / k=5–6
  temporal, not 9/4).

## 7. Suggested Next Steps (Priority Order)

1. **Fix the data-shard mismatch** between `clean_data.py`'s single-file
   output and the two-shard input expected by `validate_data.py` /
   `feature_engineering.py`.
2. **Re-run the full pipeline against real data** and refresh
   `reports/data_validation_report.txt` — the current report is
   effectively a placeholder.
3. **Wire dashboard headline metrics to `mlruns.db`** instead of hardcoded
   numbers, so the UI reflects the latest experiment results automatically.
4. **Extract clustering/DR logic into `src/models/`** as originally
   documented, so it's testable independent of Streamlit.
5. **Improve temporal clustering** via cyclical feature encoding — the
   current ~0.27 silhouette score is the weakest metric in the project.
6. **Fill in `deployment/`** (currently three empty placeholder files) to
   make the "Cloud Deployment" step in the README actually reproducible.
