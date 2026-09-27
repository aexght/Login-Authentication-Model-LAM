# V1 vs V2 — Controlled Isolation Forest Experiment Report

**Project:** AI-Based Login Authentication & Behavioral Anomaly Detection
**Experiment location:** `experiments/v1_vs_v2/` (self-contained — production `app.py`, `auth.py`,
`database.py`, `feature_extractor.py`, templates, and `models/login_anomaly_model.pkl` were **not** touched)
**Script:** `run_experiment.py` (single command: `python run_experiment.py`)

## A note on scope, stated up front

This container does not have access to your local machine or to the existing
Flask project's actual source files (`app.py`, `feature_extractor.py`, etc.),
and `D:\LoginAuthenticationModel\...` is a path on your machine, not one this
environment can read. Two things follow from that:

1. **Datasets used:** this experiment uses the same V1 and V2 CSVs already generated and
   validated earlier in this conversation (12,123-event/100-user V1, and the
   audited 112,131-event/500-user V2), copied into this experiment's own
   `data/` folder. If your local V2 folder differs from the one already
   validated here, re-run `run_experiment.py` pointed at your actual files —
   the script only expects the four CSVs in `data/` and needs no other change.
2. **"Existing ML methodology":** since the actual pipeline code wasn't available here, "the
   existing approach" is implemented as the standard methodology this project's
   own documentation already specifies for both datasets: `IsolationForest`,
   `StandardScaler` on numeric features, `OneHotEncoder` on categoricals, booleans
   as 0/1. The same implementation is applied identically to V1 and V2 so the
   comparison is controlled. If your real `feature_extractor.py` differs in some
   way, that's a gap worth checking before trusting this report as final.

---

## 1. Dataset comparison

| | V1 | V2 |
|---|---|---|
| Users | 100 | 500 |
| Events | 12,123 | 112,131 |
| Anomaly rate | 7.36% (892 events) | 7.38% (8,275 events) |
| Anomaly subtypes | none tracked (binary only) | 15 subtypes, `anomaly_type` recorded per event |
| Multi-signal anomalies | not applicable | 4,935 (single 3,340 / exactly-2 2,446 / 3+ 2,489) |
| Time span | ~5 months (synthetic) | ~20 months (synthetic) |
| Behavioral profile dimensions | 1 archetype per user | 6 independent dimensions per user (750+ combinations) |
| Raw columns (excl. IDs/timestamp) | 12 | 19 (per the schema specified for this experiment) |

## 2. Feature comparison

| | V1 | V2 |
|---|---|---|
| Numeric | `login_hour`, `failed_attempts`, `time_since_last_login_minutes`, `login_frequency_24h`, `session_duration_minutes`, `distance_from_usual_location_km` (**6**) | `failed_attempts`, `time_since_last_login_minutes`, `login_frequency_1h/6h/24h/7d`, `session_duration_minutes`, `distance_from_usual_location_km`, `usual_hour_deviation`, `usual_day_deviation`, `device_usage_frequency`, `previous_failed_attempt_rate`, `login_duration_deviation` (**13**) |
| Boolean | `new_device`, `new_ip`, `new_location` (3) | same (3) |
| Categorical | `device_type`, `browser`, `day_of_week` (3) | same (3) |
| Raw feature count | 12 | 19 |
| Encoded feature count (after one-hot) | 25 | 32 |

**Documented, unavoidable difference:** V2's schema (as specified for this experiment) **excludes raw `login_hour`**, relying instead on `usual_hour_deviation` (a per-user-normalized version of the same signal). V1 has no such per-user deviation feature, so it keeps raw `login_hour` instead. This means the two models are not seeing "the same feature, more of it" for time-of-day — V1 sees an absolute hour, V2 sees a personalized deviation — which is a genuine methodological difference, not an oversight, and is worth keeping in mind when comparing results. V2 additionally has 3 more frequency granularities (1h/6h/7d) and 4 more engineered features with no V1 equivalent at all.

## 3. Training methodology (kept identical except where noted above)

- Model: `IsolationForest(contamination=0.074, random_state=42, n_estimators=100)` for **both** datasets, all splits.
- Preprocessing: `StandardScaler` (numeric) + `OneHotEncoder(handle_unknown="ignore")` (categorical) + passthrough (boolean, cast to int), assembled via `ColumnTransformer`.
- **Unsupervised throughout:** `model.fit(X_train)` is called with no `y` argument at all. `actual_status` / `anomaly_type` are joined onto the test set **only after** `.predict()` and `.decision_function()` have been called, purely for scoring. They are never available to the preprocessor, the model, or any threshold/feature-selection step.
- Preprocessing is fit **only** on each split's training partition and then applied to that split's test partition — never fit on the full dataset before splitting.
- `contamination=0.074` is fixed and identical for V1 and V2 in this primary comparison (V1's own measured rate is 7.36%, V2's is 7.38% — both round to ~0.074), per the instruction not to tune it separately for V2. No secondary tuned experiment was run in this pass.

## 4–6. Results by split strategy

Full numbers: `results/v1_vs_v2_results.csv`. Chart: `results/v1_vs_v2_metrics.png`.

### A. Random event-level split (`test_size=0.20`, `random_state=42`)

| Metric | V1 | V2 | Diff (V2−V1) |
|---|---|---|---|
| Accuracy | 0.9485 | 0.9402 | −0.0082 |
| Precision | 0.6042 | 0.5966 | −0.0076 |
| Recall | 0.7030 | 0.5905 | **−0.1126** |
| F1 | 0.6499 | 0.5935 | −0.0563 |
| ROC-AUC | 0.9177 | 0.8783 | −0.0395 |
| PR-AUC | 0.7248 | 0.6534 | −0.0715 |

Confusion matrices: `v1_random_confusion_matrix.png`, `v2_random_confusion_matrix.png`.

**Documented leakage risk (as required):** in this split, events from the same user can appear on both sides. Several V1/V2 features (`time_since_last_login_minutes`, all `login_frequency_*` windows, and — for V2 — every engineered deviation/rate feature) are direct functions of a user's own adjacent history. When a user's events are scattered across train and test, the model can partly "recognize" that user's pattern from training and be evaluated on a near-duplicate slice of the same pattern. Both V1 and V2 score highest on this split, which is consistent with (though doesn't by itself prove) that risk.

### B. User-level split (80% train users / 20% unseen test users, `random_state=42`)

| Metric | V1 | V2 | Diff (V2−V1) |
|---|---|---|---|
| Accuracy | 0.9514 | 0.9400 | −0.0114 |
| Precision | 0.7083 | 0.5941 | **−0.1142** |
| Recall | 0.5795 | 0.5628 | −0.0168 |
| F1 | 0.6375 | 0.5780 | −0.0595 |
| ROC-AUC | 0.8821 | 0.8663 | −0.0158 |
| PR-AUC | 0.7032 | 0.6347 | −0.0685 |

Confusion matrices: `v1_user_confusion_matrix.png`, `v2_user_confusion_matrix.png`. A hard assertion in the script (`assert ... isdisjoint(...)`) confirms zero user overlap between train and test for both datasets.

### C. Temporal split (sorted chronologically, first 80% of time = train, no shuffling)

| Metric | V1 | V2 | Diff (V2−V1) |
|---|---|---|---|
| Accuracy | 0.9365 | 0.9441 | **+0.0076** |
| Precision | 0.5792 | 0.6270 | **+0.0477** |
| Recall | 0.5792 | 0.5788 | −0.0005 |
| F1 | 0.5792 | 0.6019 | **+0.0227** |
| ROC-AUC | 0.9004 | 0.8740 | −0.0265 |
| PR-AUC | 0.6423 | 0.6593 | **+0.0170** |

Confusion matrices: `v1_temporal_confusion_matrix.png`, `v2_temporal_confusion_matrix.png`.

## 7. PR-AUC comparison (all splits)

PR-AUC (average precision) is the metric to weight most heavily here, since only ~7.4% of events are anomalous and ROC-AUC can look flattering under that imbalance.

| Split | V1 PR-AUC | V2 PR-AUC | Diff |
|---|---|---|---|
| Random | 0.7248 | 0.6534 | −0.0715 |
| User | 0.7032 | 0.6347 | −0.0685 |
| Temporal | 0.6423 | 0.6593 | +0.0170 |

V1 leads on PR-AUC in 2 of 3 splits; V2 leads narrowly on the temporal split. See `results/pr_curves.png` for the full curves (`roc_curves.png` for ROC).

## 8. Confusion matrices

All six are in `results/`: `{v1,v2}_{random,user,temporal}_confusion_matrix.png`. Raw counts are also in `v1_vs_v2_results.csv` (`true_negative`, `false_positive`, `false_negative`, `true_positive` columns).

## 9. V2 anomaly-type analysis (evaluation-only; `anomaly_type` never used in training)

Computed on the **random-split** V2 test set (22,427 test events — the largest of the three V2 test sets, and drawn independently of user or time, making it the most representative single slice for this breakdown). Full table: `results/v2_anomaly_type_performance.csv`.

| Category | Test events | Detection rate (recall) |
|---|---|---|
| high_frequency | 250 | **1.0000** |
| impossible_travel | 248 | **1.0000** |
| large_distance | 266 | 0.9699 |
| new_device | 246 | 0.9065 |
| new_location | 277 | 0.7762 |
| session_anomaly | 289 | 0.7128 |
| sudden_failed_attempts_increase | 294 | 0.6565 |
| new_ip | 277 | 0.6606 |
| high_failed_attempts | 244 | 0.6475 |
| short_gap | 243 | 0.6132 |
| unusual_device_browser_combo | 288 | 0.6215 |
| unusual_hour | 272 | 0.6066 |
| unusual_day | 250 | 0.5800 |
| device_switch | 264 | 0.5795 |
| **ALL anomalies (overall)** | 1,658 | **0.5905** |

By signal count:

| Group | Test events | Detection rate (recall) |
|---|---|---|
| Single-signal (k=1) | 675 | 0.3348 |
| Exactly-2-signal (k=2) | 480 | 0.6271 |
| 3+ signal (k=3–7) | 503 | 0.8986 |

**Interpretation:** detection rate scales almost monotonically with anomaly severity — this is exactly what Isolation Forest should do, and is a good sanity check that the model is learning genuine structure rather than noise. The weakest category is `device_switch` (0.58) — a switch to a device the user has genuinely used before, just rarely — which is by design the hardest, most "legitimate-looking" anomaly in the dataset. Single-signal anomalies are caught only about a third of the time; if your use case needs to catch subtle single-signal cases specifically, this result says the current feature set / contamination setting isn't there yet.

## 10. Training time and model size

| | V1 | V2 | Ratio (V2/V1) |
|---|---|---|---|
| Training samples (random split) | 9,698 | 89,704 | 9.25× |
| Training time | 0.163 s | 0.553 s | 3.4× |
| Prediction time (test set) | 0.018 s | 0.108 s | 5.9× |
| Encoded feature count | 25 | 32 | 1.28× |
| Model file size (.pkl) | 1,383,251 bytes | 1,202,444 bytes | 0.87× (V2 is *smaller*) |

**Why training time didn't scale 9×:** `IsolationForest`'s default `max_samples="auto"` subsamples only `min(256, n_samples)` rows per tree, regardless of how large the full training set is. Both V1 (9,698 rows) and V2 (89,704 rows) train each of the 100 trees on the same 256-row subsample size — the extra V2 data mostly adds a constant per-tree sampling overhead, not additional tree-building work, which is why the increase is sub-linear (~3.4×, not ~9×). The same reasoning explains why the two model files end up close in size (V2's is even slightly smaller): serialized tree structure size depends on tree depth/count, not on how much data it was drawn from.

## 11. Model size comparison

See table above — V2's `.pkl` (1.20 MB) is marginally *smaller* than V1's (1.38 MB) despite covering 500 users and 9× the events, for the reason just given. Practically: **the storage/inference cost of moving to V2 is negligible**; the real cost is the ~3–6× increase in training/prediction wall-clock time, which is still well under a second for a dataset this size.

## 12. Limitations

- **No access to the real production pipeline.** This experiment reimplements "the existing methodology" from this project's own prior documentation, not by reading your actual `feature_extractor.py`/`app.py`. If those differ (different scaler, different encoding, different contamination), re-validate against them before trusting this report for a go/no-go decision.
- **Synthetic data throughout.** Both V1 and V2 are synthetic; behavior "realism" reflects design choices made when generating them, not observed real users. Absolute metric values should not be read as predictions of real-world performance — only the *relative* V1-vs-V2 comparison, under matched methodology, is the intended signal here.
- **`contamination=0.074` was not tuned for either dataset.** It's a reasonable prior since it matches both datasets' true anomaly rate, but a deployment would normally tune this against a validation set — no such tuning was done here, by design, to keep the comparison controlled.
- **User-level split test-set anomaly rate isn't forced to 7.4%.** Because whole users move to the test set, the test set's actual anomaly rate can drift from the global rate depending on which users are drawn; this experiment did not rebalance for that.
- **V2 anomaly-type breakdown uses only the random-split model.** The other two splits' V2 models weren't separately broken down by anomaly type; if a specific split's behavior on a specific anomaly type matters for your decision, that analysis would need to be run explicitly.
- **Single run, no confidence intervals.** All numbers come from one fixed-seed run each. Given the imbalanced classes, especially the smaller V1 anomaly counts (~178–223 anomalies per test split), individual metrics (particularly precision/recall) can be somewhat sensitive to exactly which anomalous events land in the test set — treat small differences (a percentage point or two) as within noise, and larger gaps (5+ points) as more likely to be real.

## 13. Interpretation

**Across the two splits least susceptible to same-user leakage or time-based drift issues that favor a smaller/simpler dataset (random and user-level), V1 modestly outperforms V2 on every headline metric** — accuracy, precision, recall, F1, ROC-AUC, and PR-AUC. The gaps are generally in the 1–11 percentage-point range, largest on recall (random split) and precision (user split).

**On the temporal split, the picture flips partially:** V2 has higher accuracy, precision, F1, and PR-AUC, while V1 keeps a higher ROC-AUC and the two are essentially tied on recall. This is the one split where V2's larger, more temporally-spread dataset (~20 months vs ~5) plausibly helps — training on a longer history may give the model a better sense of "normal" drift over time, at some cost to overall ranking quality (ROC-AUC) even as its precision at the chosen contamination threshold improves.

**On the random-vs-other-splits gap (the explicitly requested check):** V1's ROC-AUC drops from 0.918 (random) to 0.882 (user, −3.6 points) and 0.900 (temporal, −1.7 points). V2's ROC-AUC is comparatively flatter: 0.878 (random) → 0.866 (user, −1.2 points) → 0.874 (temporal, −0.4 points). V1 shows a larger drop-off moving away from the random split than V2 does. This is consistent with — though doesn't conclusively prove — mild same-user leakage inflating V1's random-split number somewhat more than V2's; it may also simply reflect V1 having 5× fewer users, so each held-out user represents a larger, more idiosyncratic share of the test set. Both explanations point the same direction: **the random-split numbers for both datasets, and especially V1's, are probably the most optimistic and least representative of real deployment performance**, where new users and future time periods are the norm.

**Bottom line, stated directly per the instructions:** this experiment does **not** support a claim that the larger, more diverse V2 dataset improves Isolation Forest performance. Under an identical, controlled methodology, V2 performs somewhat worse on 2 of 3 splits and only marginally better (with a mixed metric picture) on the third. If diversity were expected to help, this is evidence against that specific expectation for this feature set and contamination setting — worth investigating further (e.g., per-user feature normalization, a tuned contamination, or feature-importance analysis on why `device_switch` and single-signal anomalies are hardest) rather than a reason to adopt V2 as-is.

---

## Files produced

```
experiments/v1_vs_v2/
├── run_experiment.py
├── data/                                  (copies of the audited V1 & V2 CSVs used)
├── models/
│   ├── v1_isolation_forest.pkl            (random-split model + preprocessor bundle)
│   └── v2_isolation_forest.pkl            (random-split model + preprocessor bundle)
└── results/
    ├── v1_vs_v2_results.csv
    ├── v1_vs_v2_metrics.png
    ├── v1_random_confusion_matrix.png
    ├── v2_random_confusion_matrix.png
    ├── v1_user_confusion_matrix.png
    ├── v2_user_confusion_matrix.png
    ├── v1_temporal_confusion_matrix.png
    ├── v2_temporal_confusion_matrix.png
    ├── roc_curves.png
    ├── pr_curves.png
    ├── v2_anomaly_type_performance.csv
    └── run_summary.json
```

No files under the production application or `models/login_anomaly_model.pkl` were created, modified, or replaced.
