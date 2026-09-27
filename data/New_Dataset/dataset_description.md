# Login Behavior Dataset v2 — Documentation

Project: **AI-Based Login Authentication & Behavioral Anomaly Detection**
Generator: `generate_dataset.py` (fixed seed = 42, fully reproducible)

This is an **upgrade** of the earlier ~12,000-row / 100-user dataset. It is a
new, independent generator — not a re-sample or duplication of the old one —
built around 500 users with substantially more behavioral diversity, a
longer time span, and additional engineered features.

## Files produced

```
data/
├── login_behavior_dataset.csv   # unlabeled — feed this to the Isolation Forest
└── ground_truth.csv             # event_id, actual_status, anomaly_type — evaluation only
generate_dataset.py               # reproduces everything above (SEED = 42)
dataset_description.md            # this file
dataset_statistics.txt            # descriptive statistics (see below)
```

## Headline numbers (from the actual run)

- **Users:** 500
- **Events:** 112,131
- **Normal:** 103,856 (92.62%)
- **Anomalous:** 8,275 (7.38%) — within the requested 5–10% band
- **Time span:** 2023-12-30 to 2025-08-30 (~20 months), so random, user-level, *and* temporal train/test splits are all possible
- **Reproducibility:** re-running `python generate_dataset.py` produces byte-identical CSVs (verified via checksum across two runs)

Anomaly-type counts (primary buckets — an event counts toward a bucket if that cause is *one of* its contributing signals, so a multi-signal event counts in several buckets):

| Type | Count |
|---|---|
| unusual_hour | 1,306 |
| new_device | 1,306 |
| new_ip | 1,364 |
| new_location | 1,332 |
| high_failed_attempts | 1,332 |
| short_gap | 1,303 |
| high_frequency | 1,330 |
| multi_signal (3+ combined causes) | 2,489 |

Full statistics — events per user, failed-attempt distribution, login-hour distribution, device/browser distribution, session-duration and distance statistics, and all four frequency-window statistics — are in `dataset_statistics.txt`.

## Columns

### Base features (as specified)

| Column | Meaning |
|---|---|
| `event_id`, `user_id` | Identifiers. |
| `login_timestamp`, `login_hour`, `day_of_week` | When the login happened. |
| `failed_attempts` | Failed attempts before this login succeeded. |
| `time_since_last_login_minutes` | Minutes since this user's previous login (first login per user uses that user's own median gap, not an arbitrary constant). |
| `login_frequency_1h` / `_6h` / `_24h` / `_7d` | This user's own login count in each trailing window, including the current event. Four granularities instead of one lets the model see both bursty short-window behavior and longer-run frequency shifts separately. |
| `new_device`, `new_ip`, `new_location` | Whether the device/network/location was unfamiliar for this user. |
| `device_type`, `browser` | Laptop/Desktop/Mobile/Tablet and a browser correlated with both the device and the user's own habits. |
| `session_duration_minutes` | Session length. |
| `distance_from_usual_location_km` | Distance from this user's usual login location (never a real coordinate or place name). |

### New engineered features — and why each one earns its place

Every one of these maps directly to a requested anomaly category (temporal, device, or authentication) rather than being added for column count:

- **`usual_hour_deviation`** — how far (in hours, circular) this login's hour is from the user's *own running average hour*, scaled by their own running hour variability. Directly supports detecting the "unusual login hour" temporal anomaly *relative to that specific user*, rather than relative to a global average.
- **`usual_day_deviation`** — `1 − (this user's historical share of logins on this weekday)`. Supports the "unusual day" temporal anomaly.
- **`device_usage_frequency`** — what fraction of this user's *past* logins used this exact device type. A brand-new device naturally scores near 0; a rarely-used-but-not-new device scores low-but-nonzero, which is exactly the distinction between the `new_device` and `device_switch` anomaly types.
- **`previous_failed_attempt_rate`** — the average `failed_attempts` over this user's last up to 5 logins *before* the current one. This is what lets a model distinguish a "sudden increase" in failed attempts from a user who always has a few failed attempts.
- **`login_duration_deviation`** — how far this session's duration is from the user's own running mean, scaled by their own running variability. Supports the "session duration significantly outside normal behavior" anomaly.

**Important design point — no target leakage:** all five engineered features are computed from an **expanding, backward-only window of that user's own prior events** (a running mean/variance updated one event at a time, in chronological order). No feature ever uses the current event's own hidden label, any future event, or the ground-truth file. This mirrors exactly how a real detection system would compute "how usual is this?" from history alone. The dataset also deliberately contains **no** `risk_score`, `is_attack`, `is_suspicious`, or similar pre-computed indicator — the point of the exercise is for Isolation Forest to discover the anomalies from raw behavioral features, not to hand it the answer.

One consequence worth knowing: for a user's first ~1–5 events there isn't enough history yet, so these features fall back to neutral defaults (0.0 / 0.5 / 1.0, documented in the generator). This is normal "cold start" behavior, not a bug — the same limitation exists in any real running-statistics-based detector.

## Behavioral diversity (what changed from v1)

Instead of one archetype per user, each of the 500 users is built from **six independent dimensions**, so the population covers a very large combination space rather than a handful of clusters:

1. **Temporal pattern:** morning, afternoon, evening, night, night-shift, bimodal (twice-daily), or irregular (genuinely unpredictable hours)
2. **Weekly bias:** weekday-heavy, weekend-heavy, or balanced
3. **Frequency:** frequent, average, or infrequent (infrequent users log in roughly every 2–6 days on average, not daily)
4. **Device pattern:** single-device, multi-device (2–3, one dominant), or device-switcher (2–4 devices used fairly evenly — genuinely legitimate multi-device use)
5. **Network stability:** stable (rarely changes network) or occasional-changer (mobile/traveling users with a higher legitimate IP-change rate)
6. **Overall regularity:** stable (low variance across hour/distance/session) or irregular (higher variance across all of them, independent of the temporal-pattern dimension)

This gives 750+ possible combinations for only 500 users, so behavior is genuinely varied rather than 500 copies of ~10 templates. Every profile also carries small **background noise probabilities** (an occasional genuine failed attempt, a legitimate IP change, a legitimate new device) so normal behavior is never perfectly predictable — the point being that the model must learn that "unusual" isn't automatically "malicious."

## How anomalies were generated

For each user, 5–10% of their events (drawn per-user, so the overall rate lands in that band) are chosen as anomalous. Each anomalous event draws **k** anomaly "subtypes" from a pool of 15 (unusual hour, unusual day, short gap, high frequency, high failed attempts, sudden failed-attempt increase, new device, unusual device/browser combo, device switch, new IP, new location, large distance, impossible travel, dormant reactivation, session anomaly), with k sampled as:

- **40% — 1 subtype** (subtle, single-feature)
- **30% — 2 subtypes** (moderate)
- **20% — 3–4 subtypes** (multi-feature)
- **10% — 5–7 subtypes** (severe, multi-signal)

Some subtypes are only offered as candidates when they make physical sense for that specific event — e.g. `short_gap` and `impossible_travel` never apply to a user's very first event, and `dormant_reactivation` is only offered when that event already has a naturally large (45+ day) gap since the previous login for that user, rather than being artificially forced (which would have broken chronological ordering). `dormant_reactivation` and `impossible_travel` are intentionally rarer as a result — they only fire where the underlying schedule supports them, which is itself realistic (you can't have a "dormant account reactivation" pattern for a user who never goes dormant).

**Design simplification, stated plainly:** "repeated network changes across several events" and "repeated login attempts within minutes" are represented as *combinations* of `new_ip` + `short_gap` (+ `high_failed_attempts`/`high_frequency`) on a single event, rather than as a multi-event burst pattern. Modeling a literal multi-event burst would require session-level rather than per-event labels, which is out of scope for this per-event dataset — noted here so it isn't mistaken for an oversight.

`ground_truth.csv`'s `anomaly_type` column records the exact combination for each anomalous row (e.g. `new_ip+short_gap+sudden_failed_attempts_increase`), so you can filter to any granularity you like; the console summary and `dataset_statistics.txt` roll these up into the broader buckets shown above.

## Validation performed automatically by the generator

Every run checks and reports on:

- User count and event count
- Missing values (none allowed)
- Duplicate `event_id`s and duplicate full rows
- Anomaly percentage
- Feature ranges (`login_hour` in [0,23], no negative failed attempts/distances, positive session durations, all frequency windows ≥ 1 since a login always counts itself)
- Timestamp ordering (chronological per user)
- Logical device/browser combinations (every browser is valid for its device type)
- Logical anomaly-label consistency (every `NORMAL` row has `anomaly_type = NONE` and vice versa)

The actual run reported: **no missing values, no duplicate IDs/rows, all feature ranges valid, timestamps chronological per user, all device/browser combinations valid, ground truth consistent.**

## Train/test splitting — three options, with trade-offs

This dataset is built to support all three; none is "correct" in isolation, and a real project should try more than one:

### 1. Random event-level split
```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
```
- **Advantage:** simplest, maximizes use of the data, gives a stable estimate of average performance.
- **Limitation — leakage risk:** two events from the *same user*, sometimes minutes apart, can land on opposite sides of the split. Because several engineered features (and even `time_since_last_login_minutes`) are functions of a user's own history, the model can end up implicitly "recognizing" a user's pattern from training and testing on a near-duplicate slice of that same pattern, inflating apparent performance versus a genuinely unseen user or time period.

### 2. User-level split
```python
users = dataset["user_id"].unique()
train_users, test_users = train_test_split(users, test_size=0.2, random_state=42)
train = dataset[dataset["user_id"].isin(train_users)]
test = dataset[dataset["user_id"].isin(test_users)]
```
- **Advantage:** tests generalization to *unseen users* — closest to "will this work for a new employee/customer we've never modeled before." Avoids the leakage risk above entirely.
- **Limitation:** Isolation Forest here is unsupervised per-event, not literally a per-user profile model, so this split is most informative if you also engineer any per-user normalization consistently (e.g., always computing deviations relative to *that* user's own history, which this dataset already does). It also means the test set's anomaly rate can vary run-to-run depending on which users land in it, so check it after splitting rather than assuming it will exactly match the global 7.38%.

### 3. Temporal split
```python
dataset["login_timestamp"] = pd.to_datetime(dataset["login_timestamp"])
cutoff = dataset["login_timestamp"].quantile(0.8)
train = dataset[dataset["login_timestamp"] <= cutoff]
test = dataset[dataset["login_timestamp"] > cutoff]
```
- **Advantage:** most realistic for a deployed system — you always train on the past and evaluate on the future. Because the dataset spans ~20 months, this split is meaningful rather than trivial (there's enough runway for behavior/frequency to plausibly drift).
- **Limitation:** a user who only appears in the *later* period (or stops appearing after the cutoff) has little or no training history, so their early post-cutoff engineered features will legitimately look "cold-start" rather than reflecting a real behavior shift — inspect this before treating every temporal-split error as a model failure.

**Recommendation:** run all three and compare precision/recall/F1. If performance drops sharply from the random split to the user-level or temporal split, that gap *is* the finding — it tells you whether the model is learning genuinely generalizable behavioral patterns or partly memorizing per-user quirks it saw in training.

## Avoiding target leakage (summary)

- The main dataset file has **no label column at all** — `actual_status` and `anomaly_type` exist only in `ground_truth.csv`.
- No feature is a disguised risk score. Every engineered feature is a plain behavioral statistic (a deviation, a rate, a frequency) computed the same way for normal and anomalous events alike.
- Every engineered feature only ever looks at a user's own **past** events — never the current event's hidden label, never future events, never another user's data.
- `ground_truth.csv` should only ever be joined back in *after* generating predictions, purely for scoring.

## How to modify your existing Isolation Forest pipeline

Your current pipeline (built for the ~12,000-row v1 dataset) will need these adjustments to use v2:

1. **Read path:** the files now live under `data/` (`data/login_behavior_dataset.csv`, `data/ground_truth.csv`) instead of the project root.
2. **New numeric columns to include as features:** `login_frequency_1h`, `login_frequency_6h`, `login_frequency_7d` (v1 only had `login_frequency_24h`), plus `usual_hour_deviation`, `usual_day_deviation`, `device_usage_frequency`, `previous_failed_attempt_rate`, `login_duration_deviation`. Add all of these to whatever numeric feature list / `StandardScaler` (or similar) step your pipeline already applies to `distance_from_usual_location_km`, `session_duration_minutes`, etc. — they're plain floats/ints, no special handling required.
3. **Categorical encoding is unchanged in kind:** `device_type`, `browser`, `day_of_week` still one-hot (or similarly) encode exactly as before; there are simply more rows now.
4. **Boolean columns unchanged:** `new_device`, `new_ip`, `new_location` still cast to `{0,1}` the same way.
5. **`user_id` / `event_id` / `login_timestamp`:** continue to exclude these from the feature matrix itself (keep them alongside for joining predictions back to `ground_truth.csv` and for the user-level/temporal splits above).
6. **Contamination parameter:** if your `IsolationForest(contamination=...)` was tuned to v1's ~7.4% rate, it can stay as-is — v2 lands at essentially the same rate (7.38%) by design, so this parameter shouldn't need retuning on that basis alone.
7. **Nothing else needs to change** to run your existing model end-to-end on v2; the new columns are additive, not replacements.

## Loading the dataset

```python
import pandas as pd

df = pd.read_csv("data/login_behavior_dataset.csv", parse_dates=["login_timestamp"])
ground_truth = pd.read_csv("data/ground_truth.csv")

numeric_features = [
    "failed_attempts", "time_since_last_login_minutes",
    "login_frequency_1h", "login_frequency_6h", "login_frequency_24h", "login_frequency_7d",
    "session_duration_minutes", "distance_from_usual_location_km",
    "usual_hour_deviation", "usual_day_deviation",
    "device_usage_frequency", "previous_failed_attempt_rate", "login_duration_deviation",
]
boolean_features = ["new_device", "new_ip", "new_location"]
categorical_features = ["device_type", "browser", "day_of_week"]
```

## An important caveat, stated directly

This dataset is **larger and more behaviorally diverse** than the v1 dataset — that's what was asked for, and it's real. It is *not* automatically a better dataset for improving Isolation Forest's measured performance: more diversity can make some anomalies easier to isolate (more contrast against varied normal behavior) while making others harder (some rare-but-legitimate patterns — night-shift workers, device-switchers, occasional network changers — now look more like the subtle anomalies by design, which is the intended stress test). Whether it improves your model's precision/recall versus v1 is an empirical question this dataset is built to let you test, not a conclusion this documentation should assert in advance.
