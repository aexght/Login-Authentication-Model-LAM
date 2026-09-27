"""
generate_dataset.py
====================
Reproducible synthetic login-behavior dataset generator (v2 — upgraded scale
and diversity) for:

    AI-Based Login Authentication & Behavioral Anomaly Detection

Produces:
    data/login_behavior_dataset.csv   (unlabeled — for the Isolation Forest)
    data/ground_truth.csv             (event_id, actual_status, anomaly_type)
    dataset_statistics.txt            (descriptive statistics)

Everything is driven by a single fixed seed (SEED = 42). Running
`python generate_dataset.py` always reproduces byte-identical output.

No real people, IP addresses, emails, or GPS coordinates are used anywhere.
"Location" is represented only as a distance-from-usual-location metric;
"network"/"device" are represented only as familiarity flags plus a device
*category* — never an actual address, ID, or fingerprint.
"""

import os
import math
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

N_USERS = 500
START_DATE = datetime(2024, 1, 1)
SPAN_DAYS = 610          # ~20 months -> supports temporal train/test evaluation
MAX_SCHEDULE_DAYS = 610  # hard cap while scheduling logins

DEVICE_TYPES = ["Laptop", "Desktop", "Mobile", "Tablet"]
BROWSERS_BY_DEVICE = {
    "Laptop": ["Chrome", "Firefox", "Edge", "Safari"],
    "Desktop": ["Chrome", "Firefox", "Edge"],
    "Mobile": ["Chrome", "Safari", "Samsung Internet"],
    "Tablet": ["Safari", "Chrome"],
}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

OUT_DIR = "data"
EPS = 1e-6


# ---------------------------------------------------------------------------
# Small helper: incremental (Welford) mean/std so we never need to re-scan
# a user's whole history to compute a running statistic.
# ---------------------------------------------------------------------------
class RunningStats:
    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / self.n)


# ---------------------------------------------------------------------------
# Step 1: Build 500 distinct, multi-dimensional behavioral profiles.
#
# Profiles are built from *independent* dimensions (temporal pattern, weekly
# bias, frequency, device pattern, network stability, overall regularity)
# rather than a single fixed archetype label, so the 500 users cover a very
# large combination space instead of falling into a handful of clusters.
# ---------------------------------------------------------------------------
TEMPORAL_TYPES = ["morning", "afternoon", "evening", "night", "night_shift", "bimodal", "irregular"]
TEMPORAL_WEIGHTS = [0.20, 0.15, 0.20, 0.05, 0.05, 0.15, 0.20]

WEEKLY_TYPES = ["weekday_heavy", "weekend_heavy", "balanced"]
WEEKLY_WEIGHTS = [0.45, 0.15, 0.40]

FREQUENCY_TYPES = ["frequent", "average", "infrequent"]
FREQUENCY_WEIGHTS = [0.25, 0.50, 0.25]

DEVICE_PATTERNS = ["single_device", "multi_device", "device_switcher"]
DEVICE_PATTERN_WEIGHTS = [0.50, 0.35, 0.15]

NETWORK_STABILITY = ["stable", "occasional_changer"]
NETWORK_WEIGHTS = [0.70, 0.30]

REGULARITY_TYPES = ["stable", "irregular"]
REGULARITY_WEIGHTS = [0.60, 0.40]


def build_user_profiles(n_users):
    profiles = {}
    for i in range(1, n_users + 1):
        uid = f"USR{i:04d}"

        temporal_type = rng.choice(TEMPORAL_TYPES, p=TEMPORAL_WEIGHTS)
        weekly_type = rng.choice(WEEKLY_TYPES, p=WEEKLY_WEIGHTS)
        frequency_type = rng.choice(FREQUENCY_TYPES, p=FREQUENCY_WEIGHTS)
        device_pattern = rng.choice(DEVICE_PATTERNS, p=DEVICE_PATTERN_WEIGHTS)
        network_stability = rng.choice(NETWORK_STABILITY, p=NETWORK_WEIGHTS)
        regularity = rng.choice(REGULARITY_TYPES, p=REGULARITY_WEIGHTS)

        irregular_factor = rng.uniform(1.6, 2.3) if regularity == "irregular" else rng.uniform(0.7, 1.0)

        # --- hour pattern ---
        if temporal_type == "morning":
            hour_centers = [rng.uniform(6, 10)]
        elif temporal_type == "afternoon":
            hour_centers = [rng.uniform(12, 16)]
        elif temporal_type == "evening":
            hour_centers = [rng.uniform(17, 21)]
        elif temporal_type == "night":
            hour_centers = [rng.uniform(21, 25.5)]  # wraps past midnight
        elif temporal_type == "night_shift":
            hour_centers = [rng.uniform(1, 4.5)]
        elif temporal_type == "bimodal":
            hour_centers = [rng.uniform(6.5, 9), rng.uniform(18.5, 22)]
        else:  # irregular
            hour_centers = list(rng.uniform(0, 24, size=3))
        base_hour_std = rng.uniform(0.8, 1.8) if temporal_type != "irregular" else rng.uniform(4.5, 7.0)
        hour_std = base_hour_std * irregular_factor

        # --- weekly activity ---
        if weekly_type == "weekday_heavy":
            weekday_active_prob = rng.uniform(0.50, 0.85)
            weekend_active_prob = rng.uniform(0.05, 0.25)
        elif weekly_type == "weekend_heavy":
            weekday_active_prob = rng.uniform(0.15, 0.40)
            weekend_active_prob = rng.uniform(0.50, 0.85)
        else:
            weekday_active_prob = rng.uniform(0.35, 0.65)
            weekend_active_prob = rng.uniform(0.35, 0.65)

        # --- frequency / target event count over the whole window ---
        if frequency_type == "frequent":
            daily_rate = rng.uniform(1.8, 3.2)
            target_n = int(rng.integers(300, 460))
        elif frequency_type == "average":
            daily_rate = rng.uniform(0.8, 1.6)
            target_n = int(rng.integers(150, 270))
        else:  # infrequent
            daily_rate = rng.uniform(0.15, 0.5)
            target_n = int(rng.integers(40, 100))

        # --- devices ---
        if device_pattern == "single_device":
            n_devices = 1
        elif device_pattern == "multi_device":
            n_devices = int(rng.choice([2, 3], p=[0.7, 0.3]))
        else:  # device_switcher -> uses several devices fairly evenly
            n_devices = int(rng.choice([2, 3, 4], p=[0.4, 0.4, 0.2]))
        known_devices = list(rng.choice(DEVICE_TYPES, size=n_devices, replace=False))
        if device_pattern == "device_switcher":
            raw_w = rng.dirichlet(np.ones(n_devices) * 6)  # fairly even usage
        else:
            raw_w = rng.dirichlet(np.ones(n_devices) * 2.5)
            raw_w = np.sort(raw_w)[::-1]  # one dominant device for single/multi patterns
        device_weights = dict(zip(known_devices, raw_w))
        preferred_browser = {d: rng.choice(BROWSERS_BY_DEVICE[d]) for d in known_devices}

        # --- network stability ---
        if network_stability == "stable":
            ip_change_base_prob = rng.uniform(0.01, 0.05)
        else:
            ip_change_base_prob = rng.uniform(0.08, 0.22)

        # --- location / session baselines ---
        baseline_distance_std = rng.uniform(0.5, 8.0) * irregular_factor
        session_mean = rng.uniform(8, 45)
        session_std = rng.uniform(3, 12) * irregular_factor

        failed_attempt_base_prob = rng.uniform(0.02, 0.10)
        device_noise_prob = rng.uniform(0.01, 0.04)
        location_noise_prob = rng.uniform(0.01, 0.05)

        anomaly_rate = rng.uniform(0.05, 0.10)

        profiles[uid] = dict(
            temporal_type=temporal_type,
            weekly_type=weekly_type,
            frequency_type=frequency_type,
            device_pattern=device_pattern,
            network_stability=network_stability,
            regularity=regularity,
            hour_centers=hour_centers,
            hour_std=hour_std,
            weekday_active_prob=weekday_active_prob,
            weekend_active_prob=weekend_active_prob,
            daily_rate=daily_rate,
            target_n=target_n,
            known_devices=known_devices,
            device_weights=device_weights,
            preferred_browser=preferred_browser,
            ip_change_base_prob=ip_change_base_prob,
            baseline_distance_std=baseline_distance_std,
            session_mean=session_mean,
            session_std=session_std,
            failed_attempt_base_prob=failed_attempt_base_prob,
            device_noise_prob=device_noise_prob,
            location_noise_prob=location_noise_prob,
            anomaly_rate=anomaly_rate,
        )
    return profiles


def sample_hour(profile):
    center = rng.choice(profile["hour_centers"])
    h = rng.normal(center, profile["hour_std"])
    return int(round(h)) % 24


# ---------------------------------------------------------------------------
# Step 2: Schedule raw login timestamps for one user.
# ---------------------------------------------------------------------------
def schedule_timestamps(profile):
    target_n = profile["target_n"]
    timestamps = []
    day_offset = 0
    while len(timestamps) < target_n and day_offset < MAX_SCHEDULE_DAYS:
        current_date = START_DATE + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5
        active_prob = profile["weekend_active_prob"] if is_weekend else profile["weekday_active_prob"]
        if rng.random() < active_prob:
            n_logins_today = int(rng.poisson(profile["daily_rate"]))
            n_logins_today = max(1, min(n_logins_today, 6))
            for _ in range(n_logins_today):
                hour = sample_hour(profile)
                ts = current_date.replace(
                    hour=hour, minute=int(rng.integers(0, 60)), second=int(rng.integers(0, 60))
                )
                timestamps.append(ts)
                if len(timestamps) >= target_n:
                    break
        day_offset += 1

    while len(timestamps) < target_n:
        extra_day = START_DATE + timedelta(days=int(rng.integers(0, MAX_SCHEDULE_DAYS)))
        hour = sample_hour(profile)
        ts = extra_day.replace(hour=hour, minute=int(rng.integers(0, 60)), second=int(rng.integers(0, 60)))
        timestamps.append(ts)

    timestamps.sort()
    return timestamps


# ---------------------------------------------------------------------------
# Anomaly subtype catalogue
# ---------------------------------------------------------------------------
ALL_SUBTYPES = [
    "unusual_hour", "unusual_day", "short_gap", "high_frequency",
    "high_failed_attempts", "sudden_failed_attempts_increase",
    "new_device", "unusual_device_browser_combo", "device_switch",
    "new_ip", "new_location", "large_distance", "impossible_travel",
    "dormant_reactivation", "session_anomaly",
]
# subtypes that require a previous event in the sequence
NEEDS_PREVIOUS = {"short_gap", "impossible_travel", "dormant_reactivation"}
DORMANT_GAP_DAYS = 45  # a natural gap must already exceed this to be eligible


def eligible_pool(idx, natural_gap_days):
    pool = [s for s in ALL_SUBTYPES if s not in NEEDS_PREVIOUS]
    if idx > 0:
        pool += ["short_gap", "impossible_travel"]
        if natural_gap_days is not None and natural_gap_days > DORMANT_GAP_DAYS:
            pool.append("dormant_reactivation")
    return pool


def sample_k():
    """40% subtle (k=1), 30% moderate (k=2), 20% multi-feature (k=3-4), 10% severe (k=5-7)."""
    bucket = rng.choice(["subtle", "moderate", "multi", "severe"], p=[0.40, 0.30, 0.20, 0.10])
    if bucket == "subtle":
        return 1
    if bucket == "moderate":
        return 2
    if bucket == "multi":
        return int(rng.choice([3, 4], p=[0.6, 0.4]))
    return int(rng.choice([5, 6, 7], p=[0.5, 0.3, 0.2]))


# ---------------------------------------------------------------------------
# Step 3: Build every event for one user.
# ---------------------------------------------------------------------------
def generate_user_events(uid, profile, global_event_counter):
    target_n = profile["target_n"]
    timestamps = schedule_timestamps(profile)

    n_anom = max(1, int(round(target_n * profile["anomaly_rate"])))
    anomaly_indices = set(rng.choice(target_n, size=n_anom, replace=False).tolist())

    subtype_assignment = {}
    for idx in sorted(anomaly_indices):
        natural_gap_days = None
        if idx > 0:
            natural_gap_days = (timestamps[idx] - timestamps[idx - 1]).total_seconds() / 86400.0
        pool = eligible_pool(idx, natural_gap_days)
        k = min(sample_k(), len(pool))
        chosen = list(rng.choice(pool, size=k, replace=False))
        subtype_assignment[idx] = chosen

        if "unusual_hour" in chosen:
            centers = profile["hour_centers"]
            candidate_hour = int(rng.integers(0, 24))
            for _ in range(15):
                candidate_hour = int(rng.integers(0, 24))
                if all(4 < abs(candidate_hour - (c % 24)) < 20 for c in centers):
                    break
            timestamps[idx] = timestamps[idx].replace(hour=candidate_hour)

        if "unusual_day" in chosen:
            # nudge the date by 1-3 days to land on a weekday/weekend the user rarely uses,
            # while keeping the same time-of-day
            shift = int(rng.integers(1, 4)) * rng.choice([-1, 1])
            timestamps[idx] = timestamps[idx] + timedelta(days=int(shift))

        if ("short_gap" in chosen or "impossible_travel" in chosen) and idx > 0:
            gap_minutes = rng.uniform(1, 8)
            timestamps[idx] = timestamps[idx - 1] + timedelta(minutes=gap_minutes)

        if "dormant_reactivation" in chosen and idx > 0:
            pass  # timestamp already has a large natural gap; nothing to mutate

    # Re-sort while keeping each timestamp's original schedule index attached
    combined = sorted(zip(timestamps, range(target_n)), key=lambda x: x[0])
    sorted_ts = [c[0] for c in combined]
    order_orig_idx = [c[1] for c in combined]

    # --- gaps (minutes since previous login, in final chronological order) ---
    gaps = [None] * target_n
    for pos in range(1, target_n):
        gaps[pos] = (sorted_ts[pos] - sorted_ts[pos - 1]).total_seconds() / 60.0
    valid_gaps = [g for g in gaps if g is not None]
    gaps[0] = round(float(np.median(valid_gaps)), 1) if valid_gaps else 1440.0

    # --- sliding-window login-frequency counts (two-pointer, O(n) per window) ---
    def window_counts(hours):
        counts = [0] * target_n
        left = 0
        for pos in range(target_n):
            while sorted_ts[left] <= sorted_ts[pos] - timedelta(hours=hours):
                left += 1
            counts[pos] = pos - left + 1
        return counts

    freq_1h = window_counts(1)
    freq_6h = window_counts(6)
    freq_24h = window_counts(24)
    freq_7d = window_counts(24 * 7)

    # --- chronological feature-generation pass with running (backward-only) stats ---
    hour_stats = RunningStats()
    session_stats = RunningStats()
    weekday_counts = {d: 0 for d in DAY_NAMES}
    device_counts = {d: 0 for d in DEVICE_TYPES}
    last_failed = deque(maxlen=5)
    total_prior = 0

    rows = []
    for pos in range(target_n):
        orig_idx = order_orig_idx[pos]
        is_anom = orig_idx in anomaly_indices
        subtypes = subtype_assignment.get(orig_idx, [])
        ts = sorted_ts[pos]
        weekday_name = DAY_NAMES[ts.weekday()]

        # --- device / browser ---
        if "new_device" in subtypes:
            new_device_flag = True
            possible_new = [d for d in DEVICE_TYPES if d not in profile["known_devices"]]
            device_type = rng.choice(possible_new) if possible_new and rng.random() < 0.7 else rng.choice(DEVICE_TYPES)
            usual_browser = profile["preferred_browser"].get(device_type)
            options = [b for b in BROWSERS_BY_DEVICE[device_type] if b != usual_browser]
            browser = rng.choice(options) if options else rng.choice(BROWSERS_BY_DEVICE[device_type])
        elif "device_switch" in subtypes and len(profile["known_devices"]) > 1:
            new_device_flag = False
            device_type = rng.choice(profile["known_devices"])
            browser = profile["preferred_browser"][device_type]
        else:
            device_type = rng.choice(profile["known_devices"],
                                      p=[profile["device_weights"][d] for d in profile["known_devices"]])
            browser = profile["preferred_browser"][device_type]
            new_device_flag = rng.random() < profile["device_noise_prob"]

        if "unusual_device_browser_combo" in subtypes:
            mismatched = [b for b in BROWSERS_BY_DEVICE[device_type] if b != profile["preferred_browser"].get(device_type, browser)]
            if mismatched:
                browser = rng.choice(mismatched)

        # --- network ---
        if "new_ip" in subtypes or "impossible_travel" in subtypes or "dormant_reactivation" in subtypes:
            new_ip_flag = True
        else:
            new_ip_flag = rng.random() < profile["ip_change_base_prob"]

        # --- location / distance ---
        if "large_distance" in subtypes or "impossible_travel" in subtypes:
            distance = rng.uniform(300, 3000)
            new_location_flag = True
        elif "new_location" in subtypes:
            distance = rng.uniform(20, 150)
            new_location_flag = True
        else:
            distance = abs(rng.normal(0, profile["baseline_distance_std"]))
            new_location_flag = rng.random() < profile["location_noise_prob"]

        # --- failed attempts ---
        if "high_failed_attempts" in subtypes:
            failed_attempts = int(rng.integers(3, 7))
        elif "sudden_failed_attempts_increase" in subtypes:
            failed_attempts = int(rng.integers(2, 4))
        else:
            failed_attempts = 1 if rng.random() < profile["failed_attempt_base_prob"] else 0

        # --- session duration ---
        base_duration = abs(rng.normal(profile["session_mean"], profile["session_std"]))
        if "session_anomaly" in subtypes:
            session_duration = round(rng.uniform(0.2, 2.0), 1) if rng.random() < 0.6 else round(rng.uniform(120, 300), 1)
        elif is_anom and len(subtypes) >= 3 and rng.random() < 0.35:
            session_duration = round(rng.uniform(0.2, 2.5), 1)
        else:
            session_duration = round(max(0.3, base_duration), 1)

        # --- gap / frequency (natural from final timestamps; boosted for high_frequency) ---
        gap = gaps[pos]
        f1, f6, f24, f7d = freq_1h[pos], freq_6h[pos], freq_24h[pos], freq_7d[pos]
        if "high_frequency" in subtypes:
            f1 += int(rng.integers(2, 5))
            f6 += int(rng.integers(4, 9))
            f24 += int(rng.integers(8, 18))
            f7d += int(rng.integers(15, 35))

        # --- engineered features, computed ONLY from this user's PRIOR events ---
        if total_prior == 0:
            usual_hour_deviation = 0.0
            usual_day_deviation = 0.5
            device_usage_frequency = 1.0
            login_duration_deviation = 0.0
        else:
            hour_diff = abs(ts.hour - hour_stats.mean)
            hour_diff = min(hour_diff, 24 - hour_diff)  # circular distance
            # floors (not just +EPS) prevent a near-zero running std from blowing up
            # the z-score on small samples or unusually stable users
            hour_std_floor = max(hour_stats.std, 1.0)
            duration_std_floor = max(session_stats.std, 2.0)
            usual_hour_deviation = round(hour_diff / hour_std_floor, 3)
            usual_day_deviation = round(1 - (weekday_counts[weekday_name] / total_prior), 3)
            device_usage_frequency = round(device_counts.get(device_type, 0) / total_prior, 3)
            login_duration_deviation = round((session_duration - session_stats.mean) / duration_std_floor, 3)
        previous_failed_attempt_rate = round(sum(last_failed) / len(last_failed), 3) if last_failed else 0.0

        event_id = f"EVT{global_event_counter[0]:07d}"
        global_event_counter[0] += 1

        rows.append({
            "event_id": event_id,
            "user_id": uid,
            "login_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "login_hour": ts.hour,
            "day_of_week": weekday_name,
            "failed_attempts": failed_attempts,
            "time_since_last_login_minutes": round(gap, 1) if gap is not None else None,
            "login_frequency_1h": f1,
            "login_frequency_6h": f6,
            "login_frequency_24h": f24,
            "login_frequency_7d": f7d,
            "new_device": bool(new_device_flag),
            "new_ip": bool(new_ip_flag),
            "new_location": bool(new_location_flag),
            "device_type": device_type,
            "browser": browser,
            "session_duration_minutes": session_duration,
            "distance_from_usual_location_km": round(distance, 2),
            "usual_hour_deviation": usual_hour_deviation,
            "usual_day_deviation": usual_day_deviation,
            "device_usage_frequency": device_usage_frequency,
            "previous_failed_attempt_rate": previous_failed_attempt_rate,
            "login_duration_deviation": login_duration_deviation,
            "_actual_status": "ANOMALY" if is_anom else "NORMAL",
            "_anomaly_type": "+".join(sorted(subtypes)) if is_anom else "NONE",
        })

        # update running stats AFTER generating this event (backward-only for future events)
        hour_stats.update(ts.hour)
        session_stats.update(session_duration)
        weekday_counts[weekday_name] += 1
        device_counts[device_type] = device_counts.get(device_type, 0) + 1
        last_failed.append(failed_attempts)
        total_prior += 1

    return rows


# ---------------------------------------------------------------------------
# Step 4: Validation
# ---------------------------------------------------------------------------
def validate(dataset, ground_truth):
    issues = []

    if dataset["event_id"].duplicated().any():
        issues.append("Duplicate event_id values found in dataset.")
    if dataset.duplicated().any():
        issues.append("Fully duplicate rows found in dataset.")
    if dataset.isna().any().any():
        issues.append("Missing values found in dataset.")
    if not dataset["login_hour"].between(0, 23).all():
        issues.append("login_hour out of range [0,23].")
    if (dataset["failed_attempts"] < 0).any():
        issues.append("Negative failed_attempts found.")
    if (dataset["distance_from_usual_location_km"] < 0).any():
        issues.append("Negative distance found.")
    if (dataset["session_duration_minutes"] <= 0).any():
        issues.append("Non-positive session_duration_minutes found.")
    for f in ["login_frequency_1h", "login_frequency_6h", "login_frequency_24h", "login_frequency_7d"]:
        if (dataset[f] < 1).any():
            issues.append(f"{f} has values below 1 (a login always counts itself).")

    # logical device/browser combinations
    bad_combo = 0
    for device_type, browser in zip(dataset["device_type"], dataset["browser"]):
        if browser not in BROWSERS_BY_DEVICE.get(device_type, []):
            bad_combo += 1
    if bad_combo:
        issues.append(f"{bad_combo} rows have a browser not valid for their device_type.")

    # timestamp ordering per user
    bad_order_users = 0
    ts_parsed = pd.to_datetime(dataset["login_timestamp"])
    tmp = dataset.assign(_ts=ts_parsed).sort_values(["user_id", "_ts"])
    for uid, grp in tmp.groupby("user_id"):
        if not grp["_ts"].is_monotonic_increasing:
            bad_order_users += 1
    if bad_order_users:
        issues.append(f"{bad_order_users} users have a non-chronological event order after sorting.")

    # ground-truth logical consistency
    merged = dataset[["event_id"]].merge(ground_truth, on="event_id", how="left")
    if merged["actual_status"].isna().any():
        issues.append("Some events are missing a ground_truth row.")
    mismatch = ((merged["actual_status"] == "NORMAL") & (ground_truth.set_index("event_id").loc[merged["event_id"], "anomaly_type"].values != "NONE")).sum()
    if mismatch:
        issues.append(f"{mismatch} NORMAL rows have a non-NONE anomaly_type.")
    mismatch2 = ((merged["actual_status"] == "ANOMALY") & (ground_truth.set_index("event_id").loc[merged["event_id"], "anomaly_type"].values == "NONE")).sum()
    if mismatch2:
        issues.append(f"{mismatch2} ANOMALY rows have anomaly_type = NONE.")

    return issues


# ---------------------------------------------------------------------------
# Step 5: Statistics file
# ---------------------------------------------------------------------------
def write_statistics(dataset, ground_truth, path):
    merged = dataset.merge(ground_truth, on="event_id")
    lines = []
    lines.append("LOGIN BEHAVIOR DATASET — STATISTICAL SUMMARY")
    lines.append("=" * 50)
    lines.append(f"Total users: {dataset['user_id'].nunique()}")
    lines.append(f"Total events: {len(dataset)}")
    lines.append("")

    per_user = dataset.groupby("user_id").size()
    lines.append("Events per user:")
    lines.append(f"  mean:   {per_user.mean():.2f}")
    lines.append(f"  median: {per_user.median():.2f}")
    lines.append(f"  min:    {per_user.min()}")
    lines.append(f"  max:    {per_user.max()}")
    lines.append("")

    n_anom = (merged["actual_status"] == "ANOMALY").sum()
    lines.append(f"Anomaly rate: {n_anom} / {len(merged)} = {n_anom/len(merged)*100:.2f}%")
    lines.append("")

    lines.append("Failed-attempt distribution:")
    lines.append(dataset["failed_attempts"].value_counts().sort_index().to_string())
    lines.append("")

    lines.append("Login-hour distribution (0-23):")
    lines.append(dataset["login_hour"].value_counts().sort_index().to_string())
    lines.append("")

    lines.append("Device distribution:")
    lines.append(dataset["device_type"].value_counts().to_string())
    lines.append("")

    lines.append("Browser distribution:")
    lines.append(dataset["browser"].value_counts().to_string())
    lines.append("")

    lines.append("Session duration statistics (minutes):")
    lines.append(dataset["session_duration_minutes"].describe().to_string())
    lines.append("")

    lines.append("Distance-from-usual-location statistics (km):")
    lines.append(dataset["distance_from_usual_location_km"].describe().to_string())
    lines.append("")

    for f in ["login_frequency_1h", "login_frequency_6h", "login_frequency_24h", "login_frequency_7d"]:
        lines.append(f"{f} statistics:")
        lines.append(dataset[f].describe().to_string())
        lines.append("")

    lines.append("Anomaly subtype counts (single-cause anomalies only):")
    single_cause = merged[(merged["actual_status"] == "ANOMALY") & (~merged["anomaly_type"].str.contains(r"\+"))]
    lines.append(single_cause["anomaly_type"].value_counts().to_string())
    lines.append("")
    lines.append(f"Multi-signal anomalies (2+ combined causes): {(merged['anomaly_type'].str.contains(r'\\+')).sum()}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    profiles = build_user_profiles(N_USERS)
    global_event_counter = [1]
    all_rows = []
    for uid, profile in profiles.items():
        all_rows.extend(generate_user_events(uid, profile, global_event_counter))

    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle row order

    ground_truth = df[["event_id", "_actual_status", "_anomaly_type"]].rename(
        columns={"_actual_status": "actual_status", "_anomaly_type": "anomaly_type"}
    )
    dataset = df.drop(columns=["_actual_status", "_anomaly_type"])

    dataset_path = os.path.join(OUT_DIR, "login_behavior_dataset.csv")
    gt_path = os.path.join(OUT_DIR, "ground_truth.csv")
    dataset.to_csv(dataset_path, index=False)
    ground_truth.to_csv(gt_path, index=False)

    # ---- validation ----
    issues = validate(dataset, ground_truth)

    # ---- statistics file ----
    write_statistics(dataset, ground_truth, "dataset_statistics.txt")

    # ---- console summary ----
    merged = dataset.merge(ground_truth, on="event_id")
    n_total = len(dataset)
    n_anom = (merged["actual_status"] == "ANOMALY").sum()
    n_norm = n_total - n_anom

    print(f"Users: {dataset['user_id'].nunique()}")
    print(f"Events: {n_total}")
    print(f"Normal events: {n_norm} ({n_norm/n_total*100:.2f}%)")
    print(f"Anomalous events: {n_anom} ({n_anom/n_total*100:.2f}%)")
    print()
    print("Anomaly types (primary buckets):")
    key_types = ["unusual_hour", "new_device", "new_ip", "new_location",
                 "high_failed_attempts", "short_gap", "high_frequency"]
    anom_rows = merged[merged["actual_status"] == "ANOMALY"]
    for t in key_types:
        count = anom_rows["anomaly_type"].apply(lambda s: t in s.split("+")).sum()
        print(f"  {t}: {count}")
    multi_signal = (anom_rows["anomaly_type"].str.count(r"\+") >= 2).sum()  # 3+ combined subtypes
    print(f"  multi_signal (3+ combined causes): {multi_signal}")
    print()

    if issues:
        print("VALIDATION ISSUES FOUND:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("Validation passed: no missing values, no duplicate IDs/rows, "
              "all feature ranges valid, timestamps chronological per user, "
              "all device/browser combinations valid, ground truth consistent.")


if __name__ == "__main__":
    main()
