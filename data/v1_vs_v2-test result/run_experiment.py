"""
run_experiment.py — Controlled V1 vs V2 Isolation Forest experiment.

Trains and evaluates one Isolation Forest per (dataset, split) combination
using an identical, controlled methodology:

    datasets: V1 (12,123 events / 100 users), V2 (112,131 events / 500 users)
    splits:   random (event-level), user-level, temporal
    model:    IsolationForest(contamination=0.074, random_state=42, n_estimators=100)

Ground truth is used ONLY after predictions are generated, never during
preprocessing fitting, training, or feature/threshold selection.

Outputs (all written under this experiment's own results/ and models/
folders — nothing in the production app or its model file is touched):

    results/v1_vs_v2_results.csv
    results/v1_vs_v2_metrics.png
    results/{v1,v2}_{random,user,temporal}_confusion_matrix.png
    results/roc_curves.png
    results/pr_curves.png
    results/v2_anomaly_type_performance.csv
    results/V1_VS_V2_REPORT.md
    models/v1_isolation_forest.pkl
    models/v2_isolation_forest.pkl

Run with: python run_experiment.py
"""

import json
import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
CONTAMINATION = 0.074  # shared, controlled value for the PRIMARY comparison (see Section 7 of the brief)
TEST_SIZE = 0.20

DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature schemas
# ---------------------------------------------------------------------------
V1_NUMERIC = ["login_hour", "failed_attempts", "time_since_last_login_minutes",
              "login_frequency_24h", "session_duration_minutes", "distance_from_usual_location_km"]
V1_BOOLEAN = ["new_device", "new_ip", "new_location"]
V1_CATEGORICAL = ["device_type", "browser", "day_of_week"]

# V2 schema exactly as specified in the brief. Note login_hour is deliberately
# NOT included here (see README note in the report) -- it is superseded by
# usual_hour_deviation, which is a per-user-normalized version of the same signal.
V2_NUMERIC = ["failed_attempts", "time_since_last_login_minutes",
              "login_frequency_1h", "login_frequency_6h", "login_frequency_24h", "login_frequency_7d",
              "session_duration_minutes", "distance_from_usual_location_km",
              "usual_hour_deviation", "usual_day_deviation",
              "device_usage_frequency", "previous_failed_attempt_rate", "login_duration_deviation"]
V2_BOOLEAN = ["new_device", "new_ip", "new_location"]
V2_CATEGORICAL = ["device_type", "browser", "day_of_week"]

ANOMALY_SUBTYPES = [
    "unusual_hour", "unusual_day", "short_gap", "high_frequency",
    "high_failed_attempts", "sudden_failed_attempts_increase",
    "new_device", "unusual_device_browser_combo", "device_switch",
    "new_ip", "new_location", "large_distance", "impossible_travel",
    "dormant_reactivation", "session_anomaly",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(prefix):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}_login_behavior_dataset.csv"),
                      parse_dates=["login_timestamp"])
    gt = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}_ground_truth.csv"))
    return df, gt


# ---------------------------------------------------------------------------
# Splits — none of these look at actual_status / anomaly_type
# ---------------------------------------------------------------------------
def split_random(df):
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_user_level(df):
    users = df["user_id"].unique()
    train_users, test_users = train_test_split(users, test_size=TEST_SIZE, random_state=SEED)
    train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    test_df = df[df["user_id"].isin(test_users)].reset_index(drop=True)
    assert set(train_df["user_id"]).isdisjoint(set(test_df["user_id"])), "User leakage between train/test!"
    return train_df, test_df


def split_temporal(df):
    df_sorted = df.sort_values("login_timestamp").reset_index(drop=True)  # NOT shuffled
    cutoff = int(len(df_sorted) * (1 - TEST_SIZE))
    return df_sorted.iloc[:cutoff].reset_index(drop=True), df_sorted.iloc[cutoff:].reset_index(drop=True)


SPLIT_FUNCS = {"random": split_random, "user": split_user_level, "temporal": split_temporal}


# ---------------------------------------------------------------------------
# Preprocessing — fit ONLY on the training partition of each split
# ---------------------------------------------------------------------------
def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="passthrough",  # boolean columns (already 0/1 ints) pass through unchanged
    )


# ---------------------------------------------------------------------------
# Core experiment runner for one (dataset, split) combination
# ---------------------------------------------------------------------------
def run_one(dataset_name, df, gt, numeric_cols, boolean_cols, categorical_cols, split_type):
    train_df, test_df = SPLIT_FUNCS[split_type](df)

    # attach ground truth AFTER splitting, for evaluation only -- never used above this line
    gt_cols = ["event_id", "actual_status"] + (["anomaly_type"] if "anomaly_type" in gt.columns else [])
    train_df = train_df.merge(gt[gt_cols], on="event_id", how="left")
    test_df = test_df.merge(gt[gt_cols], on="event_id", how="left")
    if "anomaly_type" not in test_df.columns:
        test_df["anomaly_type"] = np.where(test_df["actual_status"] == "ANOMALY", "UNSPECIFIED", "NONE")

    feat_cols = numeric_cols + boolean_cols + categorical_cols
    X_train_raw = train_df[feat_cols].copy()
    X_test_raw = test_df[feat_cols].copy()
    for b in boolean_cols:
        X_train_raw[b] = X_train_raw[b].astype(int)
        X_test_raw[b] = X_test_raw[b].astype(int)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train = preprocessor.fit_transform(X_train_raw)  # fit ONLY on training data
    X_test = preprocessor.transform(X_test_raw)

    model = IsolationForest(contamination=CONTAMINATION, random_state=SEED, n_estimators=100)
    t0 = time.time()
    model.fit(X_train)  # UNSUPERVISED -- no labels passed in
    train_time = time.time() - t0

    t0 = time.time()
    raw_preds = model.predict(X_test)          # {-1, 1}, uses `contamination` as the decision threshold
    predict_time = time.time() - t0
    anomaly_scores = -model.decision_function(X_test)  # higher = more anomalous; threshold-independent

    y_pred = (raw_preds == -1).astype(int)
    y_true = (test_df["actual_status"] == "ANOMALY").astype(int).values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = dict(
        dataset=dataset_name,
        split_type=split_type,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, anomaly_scores),
        pr_auc=average_precision_score(y_true, anomaly_scores),
        true_negative=tn, false_positive=fp, false_negative=fn, true_positive=tp,
        train_time_sec=train_time, predict_time_sec=predict_time,
        n_train=len(train_df), n_test=len(test_df),
        n_features_encoded=X_train.shape[1], n_features_raw=len(feat_cols),
    )

    extra = dict(
        model=model, preprocessor=preprocessor,
        y_true=y_true, y_pred=y_pred, scores=anomaly_scores,
        test_df=test_df,
    )
    return metrics, extra


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm_counts, title, path):
    tn, fp, fn, tp = cm_counts
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_metrics_grid(results_df, path):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    splits = ["random", "user", "temporal"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    x = np.arange(len(metrics))
    width = 0.35
    for ax, split in zip(axes, splits):
        v1_vals = [results_df[(results_df.dataset == "V1") & (results_df.split_type == split)][m].values[0] for m in metrics]
        v2_vals = [results_df[(results_df.dataset == "V2") & (results_df.split_type == split)][m].values[0] for m in metrics]
        ax.bar(x - width / 2, v1_vals, width, label="V1", color="#6699CC")
        ax.bar(x + width / 2, v2_vals, width, label="V2", color="#CC6666")
        ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=35, ha="right")
        ax.set_title(f"{split} split")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[0].legend()
    fig.suptitle("V1 vs V2 — Isolation Forest metrics by split strategy (contamination=0.074, seed=42)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc_curves(all_extra, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for (dataset, split), extra in all_extra.items():
        fpr, tpr, _ = roc_curve(extra["y_true"], extra["scores"])
        auc = roc_auc_score(extra["y_true"], extra["scores"])
        ax.plot(fpr, tpr, label=f"{dataset} / {split} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — all experiments")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pr_curves(all_extra, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for (dataset, split), extra in all_extra.items():
        precision, recall, _ = precision_recall_curve(extra["y_true"], extra["scores"])
        ap = average_precision_score(extra["y_true"], extra["scores"])
        ax.plot(recall, precision, label=f"{dataset} / {split} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves — all experiments")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# V2-only anomaly-type breakdown (evaluation-only; uses the random-split model,
# since that split gives the largest, most representative test sample)
# ---------------------------------------------------------------------------
def anomaly_type_breakdown(test_df, y_pred):
    test_df = test_df.copy()
    test_df["predicted_anomaly"] = y_pred
    anomalies = test_df[test_df["actual_status"] == "ANOMALY"].copy()

    rows = []
    for subtype in ANOMALY_SUBTYPES:
        mask = anomalies["anomaly_type"].apply(lambda s: subtype in s.split("+"))
        n = mask.sum()
        if n == 0:
            continue
        recall = anomalies.loc[mask, "predicted_anomaly"].mean()
        rows.append({"category": subtype, "n_events": int(n), "detection_rate_recall": round(recall, 4)})

    anomalies["n_signals"] = anomalies["anomaly_type"].apply(lambda s: len(s.split("+")))
    for label, mask in [
        ("single-signal (k=1)", anomalies["n_signals"] == 1),
        ("exactly-2-signal (k=2)", anomalies["n_signals"] == 2),
        ("3+ signal (k=3-7)", anomalies["n_signals"] >= 3),
    ]:
        n = mask.sum()
        if n == 0:
            continue
        recall = anomalies.loc[mask, "predicted_anomaly"].mean()
        rows.append({"category": label, "n_events": int(n), "detection_rate_recall": round(recall, 4)})

    # overall, for reference
    rows.append({"category": "ALL anomalies (overall recall)", "n_events": int(len(anomalies)),
                  "detection_rate_recall": round(anomalies["predicted_anomaly"].mean(), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    v1_df, v1_gt = load_dataset("v1")
    v2_df, v2_gt = load_dataset("v2")

    datasets = {
        "V1": (v1_df, v1_gt, V1_NUMERIC, V1_BOOLEAN, V1_CATEGORICAL),
        "V2": (v2_df, v2_gt, V2_NUMERIC, V2_BOOLEAN, V2_CATEGORICAL),
    }

    all_results = []
    all_extra = {}

    for dataset_name, (df, gt, num_cols, bool_cols, cat_cols) in datasets.items():
        for split_type in ["random", "user", "temporal"]:
            print(f"Running {dataset_name} / {split_type} split ...")
            metrics, extra = run_one(dataset_name, df, gt, num_cols, bool_cols, cat_cols, split_type)
            all_results.append(metrics)
            all_extra[(dataset_name, split_type)] = extra

            cm_path = os.path.join(RESULTS_DIR, f"{dataset_name.lower()}_{split_type}_confusion_matrix.png")
            plot_confusion_matrix(
                (metrics["true_negative"], metrics["false_positive"], metrics["false_negative"], metrics["true_positive"]),
                f"{dataset_name} — {split_type} split", cm_path,
            )

    results_df = pd.DataFrame(all_results)
    ordered_cols = ["dataset", "split_type", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                    "true_negative", "false_positive", "false_negative", "true_positive",
                    "train_time_sec", "predict_time_sec", "n_train", "n_test",
                    "n_features_raw", "n_features_encoded"]
    results_df = results_df[ordered_cols]
    results_df.to_csv(os.path.join(RESULTS_DIR, "v1_vs_v2_results.csv"), index=False)

    plot_metrics_grid(results_df, os.path.join(RESULTS_DIR, "v1_vs_v2_metrics.png"))
    plot_roc_curves(all_extra, os.path.join(RESULTS_DIR, "roc_curves.png"))
    plot_pr_curves(all_extra, os.path.join(RESULTS_DIR, "pr_curves.png"))

    # V2 anomaly-type breakdown, using the random-split V2 model/test-set (largest, most representative sample)
    v2_random_extra = all_extra[("V2", "random")]
    breakdown_df = anomaly_type_breakdown(v2_random_extra["test_df"], v2_random_extra["y_pred"])
    breakdown_df.to_csv(os.path.join(RESULTS_DIR, "v2_anomaly_type_performance.csv"), index=False)

    # Persist the random-split model + preprocessor for each dataset (representative production-style artifact)
    for dataset_name in ["V1", "V2"]:
        extra = all_extra[(dataset_name, "random")]
        bundle = {"model": extra["model"], "preprocessor": extra["preprocessor"],
                  "contamination": CONTAMINATION, "seed": SEED, "split_type": "random"}
        pkl_path = os.path.join(MODELS_DIR, f"{dataset_name.lower()}_isolation_forest.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(bundle, f)

    model_sizes = {
        name: os.path.getsize(os.path.join(MODELS_DIR, f"{name.lower()}_isolation_forest.pkl"))
        for name in ["V1", "V2"]
    }

    print("\n=== RESULTS TABLE ===")
    print(results_df.to_string(index=False))
    print("\n=== V2 ANOMALY-TYPE BREAKDOWN (random split) ===")
    print(breakdown_df.to_string(index=False))
    print("\n=== MODEL FILE SIZES ===")
    for k, v in model_sizes.items():
        print(f"{k}: {v:,} bytes")

    # dump a small json summary for downstream report-writing
    summary = {"model_sizes_bytes": model_sizes, "contamination": CONTAMINATION, "seed": SEED}
    with open(os.path.join(RESULTS_DIR, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
