import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

V1_DATA_PATH = r"D:\LoginAuthenticationModel\data\login_behavior_dataset.csv"
V1_LABEL_PATH = r"D:\LoginAuthenticationModel\data\ground_truth.csv"

V2_DATA_PATH = r"D:\LoginAuthenticationModel\data\New_Dataset\login_behavior_dataset.csv"
V2_LABEL_PATH = r"D:\LoginAuthenticationModel\data\New_Dataset\ground_truth.csv"

OUT_DIR = r"D:\LoginAuthenticationModel\data\v2_feature_ablation"
CONF_MAT_DIR = os.path.join(OUT_DIR, "confusion_matrices")

V1_NUMERIC = ['login_hour', 'failed_attempts', 'time_since_last_login_minutes', 'login_frequency_24h', 'session_duration_minutes', 'distance_from_usual_location_km']
V1_BOOLEAN = ['new_device', 'new_ip', 'new_location']
V1_CATEGORICAL = ['device_type', 'browser', 'day_of_week']

V2_NUMERIC = ['failed_attempts', 'time_since_last_login_minutes', 'login_frequency_1h', 'login_frequency_6h', 'login_frequency_24h', 'login_frequency_7d', 'session_duration_minutes', 'distance_from_usual_location_km', 'usual_hour_deviation', 'usual_day_deviation', 'device_usage_frequency', 'previous_failed_attempt_rate', 'login_duration_deviation']
V2_BOOLEAN = ['new_device', 'new_ip', 'new_location']
V2_CATEGORICAL = ['device_type', 'browser', 'day_of_week']

def load_and_merge(data_path, label_path):
    df_data = pd.read_csv(data_path)
    df_labels = pd.read_csv(label_path)
    df = pd.merge(df_data, df_labels, on='event_id')
    df['login_timestamp'] = pd.to_datetime(df['login_timestamp'])
    # Convert boolean to int 0/1
    for col in df.columns:
        if df[col].dtype == 'bool' or col in V1_BOOLEAN + V2_BOOLEAN:
            df[col] = df[col].astype(int)
    # Target
    df['label'] = (df['actual_status'] != 'NORMAL').astype(int)
    return df

print("Loading V1...")
df_v1 = load_and_merge(V1_DATA_PATH, V1_LABEL_PATH)
print("Loading V2...")
df_v2 = load_and_merge(V2_DATA_PATH, V2_LABEL_PATH)

print(f"V1 row count: {len(df_v1)}")
print(f"V2 row count: {len(df_v2)}")

def get_preprocessor(numeric_cols, categorical_cols, boolean_cols):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Boolean cols are already 0/1 so we just pass them through
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('bool', 'passthrough', boolean_cols)
        ], remainder='drop'
    )
    return preprocessor

experiments = {
    'V1': {
        'df': df_v1,
        'num': V1_NUMERIC,
        'cat': V1_CATEGORICAL,
        'bool': V1_BOOLEAN
    },
    'V2_V1_FEATURES': {
        'df': df_v2,
        'num': V1_NUMERIC,
        'cat': V1_CATEGORICAL,
        'bool': V1_BOOLEAN
    },
    'V2_FULL': {
        'df': df_v2,
        'num': V2_NUMERIC,
        'cat': V2_CATEGORICAL,
        'bool': V2_BOOLEAN
    }
}

splits = ['Random', 'User', 'Temporal']

results = []

curves = {
    'ROC': {},
    'PR': {}
}

for exp_name, exp_cfg in experiments.items():
    df = exp_cfg['df']
    num_cols = exp_cfg['num']
    cat_cols = exp_cfg['cat']
    bool_cols = exp_cfg['bool']
    
    features = num_cols + cat_cols + bool_cols
    print(f"[{exp_name}] Total features: {len(features)}")
    
    for split_type in splits:
        print(f"  Running {split_type} split...")
        if split_type == 'Random':
            X_train, X_test, y_train, y_test = train_test_split(df, df['label'], test_size=0.20, random_state=42)
        elif split_type == 'User':
            users = df['user_id'].unique()
            train_users, test_users = train_test_split(users, test_size=0.20, random_state=42)
            
            # verify zero user overlap
            assert len(set(train_users).intersection(set(test_users))) == 0
            
            X_train = df[df['user_id'].isin(train_users)]
            X_test = df[df['user_id'].isin(test_users)]
            y_train = X_train['label']
            y_test = X_test['label']
        elif split_type == 'Temporal':
            df_sorted = df.sort_values(by='login_timestamp')
            split_idx = int(len(df_sorted) * 0.8)
            X_train = df_sorted.iloc[:split_idx]
            X_test = df_sorted.iloc[split_idx:]
            y_train = X_train['label']
            y_test = X_test['label']
            
            # verify temporal ordering
            assert X_train['login_timestamp'].max() <= X_test['login_timestamp'].min()
            
        preprocessor = get_preprocessor(num_cols, cat_cols, bool_cols)
        
        # Fit on training partition only!
        preprocessor.fit(X_train)
        
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        model = IsolationForest(contamination=0.074, random_state=42, n_estimators=100)
        
        # Train on transformed X_train
        model.fit(X_train_transformed)
        
        # Predict
        preds = model.predict(X_test_transformed)
        y_pred = (preds == -1).astype(int)
        
        # Scores
        decision_scores = model.decision_function(X_test_transformed)
        # sklearn IF decision_function: higher is more normal. So anomaly score is -decision_function
        y_scores = -decision_scores
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_scores)
        pr = average_precision_score(y_test, y_scores)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        results.append({
            'Dataset': exp_name.split('_')[0] if exp_name == 'V1' else 'V2' if exp_name == 'V2_V1_FEATURES' else 'V2',
            'Features': 'V1' if exp_name in ['V1', 'V2_V1_FEATURES'] else 'V2',
            'Experiment': exp_name,
            'Split': split_type,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC-AUC': roc,
            'PR-AUC': pr,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp
        })
        
        # Save confusion matrix plot
        plt.figure(figsize=(5,4))
        sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix: {exp_name} ({split_type})')
        plt.tight_layout()
        plt.savefig(os.path.join(CONF_MAT_DIR, f"{exp_name.lower().replace('_', '_')}_{split_type.lower()}.png"))
        plt.close()
        
        # Save curves for Temporal split (or maybe we can just save all in one plot, let's just save for random for curves, or all?)
        # Let's save curves data to plot them together
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_scores)
        key = f"{exp_name}_{split_type}"
        curves['ROC'][key] = (fpr, tpr, roc)
        curves['PR'][key] = (rec_curve, prec_curve, pr)

# Save results
df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

# Plot Metrics Bar Chart (Comparing Splits and Experiments)
# Let's plot F1, ROC-AUC, PR-AUC
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(data=df_res, x='Split', y='F1', hue='Experiment', ax=axes[0])
axes[0].set_title('F1 Score')
sns.barplot(data=df_res, x='Split', y='ROC-AUC', hue='Experiment', ax=axes[1])
axes[1].set_title('ROC-AUC')
sns.barplot(data=df_res, x='Split', y='PR-AUC', hue='Experiment', ax=axes[2])
axes[2].set_title('PR-AUC')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metrics.png"))
plt.close()

# Plot ROC Curves for all experiments (e.g. for Random Split)
plt.figure(figsize=(8, 6))
for exp in ['V1', 'V2_V1_FEATURES', 'V2_FULL']:
    fpr, tpr, roc = curves['ROC'][f"{exp}_Random"]
    plt.plot(fpr, tpr, label=f"{exp} (AUC = {roc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (Random Split)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"))
plt.close()

# Plot PR Curves for all experiments (Random Split)
plt.figure(figsize=(8, 6))
for exp in ['V1', 'V2_V1_FEATURES', 'V2_FULL']:
    rec_c, prec_c, pr = curves['PR'][f"{exp}_Random"]
    plt.plot(rec_c, prec_c, label=f"{exp} (AUC = {pr:.3f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves (Random Split)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_curves.png"))
plt.close()

print("Experiment completed successfully.")
