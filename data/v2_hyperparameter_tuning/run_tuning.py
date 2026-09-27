import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import joblib

# Configuration
DATA_DIR = r"D:\LoginAuthenticationModel\data\New_Dataset"
OUT_DIR = r"D:\LoginAuthenticationModel\data\v2_hyperparameter_tuning"

# Hyperparameters to test
CONTAMINATIONS = [0.04, 0.05, 0.06, 0.07, 0.074, 0.08, 0.09, 0.10]
MAX_SAMPLES = [256, 512, 1024, 2048, 4096]
N_ESTIMATORS = [100, 200, 300]

def build_preprocessor():
    categorical_features = ['device_type', 'browser', 'day_of_week']
    numeric_features = [
        'login_hour', 'failed_attempts', 'time_since_last_login_minutes',
        'login_frequency_1h', 'login_frequency_6h', 'login_frequency_24h', 'login_frequency_7d',
        'session_duration_minutes', 'distance_from_usual_location_km',
        'usual_hour_deviation', 'usual_day_deviation', 'device_usage_frequency',
        'previous_failed_attempt_rate', 'login_duration_deviation'
    ]
    boolean_features = ['new_device', 'new_ip', 'new_location']

    def bool_to_int(df):
        return df.astype(float)

    boolean_transformer = Pipeline(steps=[
        ('bool_converter', FunctionTransformer(bool_to_int, feature_names_out='one-to-one'))
    ])
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bool', boolean_transformer, boolean_features)
        ],
        remainder='drop'
    )
    return preprocessor

def evaluate(y_true, y_pred, anomaly_scores):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)
    return acc, prec, rec, f1, roc_auc, pr_auc

def evaluate_model(model_pipeline, df, gt):
    preds = model_pipeline.predict(df)
    scores = -1 * model_pipeline.decision_function(df)
    
    y_true = (gt['actual_status'] == 'ANOMALY').astype(int)
    y_pred = (preds == -1).astype(int)
    
    return evaluate(y_true, y_pred, scores)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NORMAL', 'ANOMALY'], yticklabels=['NORMAL', 'ANOMALY'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrices", filename))
    plt.close()

def main():
    print("Loading datasets...")
    df = pd.read_csv(os.path.join(DATA_DIR, "login_behavior_dataset.csv"), parse_dates=["login_timestamp"])
    gt = pd.read_csv(os.path.join(DATA_DIR, "ground_truth.csv"))
    
    # Merge GT for easier splitting, but NEVER use labels in training features
    full_df = df.merge(gt, on="event_id")
    
    splits = {}
    
    # 1. RANDOM SPLIT
    print("Creating Random Split...")
    shuffled = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    splits['Random'] = {
        'train': shuffled.iloc[:train_end].copy(),
        'val': shuffled.iloc[train_end:val_end].copy(),
        'test': shuffled.iloc[val_end:].copy()
    }
    
    # 2. USER-LEVEL SPLIT
    print("Creating User-Level Split...")
    users = full_df['user_id'].unique()
    np.random.seed(42)
    np.random.shuffle(users)
    nu = len(users)
    u_train_end = int(0.6 * nu)
    u_val_end = int(0.8 * nu)
    train_users = set(users[:u_train_end])
    val_users = set(users[u_train_end:u_val_end])
    test_users = set(users[u_val_end:])
    
    assert train_users.isdisjoint(val_users)
    assert train_users.isdisjoint(test_users)
    assert val_users.isdisjoint(test_users)
    
    splits['User-level'] = {
        'train': full_df[full_df['user_id'].isin(train_users)].copy(),
        'val': full_df[full_df['user_id'].isin(val_users)].copy(),
        'test': full_df[full_df['user_id'].isin(test_users)].copy()
    }
    
    # 3. TEMPORAL SPLIT
    print("Creating Temporal Split...")
    sorted_df = full_df.sort_values('login_timestamp').reset_index(drop=True)
    splits['Temporal'] = {
        'train': sorted_df.iloc[:train_end].copy(),
        'val': sorted_df.iloc[train_end:val_end].copy(),
        'test': sorted_df.iloc[val_end:].copy()
    }
    
    hyperparameter_results = []
    best_phase1 = {} # Split -> top configurations
    
    # PHASE 1 TUNING
    print("Starting Phase 1 Tuning...")
    for split_name, data in splits.items():
        print(f"  Tuning on {split_name} split...")
        train_df = data['train']
        val_df = data['val']
        
        preprocessor = build_preprocessor()
        X_train_transformed = preprocessor.fit_transform(train_df)
        X_val_transformed = preprocessor.transform(val_df)
        
        y_val_true = (val_df['actual_status'] == 'ANOMALY').astype(int)
        
        results = []
        for cont in CONTAMINATIONS:
            for max_s in MAX_SAMPLES:
                model = IsolationForest(contamination=cont, max_samples=max_s, n_estimators=200, random_state=42)
                model.fit(X_train_transformed)
                
                preds = model.predict(X_val_transformed)
                scores = -1 * model.decision_function(X_val_transformed)
                y_pred = (preds == -1).astype(int)
                
                acc, prec, rec, f1, roc_auc, pr_auc = evaluate(y_val_true, y_pred, scores)
                
                res_dict = {
                    'Split': split_name,
                    'Phase': 1,
                    'contamination': cont,
                    'max_samples': max_s,
                    'n_estimators': 200,
                    'F1': f1,
                    'Precision': prec,
                    'Recall': rec,
                    'PR-AUC': pr_auc,
                    'ROC-AUC': roc_auc,
                    'Accuracy': acc
                }
                results.append(res_dict)
                hyperparameter_results.append(res_dict)
                
        # Sort and take top 3 by F1
        results.sort(key=lambda x: x['F1'], reverse=True)
        best_phase1[split_name] = [(r['contamination'], r['max_samples']) for r in results[:3]]
    
    # PHASE 2 TUNING
    print("Starting Phase 2 Tuning...")
    best_config = {} # Split -> best overall dict
    
    for split_name, data in splits.items():
        print(f"  Refining on {split_name} split...")
        train_df = data['train']
        val_df = data['val']
        
        preprocessor = build_preprocessor()
        X_train_transformed = preprocessor.fit_transform(train_df)
        X_val_transformed = preprocessor.transform(val_df)
        
        y_val_true = (val_df['actual_status'] == 'ANOMALY').astype(int)
        
        results = []
        for cont, max_s in best_phase1[split_name]:
            for n_est in N_ESTIMATORS:
                # skip already evaluated
                if n_est == 200:
                    continue
                    
                model = IsolationForest(contamination=cont, max_samples=max_s, n_estimators=n_est, random_state=42)
                model.fit(X_train_transformed)
                
                preds = model.predict(X_val_transformed)
                scores = -1 * model.decision_function(X_val_transformed)
                y_pred = (preds == -1).astype(int)
                
                acc, prec, rec, f1, roc_auc, pr_auc = evaluate(y_val_true, y_pred, scores)
                
                res_dict = {
                    'Split': split_name,
                    'Phase': 2,
                    'contamination': cont,
                    'max_samples': max_s,
                    'n_estimators': n_est,
                    'F1': f1,
                    'Precision': prec,
                    'Recall': rec,
                    'PR-AUC': pr_auc,
                    'ROC-AUC': roc_auc,
                    'Accuracy': acc
                }
                hyperparameter_results.append(res_dict)
        
        # Determine best overall for this split (Phase 1 + Phase 2)
        split_all_results = [r for r in hyperparameter_results if r['Split'] == split_name]
        split_all_results.sort(key=lambda x: x['F1'], reverse=True)
        best_config[split_name] = split_all_results[0]
        
    hp_df = pd.DataFrame(hyperparameter_results)
    hp_df.to_csv(os.path.join(OUT_DIR, "hyperparameter_results.csv"), index=False)
    
    # Evaluate Baseline vs Tuned on TEST SET
    print("Evaluating on Test Set...")
    final_test_results = []
    
    # 9. FAIRNESS CHECK (Global Config)
    # Average F1 across splits for each config in Phase 1
    phase1_df = hp_df[hp_df['Phase'] == 1]
    global_avg = phase1_df.groupby(['contamination', 'max_samples'])['F1'].mean().reset_index()
    best_global_row = global_avg.sort_values('F1', ascending=False).iloc[0]
    global_cont = best_global_row['contamination']
    global_max_s = best_global_row['max_samples']
    # Let's fix global n_est to 200
    global_n_est = 200 
    
    print(f"Best Global Config (from Phase 1 averages): contamination={global_cont}, max_samples={global_max_s}")
    
    for split_name, data in splits.items():
        train_val_df = pd.concat([data['train'], data['val']])
        test_df = data['test']
        
        y_test_true = (test_df['actual_status'] == 'ANOMALY').astype(int)
        
        # Baseline Model
        preprocessor_b = build_preprocessor()
        X_tv_b = preprocessor_b.fit_transform(train_val_df)
        X_test_b = preprocessor_b.transform(test_df)
        
        baseline = IsolationForest(contamination=0.074, max_samples='auto', n_estimators=100, random_state=42)
        baseline.fit(X_tv_b)
        
        preds_b = baseline.predict(X_test_b)
        scores_b = -1 * baseline.decision_function(X_test_b)
        y_pred_b = (preds_b == -1).astype(int)
        acc_b, prec_b, rec_b, f1_b, roc_b, pr_b = evaluate(y_test_true, y_pred_b, scores_b)
        
        final_test_results.append({
            'Split': split_name,
            'Model': 'Baseline',
            'contamination': 0.074, 'max_samples': 'auto', 'n_estimators': 100,
            'F1': f1_b, 'Precision': prec_b, 'Recall': rec_b, 'PR-AUC': pr_b, 'ROC-AUC': roc_b
        })
        
        plot_confusion_matrix(y_test_true, y_pred_b, f"Baseline - {split_name}", f"baseline_{split_name.lower()}.png")
        
        # Split-specific Tuned Model
        best_cfg = best_config[split_name]
        preprocessor_t = build_preprocessor()
        X_tv_t = preprocessor_t.fit_transform(train_val_df)
        X_test_t = preprocessor_t.transform(test_df)
        
        tuned = IsolationForest(contamination=best_cfg['contamination'], max_samples=best_cfg['max_samples'], n_estimators=best_cfg['n_estimators'], random_state=42)
        tuned.fit(X_tv_t)
        
        preds_t = tuned.predict(X_test_t)
        scores_t = -1 * tuned.decision_function(X_test_t)
        y_pred_t = (preds_t == -1).astype(int)
        acc_t, prec_t, rec_t, f1_t, roc_t, pr_t = evaluate(y_test_true, y_pred_t, scores_t)
        
        final_test_results.append({
            'Split': split_name,
            'Model': 'Tuned (Split-specific)',
            'contamination': best_cfg['contamination'], 'max_samples': best_cfg['max_samples'], 'n_estimators': best_cfg['n_estimators'],
            'F1': f1_t, 'Precision': prec_t, 'Recall': rec_t, 'PR-AUC': pr_t, 'ROC-AUC': roc_t
        })
        
        plot_confusion_matrix(y_test_true, y_pred_t, f"Tuned - {split_name}", f"tuned_{split_name.lower()}.png")
        
        # Global Tuned Model
        preprocessor_g = build_preprocessor()
        X_tv_g = preprocessor_g.fit_transform(train_val_df)
        X_test_g = preprocessor_g.transform(test_df)
        
        global_tuned = IsolationForest(contamination=global_cont, max_samples=int(global_max_s), n_estimators=global_n_est, random_state=42)
        global_tuned.fit(X_tv_g)
        
        preds_g = global_tuned.predict(X_test_g)
        scores_g = -1 * global_tuned.decision_function(X_test_g)
        y_pred_g = (preds_g == -1).astype(int)
        acc_g, prec_g, rec_g, f1_g, roc_g, pr_g = evaluate(y_test_true, y_pred_g, scores_g)
        
        final_test_results.append({
            'Split': split_name,
            'Model': 'Tuned (Global)',
            'contamination': global_cont, 'max_samples': int(global_max_s), 'n_estimators': global_n_est,
            'F1': f1_g, 'Precision': prec_g, 'Recall': rec_g, 'PR-AUC': pr_g, 'ROC-AUC': roc_g
        })
        
        if split_name == 'Temporal':
            # 10. ANOMALY-TYPE ANALYSIS
            # Run this ONLY for evaluation on the Temporal test set using Global model
            # Actually, let's do it on the combined test sets to have a robust number, or just the temporal test set.
            # "For the final globally selected V2 model, evaluate detection rate for..."
            
            # Since the model is trained on Temporal train+val, let's just evaluate on Temporal test.
            test_df['pred_anomaly'] = y_pred_g
            
            anomaly_type_results = []
            
            anomalies = test_df[test_df['actual_status'] == 'ANOMALY'].copy()
            
            # Subtypes (comma separated in 'anomaly_type' if multiple, or maybe json array. Let's assume comma-separated strings as usual or evaluate by exact match).
            # The prompt says: evaluate detection rate for: unusual_hour, unusual_day...
            # We'll split the anomaly_type string.
            
            # Ensure it's a string
            anomalies['anomaly_type'] = anomalies['anomaly_type'].fillna('').astype(str)
            
            all_subtypes = [
                'unusual_hour', 'unusual_day', 'short_gap', 'high_frequency',
                'high_failed_attempts', 'sudden_failed_attempts_increase',
                'new_device', 'unusual_device_browser_combo', 'device_switch',
                'new_ip', 'new_location', 'large_distance', 'impossible_travel',
                'dormant_reactivation', 'session_anomaly'
            ]
            
            for stype in all_subtypes:
                mask = anomalies['anomaly_type'].str.contains(stype)
                subset = anomalies[mask]
                if len(subset) > 0:
                    detected = subset['pred_anomaly'].sum()
                    anomaly_type_results.append({'Type': stype, 'Count': len(subset), 'Detected': detected, 'Rate': detected/len(subset)})
            
            # Single vs Multi signal
            # Count commas + 1
            anomalies['signal_count'] = anomalies['anomaly_type'].apply(lambda x: len([i for i in x.split(',') if i.strip()]) if x.strip() else 0)
            
            for c_label, mask in [('single-signal', anomalies['signal_count'] == 1),
                                  ('exactly-2-signal', anomalies['signal_count'] == 2),
                                  ('3+ signal', anomalies['signal_count'] >= 3)]:
                subset = anomalies[mask]
                if len(subset) > 0:
                    detected = subset['pred_anomaly'].sum()
                    anomaly_type_results.append({'Type': c_label, 'Count': len(subset), 'Detected': detected, 'Rate': detected/len(subset)})
            
            at_df = pd.DataFrame(anomaly_type_results)
            at_df.to_csv(os.path.join(OUT_DIR, "anomaly_type_detection.csv"), index=False)

    final_df = pd.DataFrame(final_test_results)
    final_df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    
    # GENERATE REPORT
    report_lines = []
    report_lines.append("# Hyperparameter Tuning Report for V2_FULL")
    report_lines.append("")
    report_lines.append("## Baseline Configuration")
    report_lines.append("- contamination=0.074\n- max_samples=auto\n- n_estimators=100")
    report_lines.append("")
    
    report_lines.append("## Best Tuned Configurations (Validation Set Selected)")
    for split_name in splits.keys():
        cfg = best_config[split_name]
        report_lines.append(f"**{split_name} Split:**")
        report_lines.append(f"- contamination={cfg['contamination']}")
        report_lines.append(f"- max_samples={cfg['max_samples']}")
        report_lines.append(f"- n_estimators={cfg['n_estimators']}")
        report_lines.append("")
    
    report_lines.append(f"**Global Configuration:**")
    report_lines.append(f"- contamination={global_cont}")
    report_lines.append(f"- max_samples={global_max_s}")
    report_lines.append(f"- n_estimators={global_n_est}")
    report_lines.append("")
    
    report_lines.append("## Final Evaluation Results (Test Set)")
    report_lines.append("")
    
    for split_name in splits.keys():
        report_lines.append(f"### {split_name} Split")
        split_res = final_df[final_df['Split'] == split_name]
        
        base_r = split_res[split_res['Model'] == 'Baseline'].iloc[0]
        tune_s_r = split_res[split_res['Model'] == 'Tuned (Split-specific)'].iloc[0]
        tune_g_r = split_res[split_res['Model'] == 'Tuned (Global)'].iloc[0]
        
        report_lines.append("| Metric | Baseline | Split-Tuned | Global-Tuned | Δ (Split-Tuned vs Base) | Δ (Global-Tuned vs Base) |")
        report_lines.append("|---|---|---|---|---|---|")
        for m in ['F1', 'Precision', 'Recall', 'PR-AUC', 'ROC-AUC']:
            val_b = base_r[m]
            val_s = tune_s_r[m]
            val_g = tune_g_r[m]
            delta_s = val_s - val_b
            delta_g = val_g - val_b
            report_lines.append(f"| {m} | {val_b:.4f} | {val_s:.4f} | {val_g:.4f} | {delta_s:+.4f} | {delta_g:+.4f} |")
        report_lines.append("")
        
    report_lines.append("## Anomaly Type Analysis (Global Tuned Model on Temporal Test)")
    report_lines.append("| Type | Count | Detected | Rate |")
    report_lines.append("|---|---|---|---|")
    try:
        for idx, row in at_df.iterrows():
            report_lines.append(f"| {row['Type']} | {row['Count']} | {row['Detected']} | {row['Rate']:.2%} |")
    except Exception as e:
        report_lines.append(f"Could not load anomaly type analysis: {e}")
    report_lines.append("")
        
    with open(os.path.join(OUT_DIR, "tuning_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    # Validation curves plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hp_df[hp_df['Phase'] == 1], x='contamination', y='F1', hue='max_samples', style='Split')
    plt.title('Validation F1 by Contamination and Max Samples (Phase 1)')
    plt.savefig(os.path.join(OUT_DIR, "validation_curves.png"))
    plt.close()

if __name__ == "__main__":
    main()
