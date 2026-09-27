import pandas as pd
import os

OUT_DIR = r"D:\LoginAuthenticationModel\data\v2_hyperparameter_tuning"
hp_df = pd.read_csv(os.path.join(OUT_DIR, "hyperparameter_results.csv"))
res_df = pd.read_csv(os.path.join(OUT_DIR, "results.csv"))
at_df = pd.read_csv(os.path.join(OUT_DIR, "anomaly_type_detection.csv"))

report_lines = []
report_lines.append("# Hyperparameter Tuning Report for V2_FULL\n")

# Best Split Configurations (from Phase 2, which has highest F1)
best_config = {}
for split_name in ['Random', 'User-level', 'Temporal']:
    split_res = hp_df[hp_df['Split'] == split_name]
    best_row = split_res.sort_values('F1', ascending=False).iloc[0]
    best_config[split_name] = best_row

# Global config
phase1_df = hp_df[hp_df['Phase'] == 1]
global_avg = phase1_df.groupby(['contamination', 'max_samples'])['F1'].mean().reset_index()
best_global_row = global_avg.sort_values('F1', ascending=False).iloc[0]
global_cont = best_global_row['contamination']
global_max_s = best_global_row['max_samples']
global_n_est = 200

report_lines.append("## Best Tuned Configurations (Validation Set Selected)")
for split_name, cfg in best_config.items():
    report_lines.append(f"**{split_name} Split:**")
    report_lines.append(f"- contamination={cfg['contamination']}")
    report_lines.append(f"- max_samples={cfg['max_samples']}")
    report_lines.append(f"- n_estimators={cfg['n_estimators']}\n")

report_lines.append(f"**Global Configuration:**")
report_lines.append(f"- contamination={global_cont}")
report_lines.append(f"- max_samples={global_max_s}")
report_lines.append(f"- n_estimators={global_n_est}\n")

report_lines.append("## Final Evaluation Results (Test Set)")
for split_name in ['Random', 'User-level', 'Temporal']:
    report_lines.append(f"\n### {split_name} Split")
    split_res = res_df[res_df['Split'] == split_name]
    
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

report_lines.append("\n## Anomaly Type Analysis (Global Tuned Model on Temporal Test)")
report_lines.append("| Type | Count | Detected | Rate |")
report_lines.append("|---|---|---|---|")
for idx, row in at_df.iterrows():
    report_lines.append(f"| {row['Type']} | {row['Count']} | {row['Detected']} | {row['Rate']:.2%} |")

with open(os.path.join(OUT_DIR, "tuning_report_fixed.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Report generated successfully.")
