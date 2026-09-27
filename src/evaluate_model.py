import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "login_behavior_dataset.csv")
    gt_path = os.path.join(base_dir, "data", "ground_truth.csv")
    model_path = os.path.join(base_dir, "models", "login_anomaly_model.pkl")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading data and model...")
    df = pd.read_csv(data_path, parse_dates=["login_timestamp"])
    gt = pd.read_csv(gt_path)
    model = joblib.load(model_path)
    
    print("Making predictions...")
    # predict returns 1 (normal) and -1 (anomaly)
    preds = model.predict(df)
    
    # decision function returns higher values for normal, lower for anomaly
    # we negate it so higher values indicate anomaly for ROC-AUC
    anomaly_scores = -1 * model.decision_function(df)
    
    # Map predictions to match ground truth format or vice versa
    # gt['actual_status'] is 'NORMAL' or 'ANOMALY'
    # Let's map gt to 1 (normal) and -1 (anomaly) for sklearn metrics,
    # OR map everything to binary 0 (normal) and 1 (anomaly).
    # Since we treat ANOMALY as the positive class:
    y_true = (gt['actual_status'] == 'ANOMALY').astype(int)
    y_pred = (preds == -1).astype(int)
    
    print("Calculating metrics...")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    report_lines = [
        "Evaluation Report",
        "=================",
        f"Accuracy:  {acc:.4f}",
        f"Precision: {prec:.4f}",
        f"Recall:    {rec:.4f}",
        f"F1-score:  {f1:.4f}",
        f"ROC-AUC:   {roc_auc:.4f}",
        "",
        "Confusion Matrix:",
        f"True Negatives (Normal classified as Normal): {tn}",
        f"False Positives (Normal classified as Anomaly): {fp}",
        f"False Negatives (Anomaly classified as Normal): {fn}",
        f"True Positives (Anomaly classified as Anomaly): {tp}"
    ]
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open(os.path.join(results_dir, "evaluation_report.txt"), "w") as f:
        f.write(report_text)
        
    print("Saving confusion matrix plot...")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'ANOMALY'], 
                yticklabels=['NORMAL', 'ANOMALY'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    evaluate_model()
