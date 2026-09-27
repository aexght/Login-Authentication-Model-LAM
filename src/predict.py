import os
import joblib
import pandas as pd

class AnomalyDetector:
    def __init__(self, model_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        v2_path = os.path.join(base_dir, "models", "login_anomaly_model_v2.pkl")
        v1_path = os.path.join(base_dir, "models", "login_anomaly_model.pkl")

        if model_path is None:
            # Prefer V2 final model artifact if it exists
            if os.path.exists(v2_path):
                model_path = v2_path
            else:
                model_path = v1_path

        self.model_path = model_path
        self.model_version = "v2" if "v2" in os.path.basename(model_path) else "v1"
        self.model = joblib.load(model_path)

    def predict(self, event_dict):
        """
        Accepts a dictionary representing a single login event.
        Returns a dictionary with Anomaly Score, Prediction, and Risk Level.
        """
        # Exclude internal metadata fields starting with _meta_
        clean_event = {k: v for k, v in event_dict.items() if not k.startswith('_meta_')}
        df = pd.DataFrame([clean_event])

        # Pipeline handles ColumnTransformer (scaling, encoding, boolean) + Isolation Forest
        pred = self.model.predict(df)[0]
        raw_score = self.model.decision_function(df)[0]
        anomaly_score = -1 * raw_score

        prediction_label = "ANOMALY" if pred == -1 else "NORMAL"

        # Determine Risk Level based primarily on model anomaly score
        if anomaly_score < -0.05:
            risk_level = "LOW"
        elif anomaly_score < 0.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "Anomaly Score": round(float(anomaly_score), 4),
            "Prediction": prediction_label,
            "Risk Level": risk_level,
            "Raw Prediction": int(pred),
            "Model Version": self.model_version
        }

if __name__ == "__main__":
    detector = AnomalyDetector()
    print(f"Loaded model version: {detector.model_version} from {detector.model_path}")
    sample_event = {
        "login_hour": 14,
        "day_of_week": "Wednesday",
        "failed_attempts": 0,
        "time_since_last_login_minutes": 120.0,
        "login_frequency_1h": 1,
        "login_frequency_6h": 2,
        "login_frequency_24h": 3,
        "login_frequency_7d": 12,
        "new_device": False,
        "new_ip": False,
        "new_location": False,
        "device_type": "Laptop",
        "browser": "Chrome",
        "session_duration_minutes": 45.0,
        "distance_from_usual_location_km": 1.2,
        "usual_hour_deviation": 0.5,
        "usual_day_deviation": 0.0,
        "device_usage_frequency": 0.95,
        "previous_failed_attempt_rate": 0.0,
        "login_duration_deviation": 0.0
    }
    result = detector.predict(sample_event)
    print("Sample Prediction Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
