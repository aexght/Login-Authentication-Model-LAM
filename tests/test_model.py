import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import (
    load_and_preprocess_data, build_preprocessor,
    build_v1_preprocessor, build_v2_preprocessor,
    V1_DOMAIN_FEATURES, V2_DOMAIN_FEATURES
)
from predict import AnomalyDetector
from feature_extractor import validate_feature_schema

class TestModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.v1_data_path = os.path.join(cls.base_dir, "data", "login_behavior_dataset.csv")
        cls.v2_data_path = os.path.join(cls.base_dir, "data", "New_Dataset", "login_behavior_dataset.csv")
        cls.v1_model_path = os.path.join(cls.base_dir, "models", "login_anomaly_model.pkl")
        cls.v2_model_path = os.path.join(cls.base_dir, "models", "login_anomaly_model_v2.pkl")
        
        # Detector instances
        cls.detector_v1 = AnomalyDetector(cls.v1_model_path)
        cls.detector_v2 = AnomalyDetector(cls.v2_model_path)
        
    def test_v1_dataset_loading(self):
        df = load_and_preprocess_data(self.v1_data_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) > 0)
        self.assertIn("login_hour", df.columns)
        
    def test_v2_dataset_loading(self):
        df = load_and_preprocess_data(self.v2_data_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 112131)
        self.assertEqual(df.shape[1], 23)
        for feat in V2_DOMAIN_FEATURES:
            self.assertIn(feat, df.columns)
        
    def test_v1_preprocessing(self):
        df = load_and_preprocess_data(self.v1_data_path)
        preprocessor = build_v1_preprocessor()
        X = preprocessor.fit_transform(df.head(10))
        self.assertIsNotNone(X)
        self.assertEqual(X.shape[0], 10)
        
    def test_v2_preprocessing(self):
        df = load_and_preprocess_data(self.v2_data_path)
        preprocessor = self.detector_v2.model.named_steps['preprocessor']
        X = preprocessor.transform(df.head(10))
        self.assertIsNotNone(X)
        self.assertEqual(X.shape[0], 10)
        self.assertEqual(X.shape[1], 33) # 14 numeric + 3 bool + 16 categorical
        
    def test_v1_model_loading(self):
        self.assertIsNotNone(self.detector_v1.model)
        self.assertEqual(self.detector_v1.model_version, "v1")
        
    def test_v2_model_loading(self):
        self.assertIsNotNone(self.detector_v2.model)
        self.assertEqual(self.detector_v2.model_version, "v2")
        
    def test_v2_schema_validation_success(self):
        valid_event = {
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
        self.assertTrue(validate_feature_schema(valid_event))
        
    def test_v2_schema_validation_missing_feature(self):
        incomplete_event = {
            "login_hour": 14,
            "day_of_week": "Wednesday"
            # Missing remaining 18 features
        }
        with self.assertRaises(ValueError) as ctx:
            validate_feature_schema(incomplete_event)
        self.assertIn("Missing required feature", str(ctx.exception))
        
    def test_v2_schema_validation_unexpected_feature(self):
        valid_event = {
            "login_hour": 14, "day_of_week": "Wednesday", "failed_attempts": 0,
            "time_since_last_login_minutes": 120.0, "login_frequency_1h": 1,
            "login_frequency_6h": 2, "login_frequency_24h": 3, "login_frequency_7d": 12,
            "new_device": False, "new_ip": False, "new_location": False,
            "device_type": "Laptop", "browser": "Chrome", "session_duration_minutes": 45.0,
            "distance_from_usual_location_km": 1.2, "usual_hour_deviation": 0.5,
            "usual_day_deviation": 0.0, "device_usage_frequency": 0.95,
            "previous_failed_attempt_rate": 0.0, "login_duration_deviation": 0.0,
            "unexpected_injected_column": 123
        }
        with self.assertRaises(ValueError) as ctx:
            validate_feature_schema(valid_event)
        self.assertIn("Unexpected feature", str(ctx.exception))
        
    def test_v2_schema_validation_invalid_types(self):
        invalid_num = {
            "login_hour": "fourteen", # should be int/float
            "day_of_week": "Wednesday", "failed_attempts": 0,
            "time_since_last_login_minutes": 120.0, "login_frequency_1h": 1,
            "login_frequency_6h": 2, "login_frequency_24h": 3, "login_frequency_7d": 12,
            "new_device": False, "new_ip": False, "new_location": False,
            "device_type": "Laptop", "browser": "Chrome", "session_duration_minutes": 45.0,
            "distance_from_usual_location_km": 1.2, "usual_hour_deviation": 0.5,
            "usual_day_deviation": 0.0, "device_usage_frequency": 0.95,
            "previous_failed_attempt_rate": 0.0, "login_duration_deviation": 0.0
        }
        with self.assertRaises(TypeError):
            validate_feature_schema(invalid_num)
            
        invalid_nan = invalid_num.copy()
        invalid_nan['login_hour'] = float('nan')
        with self.assertRaises(ValueError):
            validate_feature_schema(invalid_nan)

    def test_v2_sanity_scenarios_normal_vs_suspicious(self):
        normal_event = {
            "login_hour": 14,
            "day_of_week": "Wednesday",
            "failed_attempts": 0,
            "time_since_last_login_minutes": 1440.0,
            "login_frequency_1h": 0,
            "login_frequency_6h": 0,
            "login_frequency_24h": 1,
            "login_frequency_7d": 5,
            "new_device": False,
            "new_ip": False,
            "new_location": False,
            "device_type": "Desktop",
            "browser": "Chrome",
            "session_duration_minutes": 0.0,
            "distance_from_usual_location_km": 1.5,
            "usual_hour_deviation": 0.2,
            "usual_day_deviation": 0.0,
            "device_usage_frequency": 0.95,
            "previous_failed_attempt_rate": 0.0,
            "login_duration_deviation": 0.0
        }
        
        suspicious_event = {
            "login_hour": 3,
            "day_of_week": "Sunday",
            "failed_attempts": 6,
            "time_since_last_login_minutes": 2.0,
            "login_frequency_1h": 15,
            "login_frequency_6h": 25,
            "login_frequency_24h": 35,
            "login_frequency_7d": 60,
            "new_device": True,
            "new_ip": True,
            "new_location": True,
            "device_type": "Mobile",
            "browser": "Samsung Internet",
            "session_duration_minutes": 0.0,
            "distance_from_usual_location_km": 2400.0,
            "usual_hour_deviation": 11.0,
            "usual_day_deviation": 4.0,
            "device_usage_frequency": 0.0,
            "previous_failed_attempt_rate": 0.8,
            "login_duration_deviation": 0.0
        }
        
        res_normal = self.detector_v2.predict(normal_event)
        res_suspicious = self.detector_v2.predict(suspicious_event)
        
        self.assertIn("Anomaly Score", res_normal)
        self.assertIn("Risk Level", res_normal)
        self.assertEqual(res_normal['Prediction'], 'NORMAL')
        
        self.assertIn("Anomaly Score", res_suspicious)
        self.assertIn("Risk Level", res_suspicious)
        self.assertEqual(res_suspicious['Prediction'], 'ANOMALY')
        self.assertEqual(res_suspicious['Risk Level'], 'HIGH')
        
        # Suspicious anomaly score must be significantly higher than normal anomaly score
        self.assertGreater(res_suspicious['Anomaly Score'], res_normal['Anomaly Score'])
        self.assertGreater(res_suspicious['Anomaly Score'], 0.0)

if __name__ == "__main__":
    unittest.main()
