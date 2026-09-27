import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

# Local imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from preprocess import (
    build_v1_preprocessor, build_v2_preprocessor,
    V1_NUMERIC_FEATURES, V1_BOOLEAN_FEATURES, V1_CATEGORICAL_FEATURES, V1_DOMAIN_FEATURES,
    V2_NUMERIC_FEATURES, V2_BOOLEAN_FEATURES, V2_CATEGORICAL_FEATURES, V2_DOMAIN_FEATURES,
    load_and_preprocess_data
)

def train_v2_final_model(data_path=None, model_output_path=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, "data", "New_Dataset", "login_behavior_dataset.csv")
    if model_output_path is None:
        model_output_path = os.path.join(base_dir, "models", "login_anomaly_model_v2.pkl")

    print("=" * 60)
    print("TRAINING FINAL PRODUCTION-PROTOTYPE V2 MODEL")
    print("=" * 60)

    # 1. Verification of Feature Schema
    print("\n--- [1] Feature Schema Verification ---")
    print(f"Total raw CSV columns expected: 23 (3 metadata + 20 domain features)")
    print(f"Numerical features ({len(V2_NUMERIC_FEATURES)}):")
    for f in V2_NUMERIC_FEATURES:
        print(f"  - {f}")
    print(f"Boolean features ({len(V2_BOOLEAN_FEATURES)}):")
    for f in V2_BOOLEAN_FEATURES:
        print(f"  - {f}")
    print(f"Categorical features ({len(V2_CATEGORICAL_FEATURES)}):")
    for f in V2_CATEGORICAL_FEATURES:
        print(f"  - {f}")
    print(f"Total Domain Features: {len(V2_DOMAIN_FEATURES)}")
    assert len(V2_DOMAIN_FEATURES) == 20, f"Expected 20 domain features, got {len(V2_DOMAIN_FEATURES)}"

    # 2. Loading Dataset
    print(f"\n--- [2] Loading V2 Dataset from {data_path} ---")
    df = load_and_preprocess_data(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of training samples: {len(df)}")
    print(f"Unique synthetic users: {df['user_id'].nunique()}")

    # Ensure metadata columns are present but will be dropped by remainder='drop'
    for col in ['event_id', 'user_id', 'login_timestamp']:
        assert col in df.columns, f"Metadata column {col} missing from dataset"

    # Confirm all 20 domain features exist in df
    for col in V2_DOMAIN_FEATURES:
        assert col in df.columns, f"Domain feature {col} missing from dataset"

    # 3. Preprocessor Construction & Transform Dimension Verification
    print("\n--- [3] Building V2 Preprocessor ---")
    preprocessor = build_v2_preprocessor()
    sample_transformed = preprocessor.fit_transform(df.head(100))
    transformed_dim = sample_transformed.shape[1]
    print(f"Transformed dimensionality (after scaling, boolean conversion & OneHotEncoding): {transformed_dim}")
    assert transformed_dim == 33, f"Expected 33 transformed dimensions, got {transformed_dim}"

    # 4. Isolation Forest Configuration
    print("\n--- [4] Configuring Isolation Forest (Validated Global V2 Parameters) ---")
    iso_forest = IsolationForest(
        contamination=0.06,
        max_samples=4096,
        n_estimators=200,
        random_state=42
    )
    print(f"Configuration:\n  {iso_forest}")

    # Complete pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', iso_forest)
    ])

    # 5. Model Fitting
    print("\n--- [5] Fitting Pipeline on Entire V2 Dataset (112,131 samples) ---")
    model_pipeline.fit(df)
    print("Pipeline fitting complete.")

    # 6. Artifact Serialization
    print(f"\n--- [6] Saving Model Artifact to {model_output_path} ---")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model_pipeline, model_output_path)
    print("Final V2 Model Artifact successfully saved.")

    return model_pipeline

def train_and_save_model(data_path, model_output_path):
    """Legacy V1 training function kept for backward compatibility."""
    print("Loading V1 data...")
    df = load_and_preprocess_data(data_path)
    print(f"Number of training samples: {len(df)}")
    preprocessor = build_v1_preprocessor()
    iso_forest = IsolationForest(
        contamination=0.075,
        random_state=42,
        n_estimators=100
    )
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', iso_forest)
    ])
    model_pipeline.fit(df)
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model_pipeline, model_output_path)
    print(f"V1 Model saved to {model_output_path}.")
    return model_pipeline

if __name__ == "__main__":
    train_v2_final_model()
