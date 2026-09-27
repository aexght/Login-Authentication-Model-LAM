import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

# Top-level boolean conversion function for clean pickling
def bool_to_int(df):
    return df.astype(float)

# V1 Feature definitions
V1_CATEGORICAL_FEATURES = ['device_type', 'browser', 'day_of_week']
V1_NUMERIC_FEATURES = [
    'login_hour',
    'failed_attempts',
    'time_since_last_login_minutes',
    'login_frequency_24h',
    'session_duration_minutes',
    'distance_from_usual_location_km'
]
V1_BOOLEAN_FEATURES = ['new_device', 'new_ip', 'new_location']
V1_DOMAIN_FEATURES = V1_NUMERIC_FEATURES + V1_BOOLEAN_FEATURES + V1_CATEGORICAL_FEATURES

# V2 Feature definitions (exact 20 domain features from V2_FULL)
V2_CATEGORICAL_FEATURES = ['device_type', 'browser', 'day_of_week']
V2_NUMERIC_FEATURES = [
    'login_hour',
    'failed_attempts',
    'time_since_last_login_minutes',
    'login_frequency_1h',
    'login_frequency_6h',
    'login_frequency_24h',
    'login_frequency_7d',
    'session_duration_minutes',
    'distance_from_usual_location_km',
    'usual_hour_deviation',
    'usual_day_deviation',
    'device_usage_frequency',
    'previous_failed_attempt_rate',
    'login_duration_deviation'
]
V2_BOOLEAN_FEATURES = ['new_device', 'new_ip', 'new_location']
V2_DOMAIN_FEATURES = V2_NUMERIC_FEATURES + V2_BOOLEAN_FEATURES + V2_CATEGORICAL_FEATURES

def build_v1_preprocessor():
    """
    Builds and returns ColumnTransformer for V1 feature schema (12 domain features).
    """
    boolean_transformer = Pipeline(steps=[
        ('bool_converter', FunctionTransformer(bool_to_int, feature_names_out='one-to-one'))
    ])
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, V1_NUMERIC_FEATURES),
            ('cat', categorical_transformer, V1_CATEGORICAL_FEATURES),
            ('bool', boolean_transformer, V1_BOOLEAN_FEATURES)
        ],
        remainder='drop'
    )

def build_v2_preprocessor():
    """
    Builds and returns ColumnTransformer for V2_FULL feature schema (20 domain features).
    """
    boolean_transformer = Pipeline(steps=[
        ('bool_converter', FunctionTransformer(bool_to_int, feature_names_out='one-to-one'))
    ])
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, V2_NUMERIC_FEATURES),
            ('cat', categorical_transformer, V2_CATEGORICAL_FEATURES),
            ('bool', boolean_transformer, V2_BOOLEAN_FEATURES)
        ],
        remainder='drop'
    )

def build_preprocessor(version='v1'):
    """
    Unified entry point for preprocessor creation.
    Defaults to v1 for full backward compatibility with legacy scripts.
    """
    if version == 'v2':
        return build_v2_preprocessor()
    return build_v1_preprocessor()

def load_and_preprocess_data(file_path):
    """
    Loads dataset and returns DataFrame ready for pipeline.
    """
    df = pd.read_csv(file_path, parse_dates=["login_timestamp"])
    return df
