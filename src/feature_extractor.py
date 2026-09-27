import hashlib
import json
import math
from datetime import datetime, timedelta
import random

# Exact 20 domain features required by V2_FULL
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
V2_CATEGORICAL_FEATURES = ['device_type', 'browser', 'day_of_week']

V2_DOMAIN_FEATURES = V2_NUMERIC_FEATURES + V2_BOOLEAN_FEATURES + V2_CATEGORICAL_FEATURES

VALID_DEVICE_TYPES = {'Desktop', 'Laptop', 'Mobile', 'Tablet', 'Unknown'}
VALID_BROWSERS = {'Chrome', 'Firefox', 'Safari', 'Edge', 'Samsung Internet', 'Other'}
VALID_DAYS_OF_WEEK = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}

def validate_feature_schema(features_dict):
    """
    Validates that a feature dictionary strictly conforms to the V2_FULL feature schema.
    Fails loudly (raises ValueError or TypeError) if:
    - a required feature is missing
    - an unexpected feature is present (ignoring internal metadata keys prefixed with '_meta_')
    - a categorical feature has an invalid value or type
    - a numerical feature has an invalid type, is NaN, or is infinite
    - a boolean feature is not boolean or convertible to 0/1
    """
    # Filter out metadata keys
    clean_keys = {k for k in features_dict.keys() if not k.startswith('_meta_')}

    # 1. Missing required features
    missing = set(V2_DOMAIN_FEATURES) - clean_keys
    if missing:
        raise ValueError(f"Schema validation failed: Missing required feature(s): {sorted(list(missing))}")

    # 2. Unexpected features
    unexpected = clean_keys - set(V2_DOMAIN_FEATURES)
    if unexpected:
        raise ValueError(f"Schema validation failed: Unexpected feature(s) found: {sorted(list(unexpected))}")

    # 3. Categorical validation
    for cat in V2_CATEGORICAL_FEATURES:
        val = features_dict[cat]
        if not isinstance(val, str):
            raise TypeError(f"Categorical feature '{cat}' must be a string, got {type(val).__name__}: {val}")

    if features_dict['day_of_week'] not in VALID_DAYS_OF_WEEK:
        raise ValueError(f"Invalid day_of_week: '{features_dict['day_of_week']}'. Expected one of {VALID_DAYS_OF_WEEK}")

    # 4. Numerical validation
    for num in V2_NUMERIC_FEATURES:
        val = features_dict[num]
        # In Python, isinstance(True, int) is True, so check for bool explicitly
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"Numerical feature '{num}' must be float or int, got {type(val).__name__}: {val}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Numerical feature '{num}' must not be NaN or Inf, got: {val}")

    # 5. Boolean validation
    for b in V2_BOOLEAN_FEATURES:
        val = features_dict[b]
        if not isinstance(val, (bool, int)):
            raise TypeError(f"Boolean feature '{b}' must be bool or int (0/1), got {type(val).__name__}: {val}")
        if isinstance(val, int) and val not in (0, 1):
            raise ValueError(f"Boolean feature '{b}' integer value must be 0 or 1, got {val}")

    return True


class FeatureExtractor:
    def __init__(self, db_connection_factory):
        self.get_db_connection = db_connection_factory

    def _hash_string(self, text):
        if not text:
            return None
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def parse_user_agent(self, ua_string):
        """
        Parses User-Agent string into privacy-conscious device category and browser.
        """
        ua = ua_string.lower() if ua_string else ""
        if 'mobile' in ua or 'android' in ua or 'iphone' in ua:
            device_type = 'Mobile'
        elif 'tablet' in ua or 'ipad' in ua:
            device_type = 'Tablet'
        elif ua:
            device_type = 'Laptop'
        else:
            device_type = 'Desktop'

        if 'samsung' in ua:
            browser = 'Samsung Internet'
        elif 'edg' in ua:
            browser = 'Edge'
        elif 'chrome' in ua and 'edg' not in ua:
            browser = 'Chrome'
        elif 'safari' in ua and 'chrome' not in ua:
            browser = 'Safari'
        elif 'firefox' in ua:
            browser = 'Firefox'
        else:
            browser = 'Other'

        return device_type, browser

    def extract_features(self, user_id, raw_ip, raw_ua, failed_attempts_count, simulated_distance=None):
        """
        Converts a real Flask login event into the exact 20-feature V2_FULL schema.

        Clear architectural distinction:
        ========================================================================
        1. DIRECTLY OBSERVED:
           - login_hour: Current hour of login (0-23)
           - day_of_week: Current day of week name ('Monday'-'Sunday')
           - device_type: Categorized from request User-Agent
           - browser: Categorized from request User-Agent
           - session_duration_minutes: Initialized to 0.0 at login initiation

        2. DERIVED FROM APPLICATION HISTORY:
           - failed_attempts: Consecutive failed login attempts before this success
           - time_since_last_login_minutes: Minutes elapsed since previous login
           - login_frequency_1h: Count of logins for user in previous 1 hour
           - login_frequency_6h: Count of logins for user in previous 6 hours
           - login_frequency_24h: Count of logins for user in previous 24 hours
           - login_frequency_7d: Count of logins for user in previous 7 days
           - new_device: True if device_hash has not been seen in user's history
           - new_ip: True if ip_hash has not been seen in user's history
           - usual_hour_deviation: Circular difference between login_hour and user's historical average
           - usual_day_deviation: Difference between day of week and user's typical login day
           - device_usage_frequency: Ratio of historical logins using this device_type
           - previous_failed_attempt_rate: Ratio of historic failed attempts
           - login_duration_deviation: Session duration deviation (0.0 at login initiation)

        3. SIMULATED DEMO FEATURE:
           - distance_from_usual_location_km: Simulated geographic jump (NOT real GPS coordinates)
           - new_location: True if simulated distance exceeds 30.0 km
        ========================================================================
        """
        now = datetime.now()

        # ----------------------------------------------------------------------
        # 1. DIRECTLY OBSERVED
        # ----------------------------------------------------------------------
        login_hour = int(now.hour)
        day_of_week = str(now.strftime('%A'))
        device_type, browser = self.parse_user_agent(raw_ua)
        session_duration_minutes = 0.0

        # Privacy-preserving hashes (Never store or log raw IPs or raw fingerprint data)
        ip_hash = self._hash_string(raw_ip)
        device_hash = self._hash_string(raw_ua)

        # ----------------------------------------------------------------------
        # 2. DERIVED FROM APPLICATION HISTORY (SQLite)
        # ----------------------------------------------------------------------
        time_since_last_login_minutes = 1440.0
        login_frequency_1h = 0
        login_frequency_6h = 0
        login_frequency_24h = 0
        login_frequency_7d = 0
        new_ip = True
        new_device = True
        usual_hour_deviation = 0.0
        usual_day_deviation = 0.0
        device_usage_frequency = 1.0
        previous_failed_attempt_rate = 0.0
        login_duration_deviation = 0.0

        conn = self.get_db_connection()
        cursor = conn.cursor()

        # Multi-window login frequency counts
        t_1h = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        t_6h = (now - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
        t_24h = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        t_7d = (now - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

        f1 = cursor.execute("SELECT COUNT(*) as c FROM login_events WHERE user_id = ? AND timestamp > ?", (user_id, t_1h)).fetchone()
        if f1: login_frequency_1h = int(f1['c'])

        f6 = cursor.execute("SELECT COUNT(*) as c FROM login_events WHERE user_id = ? AND timestamp > ?", (user_id, t_6h)).fetchone()
        if f6: login_frequency_6h = int(f6['c'])

        f24 = cursor.execute("SELECT COUNT(*) as c FROM login_events WHERE user_id = ? AND timestamp > ?", (user_id, t_24h)).fetchone()
        if f24: login_frequency_24h = int(f24['c'])

        f7d = cursor.execute("SELECT COUNT(*) as c FROM login_events WHERE user_id = ? AND timestamp > ?", (user_id, t_7d)).fetchone()
        if f7d: login_frequency_7d = int(f7d['c'])

        # Fetch recent login events for user behavioral profiling
        recent_logins = cursor.execute(
            "SELECT timestamp, ip_hash, device_hash, device_type, relevant_features FROM login_events WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
            (user_id,)
        ).fetchall()

        if recent_logins:
            # Time since previous login
            last_login = recent_logins[0]
            try:
                last_time = datetime.strptime(last_login['timestamp'], "%Y-%m-%d %H:%M:%S")
                time_since_last_login_minutes = max(0.0, float((now - last_time).total_seconds() / 60.0))
            except Exception:
                time_since_last_login_minutes = 1440.0

            # Familiarity check
            known_ips = {row['ip_hash'] for row in recent_logins if row['ip_hash']}
            known_devices = {row['device_hash'] for row in recent_logins if row['device_hash']}
            new_ip = bool(ip_hash not in known_ips)
            new_device = bool(device_hash not in known_devices)

            # Device usage frequency
            matching_devices = sum(1 for row in recent_logins if row['device_type'] == device_type)
            device_usage_frequency = float(matching_devices / len(recent_logins))

            # Usual hour deviation (circular distance on 24h clock)
            past_hours = []
            past_days = []
            for row in recent_logins:
                try:
                    dt = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
                    past_hours.append(dt.hour)
                    past_days.append(dt.weekday())
                except Exception:
                    pass

            if past_hours:
                avg_hour = sum(past_hours) / len(past_hours)
                h_diff = abs(login_hour - avg_hour)
                usual_hour_deviation = float(min(h_diff, 24.0 - h_diff))

            if past_days:
                # Mode day of week index
                typical_day = max(set(past_days), key=past_days.count)
                usual_day_deviation = float(abs(now.weekday() - typical_day))
        else:
            # First-ever login for user establishes baseline known device and IP
            new_ip = False
            new_device = False
            time_since_last_login_minutes = 1440.0

        # Failed attempt tracking
        fail_rec = cursor.execute("SELECT count FROM failed_attempts WHERE user_id = ?", (user_id,)).fetchone()
        tot_logins = len(recent_logins)
        curr_fails = int(failed_attempts_count)
        if tot_logins + curr_fails > 0:
            previous_failed_attempt_rate = float(curr_fails / (tot_logins + curr_fails))

        conn.close()

        # ----------------------------------------------------------------------
        # 3. SIMULATED DEMO FEATURE (Explicitly Simulated - No Real GPS Used)
        # ----------------------------------------------------------------------
        distance_from_usual_location_km = 0.0
        if simulated_distance is not None:
            distance_from_usual_location_km = float(simulated_distance)
        elif new_ip:
            # Simulate a realistic IP location variance jump for demonstration
            distance_from_usual_location_km = float(random.uniform(50.0, 500.0))

        new_location = bool(distance_from_usual_location_km > 30.0)

        # Assemble full 20 domain features
        ml_event_features = {
            "login_hour": int(login_hour),
            "day_of_week": str(day_of_week),
            "failed_attempts": int(failed_attempts_count),
            "time_since_last_login_minutes": float(time_since_last_login_minutes),
            "login_frequency_1h": int(login_frequency_1h),
            "login_frequency_6h": int(login_frequency_6h),
            "login_frequency_24h": int(login_frequency_24h),
            "login_frequency_7d": int(login_frequency_7d),
            "new_device": bool(new_device),
            "new_ip": bool(new_ip),
            "new_location": bool(new_location),
            "device_type": str(device_type),
            "browser": str(browser),
            "session_duration_minutes": float(session_duration_minutes),
            "distance_from_usual_location_km": float(distance_from_usual_location_km),
            "usual_hour_deviation": float(usual_hour_deviation),
            "usual_day_deviation": float(usual_day_deviation),
            "device_usage_frequency": float(device_usage_frequency),
            "previous_failed_attempt_rate": float(previous_failed_attempt_rate),
            "login_duration_deviation": float(login_duration_deviation),

            # Internal metadata keys (prefixed with _meta_) for DB persistence
            "_meta_ip_hash": ip_hash,
            "_meta_device_hash": device_hash
        }

        # Validate schema before returning
        validate_feature_schema(ml_event_features)

        return ml_event_features
