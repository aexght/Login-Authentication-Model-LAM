import os
import sys
import json
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_session import Session

# Initialize Flask app securely
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'

Session(app)

# Add src to path to import ML pipeline and feature extractor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from predict import AnomalyDetector
from feature_extractor import FeatureExtractor, validate_feature_schema
from database import init_db, get_db_connection
from auth import (
    register_user, authenticate_user, get_failed_attempts, 
    reset_failed_attempts, record_login_event
)

# Load final V2 ML Model ONCE on application startup
print("=" * 60)
print("INITIALIZING FLASK APPLICATION WITH V2 ML ANOMALY MODEL")
t_start = time.perf_counter()
base_dir = os.path.dirname(os.path.abspath(__file__))
v2_model_path = os.path.join(base_dir, "models", "login_anomaly_model_v2.pkl")
ml_detector = AnomalyDetector(model_path=v2_model_path)
feature_extractor = FeatureExtractor(get_db_connection)
startup_duration_ms = (time.perf_counter() - t_start) * 1000
print(f"ML Model ({ml_detector.model_version}) loaded successfully in {startup_duration_ms:.2f} ms.")
print(f"Artifact: {v2_model_path}")
print("=" * 60)

# Initialize database
init_db()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
            
        result = register_user(username, email, password)
        if result['success']:
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash(result['message'], 'danger')
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # 1. Primary credential authentication via bcrypt
        auth_result = authenticate_user(username, password)
        
        if not auth_result['success']:
            flash(auth_result['message'], 'danger')
            return render_template('login.html')
            
        user_id = auth_result['user_id']
        
        # 2. Extract context and telemetry
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        failed_attempts = get_failed_attempts(user_id)
        
        # Simulated distance parameter (Explicitly demo-only, not real GPS)
        simulated_distance = None
        if username == 'hacker_test':
            simulated_distance = 1500.0
            
        # 3. Extract exact 20 V2_FULL domain features
        ml_event_features = feature_extractor.extract_features(
            user_id, ip_address, user_agent, failed_attempts, simulated_distance
        )
        
        # Simulated anomaly injection for prototype testing
        if username == 'hacker_test':
            ml_event_features.update({
                "login_hour": 3,
                "failed_attempts": 6,
                "usual_hour_deviation": 11.0,
                "usual_day_deviation": 4.0,
                "login_frequency_1h": 12,
                "login_frequency_6h": 25,
                "login_frequency_24h": 30,
                "login_frequency_7d": 50,
                "new_device": True,
                "new_ip": True,
                "new_location": True,
                "device_usage_frequency": 0.0,
                "previous_failed_attempt_rate": 0.75,
                "distance_from_usual_location_km": 1500.0
            })
            
        # 4. Strict Schema Validation
        validate_feature_schema(ml_event_features)
            
        # 5. Model Inference (Behavioral Anomaly Risk Layer)
        t_pred = time.perf_counter()
        prediction_result = ml_detector.predict(ml_event_features)
        pred_latency_ms = (time.perf_counter() - t_pred) * 1000
        
        # 6. Record login event in SQLite
        record_login_event(
            user_id, 
            prediction_result['Anomaly Score'], 
            prediction_result['Risk Level'], 
            prediction_result['Raw Prediction'],
            ml_event_features
        )
        
        # Reset failed attempts counter on authenticated login
        reset_failed_attempts(user_id)
        
        # Set session state
        session['user_id'] = user_id
        session['username'] = auth_result['username']
        session['risk_level'] = prediction_result['Risk Level']
        session['anomaly_score'] = prediction_result['Anomaly Score']
        session['model_version'] = prediction_result.get('Model Version', 'v2')
        
        risk = prediction_result['Risk Level']
        if risk == 'LOW':
            flash('Login successful.', 'success')
        elif risk == 'MEDIUM':
            flash('Login successful, but behavior is slightly unusual. A security warning has been recorded.', 'warning')
        elif risk == 'HIGH':
            flash('SECURITY WARNING: Highly suspicious login detected. Additional verification may be required.', 'danger')
            
        return redirect(url_for('dashboard'))
        
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    user_id = session['user_id']
    conn = get_db_connection()
    
    # Get recent login activity
    recent_logins = conn.execute(
        "SELECT * FROM login_events WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10", 
        (user_id,)
    ).fetchall()
    
    # Security summary
    total_logins = conn.execute("SELECT COUNT(*) as count FROM login_events WHERE user_id = ?", (user_id,)).fetchone()['count']
    suspicious_logins = conn.execute("SELECT COUNT(*) as count FROM login_events WHERE user_id = ? AND risk_level IN ('MEDIUM', 'HIGH')", (user_id,)).fetchone()['count']
    
    failed_record = conn.execute("SELECT count FROM failed_attempts WHERE user_id = ?", (user_id,)).fetchone()
    failed_attempts = failed_record['count'] if failed_record else 0
    
    conn.close()
    
    current_risk = session.get('risk_level', 'UNKNOWN')
    anomaly_score = session.get('anomaly_score', 0.0)
    if not anomaly_score and recent_logins:
        anomaly_score = recent_logins[0]['risk_score']
    
    return render_template(
        'dashboard.html', 
        username=session['username'],
        current_risk=current_risk,
        anomaly_score=anomaly_score,
        recent_logins=recent_logins,
        total_logins=total_logins,
        suspicious_logins=suspicious_logins,
        failed_attempts=failed_attempts
    )

@app.route('/security-events')
def security_events():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    user_id = session['user_id']
    conn = get_db_connection()
    
    suspicious_events = conn.execute(
        "SELECT * FROM login_events WHERE user_id = ? AND risk_level IN ('MEDIUM', 'HIGH') ORDER BY timestamp DESC",
        (user_id,)
    ).fetchall()
    
    conn.close()
    
    events_with_features = []
    for evt in suspicious_events:
        try:
            feats = json.loads(evt['relevant_features'])
        except Exception:
            feats = {}
        events_with_features.append({"event": evt, "features": feats})
        
    return render_template('security_events.html', events=events_with_features)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, port=5000)
