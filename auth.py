import sqlite3
import bcrypt
import json
from datetime import datetime, timedelta
import random
from database import get_db_connection

def register_user(username, email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    try:
        cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                       (username, email, password_hash))
        conn.commit()
        return {"success": True, "message": "Registration successful."}
    except sqlite3.IntegrityError:
        return {"success": False, "message": "Username or email already exists."}
    finally:
        conn.close()

def get_user_by_username(username):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, username)).fetchone()
    conn.close()
    return user

def record_failed_attempt(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get current failed attempts
    record = cursor.execute("SELECT count FROM failed_attempts WHERE user_id = ?", (user_id,)).fetchone()
    if record:
        new_count = record['count'] + 1
        cursor.execute("UPDATE failed_attempts SET count = ?, timestamp = CURRENT_TIMESTAMP WHERE user_id = ?", (new_count, user_id))
    else:
        new_count = 1
        cursor.execute("INSERT INTO failed_attempts (user_id, count) VALUES (?, 1)", (user_id,))
        
    conn.commit()
    conn.close()
    return new_count

def reset_failed_attempts(user_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM failed_attempts WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

def get_failed_attempts(user_id):
    conn = get_db_connection()
    record = conn.execute("SELECT count FROM failed_attempts WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return record['count'] if record else 0

def authenticate_user(username, password):
    user = get_user_by_username(username)
    if not user:
        return {"success": False, "message": "Invalid username or password.", "user_id": None}
        
    if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        record_failed_attempt(user['id'])
        return {"success": False, "message": "Invalid username or password.", "user_id": user['id']}
        
    return {"success": True, "user_id": user['id'], "username": user['username']}

def record_login_event(user_id, risk_score, risk_level, anomaly_prediction, features_dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    ip_hash = features_dict.pop('_meta_ip_hash', None)
    device_hash = features_dict.pop('_meta_device_hash', None)
    
    features_json = json.dumps(features_dict)
    
    cursor.execute('''
        INSERT INTO login_events 
        (user_id, risk_score, risk_level, anomaly_prediction, new_ip, new_device, new_location, relevant_features, device_type, ip_hash, device_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, 
        risk_score, 
        risk_level, 
        anomaly_prediction, 
        features_dict.get('new_ip', False), 
        features_dict.get('new_device', False), 
        features_dict.get('new_location', False), 
        features_json,
        features_dict.get('device_type', 'Unknown'),
        ip_hash,
        device_hash
    ))
    
    conn.commit()
    conn.close()
