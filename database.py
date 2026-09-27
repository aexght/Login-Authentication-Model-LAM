import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "auth.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS login_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        risk_score REAL,
        risk_level TEXT,
        anomaly_prediction INTEGER,
        new_ip BOOLEAN,
        new_device BOOLEAN,
        new_location BOOLEAN,
        relevant_features TEXT,
        device_type TEXT,
        ip_hash TEXT,
        device_hash TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS failed_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        count INTEGER DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
