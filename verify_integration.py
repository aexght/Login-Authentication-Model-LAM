import sys
import os

sys.path.insert(0, r"D:\LoginAuthenticationModel")
from app import app
from database import get_db_connection

def run_verification():
    conn = get_db_connection()
    conn.execute("DELETE FROM login_events WHERE user_id IN (SELECT id FROM users WHERE username IN ('demo_user', 'hacker_test'))")
    conn.execute("DELETE FROM users WHERE username IN ('demo_user', 'hacker_test')")
    conn.commit()
    conn.close()

    client = app.test_client()

    print("=== 1. USER REGISTRATION ===")
    res_reg = client.post('/register', data={
        'username': 'demo_user',
        'email': 'demo@example.com',
        'password': 'StrongPassword123!',
        'confirm_password': 'StrongPassword123!'
    }, follow_redirects=True)
    assert res_reg.status_code == 200
    assert b'Registration successful' in res_reg.data
    print("Registration successful.")

    print("\n=== 2. NORMAL LOGIN ===")
    res_login = client.post('/login', data={
        'username': 'demo_user',
        'password': 'StrongPassword123!'
    }, follow_redirects=True)
    assert res_login.status_code == 200
    assert b'Welcome, demo_user' in res_login.data
    print("Login successful.")

    conn = get_db_connection()
    user = conn.execute("SELECT id FROM users WHERE username = 'demo_user'").fetchone()
    normal_event = conn.execute("SELECT * FROM login_events WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user['id'],)).fetchone()
    print(f"Normal Event DB Record: Risk Level = {normal_event['risk_level']}, Anomaly Score = {normal_event['risk_score']:.4f}")
    assert normal_event['risk_level'] in ('LOW', 'MEDIUM')
    assert normal_event['risk_score'] < 0.0

    print("\n=== 3. DASHBOARD VERIFICATION ===")
    res_dash = client.get('/dashboard')
    assert res_dash.status_code == 200
    assert b'Current Login Risk' in res_dash.data
    print("Dashboard accessible with risk indicators.")

    print("\n=== 4. SUSPICIOUS LOGIN SIMULATION ===")
    client.post('/register', data={
        'username': 'hacker_test',
        'email': 'hacker@example.com',
        'password': 'Password123!',
        'confirm_password': 'Password123!'
    })
    res_hacker = client.post('/login', data={
        'username': 'hacker_test',
        'password': 'Password123!'
    }, follow_redirects=True)
    assert b'SECURITY WARNING: Highly suspicious login detected' in res_hacker.data
    print("Security warning flashed on suspicious login.")

    hacker_user = conn.execute("SELECT id FROM users WHERE username = 'hacker_test'").fetchone()
    susp_event = conn.execute("SELECT * FROM login_events WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (hacker_user['id'],)).fetchone()
    print(f"Suspicious Event DB Record: Risk Level = {susp_event['risk_level']}, Anomaly Score = {susp_event['risk_score']:.4f}")
    assert susp_event['risk_level'] == 'HIGH'
    assert susp_event['risk_score'] > 0.0

    print("\n=== 5. SECURITY EVENTS PAGE VERIFICATION ===")
    res_sec = client.get('/security-events')
    assert res_sec.status_code == 200
    assert b'SECURITY ALERT: Suspicious login detected' in res_sec.data
    print("Suspicious event correctly surfaced on /security-events.")

    print("\n=== 6. LOGOUT VERIFICATION ===")
    res_logout = client.get('/logout', follow_redirects=True)
    assert res_logout.status_code == 200
    assert b'logged out' in res_logout.data
    print("Logout successful; session cleared.")

    # Cleanup test users
    conn.execute("DELETE FROM login_events WHERE user_id IN (?, ?)", (user['id'], hacker_user['id']))
    conn.execute("DELETE FROM users WHERE id IN (?, ?)", (user['id'], hacker_user['id']))
    conn.commit()
    conn.close()

    print("\n=== VERIFICATION COMPLETE: ALL INTEGRATION WORKFLOWS VERIFIED ===")

if __name__ == "__main__":
    run_verification()
