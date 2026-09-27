import os
import sys
import unittest
from flask import session

# Ensure app runs in testing mode
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app
from database import init_db, get_db_connection

class TestAuthApp(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # We can use the existing sqlite database but ideally we should use a test DB
        # For simplicity, we just initialize the real one and clean up our test users
        init_db()
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        
    def setUp(self):
        self.client = app.test_client()
        self._cleanup_test_user()
        
    def tearDown(self):
        self._cleanup_test_user()
        
    def _cleanup_test_user(self):
        conn = get_db_connection()
        user = conn.execute("SELECT id FROM users WHERE username IN ('testuser', 'hacker_test')").fetchall()
        for u in user:
            conn.execute("DELETE FROM failed_attempts WHERE user_id = ?", (u['id'],))
            conn.execute("DELETE FROM login_events WHERE user_id = ?", (u['id'],))
            conn.execute("DELETE FROM users WHERE id = ?", (u['id'],))
        conn.commit()
        conn.close()

    def test_01_registration(self):
        res = self.client.post('/register', data={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=True)
        self.assertIn(b'Registration successful', res.data)
        
    def test_02_duplicate_registration(self):
        self.client.post('/register', data={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        res = self.client.post('/register', data={
            'username': 'testuser',
            'email': 'test2@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=True)
        self.assertIn(b'already exists', res.data)

    def test_03_login_success(self):
        self.client.post('/register', data={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        res = self.client.post('/login', data={
            'username': 'testuser',
            'password': 'password123'
        }, follow_redirects=True)
        self.assertIn(b'Welcome, testuser', res.data)
        self.assertIn(b'SUCCESS', res.data)

    def test_04_login_failure(self):
        self.client.post('/register', data={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        res = self.client.post('/login', data={
            'username': 'testuser',
            'password': 'wrongpassword'
        }, follow_redirects=True)
        self.assertIn(b'Invalid username or password', res.data)

    def test_05_suspicious_login(self):
        self.client.post('/register', data={
            'username': 'hacker_test',
            'email': 'hacker@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        # Our app has a hardcoded heuristic payload for "hacker_test"
        res = self.client.post('/login', data={
            'username': 'hacker_test',
            'password': 'password123'
        }, follow_redirects=True)
        
        self.assertIn(b'Welcome, hacker_test', res.data)
        # Should be classified as HIGH risk
        self.assertIn(b'HIGH', res.data)
        
    def test_06_dashboard_access_control(self):
        res = self.client.get('/dashboard', follow_redirects=True)
        self.assertIn(b'Secure Login', res.data)
        
    def test_07_logout(self):
        self.client.post('/register', data={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        self.client.post('/login', data={
            'username': 'testuser',
            'password': 'password123'
        })
        res = self.client.get('/logout', follow_redirects=True)
        self.assertIn(b'logged out', res.data)
        
    def test_08_failed_login_tracking(self):
        self.client.post('/register', data={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        # Fail 3 times
        for _ in range(3):
            self.client.post('/login', data={
                'username': 'testuser',
                'password': 'wrongpassword'
            })
            
        # Login successfully
        res = self.client.post('/login', data={
            'username': 'testuser',
            'password': 'password123'
        }, follow_redirects=True)
        
        # Check if failed attempts are reset
        conn = get_db_connection()
        user = conn.execute("SELECT id FROM users WHERE username = 'testuser'").fetchone()
        failed = conn.execute("SELECT count FROM failed_attempts WHERE user_id = ?", (user['id'],)).fetchone()
        conn.close()
        
        self.assertIsNone(failed)

    def test_09_security_events_inspection(self):
        self.client.post('/register', data={
            'username': 'hacker_test',
            'email': 'hacker@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        self.client.post('/login', data={
            'username': 'hacker_test',
            'password': 'password123'
        }, follow_redirects=True)
        
        res = self.client.get('/security-events', follow_redirects=True)
        self.assertIn(b'SECURITY ALERT: Suspicious login detected', res.data)
        self.assertIn(b'HIGH', res.data)

    def test_10_feature_extraction(self):
        from src.feature_extractor import FeatureExtractor, V2_DOMAIN_FEATURES
        extractor = FeatureExtractor(get_db_connection)
        
        # Test basic parsing
        dt, br = extractor.parse_user_agent('Mozilla/5.0 (iPhone; CPU iPhone OS 14_0) Safari/604.1')
        self.assertEqual(dt, 'Mobile')
        self.assertEqual(br, 'Safari')
        
        # Test extraction logic for a new user establishing baseline
        features = extractor.extract_features(999, '192.168.1.1', 'Mozilla', 2)
        self.assertFalse(features['new_ip'])
        self.assertFalse(features['new_device'])
        self.assertEqual(features['failed_attempts'], 2)
        self.assertEqual(features['login_frequency_24h'], 0)
        self.assertEqual(features['login_frequency_1h'], 0)
        self.assertEqual(features['login_frequency_6h'], 0)
        self.assertEqual(features['login_frequency_7d'], 0)
        self.assertIn('_meta_ip_hash', features)
        self.assertIn('_meta_device_hash', features)
        
        # Confirm all 20 V2 domain features are present
        for feat in V2_DOMAIN_FEATURES:
            self.assertIn(feat, features)

    def test_11_v2_model_loaded_once_at_startup(self):
        from app import ml_detector
        self.assertIsNotNone(ml_detector)
        self.assertEqual(ml_detector.model_version, "v2")
        self.assertIn("login_anomaly_model_v2.pkl", ml_detector.model_path)
        
    def test_12_normal_login_invokes_v2_model(self):
        self.client.post('/register', data={
            'username': 'normal_v2_user',
            'email': 'normal_v2@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        res = self.client.post('/login', data={
            'username': 'normal_v2_user',
            'password': 'password123'
        }, follow_redirects=True)
        self.assertIn(b'Welcome, normal_v2_user', res.data)
        
        # Verify event recorded in DB with anomaly score
        conn = get_db_connection()
        user = conn.execute("SELECT id FROM users WHERE username = 'normal_v2_user'").fetchone()
        event = conn.execute("SELECT * FROM login_events WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user['id'],)).fetchone()
        conn.close()
        
        self.assertIsNotNone(event)
        self.assertIsNotNone(event['risk_score'])
        self.assertIn(event['risk_level'], ['LOW', 'MEDIUM', 'HIGH'])
        
        # Cleanup
        conn = get_db_connection()
        conn.execute("DELETE FROM login_events WHERE user_id = ?", (user['id'],))
        conn.execute("DELETE FROM users WHERE id = ?", (user['id'],))
        conn.commit()
        conn.close()

    def test_13_suspicious_login_invokes_v2_and_records_high_risk(self):
        self.client.post('/register', data={
            'username': 'hacker_test',
            'email': 'hacker_test_v2@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        res = self.client.post('/login', data={
            'username': 'hacker_test',
            'password': 'password123'
        }, follow_redirects=True)
        
        self.assertIn(b'SECURITY WARNING: Highly suspicious login detected', res.data)
        
        conn = get_db_connection()
        user = conn.execute("SELECT id FROM users WHERE username = 'hacker_test'").fetchone()
        event = conn.execute("SELECT * FROM login_events WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user['id'],)).fetchone()
        conn.close()
        
        self.assertIsNotNone(event)
        self.assertEqual(event['risk_level'], 'HIGH')
        self.assertGreater(event['risk_score'], 0.0)

if __name__ == '__main__':
    unittest.main()

