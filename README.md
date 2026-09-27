# AI-Based Login Authentication & Behavioral Anomaly Detection

## 1. Project Overview
This project integrates an unsupervised machine learning behavioral anomaly detection model into a complete Flask-based login authentication application.

**Important Definition:**
The ML model is NOT an "AI that determines whether a login is legitimate." Rather, it serves as an advisory **behavioral anomaly detection layer that assigns a risk signal to authenticated login events**.

**Note:** This is a research and portfolio prototype system trained and evaluated on synthetic login behavior data. It is not production-ready. Anomaly detection is an additional risk signal, not a replacement for credential authentication.

## 2. Problem Statement
Traditional authentication relies solely on credentials (username/password), which are vulnerable to theft. This project aims to demonstrate how analyzing user behavioral patterns (time of day, location, device, velocity) can provide a secondary layer of security to flag suspicious activities even when credentials are correct.

## 3. Architecture
```text
[User Web Client]
      ↓ (HTTP Login Submission)
[app.py (Flask Router & Session State)]
      ↓ (Credentials)
[auth.py (bcrypt Hash Verification & DB Tracking)]
      ↓ (Telemetry: IP, User-Agent, UserID)
[src/feature_extractor.py]
      ↓ (Extracts & Validates Domain Features)
[src/predict.py (AnomalyDetector)]
      ↓ (Loads models/login_anomaly_model_v2.pkl)
[Isolation Forest Risk Scoring]
      ↓ (LOW / MEDIUM / HIGH Risk Signal)
[app.py Security Handling & Dashboard]
```

## 4. Authentication Flow
1. **Registration:** Users register with a unique username, email, and password.
2. **Credential Authentication:** Primary authentication checks the supplied password against the bcrypt hash stored in the SQLite database.
3. **Telemetry Collection:** On successful credential verification, request telemetry (IP, User-Agent, simulated location) is collected.
4. **Behavioral Analysis:** The telemetry is passed to the ML pipeline.
5. **Session Allocation:** If the risk is acceptable (or requires MFA in a real system), the session is established.

## 5. Behavioral Anomaly Detection
The system employs an unsupervised anomaly detection layer that assigns a risk signal to authenticated login events. The model analyzes patterns rather than acting as a binary access control, allowing the system to flag unusual behaviors for further verification or auditing.

## 6. Dataset Description
The training data is synthetic and contains no real people, IP addresses, emails, or GPS coordinates. 

**V2 Synthetic Dataset Properties:**
- 500 synthetic users
- 112,131 login events
- 7.38% anomalies

The dataset includes normal logins and various simulated anomalies like brute-force attempts, credential stuffing, impossible travel, and atypical devices.

## 7. Feature Engineering
The model transforms raw login metadata into 20 domain features (which expand further after one-hot encoding). 
Features include:
- `login_hour` and `usual_hour_deviation`
- Geographical features (simulated location distance)
- Network features (IP reputation, ASN)
- Device features (Browser, OS, device type)
- Behavioral velocity (login frequency, failures before success)

## 8. Isolation Forest Explanation
The model uses an Isolation Forest, an unsupervised anomaly detection algorithm. Instead of profiling normal behavior to find deviations, it explicitly isolates anomalies by randomly partitioning the dataset. Anomalous points are easier to isolate and thus have shorter path lengths in the underlying decision trees.

## 9. V1 vs V2 Experimentation
The project underwent rigorous experimentation comparing a V1 dataset (12 features) against an expanded V2 dataset (20 features). V2 introduced richer features like `usual_hour_deviation` and better modeling of complex attack patterns (impossible travel, credential stuffing).

## 10. Hyperparameter Tuning
Hyperparameter tuning was conducted using validation-based grid search over completely untouched test sets. Splits included Random, User-level, and Temporal holdouts to ensure generalizability and guard against data leakage.

## 11. Final Model Configuration
The final chosen model uses the V2 dataset with the following global best configuration for the Isolation Forest:
- `contamination = 0.06`
- `max_samples = 4096`
- `n_estimators = 200`
- `random_state = 42`

## 12. Evaluation Results
The final V2 Global configuration achieved the following performance on strict holdout test sets:

| Split | F1-Score | PR-AUC | ROC-AUC |
|---|---|---|---|
| **Random Split** | 0.6324 | 0.6821 | 0.8906 |
| **User-level Split** | 0.6382 | 0.6832 | 0.8882 |
| **Temporal Split** | 0.6524 | 0.6979 | 0.8897 |

## 13. Flask Integration
The application uses Flask to serve the web interface. The final ML model (`models/login_anomaly_model_v2.pkl`) is loaded into memory exactly once at application startup by `predict.py` to minimize latency. Risk explanations are presented in the Flask dashboard based on extracted features and the final prediction.

## 14. Security Controls
- **Password Protection:** bcrypt with per-user salt.
- **Session Security:** Flask filesystem-based sessions, HTTPOnly cookies.
- **Environment Configuration:** Flask secret keys loaded from environment variables (fallback for dev only).
- **Simulated Location:** The location distance is simulated in the demo based on predefined IP mappings.

## 15. Privacy Considerations
As the dataset is entirely synthetic, there are no real user data privacy concerns in this repository. In a real-world deployment, extensive privacy controls, data anonymization, and telemetry retention policies would be mandatory.

## 16. Installation
```bash
# 1. Clone the repository
git clone https://github.com/aexght/Login-Authentication-Model-LAM.git
cd LoginAuthenticationModel

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 17. Running the Application
```bash
# Set Flask environment variables
export FLASK_APP=app.py
export FLASK_ENV=development  # Use 'production' for real deployments

# Run the server
flask run
```
Access the application at `http://127.0.0.1:5000/`.

## 18. Running Tests
The project includes a comprehensive test suite.
```bash
python -m pytest tests/
```
You can also run the integration script to verify end-to-end functionality:
```bash
python verify_integration.py
```

## 19. Project Structure
```
login-authentication-model/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── auth.py
├── database.py
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   ├── feature_extractor.py
│   ├── evaluate_model.py
│   └── explain_prediction.py
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   └── security_events.html
├── static/
│   └── style.css
├── tests/
│   ├── test_model.py
│   └── test_auth.py
├── data/
│   ├── New_Dataset/
│   ├── v1_vs_v2-test result/
│   ├── v2_feature_ablation/
│   ├── v2_hyperparameter_tuning/
│   └── final_model_comparison/
└── models/
    ├── login_anomaly_model.pkl
    └── login_anomaly_model_v2.pkl
```

## 20. Limitations
- The model is an additional risk layer and is not a production-grade authentication gate by itself.
- Training data is completely synthetic and may not capture all real-world complexities.
- Location tracking and distances are simulated in the demo.
- No actual Multi-Factor Authentication (MFA) step is implemented; high-risk logins simply trigger a warning.

## 21. Future Improvements
- Implement real IP-to-Geo integration (e.g., MaxMind).
- Add genuine Multi-Factor Authentication (MFA) enforcement for high-risk logins.
- Shift to supervised or semi-supervised learning if real labeled incident data becomes available.
- Deploy a streaming data pipeline for continuous model updates.
