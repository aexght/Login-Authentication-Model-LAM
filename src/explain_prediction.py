import os
from predict import AnomalyDetector

def explain_prediction(event_dict, detector=None):
    if detector is None:
        detector = AnomalyDetector()
        
    result = detector.predict(event_dict)
    
    explanation = []
    explanation.append(f"Login classified as {result['Risk Level']} RISK (Prediction: {result['Prediction']}).")
    explanation.append(f"Anomaly Score: {result['Anomaly Score']}")
    explanation.append("")
    explanation.append("Note: The Isolation Forest model outputs a single global anomaly score based on all features collectively.")
    explanation.append("The following observations are heuristic flags based on simple thresholds to provide potential context for the score, and do not perfectly represent the model's internal split decisions.")
    explanation.append("")
    explanation.append("Potential contributing factors:")
    
    factors = []
    
    # Simple heuristics to generate an explanation
    if event_dict.get('new_device'):
        factors.append("- Device is unfamiliar.")
    if event_dict.get('new_ip'):
        factors.append("- IP address is unfamiliar.")
    if event_dict.get('new_location'):
        factors.append("- Location is unfamiliar.")
    if event_dict.get('distance_from_usual_location_km', 0) > 30:
        factors.append(f"- Location is significantly farther ({event_dict['distance_from_usual_location_km']} km) than the user's usual location.")
    if event_dict.get('failed_attempts', 0) > 2:
        factors.append(f"- Unusually high number of failed attempts ({event_dict['failed_attempts']}).")
    if event_dict.get('time_since_last_login_minutes', 999) < 10:
        factors.append(f"- Login occurred very shortly ({event_dict['time_since_last_login_minutes']} minutes) after the previous login.")
    if event_dict.get('login_frequency_24h', 0) > 5:
        factors.append(f"- High number of logins in the last 24 hours ({event_dict['login_frequency_24h']}).")
    if event_dict.get('login_hour', 12) < 5 or event_dict.get('login_hour', 12) > 23:
        factors.append(f"- Login occurred at a potentially unusual hour ({event_dict['login_hour']}:00).")
        
    if not factors:
        factors.append("- No single extreme feature detected. The model may have flagged an unusual combination of ordinary features, or this might be a false positive.")
        
    explanation.extend(factors)
    return "\n".join(explanation)

if __name__ == "__main__":
    sample_event = {
        "login_hour": 3,
        "day_of_week": "Sunday",
        "failed_attempts": 3,
        "time_since_last_login_minutes": 4,
        "login_frequency_24h": 2,
        "new_device": True,
        "new_ip": True,
        "new_location": True,
        "device_type": "Mobile",
        "browser": "Safari",
        "session_duration_minutes": 5,
        "distance_from_usual_location_km": 350.5
    }
    
    print(explain_prediction(sample_event))
