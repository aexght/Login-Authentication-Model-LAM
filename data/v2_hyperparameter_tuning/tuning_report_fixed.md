# Hyperparameter Tuning Report for V2_FULL

## Best Tuned Configurations (Validation Set Selected)
**Random Split:**
- contamination=0.05
- max_samples=2048
- n_estimators=200

**User-level Split:**
- contamination=0.06
- max_samples=4096
- n_estimators=200

**Temporal Split:**
- contamination=0.06
- max_samples=4096
- n_estimators=100

**Global Configuration:**
- contamination=0.06
- max_samples=4096.0
- n_estimators=200

## Final Evaluation Results (Test Set)

### Random Split
| Metric | Baseline | Split-Tuned | Global-Tuned | Δ (Split-Tuned vs Base) | Δ (Global-Tuned vs Base) |
|---|---|---|---|---|---|
| F1 | 0.6135 | 0.6337 | 0.6324 | +0.0202 | +0.0188 |
| Precision | 0.6230 | 0.7944 | 0.7033 | +0.1714 | +0.0803 |
| Recall | 0.6044 | 0.5271 | 0.5745 | -0.0773 | -0.0299 |
| PR-AUC | 0.6807 | 0.6843 | 0.6821 | +0.0036 | +0.0013 |
| ROC-AUC | 0.8884 | 0.8920 | 0.8906 | +0.0036 | +0.0022 |

### User-level Split
| Metric | Baseline | Split-Tuned | Global-Tuned | Δ (Split-Tuned vs Base) | Δ (Global-Tuned vs Base) |
|---|---|---|---|---|---|
| F1 | 0.6063 | 0.6382 | 0.6382 | +0.0318 | +0.0318 |
| Precision | 0.6189 | 0.7219 | 0.7219 | +0.1030 | +0.1030 |
| Recall | 0.5943 | 0.5719 | 0.5719 | -0.0224 | -0.0224 |
| PR-AUC | 0.6710 | 0.6832 | 0.6832 | +0.0122 | +0.0122 |
| ROC-AUC | 0.8796 | 0.8882 | 0.8882 | +0.0085 | +0.0085 |

### Temporal Split
| Metric | Baseline | Split-Tuned | Global-Tuned | Δ (Split-Tuned vs Base) | Δ (Global-Tuned vs Base) |
|---|---|---|---|---|---|
| F1 | 0.6145 | 0.6545 | 0.6524 | +0.0400 | +0.0379 |
| Precision | 0.6386 | 0.7681 | 0.7656 | +0.1295 | +0.1270 |
| Recall | 0.5922 | 0.5702 | 0.5684 | -0.0220 | -0.0238 |
| PR-AUC | 0.6694 | 0.6995 | 0.6979 | +0.0302 | +0.0285 |
| ROC-AUC | 0.8817 | 0.8909 | 0.8897 | +0.0092 | +0.0080 |

## Anomaly Type Analysis (Global Tuned Model on Temporal Test)
| Type | Count | Detected | Rate |
|---|---|---|---|
| unusual_hour | 253 | 155 | 61.26% |
| unusual_day | 278 | 149 | 53.60% |
| short_gap | 279 | 168 | 60.22% |
| high_frequency | 271 | 271 | 100.00% |
| high_failed_attempts | 267 | 185 | 69.29% |
| sudden_failed_attempts_increase | 269 | 185 | 68.77% |
| new_device | 246 | 209 | 84.96% |
| unusual_device_browser_combo | 276 | 155 | 56.16% |
| device_switch | 236 | 126 | 53.39% |
| new_ip | 271 | 179 | 66.05% |
| new_location | 269 | 194 | 72.12% |
| large_distance | 270 | 251 | 92.96% |
| impossible_travel | 248 | 248 | 100.00% |
| session_anomaly | 251 | 177 | 70.52% |
| single-signal | 1638 | 931 | 56.84% |