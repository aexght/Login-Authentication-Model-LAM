# Feature Ablation Experiment Report: V1 vs V2

## Overview
This report details the findings of a controlled feature-ablation experiment designed to determine whether the observed performance drop in the V2 model was caused by the larger, more diverse dataset itself, or by the new V2 feature representation.

We compared three setups using exactly the same model configuration (`IsolationForest(contamination=0.074, random_state=42, n_estimators=100)`):
- **V1**: V1 Dataset (12,123 rows) + V1 Feature Schema (12 features)
- **V2_V1_FEATURES**: V2 Dataset (112,131 rows) + V1 Feature Schema (12 features)
- **V2_FULL**: V2 Dataset (112,131 rows) + Full V2 Feature Schema (19 features)

Each setup was evaluated across three distinct splits: Random Event-Level, User-Level (zero overlap), and Temporal.

## Results Table

| Dataset | Features | Split | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---------|----------|-------|----------|-----------|--------|----|---------|--------|
| V1 | V1 | Random | 0.9485 | 0.6042 | 0.7030 | 0.6499 | 0.9177 | 0.7248 |
| V1 | V1 | User | 0.9514 | 0.7083 | 0.5795 | 0.6375 | 0.8821 | 0.7032 |
| V1 | V1 | Temporal | 0.9365 | 0.5792 | 0.5792 | 0.5792 | 0.9004 | 0.6423 |
| V2 | V1 | Random | 0.9346 | 0.5573 | 0.5603 | 0.5588 | 0.8712 | 0.6320 |
| V2 | V1 | User | 0.9283 | 0.5083 | 0.5550 | 0.5306 | 0.8530 | 0.6027 |
| V2 | V1 | Temporal | 0.9351 | 0.5558 | 0.5562 | 0.5560 | 0.8633 | 0.6222 |
| V2 | V2 | Random | 0.9402 | 0.5966 | 0.5905 | 0.5935 | 0.8783 | 0.6534 |
| V2 | V2 | User | 0.9400 | 0.5941 | 0.5628 | 0.5780 | 0.8663 | 0.6347 |
| V2 | V2 | Temporal | 0.9441 | 0.6270 | 0.5788 | 0.6019 | 0.8740 | 0.6593 |

## Analysis

### 1. Dataset-size effect
By comparing **V1** and **V2_V1_FEATURES**, we isolate the effect of the dataset itself. Moving to the larger and more diverse V2 dataset while keeping the features identical resulted in a significant drop in performance across all metrics (e.g., Random Split F1 dropped from 0.6499 to 0.5588; PR-AUC dropped from 0.7248 to 0.6320). 

**Conclusion:** Both V2 variants underperform V1, indicating that the larger V2 dataset contains more legitimate behavioral diversity that makes anomaly separation harder, or the synthetic anomalies are more difficult to distinguish using the original limited feature set. The primary cause of the initial performance drop is the complexity of the new dataset, not the new features.

### 2. Feature-engineering effect
By comparing **V2_V1_FEATURES** and **V2_FULL**, we isolate the effect of the new engineered features. Adding the V2 features to the V2 dataset consistently improved performance (e.g., Random Split F1 increased from 0.5588 to 0.5935, and PR-AUC from 0.6320 to 0.6534). 

**Conclusion:** The additional behavioral features (e.g., login duration deviation, specific timeframe frequencies) are providing useful, discriminative information that partially recovers the performance loss caused by the more complex dataset.

### 3. Generalization to unseen users
Comparing Random and User splits across the experiments shows how well the model generalizes to completely new users. In all three experiments, the User split slightly underperformed the Random split (e.g., V2_FULL F1 dropped from 0.5935 to 0.5780). This degradation is relatively small and expected, demonstrating that the Isolation Forest generalizes reasonably well to unseen users, though it does slightly benefit from seeing some behavior from a user during training in the random split.

### 4. Temporal generalization
Temporal generalization reveals interesting stability in the new features. While the V1 setup experienced a massive drop when evaluating on future data (Random F1 0.6499 vs Temporal F1 0.5792), the V2_FULL setup actually remained highly stable and even slightly improved in F1 on the Temporal split (Random F1 0.5935 vs Temporal F1 0.6019). The V2 feature set appears significantly more robust to temporal shifts than the original V1 features.

### 5. PR-AUC differences
PR-AUC (Precision-Recall Area Under Curve) highlights the trade-off in detecting the minority anomaly class. V1 enjoyed an "easier" environment with a PR-AUC of ~0.725. The complexity of the V2 dataset caused this to plummet to ~0.632. Fortunately, the V2_FULL features provided better separation, raising PR-AUC back to ~0.653-0.659.

### 6. Recall differences
Recall dropped sharply when transitioning to the V2 dataset. V1 achieved ~0.703 recall on the random split, meaning it caught 70% of anomalies. V2_V1_FEATURES fell to ~0.560. V2_FULL recovered this slightly to ~0.590. The fixed `contamination=0.074` parameter might be overly restrictive for the denser V2 dataset, missing more subtle anomalies that are now buried in the increased legitimate behavioral variance.

### 7. Precision differences
Similar to recall, precision fell from 0.604 (V1) to 0.557 (V2_V1_FEATURES) on the random split. The V2_FULL features successfully pushed precision back up to 0.597 (Random) and even 0.627 (Temporal). This indicates the new features are particularly effective at reducing false positives by providing a tighter, more accurate definition of normal behavior.
