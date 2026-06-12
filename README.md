# Ad Performance Insight Engine

A cross-platform ML system to detect ROAS performance gaps 
and optimize ad budget allocation across Google Ads, 
Meta Ads, and TikTok Ads.

## Problem
Advertisers spend budgets across multiple platforms without 
a systematic way to identify which campaigns are worth 
scaling and which are wasting money.

## Solution
An end-to-end ML pipeline that:
- Predicts high vs low ROAS campaigns (ROC-AUC: 0.8777)
- Identifies anomalous campaigns automatically
- Surfaces actionable budget optimization insights

## Tech Stack
- Python, Pandas, Scikit-learn, XGBoost
- SHAP, Optuna, Isolation Forest
- Google Colab

## Pipeline
1. Data Loading & EDA
2. Feature Engineering
3. Model Selection (5 models benchmarked)
4. Logistic Regression (winner, ROC-AUC: 0.8777)
5. XGBoost + Optuna Tuning
6. SHAP Feature Importance
7. Anomaly Detection (Z-score + Isolation Forest)

## Key Results
| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.8777 ✅ |
| XGBoost (tuned) | 0.8633 |
| Random Forest | 0.8497 |
| Decision Tree | 0.7069 |
| KNN | 0.6256 |

## Key Findings
- TikTok Ads showed highest avg ROAS (9.54) vs Google (4.11)
- CPA identified as top ROAS predictor via SHAP
- 19 high-confidence anomalies flagged (both Z-score + Isolation Forest)

## Dataset
[Global Ads Performance Dataset](https://www.kaggle.com/datasets/nudratabbas/global-ads-performance-google-meta-tiktok)

## Author
Justin Cho | [LinkedIn](https://www.linkedin.com/in/soungbin-cho)
