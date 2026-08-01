# 🏭 TATA Steel Machine Failure Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![F1 Score](https://img.shields.io/badge/F1--Score-99.32%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Problem Statement
In steel manufacturing, unplanned machine failures lead to significant production
downtime and increased maintenance costs. This project builds a predictive 
maintenance model to anticipate machine failures before they occur, enabling 
TATA Steel to shift from reactive to proactive maintenance.

---

## 📂 Dataset
| File | Records | Description |
|---|---|---|
| train.csv | 1,36,429 | Training data with labels |
| test.csv | 90,954 | Test data for predictions |

**Class Imbalance:** Only 1.6% failure rate — handled using SMOTE

---

## 🔧 Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, SHAP, Matplotlib, Seaborn
- **Environment:** Google Colab

---

## 🚀 Approach

### 1. EDA
- Analyzed distributions of all sensor features
- Identified class imbalance (98.4% vs 1.6%)
- Correlation heatmap analysis

### 2. Feature Engineering
| Feature | Formula | Reason |
|---|---|---|
| Temp_diff | Process Temp − Air Temp | Detects overheating |
| Power | Torque × RPM | Captures machine load |

### 3. Class Imbalance Handling
- Applied **SMOTE** on training data only
- Balanced minority (failure) class for better learning

### 4. Models Trained
- Logistic Regression (Baseline)
- Random Forest
- XGBoost (Best Model)

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 81.47% | 82.90% | 79.28% | 81.05% | 81.48% |
| Random Forest | 93.31% | 94.69% | 91.76% | 93.20% | 93.31% |
| **XGBoost** | **99.32%** | **99.45%** | **99.18%** | **99.32%** | **99.32%** |

### XGBoost Confusion Matrix
True Negative → 26,733 ✅

False Positive → 146 ❌

False Negative → 221 ❌

True Positive → 26,613 ✅

---

## 🔍 SHAP Analysis — Feature Importance

Top features identified by SHAP:
1. **Temp_diff** — Most important (engineered feature ✅)
2. Process_temperature_K
3. Rotational_speed_rpm
4. Air_temperature_K
5. Torque_Nm

> Temp_diff topping the SHAP chart validates our feature engineering approach.

---

## 💼 Business Impact
- Proactive maintenance — failures predicted before they occur
- Reduced unplanned downtime
- Lower maintenance costs
- Improved product quality and operational efficiency

---

## 👤 Author
**Aankit Sharma**  
Data Science & Analytics | BFSI Domain  
[LinkedIn](https://linkedin.com/in/aankitsharma) | [GitHub](https://github.com/aankitsharma)
