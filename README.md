# 🤖 Machine Learning Projects
A collection of end-to-end machine learning projects covering real-world problems across different domains — healthcare, automotive, manufacturing, and finance.

---

## 📁 Projects

| # | Project | Domain | Techniques | Status |
|---|---------|--------|------------|--------|
| 1 | [Depression Risk Classification](#1-depression-risk-classification) | Healthcare | Classification, Imbalanced Data | ✅ Complete |
| 2 | [Tata Car Resale Price Prediction](#2-tata-car-resale-price-prediction) | Automotive | Regression, GridSearchCV | ✅ Complete |
| 3 | [Tata Steel Machine Failure Prediction](#3-tata-steel-machine-failure-prediction) | Manufacturing | Classification, XGBoost, SHAP | ✅ Complete |
| 4 | [Yes Bank Stock Price Prediction](#4-yes-bank-stock-price-prediction) | Finance | Regression, SHAP | ✅ Complete |
| 5 | More projects coming soon... | - | - | 🔄 In Progress |

---

## 1. Depression Risk Classification

**📌 Problem:** Predict whether a person is at risk of depression based on lifestyle, demographic, and behavioral features.

**📂 Dataset:** Kaggle — imbalanced binary classification dataset

**🔧 Techniques Used:**
- Exploratory Data Analysis (EDA)
- Handling class imbalance
- Multiple classification models compared
- Evaluation using Precision, Recall, F1-Score, ROC-AUC

**📊 Key Results:**

| Model | Accuracy | F1-Score (Minority) |
|-------|----------|----------------------|
| Best Model | - | - |

**⚠️ Limitations:**
- Synthetic/Kaggle dataset — may not reflect real clinical data
- Class imbalance present in original dataset

📓 [View Notebook](#) | 📁 [View Folder](#)

---

## 2. Tata Car Resale Price Prediction

**📌 Problem:** Predict the resale value of used Tata Motors cars based on vehicle attributes such as model, fuel type, kilometers driven, and ownership history.

**📂 Dataset:** Kaggle — synthetic dataset, 3,800 records, Tata Motors vehicles only

**🔧 Techniques Used:**
- Exploratory Data Analysis (EDA)
- Label Encoding & One-Hot Encoding
- Standard Scaling
- Linear Regression (baseline) vs Random Forest (final)
- Hyperparameter tuning via GridSearchCV

**📊 Key Results:**

| Model | R² Score | RMSE (₹ Lakh) |
|-------|----------|----------------|
| Linear Regression | 0.9425 | 1.163 |
| **Random Forest** | **0.9949** | **0.346** |

**⚠️ Limitations:**
- Synthetic dataset — high R² expected due to structured data
- Only Tata Motors vehicles covered

📓 [View Notebook](#) | 📁 [View Folder](#)

---

## 3. Tata Steel Machine Failure Prediction

**📌 Problem:** Predict machine/equipment failure in advance using sensor data, enabling predictive maintenance and reducing unplanned downtime.

**📂 Dataset:** Sensor-based dataset — binary classification (failure vs. no failure)

**🔧 Techniques Used:**
- Exploratory Data Analysis (EDA) on sensor readings
- Feature engineering on sensor/time-series signals
- Binary Classification using XGBoost
- Model interpretability using SHAP (feature importance & failure drivers)
- Hyperparameter tuning via GridSearchCV

**📊 Key Results:**

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| **XGBoost** | - | - | - |

**🔍 SHAP Insights:**
- Identified key sensor readings driving failure predictions, supporting proactive maintenance decisions

**⚠️ Limitations:**
- Model performance tied to sensor data quality and labeling accuracy
- Real-world deployment would require live sensor stream integration

📓 [View Notebook](#) | 📁 [View Folder](#)

---

## 4. Yes Bank Stock Price Prediction

**📌 Problem:** Predict Yes Bank's stock closing price based on historical stock market data (open, high, low, and other price features).

**📂 Dataset:** Historical Yes Bank stock price data

**🔧 Techniques Used:**
- Exploratory Data Analysis (EDA) on stock price trends
- Feature scaling and preprocessing
- Linear Regression (final model)
- Model interpretability using SHAP

**📊 Key Results:**

| Model | R² Score |
|-------|----------|
| **Linear Regression** | **0.9907** |

**🔍 SHAP Insights:**
- "Low" price identified as the dominant feature driving closing price predictions

**⚠️ Limitations:**
- High R² driven by strong correlation between daily price features (Open/High/Low/Close)
- Model reflects historical patterns, not suited for real-time trading without further validation

📓 [View Notebook](#) | 📁 [View Folder](#)

---

## 🛠️ Libraries & Tools

```
Python | pandas | numpy | scikit-learn | xgboost | shap | matplotlib | seaborn
```

---

## 👤 About

**Aankit**
Advanced Certification in Full Stack Data Science & AI — AlmaBetter
12+ years background in Financial Services | Transitioning to Data Science

[LinkedIn](#) | [GitHub](#)
