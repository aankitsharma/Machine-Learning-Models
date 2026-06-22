# 🤖 Machine Learning Projects

A collection of end-to-end machine learning projects covering real-world problems across different domains — healthcare, automotive, and finance.

---

## 📁 Projects

| # | Project | Domain | Techniques | Status |
|---|---------|--------|------------|--------|
| 1 | [Depression Risk Classification](#1-depression-risk-classification) | Healthcare | Classification, Imbalanced Data | ✅ Complete |
| 2 | [Tata Car Resale Price Prediction](#2-tata-car-resale-price-prediction) | Automotive | Regression, GridSearchCV | ✅ Complete |
| 3 | More projects coming soon... | - | - | 🔄 In Progress |

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

## 🛠️ Libraries & Tools

```
Python | pandas | numpy | scikit-learn | matplotlib | seaborn
```

---

## 👤 About

**Aankit**
Advanced Certification in Full Stack Data Science & AI — AlmaBetter
12+ years background in Financial Services | Transitioning to Data Science

[LinkedIn](#) | [GitHub](#)
