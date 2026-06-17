# 🧠 Social Media & Depression Prediction — Machine Learning Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)
![Type](https://img.shields.io/badge/Type-Binary%20Classification-purple?style=flat)

---

## 📌 Project Overview

This project builds a **supervised machine learning classification model** to predict whether an individual is likely to experience **depression** based on their demographic information, social media usage patterns, lifestyle habits, and psychological factors.

The increasing use of social media among students and young adults has raised concerns about its impact on mental health. This model aims to identify key risk factors associated with depression and assist in early detection of mental health risks.

---

## 🎯 Objective

- Build a binary classification model to predict depression (`depression_label`: 0 or 1)
- Identify the most influential features driving depression risk
- Handle class imbalance using resampling techniques
- Evaluate model performance using multiple metrics

---

## 📂 Project Structure

```
├── Stress.ipynb              # Main Jupyter Notebook (full pipeline)
├── mental_health.csv         # Dataset
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## 📊 Dataset Information

| Property | Detail |
|---|---|
| Source | [Social Media Impact on Mental Health — Kaggle](https://www.kaggle.com/datasets/sunil123kumar/social-media-impact-on-mental-health) by Sunil Kumar |
| Total Records | 1,200 |
| Total Features | 13 (12 input + 1 target) |
| Missing Values | 0 |
| Duplicate Records | 0 |
| Target Variable | `depression_label` (0 = Not Depressed, 1 = Depressed) |

### Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the individual (13–19 years) |
| `gender` | Categorical | Gender (Male / Female) |
| `daily_social_media_hours` | Numerical | Average hours spent on social media per day |
| `platform_usage` | Categorical | Most used platform (Instagram / TikTok / Both) |
| `sleep_hours` | Numerical | Average sleep hours per day |
| `screen_time_before_sleep` | Numerical | Screen time before bedtime (hours) |
| `academic_performance` | Numerical | Academic performance score |
| `physical_activity` | Numerical | Level of physical activity |
| `social_interaction_level` | Categorical | Frequency of real-world social interactions |
| `stress_level` | Numerical | Measured stress level |
| `anxiety_level` | Numerical | Measured anxiety level |
| `addiction_level` | Numerical | Degree of social media addiction |
| `depression_label` | Target | 0 = Not Depressed, 1 = Depressed |

---

## ⚙️ ML Pipeline

```
Raw Data
   ↓
Exploratory Data Analysis (EDA)
   ↓
Label Encoding (Gender, Platform Usage, Social Interaction Level)
   ↓
Train-Test Split (80-20, Stratified)
   ↓
SMOTE — Class Imbalance Handling (applied on train set only)
   ↓
Model Training
   ├── Logistic Regression
   └── Random Forest + GridSearchCV
   ↓
Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
   ↓
Feature Importance Analysis
```

---

## 📈 Model Results

### Logistic Regression

| Metric | Score |
|---|---|
| Accuracy | 96.67% |
| Precision | 41.67% |
| Recall | 83.33% |
| F1-Score | 55.56% |

### Random Forest (Default)

| Metric | Score |
|---|---|
| Accuracy | 99.58% |
| Precision | 100% |
| Recall | 83.33% |
| F1-Score | 90.91% |

### ✅ Random Forest + GridSearchCV (Best Model)

| Metric | Score |
|---|---|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1-Score | 100% |
| CV F1-Score | 99.95% |

> **Best Parameters:** `n_estimators=50`, `random_state=42`

---

## 🔑 Feature Importance (Top Features)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Sleep Hours | 28.48% |
| 2 | Stress Level | 25.94% |
| 3 | Daily Social Media Hours | 19.47% |
| 4 | Anxiety Level | 17.06% |
| 5 | Academic Performance | 2.70% |
| 6 | Social Interaction Level | 1.81% |
| 7 | Screen Time Before Sleep | 1.48% |
| 8 | Physical Activity | 1.17% |
| 9 | Addiction Level | 0.97% |
| 10 | Age | 0.54% |
| 11 | Platform Usage | 0.31% |
| 12 | Gender | 0.08% |

> The top 4 features together account for **~91%** of the model's predictive power.

---

## 🔍 Key EDA Insights

- **Sleep Hours vs Screen Time:** No strong correlation observed — screen time alone does not directly determine sleep duration
- **Stress vs Gender:** Male participants show marginally higher average stress than females, but the difference is minimal
- **Addiction by Platform:** Instagram users show the highest addiction levels, closely followed by TikTok users
- **Stress by Age:** Stress levels remain relatively stable across ages 13–19; age alone is not a strong stress predictor
- **Correlation Heatmap:** All features show weak linear correlations (~0.17 max), suggesting depression is influenced by multiple interacting factors

---

## ⚠️ Limitations

1. **Severe Class Imbalance:** Only 31 positive (depressed) cases out of 1,200 records (2.6%), leaving just 6 positive cases in the test set. Perfect scores on 6 samples are not statistically robust.

2. **Small Minority Class:** With only 31 genuine positive samples, evaluation metrics are not reliable indicators of real-world performance.

3. **Likely Synthetic Dataset:** Near-zero feature correlations (~0.17) combined with perfect model performance strongly suggest this dataset is programmatically generated rather than collected from real clinical observations.

4. **Overfitting Risk:** Perfect test scores without external validation indicate possible memorization of dataset patterns rather than true generalization.

5. **Limited Scope:** Dataset covers only ages 13–19 with 1,200 records, limiting demographic diversity and generalizability.

6. **Binary Classification:** Depression is a complex condition — reducing it to a binary label oversimplifies clinical reality. Severity-based classification (PHQ-9 scale) would be more meaningful.

7. **Single Train-Test Split:** Full Stratified K-Fold cross-validation across the complete dataset would provide more reliable performance estimates.

---

## 🚀 Future Scope

- Collect real-world, clinically validated data with balanced class distribution
- Apply Stratified K-Fold cross-validation across the full dataset
- Explore multi-class severity classification (Mild / Moderate / Severe)
- Validate model on an independent external dataset
- Integrate deep learning approaches (LSTM for temporal social media patterns)
- Deploy as a web application using Flask or Streamlit

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` & `seaborn` | Data visualization |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `imbalanced-learn` | SMOTE for class balancing |

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/depression-prediction-ml.git
cd depression-prediction-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook Stress.ipynb
```

Or open directly in **Google Colab**:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 👤 Author

**Aankit**
- 📧 anks.sha26@gmail.com
- 💼 https://www.linkedin.com/in/aankit-sharma-373784409/
- 🐙 https://github.com/aankitsharma

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
