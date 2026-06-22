# 🚗 Tata Motors Used Car Resale Price Prediction

A machine learning project to predict the resale value of used Tata Motors cars based on key vehicle attributes. The project covers end-to-end ML pipeline — from EDA to model evaluation.

---

## 📌 Problem Statement

The used car market often suffers from inconsistent and unfair pricing. Buyers overpay, sellers underprice, and dealers lack a reliable valuation system. This project builds a regression model to estimate the resale price of Tata Motors cars using historical vehicle data.

---

## 🎯 Objective

- Identify key factors affecting used car resale prices
- Perform exploratory data analysis to understand patterns and relationships
- Build and compare regression models for price prediction
- Select the best-performing model based on evaluation metrics

---

## 📂 Dataset

- **Source:** Kaggle
- **Note:** This is a synthetic dataset representing Tata Motors vehicles only
- **Size:** 3,800 rows × 16 columns
- **Target Variable:** `resale_price_lakh`

### Features Used

| Feature | Description |
|---------|-------------|
| `model` | Car model (Tiago, Nexon, Safari, etc.) |
| `variant` | Variant of the model |
| `fuel_type` | Petrol / Diesel / CNG |
| `transmission` | Manual / Automatic |
| `body_type` | Hatchback / SUV / Compact SUV, etc. |
| `engine_cc` | Engine displacement in CC |
| `power_bhp` | Engine power in BHP |
| `mileage_kmpl` | Fuel efficiency |
| `year` | Manufacturing year |
| `kilometers_driven` | Total kilometers driven |
| `owner_count` | Number of previous owners |
| `accident_history` | Whether car had accident history (0/1) |

> `car_id`, `brand`, and `ex_showroom_price_lakh` were excluded from features.

---

## 🔍 EDA Highlights

- No missing values or duplicates found
- Petrol cars dominate the dataset (~58%), followed by Diesel (~32%) and CNG (~10%)
- Tiago is the most represented model (21%), followed by Altroz, Punch, and Nexon
- Outliers in `ex_showroom_price_lakh` retained — genuine premium car values
- Weak direct relationship between ex-showroom price and resale price; resale depends more on age, km driven, and fuel type

---

## ⚙️ Preprocessing

- Label encoding for `fuel_type` and `transmission`
- One-hot encoding (`get_dummies`) for `variant`, `model`, and `body_type`
- Standard scaling applied on training data (`StandardScaler`)
- 80-20 train-test split with `random_state=42`

---

## 🤖 Models

### 1. Linear Regression (Baseline)
- Simple, interpretable baseline model
- Scaled features used

### 2. Random Forest Regressor (Final Model)
- Ensemble model capable of capturing non-linear relationships
- Hyperparameter tuning via **GridSearchCV** (5-fold cross-validation)
- Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`

---

## 📊 Results

| Model | R² Score | RMSE (₹ Lakh) | MSE |
|-------|----------|----------------|-----|
| Linear Regression | 0.9425 | 1.163 | 1.352 |
| **Random Forest** | **0.9949** | **0.346** | **0.119** |

**Random Forest** was selected as the final model due to significantly lower error and better ability to capture complex, non-linear patterns in the data.

---

## 📈 Feature Importance

Top features influencing resale price (from Random Forest):
- Ex-showroom price grouping via model/variant
- Manufacturing year
- Kilometers driven
- Engine CC and Power BHP
- Owner count and accident history

*(See notebook for feature importance bar chart)*

---

## ⚠️ Limitations

- Dataset is **synthetic** and sourced from Kaggle — real-world performance may differ
- Only **Tata Motors** vehicles are covered — model is not generalized across brands
- High R² values are expected given the structured nature of synthetic data
- Fuel type encoded ordinally — may not perfectly represent category relationships

---

## 🛠️ Libraries Used

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 📁 Project Structure

```
├── car_price_prediction.ipynb   # Main notebook
├── car_price_prediction_.csv    # Dataset
└── README.md                    # Project documentation
```

---

## 👤 Author

**Aankit**
Advanced Certification in Full Stack Data Science & AI — AlmaBetter
[LinkedIn Profile](#) | [GitHub Profile](#)
