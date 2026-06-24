import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score
import pickle
import os

# ============================================================
# STEP 1 - LOAD DATA
# ============================================================
def load_data():
    df = pd.read_csv('data/car_price_prediction.csv')
    print(f'Data Loaded! Shape:{df.shape}')

    return df

# ============================================================
# STEP 2 - PREPROCESS DATA
# ============================================================

def preprocess_data(df):
    df['transmission'] = df['transmission'].map({'Manual': 0, 'Automatic': 1})
    df = pd.get_dummies(df,columns=['fuel_type', 'variant', 'model', 'body_type'], dtype=int)
    df = df.drop(['car_id', 'brand'], axis=1)
    print(f"Preprocessing done! Total columns: {df.shape[1]}")
    return df

# ============================================================
# STEP 3 - SPLIT DATA
# ============================================================

def split_data(df):
    X =df.drop('resale_price_lakh',axis = 1)
    y = df['resale_price_lakh']
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.20)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

#===============================================================
# STEP 4 - Train Model
#===============================================================
def train_model(X_train,y_train):
    print("\nTraining Random Forest with GridSearchCV...")
    print("Please wait...\n")
    rf_param = {'n_estimators': [50,100,200]
                ,'max_depth':[5,10,15],
                'min_samples_split':[5,10,15],
                'min_samples_leaf':[5,10,15]}
    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf,param_grid=rf_param,cv=5,verbose=1,n_jobs =-1)
    rf_grid.fit(X_train,y_train)
    print(f"\nBest Parameters: {rf_grid.best_params_}")
    return rf_grid.best_estimator_

# ============================================================
# STEP 6 - EVALUATE MODEL
# ============================================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse    = mean_squared_error(y_test, y_pred)
    rmse   = np.sqrt(mse)
    r2     = r2_score(y_test, y_pred)
 
    print("\n--- Model Evaluation ---")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f} lakh")
    print(f"R²   : {r2:.4f}")
 
# ============================================================
# STEP 7 - SAVE MODEL
# ============================================================
def save_model(model, X_train):
    os.makedirs('model', exist_ok=True)

    with open('model/rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/feature_columns.pkl', 'wb') as f:
        pickle.dump(list(X_train.columns), f)

    print("\nModel saved in /model folder!")
    print("Files: rf_model.pkl | feature_columns.pkl")
 
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("   CAR PRICE PREDICTION - MODEL TRAINING")
    print("=" * 50)

    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, X_train)

    print("\n" + "=" * 50)
    print("   DONE! Now run: streamlit run app.py")
    print("=" * 50)