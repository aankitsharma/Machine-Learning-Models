import streamlit as st
import pandas as pd
import pickle

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Tata Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ============================================================
# LOAD MODEL FILES
# ============================================================
@st.cache_resource
def load_model():
    with open('model/rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('model/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)

    return model, feature_columns


model, feature_columns = load_model()

# ============================================================
# OPTIONS
# ============================================================
MODEL_OPTIONS = [
    'Tiago', 'Tigor', 'Altroz', 'Punch',
    'Nexon', 'Harrier', 'Safari'
]

FUEL_OPTIONS = [
    'Petrol', 'Diesel', 'CNG'
]

BODY_OPTIONS = [
    'Hatchback', 'Sedan', 'Micro SUV',
    'Compact SUV', 'Mid-Size SUV', 'SUV'
]

VARIANT_OPTIONS = [
    'XE', 'XT', 'XTA', 'XE CNG', 'XZA',
    'XZA DCA', 'XT Diesel', 'Pure',
    'Adventure AMT', 'Adventure CNG',
    'Smart', 'Creative DCA', 'Pure Diesel',
    'Fearless AT', 'Accomplished AT'
]

ENGINE_OPTIONS = [1199, 1497, 1956]

# ============================================================
# HEADER
# ============================================================
st.title("🚗 Tata Motors Car Resale Price Predictor")
st.markdown("Fill the details below to estimate resale value.")
st.divider()

# ============================================================
# INPUT FORM
# ============================================================
st.subheader("📋 Car Details")

col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox("Car Model", MODEL_OPTIONS)
    fuel_type = st.selectbox("Fuel Type", FUEL_OPTIONS)
    body_type = st.selectbox("Body Type", BODY_OPTIONS)
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    year = st.slider("Manufacturing Year", 2020, 2026, 2022)
    variant = st.selectbox("Variant", VARIANT_OPTIONS)

with col2:
    engine_cc = st.selectbox("Engine CC", ENGINE_OPTIONS)

    ex_showroom_price_lakh = st.number_input(
        "Ex-showroom Price (₹ Lakh)",
        min_value=4.0,
        max_value=35.0,
        value=8.0,
        step=0.1
    )

    power_bhp = st.number_input(
        "Power (BHP)",
        min_value=72,
        max_value=168,
        value=85
    )

    mileage_kmpl = st.number_input(
        "Mileage (KMPL)",
        min_value=14.5,
        max_value=26.9,
        value=19.0,
        step=0.1
    )

    kilometers_driven = st.number_input(
        "Kilometers Driven",
        min_value=4001,
        max_value=129855,
        value=30000,
        step=1000
    )

    owner_count = st.selectbox(
        "Number of Owners",
        [1, 2, 3, 4, 5]
    )

    accident_history = st.selectbox(
        "Accident History",
        ['No', 'Yes']
    )

st.divider()

# ============================================================
# PREPARE INPUT
# ============================================================
def prepare_input():
    input_dict = {col: 0 for col in feature_columns}

    # Numeric features
    input_dict['transmission'] = 0 if transmission == 'Manual' else 1
    input_dict['engine_cc'] = engine_cc
    input_dict['ex_showroom_price_lakh'] = ex_showroom_price_lakh
    input_dict['power_bhp'] = power_bhp
    input_dict['mileage_kmpl'] = mileage_kmpl
    input_dict['year'] = year
    input_dict['kilometers_driven'] = kilometers_driven
    input_dict['owner_count'] = owner_count
    input_dict['accident_history'] = 1 if accident_history == 'Yes' else 0

    # One-hot encoded features
    for prefix, value in [
        ('fuel_type', fuel_type),
        ('model', model_name),
        ('body_type', body_type),
        ('variant', variant)
    ]:
        col_name = f"{prefix}_{value}"
        if col_name in input_dict:
            input_dict[col_name] = 1

    return pd.DataFrame([input_dict])

# ============================================================
# PREDICT
# ============================================================
if st.button("🔍 Predict Resale Price", use_container_width=True):

    input_df = prepare_input()
    prediction = model.predict(input_df)[0]

    st.divider()
    st.subheader("💰 Predicted Resale Price")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Estimated Price",
            f"₹ {prediction:.2f} Lakh"
        )

    with col2:
        st.metric(
            "Lower Estimate",
            f"₹ {max(0, prediction - 0.35):.2f} Lakh"
        )

    with col3:
        st.metric(
            "Upper Estimate",
            f"₹ {prediction + 0.35:.2f} Lakh"
        )

    st.info("📊 Price range based on model RMSE ± ₹0.35 Lakh")

    # Summary
    st.subheader("📋 Input Summary")

    summary = {
        "Model": model_name,
        "Variant": variant,
        "Fuel Type": fuel_type,
        "Transmission": transmission,
        "Body Type": body_type,
        "Year": year,
        "Ex-showroom Price": f"₹ {ex_showroom_price_lakh} Lakh",
        "Engine CC": engine_cc,
        "Power (BHP)": power_bhp,
        "Mileage (KMPL)": mileage_kmpl,
        "Kilometers Driven": f"{kilometers_driven:,}",
        "Owner Count": owner_count,
        "Accident History": accident_history
    }

    st.table(
        pd.DataFrame(summary.items(), columns=["Feature", "Value"])
    )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("⚠️ Based on synthetic Tata Motors dataset.")
st.caption("Model: Random Forest + GridSearchCV")