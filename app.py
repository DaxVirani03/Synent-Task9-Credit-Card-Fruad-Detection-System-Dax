import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="centered")

st.title("Credit Card Fraud Detection")
st.caption("23AIML076 - Enter transaction details to check if it's fraudulent")

with st.expander("ℹ️ About the Model & Project Details"):
    st.markdown("""
    **Objective:** Identify fraudulent credit card transactions using Machine Learning.
    
    **Model Architecture:** 
    - **Algorithm:** Random Forest Classifier
    - **Preprocessing:** `sklearn.pipeline.Pipeline` with `ColumnTransformer` (StandardScaler for numerical, OneHotEncoder for categorical)
    - **Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique) used to balance the severe 1:170 fraud ratio in training data.
    
    **Feature Engineering:**
    - Calculates geographical **Distance** between Customer and Merchant
    - Uses Log-transformed Transaction Amount (`log_amt`)
    - Extracts temporal behaviors (Age, Hour of Transaction)
    """)

@st.cache_resource
def load_model():
    model = joblib.load('best_ml_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

categories = ['entertainment', 'food_dining', 'gas_transport', 'grocery_net',
              'grocery_pos', 'health_fitness', 'home', 'kids_pets',
              'misc_net', 'misc_pos', 'personal_care', 'shopping_net',
              'shopping_pos', 'travel']

with st.form("txn_form"):
    col1, col2 = st.columns(2)
    with col1:
        amt = st.number_input("Amount ($)", min_value=0.01, value=50.0)
        category = st.selectbox("Category", categories)
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        hour = st.slider("Hour of Transaction", 0, 23, 12)
    with col2:
        lat = st.number_input("Customer Latitude", value=40.0, format="%.4f")
        long = st.number_input("Customer Longitude", value=-74.0, format="%.4f")
        merch_lat = st.number_input("Merchant Latitude", value=40.1, format="%.4f")
        merch_long = st.number_input("Merchant Longitude", value=-74.1, format="%.4f")
        city_pop = st.number_input("City Population", min_value=1, value=50000)

    submitted = st.form_submit_button("Predict")

if submitted:
    distance = np.sqrt((lat - merch_lat)**2 + (long - merch_long)**2)
    log_amt = np.log1p(amt)
    unix_time = 1325376000
    day_of_week = 2
    month = 6

    input_data = pd.DataFrame([{
        'amt': amt, 'lat': lat, 'long': long, 'city_pop': city_pop,
        'unix_time': unix_time, 'merch_lat': merch_lat, 'merch_long': merch_long,
        'age': age, 'hour': hour, 'day_of_week': day_of_week, 'month': month,
        'distance': distance, 'log_amt': log_amt, 'category': category, 'gender': gender
    }])

    processed = preprocessor.transform(input_data)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"**FRAUDULENT** (Confidence: {probability*100:.1f}%)")
    else:
        st.success(f"**LEGITIMATE** (Confidence: {(1-probability)*100:.1f}%)")

    st.progress(probability, text=f"Fraud Probability: {probability*100:.1f}%")
