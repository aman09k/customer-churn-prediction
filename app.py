import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('lr_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details to predict whether they will churn or not.")
st.divider()

# Sidebar inputs
st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", 
                    ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", 
                    ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", 
                    ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", 
                    ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", 
                    ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", 
                    ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", 
                    ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", 
                    ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract Type", 
                    ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", 
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)",
                     "Credit card (automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 
                    18.0, 120.0, 65.0)
total_charges = st.sidebar.slider("Total Charges ($)", 
                    0.0, 9000.0, monthly_charges * tenure)

# Encode inputs
def encode_input():
    data = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'MultipleLines': 0 if multiple_lines == "No" else 
                         2 if multiple_lines == "No phone service" else 1,
        'InternetService': 0 if internet_service == "DSL" else 
                           1 if internet_service == "Fiber optic" else 2,
        'OnlineSecurity': 0 if online_security == "No" else 
                          2 if online_security == "No internet service" else 1,
        'OnlineBackup': 0 if online_backup == "No" else 
                        2 if online_backup == "No internet service" else 1,
        'DeviceProtection': 0 if device_protection == "No" else 
                            2 if device_protection == "No internet service" else 1,
        'TechSupport': 0 if tech_support == "No" else 
                       2 if tech_support == "No internet service" else 1,
        'StreamingTV': 0 if streaming_tv == "No" else 
                       2 if streaming_tv == "No internet service" else 1,
        'StreamingMovies': 0 if streaming_movies == "No" else 
                           2 if streaming_movies == "No internet service" else 1,
        'Contract': 0 if contract == "Month-to-month" else 
                    1 if contract == "One year" else 2,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'PaymentMethod': 0 if payment_method == "Electronic check" else 
                         3 if payment_method == "Mailed check" else 
                         1 if payment_method == "Bank transfer (automatic)" else 2,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return pd.DataFrame([data])

# Predict button
if st.sidebar.button("🔮 Predict Churn", use_container_width=True):
    input_df = encode_input()
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN!")
        st.metric("Churn Probability", 
                  f"{round(probability[1] * 100, 1)}%")
    else:
        st.success(f"✅ This customer will likely STAY!")
        st.metric("Retention Probability", 
                  f"{round(probability[0] * 100, 1)}%")

    st.divider()
    st.subheader("Customer Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tenure", f"{tenure} months")
    with col2:
        st.metric("Monthly Charges", f"${monthly_charges}")
    with col3:
        st.metric("Contract Type", contract)
else:
    st.info("👈 Fill in customer details in the sidebar and click Predict!")