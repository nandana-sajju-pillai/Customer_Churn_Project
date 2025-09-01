import streamlit as st
import pandas as pd
import joblib
from os import path

# -----------------------------
# Load trained pipeline
# -----------------------------
model_path = path.join("model", "customer_churn_logreg.pkl")
churn_predictor = joblib.load(model_path)

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn or stay, based on their details.")

# -----------------------------
# Input Form
# -----------------------------
st.header("Enter Customer Details:")

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.radio("Senior Citizen", ["Yes", "No"])
partner = st.radio("Partner", ["Yes", "No"])
dependents = st.radio("Dependents", ["Yes", "No"])

tenure = st.slider(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    step=1,
    value=0
)

phone_service = st.radio("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

monthly_charges = st.number_input(
    "Monthly Charges ($)",
    min_value=0.0,
    max_value=200.0,
    placeholder="Enter a value between 0.0 and 200.0",
    value=None
)

total_charges = st.number_input(
    "Total Charges ($)",
    min_value=0.0,
    max_value=10000.0,
    placeholder="Enter a value between 0.0 and 10000.0",
    value=None
)

# -----------------------------
# Prepare Input Data
# -----------------------------
df_input = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method
}])

# Fix "No internet/phone service" → "No"
replace_map = {"No internet service": "No", "No phone service": "No"}
for col in df_input.columns:
    if df_input[col].dtype == object:
        df_input[col] = df_input[col].replace(replace_map)

# -----------------------------
# Show Entered Data
# -----------------------------
st.subheader("Entered Customer Data")
st.dataframe(df_input, use_container_width=True)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Churn", use_container_width=True):
    prediction = churn_predictor.predict(df_input)[0]
    if prediction == 1:
        st.error("This customer is likely to CHURN.")
    else:
        st.success("This customer is likely to STAY.")
