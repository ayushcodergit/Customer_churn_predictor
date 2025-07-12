import streamlit as st
import pandas as pd
import joblib

# Load model and trained column names
model = joblib.load('churn_model.pkl')
trained_columns = joblib.load('trained_columns.pkl')

# App Title & Info
st.title("📉 Customer Churn Predictor App")
st.markdown("Enter customer details to predict the likelihood of churn.")

# User Input Function
def user_input():
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', [0, 1])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.slider('Tenure (months)', 0, 72, 1)
    monthly_charges = st.slider('Monthly Charges', 0.0, 200.0, 50.0)
    total_charges = st.slider('Total Charges', 0.0, 10000.0, 500.0)
    contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('Payment Method', [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ])

    #  Build the input dictionary based on encoded columns
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': senior_citizen,
        'gender_Male': 1 if gender == 'Male' else 0,
        'Partner_Yes': 1 if partner == 'Yes' else 0,
        'Dependents_Yes': 1 if dependents == 'Yes' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == 'Bank transfer (automatic)' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
    }

    # Fill missing features with 0
    for col in trained_columns:
        if col not in data:
            data[col] = 0

    # Return input as DataFrame with correct column order
    input_df = pd.DataFrame([data])[trained_columns]
    return input_df

# Generate input DataFrame
input_df = user_input()

#  Predict on button click
if st.button('Predict Churn'):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn**.\n\n**Probability: {probability:.2%}**")
    else:
        st.success(f"✅ This customer is **unlikely to churn**.\n\n**Probability: {probability:.2%}**")
