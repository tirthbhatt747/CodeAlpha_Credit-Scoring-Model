import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("../models/random_forest_model.pkl")

st.title("Credit Scoring Prediction App")

income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")
credit_history = st.number_input("Credit History")
debt_ratio = st.number_input("Debt Ratio")
age = st.number_input("Age")

input_data = pd.DataFrame({
    'Income': [income],
    'LoanAmount': [loan_amount],
    'CreditHistory': [credit_history],
    'DebtRatio': [debt_ratio],
    'Age': [age]
})

if st.button("Predict Creditworthiness"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Good Creditworthiness")
    else:
        st.error("Bad Creditworthiness")