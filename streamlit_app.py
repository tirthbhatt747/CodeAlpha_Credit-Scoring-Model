import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("models/random_forest_model.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Credit Scoring Prediction",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("💳 Credit Scoring Prediction App")
st.markdown("Enter customer details to predict creditworthiness.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
income = st.number_input(
    "Income (₹)",
    min_value=0,
    value=50000,
    step=1000
)

loan_amount = st.number_input(
    "Loan Amount (₹)",
    min_value=0,
    value=10000,
    step=1000
)

credit_history = st.selectbox(
    "Credit History",
    ["Good", "Bad"]
)

debt_ratio = st.number_input(
    "Debt Ratio (%)",
    min_value=0,
    max_value=100,
    value=20,
    step=1
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)

# Convert Credit History
credit_history_value = 1 if credit_history == "Good" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Creditworthiness"):

    input_data = pd.DataFrame({
        "Income": [income],
        "LoanAmount": [loan_amount],
        "CreditHistory": [credit_history_value],
        "DebtRatio": [debt_ratio],
        "Age": [age]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approval Recommended")
    else:
        st.error("❌ Loan Approval Not Recommended")

    st.write("### Confidence Scores")

    st.write(f"Good Creditworthiness: **{probability[1]*100:.2f}%**")
    st.write(f"Bad Creditworthiness: **{probability[0]*100:.2f}%**")

    st.divider()

    st.write("### Input Summary")
    st.dataframe(input_data, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit & Random Forest Classifier")