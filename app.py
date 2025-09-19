import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline (includes preprocessing + model)
model = joblib.load("german_credit.joblib")

st.title("German Credit Risk Prediction")
st.write("Fill in the details below to predict creditworthiness.")

# -------- Feature Inputs -------- #
# 1. Status of existing checking account
status_checking = st.selectbox(
    "Status of existing checking account",
    ["<0 DM", "0<=X<200 DM", ">=200 DM", "no checking account"]
)

# 2. Duration in month
duration = st.number_input("Duration in month", min_value=1, max_value=72, value=12)

# 3. Credit history
credit_history = st.selectbox(
    "Credit history",
    ["no credits", "all paid", "existing paid", "delayed", "critical/other"]
)

# 4. Purpose
purpose = st.selectbox(
    "Purpose",
    ["car (new)", "car (used)", "furniture/equipment", "radio/TV", "domestic appliance",
     "repairs", "education", "vacation", "retraining", "business", "other"]
)

# 5. Credit amount
credit_amount = st.number_input("Credit amount", min_value=100, max_value=20000, value=1000)

# 6. Savings account/bonds
savings = st.selectbox(
    "Savings account/bonds",
    ["<100 DM", "100<=X<500 DM", "500<=X<1000 DM", ">=1000 DM", "unknown"]
)

# 7. Present employment since
employment = st.selectbox(
    "Present employment since",
    ["unemployed", "<1 year", "1<=X<4 years", "4<=X<7 years", ">=7 years"]
)

# 8. Installment rate (% of disposable income)
installment_rate = st.number_input("Installment rate (% of disposable income)", min_value=1, max_value=4, value=2)

# 9. Personal status and sex
personal_status = st.selectbox(
    "Personal status and sex",
    ["male: divorced/separated", "female: divorced/separated/married",
     "male: single", "male: married/widowed", "female: single"]
)

# 10. Other debtors/guarantors
other_debtors = st.selectbox("Other debtors/guarantors", ["none", "co-applicant", "guarantor"])

# 11. Present residence since
residence_since = st.number_input("Present residence since (years)", min_value=1, max_value=4, value=2)

# 12. Property
property_type = st.selectbox(
    "Property",
    ["real estate", "life insurance", "car or other", "no property"]
)

# 13. Age in years
age = st.number_input("Age in years", min_value=18, max_value=100, value=30)

# 14. Other installment plans
other_installments = st.selectbox("Other installment plans", ["bank", "stores", "none"])

# 15. Housing
housing = st.selectbox("Housing", ["own", "for free", "rent"])

# 16. Number of existing credits at this bank
existing_credits = st.number_input("Number of existing credits at this bank", min_value=1, max_value=4, value=1)

# 17. Job
job = st.selectbox(
    "Job",
    ["unemployed/unskilled", "unskilled resident", "skilled employee", "management/self-employed/highly qualified"]
)

# 18. Number of people being liable to provide maintenance
dependents = st.number_input("Number of dependents", min_value=1, max_value=2, value=1)

# 19. Telephone
telephone = st.selectbox("Telephone", ["none", "yes"])

# 20. Foreign worker
foreign_worker = st.selectbox("Foreign worker", ["yes", "no"])

# -------- Prediction -------- #
if st.button("Predict Credit Risk"):
    # Put all inputs into dataframe
    input_data = pd.DataFrame([[
        status_checking, duration, credit_history, purpose, credit_amount,
        savings, employment, installment_rate, personal_status, other_debtors,
        residence_since, property_type, age, other_installments, housing,
        existing_credits, job, dependents, telephone, foreign_worker
    ]], columns=[
        "Status_of_existing_checking_account",
        "Duration_in_month",
        "Credit_history",
        "Purpose",
        "Credit_amount",
        "Savings_account_bonds",
        "Present_employment_since",
        "Installment_rate_in_percentage_of_disposable_income",
        "Personal_status_and_sex",
        "Other_debtors_guarantors",
        "Present_residence_since",
        "Property",
        "Age_in_years",
        "Other_installment_plans",
        "Housing",
        "Number_of_existing_credits_at_this_bank",
        "Job",
        "Number_of_people_being_liable_to_provide_maintenance_for",
        "Telephone",
        "Foreign_worker"
    ])

    # Predict using pipeline
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("Good Credit Risk ✅")
    else:
        st.error("Bad Credit Risk ❌")

    st.write(f"Probability of Good Credit: {proba[1]:.2f}")
    st.write(f"Probability of Bad Credit: {proba[0]:.2f}")

