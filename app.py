
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
df = pd.read_csv("credit_risk_dataset.csv")

df["person_emp_length"] = pd.to_numeric(df["person_emp_length"], errors='coerce').fillna(0).astype(int)
df["loan_int_rate"].fillna(df["loan_int_rate"].mean(), inplace=True)
df.drop_duplicates(inplace=True)

# Encode categorical columns
categorical_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Streamlit app
st.title("Credit Risk Prediction App")

st.write("Enter the applicant's information to predict the risk level:")

# User Inputs
person_age = st.number_input("Person Age", min_value=18, max_value=100, value=30)
person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
person_income = st.number_input("Annual Income", min_value=0, value=50000)
loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
loan_int_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
loan_percent_income = st.number_input("Loan as % of Income", min_value=0.0, max_value=1.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, max_value=50, value=5)

# Encoded categorical features
person_home_ownership = st.selectbox("Home Ownership", le_dict["person_home_ownership"].classes_)
loan_intent = st.selectbox("Loan Intent", le_dict["loan_intent"].classes_)
loan_grade = st.selectbox("Loan Grade", le_dict["loan_grade"].classes_)
cb_person_default_on_file = st.selectbox("Default on File", le_dict["cb_person_default_on_file"].classes_)


input_data = pd.DataFrame({
    "person_age": [person_age],
    "person_emp_length": [person_emp_length],
    "person_income": [person_income],
    "loan_amnt": [loan_amnt],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "cb_person_cred_hist_length": [cb_person_cred_hist_length],
    "person_home_ownership": [le_dict["person_home_ownership"].transform([person_home_ownership])[0]],
    "loan_intent": [le_dict["loan_intent"].transform([loan_intent])[0]],
    "loan_grade": [le_dict["loan_grade"].transform([loan_grade])[0]],
    "cb_person_default_on_file": [le_dict["cb_person_default_on_file"].transform([cb_person_default_on_file])[0]]
})

# Match column order to training set
input_data = input_data[X.columns]

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("High Risk: The applicant is likely to default.")
    else:
        st.success("Low Risk: The applicant is likely to repay the loan.")
