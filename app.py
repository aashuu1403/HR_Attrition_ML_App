import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained Random Forest model and feature list
model = joblib.load('hr_model.pkl')
model_features = joblib.load('model_features.pkl')

# 2. Set up the web page configuration
st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

# 3. Build the header
st.title("🧑‍💼 HR Employee Attrition Prediction App")
st.markdown("### Interactive Machine Learning Dashboard")
st.write("Use the sidebar to adjust employee parameters and predict their flight risk in real-time.")

# 4. Build the Sidebar with our Top 5 Features
st.sidebar.header("Employee Parameters")
st.sidebar.write("Adjust the top drivers of attrition:")

stock_option = st.sidebar.slider("Stock Option Level", 0, 3, 1, 1)
monthly_income = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 5000, 100)
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
job_satisfaction = st.sidebar.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3, 1)
years_manager = st.sidebar.slider("Years With Current Manager", 0, 20, 2, 1)

# 5. Prepare the input data for the model
# Create a base dictionary with 0s for all 44 features to prevent missing column errors
input_data = {col: 0 for col in model_features}

# Overwrite the base dictionary with the user's sidebar inputs
input_data['StockOptionLevel'] = stock_option
input_data['MonthlyIncome'] = monthly_income
input_data['JobSatisfaction'] = job_satisfaction
input_data['YearsWithCurrManager'] = years_manager

# Handle the One-Hot Encoded Marital Status logic
if marital_status == "Married":
    input_data['MaritalStatus_Married'] = 1
elif marital_status == "Single":
    input_data['MaritalStatus_Single'] = 1

# Convert the dictionary into a pandas DataFrame (which the model expects)
input_df = pd.DataFrame([input_data])

# 6. Make the Prediction
st.markdown("---")
st.subheader("Prediction Results")

if st.button("Predict Flight Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ **High Flight Risk!** This employee is likely to leave. (Churn Probability: {probability:.1%})")
    else:
        st.success(f"✅ **Low Flight Risk.** This employee is likely to stay. (Churn Probability: {probability:.1%})")