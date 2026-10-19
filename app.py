import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('attrition_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Employee Attrition Prediction")
st.write("Predict if an employee is likely to leave the company.")

# Example: numeric input for only key features (you can expand)
Age = st.sidebar.number_input("Age", min_value=18, max_value=65, value=30)
MonthlyIncome = st.sidebar.number_input("Monthly Income", value=5000)
DistanceFromHome = st.sidebar.number_input("Distance From Home", value=10)
Gender = st.sidebar.selectbox("Gender", ["Male","Female"])
OverTime = st.sidebar.selectbox("OverTime", ["Yes","No"])

# Convert to numeric
Gender = 1 if Gender=="Male" else 0
OverTime = 1 if OverTime=="Yes" else 0

# Create input dataframe
input_df = pd.DataFrame({
    'Age':[Age],
    'MonthlyIncome':[MonthlyIncome],
    'DistanceFromHome':[DistanceFromHome],
    'Gender':[Gender],
    'OverTime':[OverTime]
})

for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]


prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write("Will the employee leave? **Yes**" if prediction[0]==1 else "**No**")

st.subheader("Prediction Probability")
st.write(f"Chance of Leaving: {prediction_proba[0][1]*100:.2f}%")
st.write(f"Chance of Staying: {prediction_proba[0][0]*100:.2f}%")
