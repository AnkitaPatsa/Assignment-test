
import streamlit as st
import pandas as pd
import joblib

st.title("Mental Health Care Options Classifications")

uploaded_file = st.file_uploader("Upload your data (CSV)", type="csv")

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    model = joblib.load(f"model/{model_name}.pkl")
    predictions = model.predict(data)

    st.write("Predictions:", predictions)
