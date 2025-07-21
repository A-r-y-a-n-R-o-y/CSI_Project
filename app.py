
import streamlit as st
import joblib
import numpy as np

model = joblib.load('cltv_model.pkl')

st.title("Customer Lifetime Value Prediction")

recency = st.number_input("Recency (days)", min_value=0)
frequency = st.number_input("Frequency", min_value=1)
monetary = st.number_input("Monetary Value", min_value=0.0)

if st.button("Predict"):
    x = np.array([[recency, frequency, monetary]])
    pred = model.predict(x)[0]
    st.write(f"Predicted CLTV: ${pred:.2f}")
