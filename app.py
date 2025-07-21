code = """
import streamlit as st
import joblib
import numpy as np

# Load trained sklearn-style model
model = joblib.load('cltv_model.pkl')

st.title("Customer Lifetime Value Prediction")

recency = st.number_input("Recency (days)", min_value=0)
frequency = st.number_input("Frequency", min_value=1)
monetary = st.number_input("Monetary Value", min_value=0.0)

if st.button("Predict"):
    x = np.array([[recency, frequency, monetary]])
    pred = model.predict(x)[0]
    if pred < 0:
        st.warning("Predicted CLV is negative, which may indicate bad input or model instability.")
    st.success(f"Predicted CLV: ${pred:,.2f}")
"""

with open("app.py", "w") as f:
    f.write(code)
