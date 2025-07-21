import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("model.pkl not found. Please upload it to the same directory.")
    st.stop()

st.set_page_config(page_title="Retail Purchase Predictor", layout="centered")
st.title("Retail Purchase Prediction App")

st.markdown("""
This app predicts the likelihood of a transaction based on retail data.
Fill in the inputs below and click **Predict** to see the result.
""")

invoice = st.text_input("Invoice (any string)", value="536365")
stock_code = st.text_input("StockCode", value="85123A")
description = st.text_input("Product Description", value="WHITE HANGING HEART T-LIGHT HOLDER")
quantity = st.number_input("Quantity", value=6, min_value=1)
invoice_date = st.text_input("Invoice Date (optional)", value="01-12-2010 08:26")
price = st.number_input("Price", value=2.55, min_value=0.0)
customer_id = st.text_input("Customer ID", value="17850")
country = st.text_input("Country", value="United Kingdom")

if st.button("Predict"):
    try:
        input_df = pd.DataFrame([{
            "Invoice": invoice,
            "StockCode": stock_code,
            "Description": description,
            "Quantity": quantity,
            "InvoiceDate": invoice_date,
            "Price": price,
            "Customer ID": customer_id,
            "Country": country
        }])
        prediction = model.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")