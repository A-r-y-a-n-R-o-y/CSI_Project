import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


st.set_page_config(page_title="Customer Lifetime Value Prediction", layout="wide")


st.title(" Customer Lifetime Value (CLV) Prediction")


uploaded_file = st.file_uploader("Upload Online Retail Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    

    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df.dropna(subset=['Customer_ID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']


    customer_features = df.groupby('Customer_ID').agg({
        'InvoiceDate': ['min', 'max', 'count'],
        'Invoice': 'nunique',
        'Quantity': 'sum',
        'TotalPrice': 'sum',
        'Country': 'first'
    })
    customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]
    customer_features.reset_index(inplace=True)

    customer_features['Customer_Age_Days'] = (df['InvoiceDate'].max() - customer_features['InvoiceDate_min']).dt.days
    customer_features['Frequency'] = customer_features['Invoice_nunique']
    customer_features['Avg_Purchase_Value'] = customer_features['TotalPrice_sum'] / customer_features['Frequency']

 
    X = customer_features[['Customer_Age_Days', 'Frequency', 'Quantity_sum', 'Avg_Purchase_Value']]
    y = customer_features['TotalPrice_sum']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    st.success(f" Model Trained | RMSE: £{rmse:.2f}")


    customer_id = st.text_input("🔍 Enter Customer ID to Predict CLV:")

    if customer_id:
        try:
            customer_id = int(customer_id)
            row = customer_features[customer_features['Customer_ID'] == customer_id]

            if not row.empty:
                st.subheader(" Customer Details")
                st.write(f"**Country**: {row['Country_first'].values[0]}")
                st.write(f"**Total Purchases**: {row['Frequency'].values[0]}")
                st.write(f"**Total Quantity Purchased**: {row['Quantity_sum'].values[0]}")
                st.write(f"**Avg Purchase Value**: {row['Avg_Purchase_Value'].values[0]:.2f}")
                st.write(f"**Customer Since (Days)**: {row['Customer_Age_Days'].values[0]}")

 
                X_input = row[['Customer_Age_Days', 'Frequency', 'Quantity_sum', 'Avg_Purchase_Value']]
                clv_pred = model.predict(X_input)[0]
                st.subheader(f" Predicted CLV: {clv_pred:.2f}")


                cust_df = df[df['Customer_ID'] == customer_id].copy()
                cust_df['InvoiceDate'] = pd.to_datetime(cust_df['InvoiceDate'])
                cust_df['TotalPrice'] = cust_df['Quantity'] * cust_df['Price']


                spend_over_time = cust_df.groupby('InvoiceDate')['TotalPrice'].sum()
                st.subheader(" Total Spend Over Time")
                fig1, ax1 = plt.subplots()
                spend_over_time.plot(marker='o', ax=ax1)
                ax1.set_ylabel("Total Spend ()")
                ax1.set_xlabel("Date")
                ax1.grid(True)
                st.pyplot(fig1)

      
                quantity_over_time = cust_df.groupby('InvoiceDate')['Quantity'].sum()
                st.subheader(" Quantity Purchased Over Time")
                fig2, ax2 = plt.subplots()
                quantity_over_time.plot(marker='o', color='green', ax=ax2)
                ax2.set_ylabel("Quantity")
                ax2.set_xlabel("Date")
                ax2.grid(True)
                st.pyplot(fig2)

   
                basket = cust_df.groupby('Invoice').agg({'Quantity': 'sum', 'TotalPrice': 'sum'})
                basket['AvgPricePerItem'] = basket['TotalPrice'] / basket['Quantity']
                st.subheader(" Average Price per Item")
                fig3, ax3 = plt.subplots()
                sns.histplot(basket['AvgPricePerItem'], kde=True, bins=30, color='purple', ax=ax3)
                ax3.set_xlabel("£ per Item")
                st.pyplot(fig3)

   
                st.subheader(" Top 10 Products Purchased")
                top_products = cust_df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
                fig4, ax4 = plt.subplots()
                top_products.plot(kind='barh', color='orange', ax=ax4)
                ax4.set_xlabel("Total Quantity")
                ax4.invert_yaxis()
                st.pyplot(fig4)

            else:
                st.warning("Customer ID not found.")
        except ValueError:
            st.error("Please enter a valid numeric Customer ID.")
