Customer Lifetime Value (CLV) Prediction and Customer Analytics
This project focuses on analyzing transactional data from an online retail dataset to predict Customer Lifetime Value (CLV) and generate insights into customer purchasing behavior.

Key Steps in the Project:
1.Data Preparation & Cleaning
        Imported retail data from Excel into a Pandas DataFrame.
        Cleaned column names and removed invalid records (e.g., missing Customer_ID, negative quantities).
        Converted InvoiceDate to datetime format and calculated TotalPrice (Quantity × Price).
2.Feature Engineering
-Aggregated customer-level features including:
        First and last purchase dates.
        Number of invoices and purchases.
        Total spend and total quantity purchased.
        Country of the customer.
-Derived new features such as:
        Customer_Age_Days (days since first purchase).
        Frequency (unique invoices).
        Average Purchase Value.
3.Modeling CLV
        Defined Total Historical Spend (TotalPrice_sum) as a proxy for CLV.
        Selected features: Customer_Age_Days, Frequency, Quantity_sum, Avg_Purchase_Value.
        Trained a Random Forest Regressor to predict CLV.
        Evaluated performance using RMSE (Root Mean Squared Error), achieving an RMSE of ~3474.32.
4.Customer-Level Predictions
        Enabled user input of a Customer_ID to view details such as:
        Country, total purchases, quantity, average purchase value, and customer age.
        Predicted Customer Lifetime Value (CLV) using the trained model.

5.Customer Analytics & Visualization
        For a given customer, generated insightful plots:
        Total Spend Over Time – trend of purchase value.
        Quantity Purchased Over Time – purchase volume trends.
        Average Basket Size per Invoice – distribution of item prices.
        Top Products Purchased – top 10 items based on total quantity.

Project Objectives:
        Build a pipeline to predict Customer Lifetime Value (CLV) using machine learning.
        Provide actionable customer insights through visualizations.
        Support data-driven decision-making in customer segmentation, retention strategies, and marketing campaigns.


# CSI_Project(https://csiproject-8r9srntnpmzhek3qbcekse.streamlit.app/)
This repository contains the project for Celebal Tech.
The Link of the web app contains on one of the files(This is the main project link).
I have also uploaded the colab file for the same which is extra.
