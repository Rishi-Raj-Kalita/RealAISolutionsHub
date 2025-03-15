import streamlit as st
import pandas as pd
import os
from main import *
import plotly.express as px



st.title('Visualize Expense')

# Path to the curated data file
curated_file_path = './data/curated/curated_df.csv'

if st.button("Curate Data"):
    curate_data()
    # Check if the file exists
    if os.path.exists(curated_file_path):
        # Read the curated data file
        curated_df = pd.read_csv(curated_file_path)
        
        # Total Expense
        total_expense = curated_df['my_expenses'].sum()
        st.write(f"Total Expense: {total_expense}")

        # Total expense spent in each category
        categories = ['category_food', 'category_rent', 'category_family', 'category_shopping', 'category_self-care', 'category_transport', 'category_other', 'catagory_investment']
        category_expenses = curated_df[categories].sum().reset_index()
        category_expenses.columns = ['category', 'amount']
        fig = px.bar(category_expenses, x='category', y='amount', title='Total Expense by Category')
        st.plotly_chart(fig)

        # Total money owed by each user
        users = ['user_pallavi', 'user_prateek', 'user_aws', 'user_arshad', 'user_manas']
        user_owed = curated_df[users].sum().reset_index()
        user_owed.columns = ['user', 'amount']
        fig = px.bar(user_owed, x='user', y='amount', title='Total Money Owed by Users')
        st.plotly_chart(fig)

        lent_df=extract_owe()
        st.write("Transactions owed from various users:")
        st.dataframe(lent_df)
    else:
        st.error(f"File not found: {curated_file_path}")
