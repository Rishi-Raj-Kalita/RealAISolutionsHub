import streamlit as st
import os
import pandas as pd
from main import *

@st.fragment
def generate_artifacts():
    st.success("Curating Data...")
    curated_df=curate_data()
    st.dataframe(curated_df)
    status=check_balance()
    st.write(status)

st.title('Expense Manager')

# Sidebar for file upload
st.sidebar.title('Upload CSV')
with st.sidebar:
    st.markdown('Upload your CSV file here')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Display categories and users in the sidebar
    categories = ['food', 'rent', 'family', 'shopping', 'self-care', 'transport', 'other', 'unknown']
    users = ['pallavi', 'prateek', 'aws', 'arshad', 'manas']

    st.sidebar.subheader('Categories')
    for category in categories:
        st.sidebar.checkbox(category)

    st.sidebar.subheader("Users")
    for user in users:
        st.sidebar.checkbox(user)

# Main content
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if uploaded_file is not None:
    # Save the uploaded file to the specified path
    save_path = '/Users/rishirajkalita/Desktop/expsense-manager/data/source'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    file_path = os.path.join(save_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File saved at {file_path}")

    # Display the uploaded file in the main section
    df = pd.read_csv(file_path)
    st.write("Uploaded CSV file:")
    st.dataframe(df)

    if st.button('Start Process'):
        with st.spinner('Processing...'):
            df=get_data(file_path)
            target_df, current_balance=traverse_expense(categories, users, df)
            print("-"*80)
            print("current balance received",current_balance)

        target_df.to_csv('./data/raw/target_df.csv', index=False)
        get_cleaned_data()
        get_input_amount()
        st.success('Please verify the data in your machine and save.')



        

