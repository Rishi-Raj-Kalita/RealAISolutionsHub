import streamlit as st
import os
import pandas as pd

st.title('Expense Manager')

# Sidebar for file upload
st.sidebar.title('Upload CSV')
with st.sidebar:
    st.markdown('Upload your CSV file here')
    uploaded_file = st.file_uploader("Choose a CSV file", type="DELIMITED")

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
        st.success('Process started')