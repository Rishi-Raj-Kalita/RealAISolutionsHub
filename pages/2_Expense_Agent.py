import streamlit as st
from main import *
# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_agent" not in st.session_state:
    st.session_state.rag_agent=None

with st.sidebar:
    if st.button("Invoke RAG Expense"):
        with st.spinner('Invoking RAG Expense...'):
            st.session_state.rag_agent = rag_expense()
        st.success('RAG Expense invoked successfully!')




# Accept user input
if prompt := st.chat_input("Ask a question about your expenses..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from RAG agent
    if st.session_state.rag_agent:
        response = st.session_state.rag_agent.invoke({"input": prompt})
        answer = response['answer']
        context=response['context']

        context_display="\n\n".join(doc.page_content for doc in context)
        print(context_display)
        
        # Display the formatted response
        formatted_response = f"**Answer:** {answer}\n\n**Context:**\n{context_display}"
    else:
        formatted_response = "RAG agent is not initialized. Please invoke RAG Expense from the sidebar."

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
    with st.chat_message("assistant"):
        st.markdown(formatted_response)