import streamlit as st
import requests
from main import *


if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph=fetch_graph()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Streamlit app
def main():
    st.title("LinkedCastAI (Linkedin + Podcast + AI)")
    st.sidebar.header("Input Public IDs")

    # Input: List of LinkedIn public IDs
    public_ids = st.sidebar.text_area(
        "Enter LinkedIn Public IDs (one per line):",
        placeholder="e.g., john-doe-123\njane-smith-456"
    )

    # Convert input to a list
    public_id_list = [id.strip() for id in public_ids.split("\n") if id.strip()]

    if st.sidebar.button("Play Podcast"):
        audio_file_path = './podcast/audio.wav'  # Ensure this matches the saved audio file path
        if os.path.exists(audio_file_path):
            audio_file = open(audio_file_path, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        else:
            st.sidebar.error("Audio file not found. Please generate it first." )

    

    if st.sidebar.button("Fetch Feeds"):
        if public_id_list:
            st.write("### Users:")
            with st.chat_message("user"):
                st.write(f"User IDs entered: {', '.join(public_id_list)}")

            st.session_state.messages.append({"role": "user", "content": f"User IDs entered: {', '.join(public_id_list)}"})
            st.write("### Summaries:")
            with st.spinner("Fetching feeds..."):
                feeds_list=fetch_feeds(public_id_list)
                for feed in feeds_list:
                    with st.chat_message("assistant"):
                        st.write(feed['post'])
                st.session_state.messages.append({"role": "user", "content": feeds_list})

            with st.spinner("Summarizing feeds and creating the podcast..."):
                        response = st.session_state.graph.invoke(
                            {"feeds": feeds_list}, 
                            config={"configurable": {"thread_id": "1"}}
                        )
                        st.write("### Podcast:")
                        with st.chat_message("assistant"):
                            st.write(response['top_feeds'])
                        st.session_state.messages.append({"role": "assistant", "content": response['top_feeds']})

            if response['top_feeds'] is not None:
                 
                with st.spinner("Saving audio"):
                    fetch_audio(response['top_feeds'])

            
        else:
            st.warning("Please enter at least one LinkedIn public ID.")

if __name__ == "__main__":
    main()