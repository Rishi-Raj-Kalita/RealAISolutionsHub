import os
import base64
import gc
import uuid
import streamlit as st
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer
from llama_cloud_services import LlamaParse
from llama_index.core.query_engine import CitationQueryEngine
import fitz
from PIL import Image
from main import *

load_dotenv()

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.collection_name = "pdf_collection" + str(
        st.session_state.id)
    st.session_state.file_cache = {}
    st.session_state.pdf_displayed = False  # Track if PDF is displayed
    st.session_state.pdf_path = None  # Store path to uploaded PDF
    st.session_state.pages = None

session_id = st.session_state.id


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    # Don't reset PDF state when clearing chat
    gc.collect()


def get_multimodal_sources(response):
    images = []
    metadata = []

    for citation in response.source_nodes:
        bbox = citation.node.metadata["bbox"]
        page_index = citation.node.metadata["page"]
        score = citation.score if hasattr(citation, "score") else None

        # Open the PDF file
        pdf = fitz.open(st.session_state.pdf_path)
        page = pdf[page_index]

        # Take screenshot of the page
        pix = page.get_pixmap(dpi=200)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Resize the page to align the bbox coordinates
        parsed_page = st.session_state.pages[page_index]
        image = image.resize(
            (int(parsed_page["width"]), int(parsed_page["height"])))

        # Crop the image to the bounding box
        cropped_image = image.crop(
            (bbox["x"], bbox["y"], (bbox["x"] + bbox["w"]),
             (bbox["y"] + bbox["h"])))

        # Save image and metadata
        images.append(cropped_image)
        metadata.append({
            "page":
            page_index + 1,  # Adding 1 for human-readable page numbers
            "score":
            round(score, 3) if score is not None else "N/A",
            "text":
            citation.node.text[:100] +
            "..." if len(citation.node.text) > 100 else citation.node.text
        })

    return images, metadata


def display_pdf(file):
    # Opening file from file path
    if isinstance(file, str):
        # If file is a path
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    else:
        # If file is already a file object/buffer
        base64_pdf = base64.b64encode(file.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f"""<embed
        class="pdfobject"
        type="application/pdf"
        title="Embedded PDF"
        src="data:application/pdf;base64,{base64_pdf}"
        style="overflow: auto; width: 100%; height: 800px;">"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    st.header(f"Upload your PDF")

    # Add PDF file uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    start_rag = st.button("Start RAG")

    if start_rag and uploaded_file:
        try:
            status_container = st.empty()
            with status_container.status("Processing PDF...",
                                         expanded=True) as status:
                status.write("📄 Saving uploaded PDF...")

                # Save the uploaded PDF to a temporary file
                pdf_path = f"./data/uploaded_{session_id}.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.pdf_path = pdf_path
                st.session_state.pdf_displayed = True  # Mark PDF as ready to display

                # Process the PDF for RAG
                file_key = f"{session_id}-{uploaded_file.name}"

                if file_key not in st.session_state.get('file_cache', {}):
                    status.update(label="Indexing content", state="running")
                    status.write("🔎 Indexing content.")

                    with st.spinner("Parsing PDF..."):
                        parser = LlamaParse(
                            api_key=os.getenv("LAMA_PARSE_API_KEY"),
                            parse_mode="parse_page_with_layout_agent")

                        pages = parse_pages(st.session_state.pdf_path)
                        index = index_document(pages)

                        query_engine = CitationQueryEngine.from_args(
                            index, similarity_top_k=3)

                    st.success("Parsed PDF Successfully!")
                    st.session_state.query_engine = query_engine
                    st.session_state.pages = pages

                    st.session_state.file_cache[
                        file_key] = st.session_state.query_engine
                else:
                    st.session_state.query_engine = st.session_state.file_cache[
                        file_key]

                status.update(label="Processing complete!", state="complete")
            st.success("Ready to Chat!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

    # Always show PDF if it exists
    if st.session_state.get('pdf_displayed',
                            False) and st.session_state.pdf_path:
        pdf_viewer(st.session_state.pdf_path)

col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("""
    ## Multimodal RAG with LlamaParse and LlamaIndex""",
                unsafe_allow_html=True)

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if 'metadata' and 'images' in message:
            for i, (image, meta) in enumerate(
                    zip(message['images'], message['metadata'])):
                if image and meta:
                    with st.expander(
                            f"Source {i+1} (Page {meta['page']}, Score: {meta['score']})"
                    ):
                        st.image(image, use_container_width=True)
                        st.markdown(f"**Text excerpt:** {meta['text']}")

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Check if query engine is initialized
        if not hasattr(st.session_state, "query_engine"):
            full_response = "Please upload a PDF and click 'Start RAG' before asking questions."
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
        else:

            with st.spinner("Generating Response..."):
                streaming_response = st.session_state.query_engine.query(
                    prompt)

                message_placeholder.markdown(streaming_response.response)
                images, metadata = get_multimodal_sources(streaming_response)

                # Display all sources in a presentable way
                st.markdown("### Sources")

                for i, (image, meta) in enumerate(zip(images, metadata)):
                    with st.expander(
                            f"Source {i+1} (Page {meta['page']}, Score: {meta['score']})"
                    ):
                        st.image(image, use_container_width=True)
                        st.markdown(f"**Text excerpt:** {meta['text']}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": streaming_response.response,
                "images": images,
                "metadata": metadata
            })
    # Add assistant response to chat history
