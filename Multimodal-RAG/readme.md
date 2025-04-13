# Multimodal RAG with LlamaParse and LlamaIndex

A Streamlit application that enables document question-answering with multimodal capabilities, retrieving both textual answers and relevant visual context from PDF documents.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system with multimodal capabilities. Users can upload PDF documents, ask questions about their content, and receive answers with relevant visual context extracted directly from the document pages.

Key features:
- PDF document parsing with layout-aware extraction using LlamaParse
- Semantic indexing and retrieval with LlamaIndex
- Citation-based query engine that tracks source locations
- Multimodal response generation showing both textual answers and visual evidence
- Interactive UI with PDF viewing and chat interface

## Prerequisites

- Python 3.8+
- Streamlit
- LlamaIndex
- PyMuPDF (fitz)
- Pillow
- Python-dotenv
- LlamaParse API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rishi-Raj-Kalita/RealAISolutionsHub
cd multimodal-rag
```

2. Create a `.env` file in the project root with your API keys:
```
LAMA_PARSE_API_KEY=your_llama_parse_api_key
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run App.py
```

2. Use the web interface to:
   - Upload a PDF document
   - Click "Start RAG" to parse and index the document
   - Ask questions about the document in the chat interface
   - View answers with multimodal context showing the relevant parts of the document

## How It Works

1. **Document Processing**: 
   - The uploaded PDF is processed using LlamaParse to extract text with layout awareness
   - Each page is parsed with its original layout and coordinates preserved

2. **Indexing**:
   - The parsed content is indexed using LlamaIndex
   - Document sections are stored with their corresponding bounding box coordinates and page numbers

3. **Query Processing**:
   - User queries are processed by a CitationQueryEngine
   - The engine retrieves the top-k most relevant document sections
   - Each retrieval includes metadata about where the information appears in the document

4. **Multimodal Response Generation**:
   - The system generates a textual response based on retrieved passages
   - For each source passage, the system extracts the corresponding visual area from the PDF
   - Both the answer and visual evidence are presented to the user

## Project Structure

- `App.py`: Main Streamlit application
- `main.py`: Core functionality for document parsing and indexing
- `data/`: Directory for storing uploaded PDF files
- `requirements.txt`: Project dependencies

## Future Enhancements

- Document comparison capabilities
- Support for additional file formats (PPTX, DOCX)
- Enhanced visual context with highlighting
- Custom knowledge base creation and management
- Export functionality for answers with citations
