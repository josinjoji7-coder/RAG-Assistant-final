## PDF Question Answering Assistant using RAG

Josin Joji
MUID: josinjoji@mulearn

### Project Overview

This project is a PDF Question Answering Application built using Retrieval-Augmented Generation (RAG).
The application allows users to upload a PDF document, understand its contents, and ask questions based on the uploaded document. The system retrieves relevant information from the PDF and uses a Large Language Model (LLM) to generate accurate, context-based answers.
The application also maintains conversation history, allowing users to ask follow-up questions naturally.

### Technologies Used

Programming Language

* Python

Frontend

* Streamlit

RAG Framework

* LangChain

PDF Processing

* PyPDFLoader

Text Processing

* Recursive Character Text Splitter

Embedding Model

* Sentence Transformers (`all-MiniLM-L6-v2`)

Vector Database

* ChromaDB

Large Language Model

* Google Gemini API (Free Tier)

Environment Management

* python-dotenv

### Memory Implementation

Conversation memory is implemented using Streamlit session state.

The application stores:

* Previous user questions
* Previous assistant responses

This helps the assistant understand the previous conversation and provide better responses for follow-up questions.

Example:

**User:** What is this document about?

**Assistant:** This document explains a software agreement.

**User:** Who owns the software?

The assistant uses the previous conversation context to answer correctly.

### Challenges Faced

* Managing package compatibility between LangChain, Streamlit, and Gemini API.
* Handling PDF text extraction and chunking.
* Understanding how embeddings and vector databases work.
* Managing conversation history for follow-up questions.
* Deploying the application with required environment variables.

### Future Improvements

* Support multiple PDF uploads.
* Add PDF preview functionality.
* Improve answer accuracy using advanced retrieval techniques.
* Add user authentication.
* Add source references with page numbers.
* Deploy with cloud-based vector databases for scalability.