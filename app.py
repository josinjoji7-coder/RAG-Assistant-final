import os
import shutil

import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚"
)


st.title("PDF Question Answering Assistant")


# Gemini setup

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Session memory

if "database" not in st.session_state:
    st.session_state.database = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# PDF processing function

def process_pdf(pdf_path):

    if os.path.exists("vectorstore"):
        try:
            shutil.rmtree("vectorstore")
        except PermissionError:
            pass

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    database = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="vectorstore"
    )

    return database

# Upload PDF

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file:

    pdf_path = "uploaded.pdf"


    with open(pdf_path, "wb") as file:
        file.write(uploaded_file.getbuffer())


    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            st.session_state.database = process_pdf(
                pdf_path
            )

        st.success(
            "PDF processed successfully!"
        )

# Question section

question = st.text_input(
    "Ask a question about the PDF"
)

if st.button("Get Answer"):


    if st.session_state.database is None:

        st.warning(
            "Please upload and process a PDF first."
        )

    else:

        results = st.session_state.database.similarity_search(
            question,
            k=3
        )

        context = ""

        for result in results:
            context += result.page_content + "\n"

        history = ""

        for chat in st.session_state.chat_history:

            history += (
                "User: "
                + chat["question"]
                + "\nAssistant: "
                + chat["answer"]
                + "\n"
            )

        prompt = f"""

You are a helpful PDF assistant.

Use only the context to answer.

Previous conversation:

{history}

Context:

{context}

Question:

{question}

"""
        response = llm.invoke(prompt)

        answer = response.content

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        st.write("### Answer")

        st.write(answer)

# Display chat history

if st.session_state.chat_history:

    st.write("## Conversation History")


    for chat in st.session_state.chat_history:

        st.write(
            "👤 User:",
            chat["question"]
        )

        st.write(
            "Assistant:",
            chat["answer"]
        )