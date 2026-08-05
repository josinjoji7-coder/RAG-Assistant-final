import os
import shutil
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
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

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Create vector database

def create_vectorstore(_chunks, _embeddings):

    database = Chroma.from_documents(
        documents=_chunks,
        embedding=_embeddings,
        collection_name="current_pdf"
    )

    return database

# PDF processing function

def process_pdf(pdf_path):

    # Remove old vector database
    if os.path.exists("vectorstore"):
        shutil.rmtree("vectorstore")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    for doc in documents:
        doc.page_content = " ".join(doc.page_content.split())


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)


    database = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="new_pdf_database"
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

        st.session_state.database = process_pdf(pdf_path)

        # Store uploaded PDF name
        st.session_state.pdf_name = uploaded_file.name

        # Clear old conversation
        st.session_state.chat_history = []

     st.success(f"{uploaded_file.name} processed successfully!")

if st.session_state.pdf_name:
    st.info(f"Current PDF: {st.session_state.pdf_name}")

# Question section

question = st.text_input(
    "Ask a question about the PDF"
)


if st.button("Get Answer"):

    if st.session_state.database is None:

        st.warning(
            "Please upload and process a PDF first."
        )

    elif question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        # Retrieve relevant chunks
        if "what is" in question.lower() or "about" in question.lower():

            results = st.session_state.database.similarity_search_with_score(
            "title introduction overview",
            k=3
            )

        else:

            results = st.session_state.database.similarity_search_with_score(
            question,
            k=5
          )


        st.write("Retrieved Chunks:")

        context = ""

        for i, (doc, score) in enumerate(results):

            st.write(f"Chunk {i+1}")
            st.write(f"Score: {score}")
            st.write(doc.page_content[:500])
            st.write("---")

            context += (
              f"Page {doc.metadata.get('page',0)+1}:\n"
              + doc.page_content
              + "\n\n"
            )


        st.write("FULL CONTEXT:")
        st.write(context)

        prompt = f"""

        You are a PDF document assistant.

        Answer the question based only on the provided PDF context.

        Context:
        {context}

        Question:
        {question}

        Instructions:
        - First identify what the document is about.
        - Give a short summary when asked about the document.
        - Do not mention information outside the context.
        - Do not guess.

        Answer:

    """


        response = llm.invoke(prompt)

        answer = response.content


        st.write("## Answer")
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