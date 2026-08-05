import os

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

# Embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Create vector database

@st.cache_resource
def create_vectorstore(_chunks, _embeddings):

    database = Chroma.from_documents(
        documents=_chunks,
        embedding=_embeddings
    )

    return database

# PDF processing function

def process_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Clean extracted text
    for doc in documents:
        doc.page_content = " ".join(doc.page_content.split())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    database = create_vectorstore(
        chunks,
        embeddings
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
            st.session_state.chat_history = []

        st.success("PDF processed successfully!")

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
        results = st.session_state.database.similarity_search_with_score(
            question,
            k=4
        )


        st.write("## Retrieved Chunks")

        context = ""


        for i, (doc, score) in enumerate(results):

            st.write(f"### Chunk {i+1}")
            st.write("Score:", score)
            st.write(doc.page_content[:500])
            st.write("---")


            # Add retrieved text to context
            context += f"""
Source {i+1}:

{doc.page_content}

"""


        prompt = f"""
You are a helpful PDF assistant.

Answer the question only using the provided context.

Rules:
- Do not use outside knowledge.
- Do not guess.
- If the answer is not available in the context, say:
"I could not find this information in the document."

Context:

{context}

Question:

{question}

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