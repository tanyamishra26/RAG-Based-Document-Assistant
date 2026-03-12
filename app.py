import streamlit as st
import os

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.vector_store import create_vector_store
from rag.retriever import get_retriever
from rag.rag_chain import create_rag_chain


UPLOAD_FOLDER = "uploads"


st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)


st.title("📄 RAG-Based Document Assistant")
st.markdown("Chat with your documents using Retrieval-Augmented Generation.")


# Sidebar
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)


if uploaded_files:

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    for file in uploaded_files:

        file_path = os.path.join(UPLOAD_FOLDER, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    st.sidebar.success("Documents uploaded successfully!")


    docs = load_documents(UPLOAD_FOLDER)
    chunks = split_documents(docs)
    vector_store = create_vector_store(chunks)
    retriever = get_retriever(vector_store)
    rag_chain = create_rag_chain(retriever)


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
question = st.chat_input("Ask a question about the documents")

if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)


    with st.spinner("Thinking..."):

        result = rag_chain.invoke(question)

        answer = result["answer"]
        sources = result["sources"]


    response = answer

    if sources:

        unique_sources = set()

        for doc in sources:
            source = doc.metadata["source"]
            page = doc.metadata.get("page", "N/A")

            unique_sources.add((source, page))

        source_text = "\n\n**Sources:**\n"

        for source, page in unique_sources:
            source_text += f"- {source} — page {page}\n"

        response += source_text


    with st.chat_message("assistant"):
        st.markdown(response)


    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )