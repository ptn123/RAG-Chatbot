import streamlit as st
import os
import faiss
import pickle
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Document Chatbot By PTN")
st.sidebar.header("Upload your documents")

uploaded_files = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

def process_documents(files):
    documents = []
    if not os.path.exists("temp"):
        os.makedirs("temp")
    for file in files:
        file_path = os.path.join("temp", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)
        documents.extend(loader.load())
    return documents

if uploaded_files:
    st.sidebar.write("📂 Processing files...")
    raw_docs = process_documents(uploaded_files)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(raw_docs)
    texts = [doc.page_content for doc in split_docs]
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(texts)
    doc_vectors_np = doc_vectors.toarray().astype('float32')
    index = faiss.IndexFlatL2(doc_vectors_np.shape[1])
    index.add(doc_vectors_np)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    faiss.write_index(index, "vectorstore.index")
    st.sidebar.success("✅ Documents processed!")

if os.path.exists("vectorizer.pkl") and os.path.exists("vectorstore.index"):
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    index = faiss.read_index("vectorstore.index")
    user_query = st.text_input("💬 Ask something about your document:")
    if user_query:
        query_vector = vectorizer.transform([user_query]).toarray().astype('float32')
        _, indices = index.search(query_vector, k=3)
        results = [texts[i] for i in indices[0] if i < len(texts)]
        st.subheader("📑 Relevant Information:")
        for i, text in enumerate(results):
            st.write(f"**Snippet {i+1}:** {text[:300]}...")

