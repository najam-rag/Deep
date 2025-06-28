import streamlit as st
import os
import tempfile
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# === App Config ===
st.set_page_config(page_title="📘 Ask Your Standards Assistant", layout="wide")
st.title("📘 Ask Your Standards Assistant")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# === File Upload ===
st.markdown("Upload PDF (e.g. AS3000)")
uploaded_file = st.file_uploader("Drag and drop file here", type=["pdf"])

# === Prompt Template ===
def create_prompt_template(is_complex: bool):
    template = """
    {instruction}
    Context: {context}
    Question: {question}
    Rules:
    1. Cite sources like [Clause X.Y]
    {rules}
    """
    return PromptTemplate.from_template(template).partial(
        instruction="Answer in detail with technical precision:" if is_complex else "Give concise answer:",
        rules="2. Explain underlying principles\n3. Compare to related standards if relevant" if is_complex else "2. Keep answer under 2 sentences"
    )

# === Document Chunking ===
def chunk_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# === QA System ===
def create_qa_system(chunks: List[Document], is_complex: bool):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    llm = ChatOpenAI(model="gpt-4-turbo-preview" if is_complex else "gpt-3.5-turbo", temperature=0.2 if is_complex else 0, openai_api_key=OPENAI_API_KEY)
    prompt = create_prompt_template(is_complex)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa

# === Main Workflow ===
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.success(f"Uploaded: {uploaded_file.name}")
    st.markdown("---")

    with st.spinner("🔍 Processing document and building QA system..."):
        chunks = chunk_pdf(file_path)
        qa = create_qa_system(chunks, is_complex=True)
        st.session_state.qa = qa

# === Ask Questions ===
if "qa" in st.session_state:
    question = st.text_input("🔎 Enter your question")
    if question:
        with st.spinner("Generating answer..."):
            response = st.session_state.qa.invoke({"query": question})
            st.markdown("### 📄 Answer")
            st.write(response["result"])

            st.markdown("---")
            st.markdown("#### 📚 Source Excerpts")
            for doc in response["source_documents"]:
                st.markdown(f"**Page**: {doc.metadata.get('page', 'N/A')}")
                st.markdown(f"```\n{doc.page_content[:1000]}\n```")
