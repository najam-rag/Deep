import streamlit as st
import os
import hashlib
import tempfile
import re
import time
from datetime import datetime
from functools import wraps
from typing import List, Tuple, Optional

# Document Processing
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document

# Vectorstores and Retrieval
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever

# LLM Components
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# UI and Utilities
import pandas as pd
import requests

# ===== Configuration =====
st.set_page_config(page_title="⚡ Hybrid Codes Assistant", layout="wide")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ===== Authentication =====
def check_password():
    if 'authenticated' not in st.session_state:
        st.sidebar.header("🔐 Login")
        password = st.sidebar.text_input("Enter password", type="password")
        if password == "password":
            st.session_state.authenticated = True
            st.rerun()
        elif password:
            st.warning("🚫 Access denied")
            st.stop()
        else:
            st.stop()
    return True

if not check_password():
    st.stop()

# ===== Adaptive Document Processor =====
class SmartDocumentProcessor:
    def __init__(self):
        self.ocr_fallback = False

    def process(self, pdf_bytes: bytes) -> List[Document]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            temp_path = tmp_file.name

        try:
            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                if self._is_quality_acceptable(docs):
                    return self._add_metadata(docs)
            except Exception as e:
                st.warning(f"PyPDF failed: {str(e)}")

            try:
                loader = UnstructuredPDFLoader(temp_path, mode="elements")
                docs = loader.load()
                if self._is_quality_acceptable(docs):
                    return self._add_metadata(docs)
            except Exception as e:
                st.warning(f"Unstructured failed: {str(e)}")

            self.ocr_fallback = True
            st.warning("⚠️ Using OCR...")
            images = convert_from_path(temp_path, dpi=200)
            texts = [pytesseract.image_to_string(img) for img in images]
            docs = [Document(page_content=t, metadata={"page": i + 1, "source": "OCR"})
                    for i, t in enumerate(texts) if t.strip()]
            return self._add_metadata(docs)

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.stop()
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    def _is_quality_acceptable(self, docs: List[Document]) -> bool:
        if not docs:
            return False
        return any(len(doc.page_content.strip()) > 100 for doc in docs)

    def _add_metadata(self, docs: List[Document]) -> List[Document]:
        for doc in docs:
            clause_match = re.search(r"(Clause|Section|Part)\s*(\d+(?:\.\d+)*)", doc.page_content, re.IGNORECASE)
            if clause_match:
                doc.metadata["clause"] = f"{clause_match.group(1)} {clause_match.group(2)}"
            if any(x in doc.page_content[:100] for x in ["Table", "Figure", "Diagram"]):
                doc.metadata["content_type"] = "visual"
        return docs

# ===== Adaptive Chunker =====
def adaptive_chunking(docs: List[Document], is_ocr: bool) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200 if is_ocr else 1000,
        chunk_overlap=300 if is_ocr else 200,
        separators=["\n\n", "•", "□", "■"] if is_ocr else ["\nClause", "\nSection", "\nTable", "\nFigure", "\n\n"],
        keep_separator=not is_ocr
    )

    chunks = splitter.split_documents(docs)
    merged_chunks = []
    buffer = ""
    for chunk in chunks:
        if chunk.metadata.get("content_type") == "visual":
            if buffer:
                merged_chunks.append(Document(page_content=buffer, metadata=chunks[0].metadata))
                buffer = ""
            merged_chunks.append(chunk)
        else:
            buffer += "\n\n" + chunk.page_content

    if buffer:
        merged_chunks.append(Document(page_content=buffer, metadata=chunks[0].metadata))

    return merged_chunks

# ===== Smart Retriever =====
class HybridRetriever:
    def __init__(self, chunks: List[Document]):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        self.faiss = FAISS.from_documents(chunks, self.embeddings)
        texts = [doc.page_content for doc in chunks]
        self.bm25 = BM25Retriever.from_texts(texts)
        self.bm25.k = 5

    def query(self, question: str) -> List[Document]:
        if re.search(r"(clause|section|table)\s*[\d\.]+", question, re.IGNORECASE):
            return self.bm25.invoke(question)
        faiss_results = self.faiss.similarity_search(question, k=5)
        bm25_results = self.bm25.invoke(question)
        combined = {doc.metadata.get("page", ""): doc for doc in faiss_results + bm25_results}
        return list(combined.values())[:5]

# ===== Adaptive QA System =====
def create_qa_system(retriever: HybridRetriever, is_complex: bool):
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview" if is_complex else "gpt-3.5-turbo",
        temperature=0.2 if is_complex else 0,
        openai_api_key=OPENAI_API_KEY
    )

    prompt_text = """Answer in detail with technical precision:
Context: {context}
Question: {question}
Rules:
1. Cite sources like [Clause X.Y]
2. Explain underlying principles
3. Compare to related standards if relevant""" if is_complex else """Give concise answer:
Context: {context}
Question: {question}
Rules:
1. Cite source if available
2. Keep answer under 2 sentences"""

    prompt = PromptTemplate.from_template(prompt_text)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa
