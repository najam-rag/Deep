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
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever

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
    """Simplified authentication for demo purposes"""
    if 'authenticated' not in st.session_state:
        st.sidebar.header("🔐 Login")
        password = st.sidebar.text_input("Enter password", type="password")
        if password == "password":  # Replace with secure auth in production
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
        """Intelligently process PDF with format detection"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            temp_path = tmp_file.name
        
        try:
            # First attempt - structured extraction
            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                if self._is_quality_acceptable(docs):
                    return self._add_metadata(docs)
            except Exception as e:
                st.warning(f"PyPDF failed: {str(e)}")
            
            # Second attempt - unstructured extraction
            try:
                loader = UnstructuredPDFLoader(temp_path, mode="elements")
                docs = loader.load()
                if self._is_quality_acceptable(docs):
                    return self._add_metadata(docs)
            except Exception as e:
                st.warning(f"Unstructured failed: {str(e)}")
            
            # Final fallback - OCR
            self.ocr_fallback = True
            st.warning("⚠️ Using OCR...")
            images = convert_from_path(temp_path)
            texts = [pytesseract.image_to_string(img) for img in images]
            docs = [Document(page_content=t, metadata={"page": i+1, "source": "OCR"}) 
                   for i, t in enumerate(texts) if t.strip()]
            return self._add_metadata(docs)
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.stop()
        finally:
            try: os.unlink(temp_path)
            except: pass
    
    def _is_quality_acceptable(self, docs: List[Document]) -> bool:
        """Check if extraction produced usable content"""
        if not docs: return False
        return any(len(doc.page_content.strip()) > 100 for doc in docs)
    
    def _add_metadata(self, docs: List[Document]) -> List[Document]:
        """Enhanced metadata extraction"""
        for doc in docs:
            # Extract clause/section numbers
            clause_match = re.search(r"(Clause|Section|Part)\s*(\d+(?:\.\d+)*)", doc.page_content, re.IGNORECASE)
            if clause_match:
                doc.metadata["clause"] = f"{clause_match.group(1)} {clause_match.group(2)}"
            
            # Flag visual elements
            if any(x in doc.page_content[:100] for x in ["Table", "Figure", "Diagram"]):
                doc.metadata["content_type"] = "visual"
        return docs

# ===== Adaptive Chunker =====
def adaptive_chunking(docs: List[Document], is_ocr: bool) -> List[Document]:
    """Choose chunking strategy based on content type"""
    if is_ocr:
        # Larger chunks for OCR to maintain context
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            separators=["\n\n", "•", "□", "■"]
        )
    else:
        # Precise chunking for native text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\nClause", "\nSection", "\nTable", "\nFigure", "\n\n"],
            keep_separator=True
        )
    
    chunks = splitter.split_documents(docs)
    
    # Post-process to merge fragmented tables/figures
    merged_chunks = []
    buffer = ""
    for chunk in chunks:
        if chunk.metadata.get("content_type") == "visual":
            if buffer:
                merged_chunks.append(Document(
                    page_content=buffer,
                    metadata=chunks[0].metadata
                ))
                buffer = ""
            merged_chunks.append(chunk)
        else:
            buffer += "\n\n" + chunk.page_content
    
    if buffer:
        merged_chunks.append(Document(
            page_content=buffer,
            metadata=chunks[0].metadata
        ))
    
    return merged_chunks

# ===== Smart Retriever =====
class HybridRetriever:
    def __init__(self, chunks: List[Document]):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.faiss = FAISS.from_documents(chunks, self.embeddings)
        self.bm25 = BM25Retriever.from_documents(chunks)
        self.bm25.k = 5
    
    def query(self, question: str) -> List[Document]:
        """Dynamically choose retrieval method"""
        # Exact clause/section queries
        if re.search(r"(clause|section|table)\s*[\d\.]+", question, re.IGNORECASE):
            return self.bm25.invoke(question)
        
        # Conceptual queries
        faiss_results = self.faiss.similarity_search(question, k=5)
        bm25_results = self.bm25.invoke(question)
        
        # Hybrid reranking
        combined = {doc.metadata.get("page", ""): doc for doc in faiss_results + bm25_results}
        return list(combined.values())[:5]

# ===== Adaptive QA System =====
def create_qa_system(retriever: HybridRetriever, is_complex: bool):
    """Configure QA chain based on query complexity"""
    if is_complex:
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.2,
            openai_api_key=OPENAI_API_KEY
        )
        prompt = """Answer in detail with technical precision:
        Context: {context}
        Question: {question}
        Rules:
        1. Cite sources like [Clause X.Y]
        2. Explain underlying principles
        3. Compare to related standards if relevant"""
    else:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        prompt = """Give concise answer:
        Context: {context}
        Question: {question}
        Rules:
        1. Cite source if available
        2. Keep answer under 2 sentences"""
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt)},
        return_source_documents=True
    )

# ===== Main Application =====
st.title("⚡ Hybrid Codes & Standards Assistant")
st.markdown("""
⚠️ _This adaptive tool provides optimized interpretations based on your document type and questions._
""")

# Document Selection
PRELOADED_STANDARDS = {
    "None (Upload Your Own)": None,
    "AS/NZS 3000:2018 (Sample)": "https://example.com/as3000.pdf"
}
selected_std = st.selectbox("Choose standard:", list(PRELOADED_STANDARDS.keys()))

# File Upload
uploaded_file = st.file_uploader("📎 Or upload PDF", type="pdf")
if not uploaded_file and selected_std == "None (Upload Your Own)":
    st.warning("⚠️ Please upload a PDF or select a standard.")
    st.stop()

# Document Processing
with st.spinner("🔍 Analyzing document..."):
    processor = SmartDocumentProcessor()
    
    if selected_std != "None (Upload Your Own)":
        pdf_bytes = requests.get(PRELOADED_STANDARDS[selected_std]).content
    else:
        pdf_bytes = uploaded_file.read()
    
    docs = processor.process(pdf_bytes)
    chunks = adaptive_chunking(docs, processor.ocr_fallback)
    retriever = HybridRetriever(chunks)
    st.success(f"✅ Document ready! (OCR: {'Yes' if processor.ocr_fallback else 'No'})")

# Query Handling
query = st.text_input("💬 Ask about the standard:")
if query:
    start_time = time.time()
    
    # Determine query complexity
    is_complex = any(x in query.lower() for x in ["explain", "compare", "why", "how"])
    
    with st.spinner("🔍 Generating answer..."):
        qa = create_qa_system(retriever, is_complex)
        result = qa({"query": query})
        
        # Display Results
        st.subheader("🔍 Answer")
        st.write(result["result"])
        
        st.subheader("⚙️ Performance Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Response Time", f"{time.time() - start_time:.2f}s")
        col2.metric("Sources Used", len(result["source_documents"]))
        
        st.subheader("📚 Source References")
        sources = []
        for doc in result["source_documents"][:3]:
            clause = doc.metadata.get("clause", "N/A")
            sources.append({
                "Page": doc.metadata.get("page", "N/A"),
                "Clause": clause,
                "Preview": doc.page_content[:200] + "..."
            })
        
        st.dataframe(pd.DataFrame(sources), hide_index=True)
