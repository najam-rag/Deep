"""
Deepseek.py - Hybrid Codes & Standards Assistant (Robust Version)
"""

import streamlit as st
import os
import hashlib
import tempfile
import re
import time
from datetime import datetime
from functools import wraps
from typing import List, Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAIError

# ===== DOCUMENT PROCESSING IMPORTS =====
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document

# ===== VECTORSTORES AND RETRIEVAL =====
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# ===== LLM COMPONENTS =====
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ===== UI AND UTILITIES =====
import pandas as pd
import requests

# ===== CONFIGURATION =====
st.set_page_config(page_title="⚡ Hybrid Codes Assistant", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in secrets")
    st.stop()

# ===== ENHANCED DOCUMENT PROCESSOR =====
class SmartDocumentProcessor:
    def __init__(self):
        self.ocr_fallback = False
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def process(self, pdf_bytes: bytes) -> List[Document]:
        """Robust document processing with retry logic"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            temp_path = tmp_file.name
        
        try:
            # Attempt 1: Structured PDF extraction
            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                if self._is_quality_acceptable(docs):
                    return self._add_metadata(docs)
            except Exception as e:
                st.warning(f"PyPDF failed: {str(e)}")
            
            # Attempt 2: Unstructured PDF extraction
            try:
                loader = UnstructuredPDFLoader(temp_path, mode="elements")
                docs = loader.load()
                if self._is_quality_acceptable(docs):
                    return self._add_metadata(docs)
            except Exception as e:
                st.warning(f"Unstructured failed: {str(e)}")
            
            # Final fallback: OCR processing
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
        if not docs: return False
        return any(len(doc.page_content.strip()) > 100 for doc in docs)
    
    def _add_metadata(self, docs: List[Document]) -> List[Document]:
        for doc in docs:
            clause_match = re.search(r"(Clause|Section|Part)\s*(\d+(?:\.\d+)*)", doc.page_content, re.IGNORECASE)
            if clause_match:
                doc.metadata["clause"] = f"{clause_match.group(1)} {clause_match.group(2)}"
            
            if any(x in doc.page_content[:100] for x in ["Table", "Figure", "Diagram"]):
                doc.metadata["content_type"] = "visual"
        return docs

# ===== ROBUST CHUNKING =====
def adaptive_chunking(docs: List[Document], is_ocr: bool) -> List[Document]:
    """Safe chunking with token limits"""
    try:
        if is_ocr:
            splitter = TokenTextSplitter(
                chunk_size=800,  # Reduced for OCR
                chunk_overlap=200,
                encoding_name="cl100k_base"
            )
        else:
            splitter = TokenTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                encoding_name="cl100k_base"
            )
        
        chunks = splitter.split_documents(docs)
        
        # Validate chunks
        if not chunks:
            st.error("No valid chunks created from document")
            st.stop()
            
        return chunks
        
    except Exception as e:
        st.error(f"Chunking failed: {str(e)}")
        st.stop()

# ===== FAILSAFE RETRIEVER =====
class HybridRetriever:
    def __init__(self, chunks: List[Document]):
        self.embeddings = self._get_embeddings()
        self.faiss = self._create_faiss(chunks)
        self.bm25 = BM25Retriever.from_documents(chunks)
        self.bm25.k = 5
    
    def _get_embeddings(self):
        """Fallback to local embeddings if OpenAI fails"""
        try:
            return OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model="text-embedding-3-small"  # More efficient
            )
        except Exception as e:
            st.warning(f"OpenAI embeddings failed: {str(e)}")
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings()
            except:
                st.error("No embedding model available")
                st.stop()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _create_faiss(self, chunks):
        """Safe FAISS creation with chunk validation"""
        if not chunks:
            st.error("No chunks provided for vector store")
            st.stop()
            
        if sum(len(c.page_content) for c in chunks) > 1000000:  # ~1M characters
            st.warning("Large document - processing first 50 chunks only")
            chunks = chunks[:50]
            
        try:
            return FAISS.from_documents(chunks, self.embeddings)
        except OpenAIError as e:
            st.error(f"OpenAI API error: {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"Vector store creation failed: {str(e)}")
            st.stop()
    
    def query(self, question: str) -> List[Document]:
        try:
            if re.search(r"(clause|section|table)\s*[\d\.]+", question, re.IGNORECASE):
                return self.bm25.invoke(question)
            
            faiss_results = self.faiss.similarity_search(question, k=5)
            bm25_results = self.bm25.invoke(question)
            
            combined = {doc.metadata.get("page", ""): doc for doc in faiss_results + bm25_results}
            return list(combined.values())[:5]
            
        except Exception as e:
            st.error(f"Retrieval failed: {str(e)}")
            return []

# ===== MAIN APPLICATION =====
def main():
    st.title("⚡ Hybrid Codes & Standards Assistant")
    
    # Document selection
    PRELOADED_STANDARDS = {
        "None (Upload Your Own)": None,
        "AS/NZS 3000:2018 (Sample)": "https://example.com/as3000.pdf"
    }
    
    selected_std = st.selectbox("Choose standard:", list(PRELOADED_STANDARDS.keys()))
    uploaded_file = st.file_uploader("📎 Or upload PDF", type="pdf")
    
    if not uploaded_file and selected_std == "None (Upload Your Own)":
        st.warning("⚠️ Please upload a PDF or select a standard.")
        return
    
    # Document processing
    with st.spinner("🔍 Analyzing document..."):
        try:
            processor = SmartDocumentProcessor()
            
            if selected_std != "None (Upload Your Own)":
                pdf_bytes = requests.get(PRELOADED_STANDARDS[selected_std]).content
            else:
                pdf_bytes = uploaded_file.read()
            
            docs = processor.process(pdf_bytes)
            chunks = adaptive_chunking(docs, processor.ocr_fallback)
            retriever = HybridRetriever(chunks)
            st.success(f"✅ Document ready! (OCR: {'Yes' if processor.ocr_fallback else 'No'})")
            
            # Query handling
            query = st.text_input("💬 Ask about the standard:")
            if query:
                start_time = time.time()
                
                is_complex = any(x in query.lower() for x in ["explain", "compare", "why", "how"])
                llm = ChatOpenAI(
                    model="gpt-4-turbo-preview" if is_complex else "gpt-3.5-turbo",
                    temperature=0.2 if is_complex else 0,
                    openai_api_key=OPENAI_API_KEY
                )
                
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff",
                    return_source_documents=True
                )
                
                with st.spinner("🔍 Generating answer..."):
                    result = qa({"query": query})
                    
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
                    
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()
