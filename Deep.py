"""
Deepseek.py - Hybrid Codes & Standards Assistant

A Streamlit application that provides intelligent document analysis and Q&A capabilities
for technical standards and codes using RAG (Retrieval-Augmented Generation) architecture.
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

# ===== DOCUMENT PROCESSING IMPORTS =====
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path  # For OCR fallback
import pytesseract  # OCR engine
from langchain.schema import Document  # LangChain document format

# ===== VECTORSTORES AND RETRIEVAL =====
from langchain_community.vectorstores import FAISS  # Vector similarity search
from langchain.embeddings import OpenAIEmbeddings  # Text embeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever  # Keyword-based retrieval

# ===== LLM COMPONENTS =====
from langchain_openai import ChatOpenAI  # GPT models interface
from langchain.chains import RetrievalQA  # Question-answering chain
from langchain.prompts import PromptTemplate  # Prompt engineering

# ===== UI AND UTILITIES =====
import pandas as pd  # Data display
import requests  # For fetching preloaded standards

# ===== CONFIGURATION =====
st.set_page_config(page_title="⚡ Hybrid Codes Assistant", layout="wide")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Get API key from Streamlit secrets

# ===== ADAPTIVE DOCUMENT PROCESSOR =====
class SmartDocumentProcessor:
    """
    Intelligent document processor that handles both native text PDFs and scanned documents
    using OCR fallback when needed.
    """
    def __init__(self):
        self.ocr_fallback = False  # Flag to track if OCR was used
        
    def process(self, pdf_bytes: bytes) -> List[Document]:
        """
        Process PDF bytes through multiple extraction methods:
        1. First try structured extraction (PyPDF)
        2. Fallback to unstructured extraction
        3. Final fallback to OCR if needed
        
        Args:
            pdf_bytes: Binary PDF content
            
        Returns:
            List of processed LangChain documents with metadata
        """
        # Create temp file for processing
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
        """Check if extracted content meets minimum quality standards"""
        if not docs: return False
        return any(len(doc.page_content.strip()) > 100 for doc in docs)
    
    def _add_metadata(self, docs: List[Document]) -> List[Document]:
        """
        Enhance documents with additional metadata:
        - Extracts clause/section numbers
        - Identifies visual elements (tables, figures)
        """
        for doc in docs:
            # Extract clause/section numbers
            clause_match = re.search(r"(Clause|Section|Part)\s*(\d+(?:\.\d+)*)", doc.page_content, re.IGNORECASE)
            if clause_match:
                doc.metadata["clause"] = f"{clause_match.group(1)} {clause_match.group(2)}"
            
            # Flag visual elements
            if any(x in doc.page_content[:100] for x in ["Table", "Figure", "Diagram"]):
                doc.metadata["content_type"] = "visual"
        return docs

# ===== ADAPTIVE CHUNKING =====
def adaptive_chunking(docs: List[Document], is_ocr: bool) -> List[Document]:
    """
    Intelligent document chunking that adapts based on content type:
    - Larger chunks for OCR content to maintain context
    - Precise chunking for native text documents
    
    Args:
        docs: List of documents to chunk
        is_ocr: Boolean indicating if content came from OCR
        
    Returns:
        List of optimally chunked documents
    """
    if is_ocr:
        # OCR-optimized chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Larger chunks
            chunk_overlap=300,  # More overlap for context
            separators=["\n\n", "•", "□", "■"]  # OCR-specific separators
        )
    else:
        # Native text chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\nClause", "\nSection", "\nTable", "\nFigure", "\n\n"],
            keep_separator=True  # Keep structural markers
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

# ===== HYBRID RETRIEVER =====
class HybridRetriever:
    """
    Intelligent retriever that combines:
    - Vector similarity search (FAISS)
    - Keyword search (BM25)
    Dynamically selects best method based on query type
    """
    def __init__(self, chunks: List[Document]):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.faiss = FAISS.from_documents(chunks, self.embeddings)  # Vector store
        self.bm25 = BM25Retriever.from_documents(chunks)  # Keyword store
        self.bm25.k = 5  # Number of results to return
    
    def query(self, question: str) -> List[Document]:
        """
        Route queries to optimal retrieval method:
        - Exact clause/section references → BM25
        - Conceptual questions → Hybrid approach
        """
        # Exact clause/section queries
        if re.search(r"(clause|section|table)\s*[\d\.]+", question, re.IGNORECASE):
            return self.bm25.invoke(question)
        
        # Conceptual queries use hybrid approach
        faiss_results = self.faiss.similarity_search(question, k=5)
        bm25_results = self.bm25.invoke(question)
        
        # Combine and deduplicate results
        combined = {doc.metadata.get("page", ""): doc for doc in faiss_results + bm25_results}
        return list(combined.values())[:5]

# ===== ADAPTIVE QA SYSTEM =====
def create_qa_system(retriever: HybridRetriever, is_complex: bool):
    """
    Creates question-answering system configured for either:
    - Detailed technical answers (GPT-4)
    - Concise responses (GPT-3.5)
    
    Args:
        retriever: Configured HybridRetriever instance
        is_complex: Boolean indicating if question requires detailed answer
        
    Returns:
        Configured RetrievalQA chain
    """
    if is_complex:
        # Detailed answer configuration
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",  # More capable model
            temperature=0.2,  # Slightly creative
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
        # Concise answer configuration
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Faster model
            temperature=0,  # More deterministic
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
        chain_type="stuff",  # Simple question-answering
        chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt)},
        return_source_documents=True  # Return references
    )

# ===== MAIN APPLICATION UI =====
st.title("⚡ Hybrid Codes & Standards Assistant")
st.markdown("""
⚠️ _This adaptive tool provides optimized interpretations based on your document type and questions._
""")

# Document selection UI
PRELOADED_STANDARDS = {
    "None (Upload Your Own)": None,
    "AS/NZS 3000:2018 (Sample)": "https://example.com/as3000.pdf"  # Replace with actual URL
}
selected_std = st.selectbox("Choose standard:", list(PRELOADED_STANDARDS.keys()))

# File upload UI
uploaded_file = st.file_uploader("📎 Or upload PDF", type="pdf")
if not uploaded_file and selected_std == "None (Upload Your Own)":
    st.warning("⚠️ Please upload a PDF or select a standard.")
    st.stop()

# Document processing pipeline
with st.spinner("🔍 Analyzing document..."):
    processor = SmartDocumentProcessor()
    
    # Get document content
    if selected_std != "None (Upload Your Own)":
        pdf_bytes = requests.get(PRELOADED_STANDARDS[selected_std]).content
    else:
        pdf_bytes = uploaded_file.read()
    
    # Process and prepare document
    docs = processor.process(pdf_bytes)
    chunks = adaptive_chunking(docs, processor.ocr_fallback)
    retriever = HybridRetriever(chunks)
    st.success(f"✅ Document ready! (OCR: {'Yes' if processor.ocr_fallback else 'No'})")

# Question handling
query = st.text_input("💬 Ask about the standard:")
if query:
    start_time = time.time()
    
    # Determine query complexity
    is_complex = any(x in query.lower() for x in ["explain", "compare", "why", "how"])
    
    with st.spinner("🔍 Generating answer..."):
        qa = create_qa_system(retriever, is_complex)
        result = qa({"query": query})
        
        # Display results
        st.subheader("🔍 Answer")
        st.write(result["result"])
        
        # Show performance metrics
        st.subheader("⚙️ Performance Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Response Time", f"{time.time() - start_time:.2f}s")
        col2.metric("Sources Used", len(result["source_documents"]))
        
        # Display source references
        st.subheader("📚 Source References")
        sources = []
        for doc in result["source_documents"][:3]:  # Show top 3 sources
            clause = doc.metadata.get("clause", "N/A")
            sources.append({
                "Page": doc.metadata.get("page", "N/A"),
                "Clause": clause,
                "Preview": doc.page_content[:200] + "..."  # Preview snippet
            })
        
        st.dataframe(pd.DataFrame(sources), hide_index=True)
