import streamlit as st
import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import PromptTemplate
import tempfile
import re
from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document
import pandas as pd

# ===== Configuration =====
st.set_page_config(page_title="⚡ Codes & Standards Assistant", layout="wide")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Set in Streamlit secrets

# ===== Authentication =====
st.sidebar.header("🔐 Login")
password = st.sidebar.text_input("Enter password", type="password")
if password != "Password":
    st.warning("🚫 Access denied. Please enter the correct password.")
    st.stop()

# ===== UI Setup =====
st.title("⚡ Codes & Standards Assistant")
st.markdown("""
⚠️ _This tool provides interpretations based on standards. Always verify with official documents._
""")

# ===== Preloaded Standards =====
PRELOADED_STANDARDS = {
    "None (Upload Your Own)": None,
    "AS/NZS 3000:2018 (Sample)": "https://example.com/as3000.pdf"  # Replace with actual URL
}
selected_std = st.selectbox("Choose a preloaded standard:", list(PRELOADED_STANDARDS.keys()))

# ===== File Upload =====
uploaded_file = st.file_uploader("📎 Or upload your own PDF", type="pdf")
if not uploaded_file and selected_std == "None (Upload Your Own)":
    st.warning("⚠️ Please upload a PDF or select a preloaded standard.")
    st.stop()

# ===== Document Processing =====
@st.cache_resource
def load_vectorstore(pdf_bytes: bytes):
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
    vectorstore_dir = f"vectorstores/{pdf_hash}"
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Try loading existing vectorstore
    if os.path.exists(vectorstore_dir):
        try:
            return FAISS.load_local(vectorstore_dir, embeddings).as_retriever()
        except:
            os.system(f"rm -rf {vectorstore_dir}")  # Clear corrupted data

    # Process new PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        temp_pdf_path = tmp_file.name

    # Load with PyPDF first, fallback to OCR
    try:
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
    except:
        st.warning("⚠️ Using OCR for scanned PDF...")
        try:
            images = convert_from_path(temp_pdf_path)
            texts = [pytesseract.image_to_string(img) for img in images]
            docs = [Document(page_content=t, metadata={"page": i+1}) for i,t in enumerate(texts)]
        except Exception as e:
            st.error(f"❌ OCR failed: {str(e)}")
            st.stop()

    # Optimized chunking for standards
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\nClause", "\nSection", "\nTable", "\nFigure"]
    )
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        st.error("❌ No extractable text found. Try a different PDF.")
        st.stop()

    # Build retrievers
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vectorstore_dir)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    
    return EnsembleRetriever(
        retrievers=[bm25_retriever, db.as_retriever()],
        weights=[0.3, 0.7]
    )

# ===== Load Data =====
if selected_std != "None (Upload Your Own)":
    import requests
    response = requests.get(PRELOADED_STANDARDS[selected_std])
    pdf_bytes = response.content
else:
    pdf_bytes = uploaded_file.read()

retriever = load_vectorstore(pdf_bytes)
st.success("✅ Document loaded successfully!")

# ===== QA System =====
prompt_template = """
You are an expert in technical standards. Answer ONLY using these rules:
1. If the answer isn't in the context, say "Not found in the document."
2. Always cite the exact clause like [Clause 5.5.3].
3. Be concise and technical.

CONTEXT: {context}
QUESTION: {question}
ANSWER:
"""
prompt = PromptTemplate.from_template(prompt_template)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,  # Reduce creativity for factual answers
    openai_api_key=OPENAI_API_KEY
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ===== Query Interface =====
query = st.text_input("💬 Ask a question about the standard:")
if query:
    result = qa.invoke({"query": query})
    
    # Display answer
    st.subheader("🔍 Answer")
    st.write(result["result"])
    
    # Confidence score
    confidence = min(len(result["source_documents"]) / 3, 1.0)
    st.metric("Confidence Score", f"{confidence:.0%}")
    
    # Source preview
    st.subheader("📚 Source Clauses")
    sources_df = pd.DataFrame([{
        "Page": doc.metadata.get("page", "N/A"),
        "Clause": re.search(r"(Clause\s*\d+\.\d+(?:\.\d+)*)", doc.page_content).group(1) if re.search(r"Clause\s*\d+\.\d+", doc.page_content) else "N/A",
        "Preview": doc.page_content[:200] + "..."
    } for doc in result["source_documents"][:3]])
    
    st.dataframe(
        sources_df,
        column_config={"Preview": st.column_config.TextColumn("Preview", width="large")},
        hide_index=True
    )
