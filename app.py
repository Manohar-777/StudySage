import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Setup page layout and branding
st.set_page_config(
    page_title="StudySage — AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Glassmorphism & Sleek Dark theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global font setup */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background gradients */
    .stApp {
        background: radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.08), transparent 40%),
                    radial-gradient(circle at 20% 80%, rgba(139, 92, 246, 0.08), transparent 40%),
                    #0b0f19;
        color: #f1f5f9;
    }
    
    /* Sidebar premium coloring */
    [data-testid="stSidebar"] {
        background: #0f1322 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Titles and typography */
    h1 {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }
    
    /* Card containers / alerts */
    div.stAlert {
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(12px);
        color: #f1f5f9;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Inputs text fields */
    .stTextInput>div>div>input {
        background-color: #13192b !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        transition: border 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2) !important;
    }
    
    /* Source badges and cards */
    .source-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 4px solid #818cf8;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
        margin-bottom: 10px;
        transition: transform 0.2s ease;
    }
    
    .source-card:hover {
        transform: translateX(4px);
        background: rgba(255, 255, 255, 0.03);
    }
    
    .source-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #c084fc;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .source-content {
        font-size: 0.88rem;
        color: #cbd5e1;
        margin-top: 6px;
        line-height: 1.5;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Optional OCR Setup -----------------
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def ocr_pdf(path):
    """Fallback OCR pipeline for scanned/image-only PDFs."""
    if not OCR_AVAILABLE:
        raise ImportError("pytesseract or pdf2image package not installed.")
    
    # Convert PDF to PIL Images
    images = convert_from_path(path)
    documents = []
    for idx, img in enumerate(images):
        page_num = idx + 1
        # Try bilingual extraction (Telugu + English)
        try:
            text = pytesseract.image_to_string(img, lang="tel+eng")
        except Exception:
            try:
                text = pytesseract.image_to_string(img, lang="eng")
            except Exception as e:
                text = f"[OCR Failed: Tesseract binaries not configured properly or 'tel'/'eng' data missing: {e}]"
        
        doc = Document(
            page_content=text,
            metadata={"source": path, "page": page_num}
        )
        documents.append(doc)
    return documents

# ----------------- Environment / Key Configuration -----------------
load_dotenv()
api_key_from_state = st.session_state.get("custom_api_key", "")
active_api_key = os.environ.get("GOOGLE_API_KEY") or api_key_from_state

# Try streamlit secrets
if not active_api_key:
    try:
        active_api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass

# Propagate environment variable if found
if active_api_key:
    os.environ["GOOGLE_API_KEY"] = active_api_key
    # Check if we should initialize embeddings & LLM
    try:
        EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        MODELS_LOADED = True
    except Exception as e:
        MODELS_LOADED = False
        LOAD_ERROR = str(e)
else:
    MODELS_LOADED = False
    LOAD_ERROR = "API Key not provided yet."

# ----------------- Sidebar Configuration -----------------
with st.sidebar:
    st.image("https://img.icons8.com/gradient/96/000000/bookmark.png", width=70)
    st.markdown("## StudySage Panel")
    st.markdown("🤖 *An AI-powered multi-lingual RAG document search*")
    st.markdown("---")
    
    # API Key Input if not set
    if not active_api_key:
        st.warning("⚠️ Google API Key is missing!")
        user_key = st.text_input("Enter your Gemini API Key:", type="password", help="Paste your GOOGLE_API_KEY from Google AI Studio")
        if user_key:
            st.session_state["custom_api_key"] = user_key
            os.environ["GOOGLE_API_KEY"] = user_key
            st.success("API Key saved for this session!")
            st.rerun()
        st.markdown("---")
        
    st.header("1) Ingest Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Textbook / Notes (PDFs)", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    # Advanced ingestion settings
    st.markdown("#### Settings")
    chunk_size = st.slider("Chunk Size (characters)", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Overlap Size (characters)", 50, 400, 150, step=50)
    enable_ocr = st.checkbox("Enable OCR Fallback", value=True, help="Scan PDFs for text, fallback to Tesseract if empty pages are found")
    
    build_btn = st.button("Build Knowledge Base")

# ----------------- RAG Construction Helper -----------------
def build_vectorstore(pdf_paths, size, overlap):
    """Load, split, embed and index pages in FAISS."""
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        try:
            docs = loader.load()
        except Exception as e:
            docs = []
            st.sidebar.error(f"Error loading {os.path.basename(path)}: {e}")
            
        # Check if PDF text is empty/scanned
        total_len = sum(len(d.page_content.strip()) for d in docs)
        
        if total_len < 150 and enable_ocr:
            st.sidebar.info(f"'{os.path.basename(path)}' appears to be scanned. Running OCR fallback...")
            try:
                ocr_docs = ocr_pdf(path)
                all_docs.extend(ocr_docs)
            except Exception as ocr_err:
                st.sidebar.error(f"OCR failed for {os.path.basename(path)}: {ocr_err}. Loaded empty content.")
                all_docs.extend(docs)
        else:
            all_docs.extend(docs)
            
    if not all_docs:
        return None

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_documents(all_docs)
    
    # Embed and index
    vectorstore = FAISS.from_documents(chunks, EMBEDDINGS)
    return vectorstore

# Perform ingestion
if build_btn:
    if not MODELS_LOADED:
        st.sidebar.error(f"Cannot build knowledge base: {LOAD_ERROR}")
    elif not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF file first.")
    else:
        os.makedirs("data", exist_ok=True)
        paths = []
        for file in uploaded_files:
            file_path = os.path.join("data", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            paths.append(file_path)
            
        with st.sidebar.spinner("Structuring, embedding and storing your knowledge..."):
            try:
                st.session_state.vs = build_vectorstore(paths, chunk_size, chunk_overlap)
                if st.session_state.vs:
                    st.sidebar.success(f"Success! Indexed {len(paths)} document(s). Ready to search! 📚")
                else:
                    st.sidebar.error("Failed to index: No text extracted from PDFs.")
            except Exception as e:
                st.sidebar.error(f"Indexing failed: {e}")

# ----------------- Main UI and Search Interface -----------------
st.markdown("<h1>StudySage — AI Study Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload your lecture notes/textbooks and ask questions in Telugu (తెలుగు) or English. Answers are verified and cited from your documents.</p>", unsafe_allow_html=True)

# App status check
if not MODELS_LOADED:
    st.info("👋 Welcome to StudySage! To get started, please paste a **Google Gemini API Key** in the sidebar. You can get one for free at Google AI Studio.")
elif "vs" not in st.session_state:
    st.info("💡 **Knowledge Base is Empty**: Upload one or more study PDFs (e.g. an NCERT or AP SCERT textbook) and click **Build Knowledge Base** in the sidebar to start asking questions.")
else:
    st.success("✅ **Knowledge Base active**: StudySage is ready to answer questions based on your documents.")

# Form for question asking
st.markdown("### 🔍 Ask a Question")
question = st.text_input(
    "Enter your question here (in Telugu or English):", 
    placeholder="e.g. What is photosynthesis? or కిరణజన్య సంయోగ క్రియ అంటే ఏమిటి?"
)

if question:
    if not MODELS_LOADED:
        st.error("Please configure your Gemini API Key in the sidebar.")
    elif "vs" not in st.session_state:
        st.warning("Please upload PDFs and build the knowledge base in the sidebar first.")
    else:
        retriever = st.session_state.vs.as_retriever(search_kwargs={"k": 4})
        
        # Multilingual Prompt Template
        PROMPT = PromptTemplate(
            template=(
                "You are an expert study assistant. Answer the user's question using ONLY the provided document context.\n"
                "Provide detailed, structured explanations. If the answer is not contained in the context, say: "
                "'I couldn't find the answer in the provided documents.'\n"
                "Crucial instruction: Answer in the EXACT SAME language that the user asked their question in (Telugu/తెలుగు or English).\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=LLM,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        
        with st.spinner("Analyzing context and drafting answer..."):
            try:
                response = qa.invoke({"query": question})
                
                st.markdown("### 📚 StudySage Answer")
                st.write(response["result"])
                
                st.markdown("---")
                # Show reference citations with styling
                with st.expander("📄 Citations & Sources Used", expanded=True):
                    src_docs = response.get("source_documents", [])
                    if not src_docs:
                        st.caption("No sources matched the query.")
                    else:
                        for doc in src_docs:
                            filename = os.path.basename(doc.metadata.get("source", "Unknown Document"))
                            # page index is 0-indexed in pypdf, add 1 for human page representation
                            page_raw = doc.metadata.get("page", 0)
                            page = page_raw + 1 if isinstance(page_raw, int) else page_raw
                            snippet = doc.page_content[:350].replace('\n', ' ').strip() + "..."
                            
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-header">
                                    <span>📄 {filename} (Page {page})</span>
                                </div>
                                <div class="source-content">
                                    "{snippet}"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred while answering your question: {e}")

# Footer
st.markdown("<br><br><hr><center style='color:#64748b; font-size:0.85rem;'>StudySage — Local & Private RAG Assistant • Built with Streamlit, LangChain and Google Gemini</center>", unsafe_allow_html=True)
