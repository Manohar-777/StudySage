# 📚 StudySage — AI Study Assistant (Multilingual RAG) 
vist here #https://studysage-by-manohar.streamlit.app/

**One-line pitch:** An AI study assistant where you upload your textbooks/lecture notes (PDFs) and ask questions in **Telugu (తెలుగు) or English**, and it answers *from your documents* with the exact source page cited.

This project demonstrates a production-grade **Retrieval-Augmented Generation (RAG)** pipeline designed to solve LLM hallucination problems for custom user data. It implements semantic text chunking, local vector storage, context-based prompting, and bilingual query processing.

---

## 🎯 JD Alignment Checklist

- **RAG Architecture**: Built with LangChain and FAISS vector databases.
- **Vector Search**: Local vector embeddings computed and searched using cosine/L2 distance.
- **Google Gemini Integration**: Utilizing `gemini-1.5-flash` and Google AI Embeddings.
- **FastPrototyping**: Responsive and high-fidelity Streamlit UI.
- **Multilingual (English & Telugu)**: Dynamic Telugu text querying and translation-free native matching.
- **OCR Support**: Optional Tesseract OCR fallback for scanned PDF documents.
- **Containerization**: Includes a pre-configured Docker build.
- **DPDP 2023 Compliant**: Explicit data privacy section detailing safe storage rules.

---

## 🏗️ System Architecture

```
                 ┌─────────────── INGESTION (done once per document) ──────────────┐
   PDF/notes  →  Load text  →  Split into chunks  →  Embed each chunk  →  Store in FAISS
                 (PyPDFLoader)  (TextSplitter)       (Gemini embeddings)   (vector DB)

                 ┌─────────────────────── QUERY (every question) ──────────────────┐
   User Q     →  Embed the question  →  Find top-k similar chunks  →  Stuff chunks + Q
                 (same embedder)         (FAISS similarity search)     into a prompt
                                                                            │
                                                            Gemini LLM generates answer
                                                                            │
                                                          Show answer + source pages
```

- **Why RAG?**: LLMs only know their public training data. RAG retrieves relevant paragraphs from uploaded textbooks and forces the LLM to write answers solely from that local context, eliminating hallucinations.
- **Why Chunking?**: Textbooks can have hundreds of pages (over token limits). We partition documents into overlapping `1000-character` chunks so that we only extract the most relevant snippets.

---

## 🛠️ Technology Stack

- **Frontend/UI**: Streamlit (with Custom Glassmorphic Dark UI styles)
- **Framework**: LangChain (`langchain`, `langchain-community`, `langchain-google-genai`)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: `models/embedding-001` (Google AI Embeddings)
- **LLM Engine**: `gemini-1.5-flash` (free tier)
- **OCR System (Optional)**: Tesseract OCR + `pdf2image` + Poppler
- **Containerization**: Docker

---

## 🚀 Running Locally

### 1. Clone & Set Up Directory
Navigate to the project folder:
```powershell
cd studysage
```

### 2. Create Virtual Environment & Install Dependencies
```powershell
# Create environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Add API Keys
Create a file named `.env` in the root of the `studysage/` directory and paste your API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(Alternatively, you can paste the API key directly into the StudySage sidebar in the web app.)*

### 4. Run Verification Suite (Backend Test)
Run the script to verify vector embeddings, FAISS indexing, and Gemini API calls programmatically:
```powershell
python verify_rag.py
```

### 5. Launch the Web Application
```powershell
streamlit run app.py
```
This opens a new tab in your default browser (usually at `http://localhost:8501`).

---

## 🔍 OCR Configuration (Optional fallback for scanned image PDFs)
To read scanned/image PDFs, Tesseract and Poppler are required:
1. Install Tesseract OCR: [UB-Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install Poppler: [Oschwartz Poppler Releases](https://github.com/oschwartz10612/poppler-windows/releases).
3. Add both installation directories to your system's environment `PATH` variable.

---

## 🐳 Docker Deployment

1. **Build the Docker Image**:
   ```powershell
   docker build -t studysage .
   ```

2. **Run the Container**:
   ```powershell
   docker run -p 8501:8501 --env GOOGLE_API_KEY=your_gemini_api_key studysage
   ```
   Open `http://localhost:8501` to use the containerized app.

---

## 🔐 Data Privacy & Compliance (DPDP Act 2023)

StudySage is designed with strict data isolation matching modern compliance requirements:
- **No Persistence of PII**: Documents uploaded are split and embedded in-memory or saved in transient local directories (`data/` cache). No documents are uploaded to long-term databases or external servers other than transient API requests for embeddings.
- **Key Security**: API keys are isolated via local `.env` files or session-only Streamlit state.
- **Zero-Data Training**: Google Gemini APIs run in inference mode; documents sent as prompt context are not utilized to retrain or fine-tune public models.

---

## 🎓 Key Interview Discussion Points

- **Hyperparameters (k=4)**: We fetch the top `4` matches. Retrieving too few chunks (e.g. 1-2) misses cross-page context; retrieving too many (e.g. 10+) causes prompt clutter and increases latency.
- **Temperature (0.2)**: Set to `0.2` (low temperature) to keep responses factual, grounded, and aligned with source texts, avoiding creative hallucinations.
- **Bilingual RAG Matching**: Embedding models project words into a language-agnostic space. This means English queries can successfully retrieve Telugu content (and vice-versa), while the custom prompt template commands the LLM to format the response back to the question's native language.
