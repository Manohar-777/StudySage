import os
import sys

# Reconfigure output encoding to handle unicode Telugu text and emojis safely on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def run_test():
    print("--- StudySage Backend Verification ---")
    load_dotenv()
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set in .env")
        print("Please add GOOGLE_API_KEY=your_key to studysage/.env and retry.")
        sys.exit(1)
        
    print("✓ Google API Key detected.")
    
    # 1. Prepare Mock Data (Bilingual: Telugu + English)
    mock_data = [
        Document(
            page_content=(
                "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy "
                "that, through cellular respiration, can later be released to fuel the organism's activities. "
                "This chemical energy is stored in carbohydrate molecules, such as sugars and starches."
            ),
            metadata={"source": "biology_notes.pdf", "page": 1}
        ),
        Document(
            page_content=(
                "కిరణజన్య సంయోగ క్రియ (Photosynthesis) అనేది మొక్కలు మరియు కొన్ని రకాల బ్యాక్టీరియాలు కాంతి శక్తిని "
                "రసాయన శక్తిగా మార్చే ఒక జీవక్రియ. ఈ రసాయన శక్తి పిండిపదార్థాలుగా (కార్బోహైడ్రేట్లు) నిల్వ చేయబడుతుంది. "
                "మొక్కలు పత్రహరితం (Chlorophyll), నీరు, మరియు కార్బన్ డై ఆక్సైడ్ సహాయంతో సూర్యకాంతి సమక్షంలో ఆహారాన్ని తయారుచేస్తాయి."
            ),
            metadata={"source": "telugu_science.pdf", "page": 12}
        )
    ]
    print(f"✓ Mock documents loaded: {len(mock_data)} pages.")
    
    # 2. Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(mock_data)
    print(f"✓ Documents split into {len(chunks)} chunks.")
    
    # 3. Embeddings & FAISS Vectorstore
    print("Embedding chunks and building FAISS vector database (Gemini Embedding API)...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("✓ Vectorstore built successfully.")
    except Exception as e:
        print(f"❌ Embeddings/Vectorstore failed: {e}")
        sys.exit(1)
        
    # 4. LLM & Retrieval Setup
    print("Setting up LLM and RetrievalQA (Gemini-2.5-Flash)...")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
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
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        print("✓ Retrieval pipeline initialized.")
    except Exception as e:
        print(f"❌ LLM initialization failed: {e}")
        sys.exit(1)
        
    # 5. Queries
    test_queries = [
        "What is photosynthesis and where is the chemical energy stored?",
        "కిరణజన్య సంయోగ క్రియ అనగా ఏమిటి? దానికి ఏమి కావాలి?"
    ]
    
    for q in test_queries:
        print(f"\n👉 Asking: '{q}'")
        try:
            res = qa.invoke({"query": q})
            print("\n🤖 StudySage Response:")
            print(res["result"])
            print("\n📄 Sources:")
            for doc in res["source_documents"]:
                print(f" - {doc.metadata['source']} (Page {doc.metadata['page']})")
        except Exception as e:
            print(f"❌ Query failed: {e}")
            
    print("\n--- Verification Complete! ---")

if __name__ == "__main__":
    run_test()
