import os
import uuid
import time
import sys
import pymupdf
import pymupdf4llm
from google import genai
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. SETUP & SECURITY
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DIR = "chroma_db"  # Matches app.py directory name
COLLECTION_NAME = "pdf_knowledge"

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# --- THE FIX: LOCAL EMBEDDINGS (Matches app.py) ---
# Using the CPU-based model to avoid Google Quota limits
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def ingest_and_summarize(pdf_path):
    print(f"\n🚀 Ingesting: {pdf_path}")
    
    # Step 1: Extraction (OCR False for better speed and stability)
    start_time = time.time()
    try:
        md_content = pymupdf4llm.to_markdown(pdf_path, force_ocr=False)
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None
    
    if not md_content or not md_content.strip():
        print("❌ No text found in PDF.")
        return None

    # Step 2: Topic Discovery (Updated to Gemini 2.0 Flash)
    print("📋 Identifying main topics...")
    try:
        summary_resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Extract a list of the 5 main topics from this text:\n\n{md_content[:8000]}"
        )
        print("\n--- DOCUMENT OVERVIEW ---")
        print(summary_resp.text)
        print("-------------------------\n")
    except Exception as e:
        print(f"⚠️ Summary failed: {e}")

    # Step 3: Chunking & Storage
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(md_content)
    
    # Persistent Client
    db_client = PersistentClient(path=CHROMA_DIR)
    collection = db_client.get_or_create_collection(
        name=COLLECTION_NAME, 
        embedding_function=emb_fn
    )
    
    collection.add(
        documents=chunks,
        ids=[f"id-{uuid.uuid4()}" for _ in chunks],
        metadatas=[{"source": pdf_path} for _ in chunks]
    )
    
    print(f"✅ Ingestion Complete. {len(chunks)} chunks stored in {time.time() - start_time:.2f}s.")
    return collection

def chat_interface(collection):
    print("\n🤖 Assistant Ready. (Type 'exit' to quit)")
    while True:
        query = input("\n👤 You: ").strip()
        if query.lower() in ['exit', 'quit']: break
        if not query: continue

        # Local Vector Search
        results = collection.query(query_texts=[query], n_results=5)
        context = "\n\n".join(results['documents'][0])
        
        # Cloud AI Generation (Updated to Gemini 2.0 Flash)
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"Answer the question using ONLY the context provided.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}"
            )
            print(f"\n🤖 Gemini: {response.text}")
        except Exception as e:
            print(f"❌ Chat Error: {e}")

if __name__ == "__main__":
    path = input("📁 Enter PDF path: ").strip().replace('"', '')
    if os.path.exists(path):
        knowledge_base = ingest_and_summarize(path)
        if knowledge_base:
            chat_interface(knowledge_base)
    else:
        print("❌ File not found!")
