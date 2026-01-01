import os
import uuid
import time
import pymupdf
import pymupdf.layout  # Enable OCR/Layout detection
import pymupdf4llm
from google import genai
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils import embedding_functions # Import embedding utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pytesseract

# 1. SETUP & SECURITY
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DIR = "master_knowledge_db"
COLLECTION_NAME = "pdf_rag_index"

# --- OCR CONFIG (FIXED FOR CLOUD/WINDOWS) ---
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

# --- MEMORY FIX: GOOGLE EMBEDDINGS ---
# We remove 'all-MiniLM-L6-v2' to save ~400MB of RAM
emb_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GEMINI_API_KEY
)

def ingest_and_summarize(pdf_path):
    print(f"\n🚀 Ingesting: {pdf_path}")
    
    # Step 1: Extraction with Forced OCR
    start_time = time.time()
    md_content = pymupdf4llm.to_markdown(pdf_path, force_ocr=False)
    
    if not md_content or not md_content.strip():
        print("❌ OCR failed to find text. Check your Tesseract installation.")
        return None

    # Step 2: Immediate Topic Discovery
    print("📋 Identifying main topics in the document...")
    summary_resp = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=f"Extract a list of the 5 main topics from this document text:\n\n{md_content[:8000]}"
    )
    print("\n--- DOCUMENT OVERVIEW ---")
    print(summary_resp.text)
    print("-------------------------\n")

    # Step 3: Chunking & Storage
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
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
        
        # Cloud AI Generation
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=f"Use ONLY the following context to answer.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}"
        )
        print(f"\n🤖 Gemini: {response.text}")

if __name__ == "__main__":
    path = input("📁 Enter PDF path: ").strip().replace('"', '')
    if os.path.exists(path):
        knowledge_base = ingest_and_summarize(path)
        if knowledge_base:
            chat_interface(knowledge_base)
    else:
        print("❌ File not found!")

