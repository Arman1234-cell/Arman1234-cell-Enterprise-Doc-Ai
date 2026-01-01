import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pymupdf4llm
from google import genai
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# 1. INITIALIZATION & CONFIG
load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ChromaDB Setup
# 'all-MiniLM-L6-v2' is a reliable, lightweight embedding model
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
db_client = PersistentClient(path="chroma_db")
collection = db_client.get_or_create_collection(name="pdf_knowledge", embedding_function=emb_fn)

# Tesseract OCR Setup (Ensure this path is correct for your PC)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# --- ROUTES ---

@app.route("/")
def index():
    """Renders the Landing Page"""
    return render_template("index.html")

@app.route("/chat", methods=["GET"])
def chat_page():  # <--- This name MUST match url_for('chat_page')
    return render_template("chat.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Handles PDF upload, OCR, and Topic Tagging"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Step A: Extract text using OCR
    md_content = pymupdf4llm.to_markdown(filepath, force_ocr=True)
    
    # Step B: Generate dynamic Topic Tags using Gemini
    tag_prompt = f"List 5 main topics from this text as a comma-separated list:\n\n{md_content[:4000]}"
    tag_resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=tag_prompt)
    tags = [t.strip() for t in tag_resp.text.split(",")]

    # Step C: Chunk and store in Vector DB
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(md_content)
    collection.add(
        documents=chunks,
        ids=[f"{filename}-{uuid.uuid4()}" for _ in chunks],
        metadatas=[{"source": filename} for _ in chunks]
    )

    return jsonify({"message": "Ready!", "tags": tags})

@app.route("/chat", methods=["POST"])
def chat_logic():
    """Handles the RAG search and Gemini response generation"""
    user_query = request.json.get("message")
    
    # 1. Retrieve relevant context chunks from ChromaDB
    results = collection.query(query_texts=[user_query], n_results=5)
    context_text = "\n\n".join(results['documents'][0])
    
    # 2. Generate a grounded answer using Gemini
    prompt = f"""
    Answer the question based ONLY on the following context. 
    If you don't know the answer from the context, say so.
    
    CONTEXT:
    {context_text}
    
    QUESTION:
    {user_query}
    """
    
    response = client.models.generate_content(
        model="gemini-flash-lite-latest", 
        contents=prompt
    )
    
    return jsonify({"response": response.text})

if __name__ == "__main__":
    app.run(debug=True)