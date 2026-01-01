import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pymupdf4llm
from google import genai
import chromadb
from chromadb.utils import embedding_functions # Import the embedding utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pytesseract

# 1. INITIALIZATION & CONFIG
load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Gemini API Key
API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini Client
client = genai.Client(api_key=API_KEY)

# --- MEMORY FIX: GOOGLE EMBEDDINGS ---
# This uses an API call instead of loading a model locally to save RAM
emb_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=API_KEY
)

# ChromaDB Setup
db_client = chromadb.PersistentClient(path="chroma_db")
collection = db_client.get_or_create_collection(
    name="pdf_knowledge", 
    embedding_function=emb_fn
)

# --- OCR SETUP ---
if os.name == 'nt': 
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("chat.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Convert PDF to Markdown using OCR
    md_content = pymupdf4llm.to_markdown(filepath, force_ocr=False)    
    # Generate Topic Tags
    tag_prompt = f"List 5 main topics from this text as a comma-separated list:\n\n{md_content[:4000]}"
    tag_resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=tag_prompt)
    tags = [t.strip() for t in tag_resp.text.split(",")]

    # Split and Store
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(md_content)
    
    # Chroma handles embedding via 'emb_fn' automatically here
    collection.add(
        documents=chunks,
        ids=[f"{filename}-{uuid.uuid4()}" for _ in chunks],
        metadatas=[{"source": filename} for _ in chunks]
    )

    return jsonify({"message": "Ready!", "tags": tags})

@app.route("/chat", methods=["POST"])
def chat_logic():
    user_query = request.json.get("message")
    
    # Retrieve relevant chunks
    results = collection.query(query_texts=[user_query], n_results=5)
    context_text = "\n\n".join(results['documents'][0])
    
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

# --- DYNAMIC PORT FOR CLOUD ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

