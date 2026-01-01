import os
import uuid
import sys
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pymupdf4llm
from google import genai
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# 1. INITIALIZATION
load_dotenv()
app = Flask(__name__)

# Directory setup for cloud hosting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Gemini Client for Chat/Tags
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# --- THE QUOTA FIX: LOCAL EMBEDDINGS ---
# This runs on Railway's CPU. It does NOT count against your Google 429 limit.
# Note: On the first run, Railway will take a minute to download this model (~80MB).
try:
    # Using a faster, smaller model to stay under Railway's 1GB RAM limit
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
except Exception as e:
    print(f"Error loading local embedding model: {e}", file=sys.stderr)

# Database setup
db_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
collection = db_client.get_or_create_collection(name="pdf_knowledge", embedding_function=emb_fn)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("chat.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Step 1: Extract Text (OCR False for Railway compatibility)
        md_content = pymupdf4llm.to_markdown(filepath, force_ocr=False)

        if not md_content.strip():
            return jsonify({"error": "PDF contains no readable text."}), 400

        # Step 2: Gemini Topic Tagging (Uses only 1 API call per PDF)
        # We use Gemini 1.5 Flash for better free-tier stability
        print("DEBUG: Requesting tags from Gemini...", file=sys.stderr)
        tag_prompt = f"List 5 short main topics from this text as a comma-separated list:\n\n{md_content[:3000]}"
        tag_resp = client.models.generate_content(model="gemini-2.0-flash", contents=tag_prompt)
        tags = [t.strip() for t in tag_resp.text.split(",")]

        # Step 3: Local Vector Storage (Zero Quota usage)
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_text(md_content)
        
        collection.add(
            documents=chunks,
            ids=[f"{filename}-{uuid.uuid4()}" for _ in chunks],
            metadatas=[{"source": filename} for _ in chunks]
        )
        
        print(f"DEBUG: Successfully indexed {len(chunks)} chunks locally.", file=sys.stderr)
        return jsonify({"message": "Document ready!", "tags": tags})

    except Exception as e:
        print(f"CRITICAL UPLOAD ERROR: {str(e)}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat_logic():
    try:
        user_query = request.json.get("message")
        if not user_query:
            return jsonify({"error": "No message provided"}), 400

        # 1. Search Local Database
        results = collection.query(query_texts=[user_query], n_results=5)
        context_text = "\n\n".join(results['documents'][0])
        
        # 2. Ask Gemini 1.5 Flash
        prompt = f"Answer the question using only the context below:\n\nCONTEXT:\n{context_text}\n\nQUESTION: {user_query}"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        
        return jsonify({"response": response.text})

    except Exception as e:
        print(f"DEBUG CHAT ERROR: {str(e)}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Railway dynamic port binding
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


