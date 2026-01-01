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



