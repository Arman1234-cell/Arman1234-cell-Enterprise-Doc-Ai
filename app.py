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

load_dotenv()
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# --- THE FIX: LOCAL EMBEDDINGS ---
# This runs on Railway's CPU. It does NOT use your Google API Quota.
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

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
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 1. Text Extraction
        md_content = pymupdf4llm.to_markdown(filepath, force_ocr=False)

        # 2. Topic Tagging (Uses 1 request to Gemini)
        tag_prompt = f"List 3-5 main topics from this text as a comma-separated list:\n\n{md_content[:2000]}"
        tag_resp = client.models.generate_content(model="gemini-1.5-flash", contents=tag_prompt)
        tags = [t.strip() for t in tag_resp.text.split(",")]

        # 3. Vector Storage (Local - No API Quota used here!)
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = splitter.split_text(md_content)
        
        collection.add(
            documents=chunks,
            ids=[f"{filename}-{uuid.uuid4()}" for _ in chunks],
            metadatas=[{"source": filename} for _ in chunks]
        )
        
        return jsonify({"message": "Success!", "tags": tags})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat_logic():
    try:
        user_query = request.json.get("message")
        results = collection.query(query_texts=[user_query], n_results=3)
        context_text = "\n\n".join(results['documents'][0])
        
        prompt = f"Answer based on context:\n{context_text}\n\nQuestion: {user_query}"
        response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
