import os
import uuid
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pymupdf4llm
from google import genai
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# 1. INITIALIZATION & CONFIG
load_dotenv()
app = Flask(__name__)

# Ensure the upload path is absolute for server reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Gemini API Key
API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini Client
client = genai.Client(api_key=API_KEY)

# --- MEMORY FIX: GOOGLE EMBEDDINGS ---
emb_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=API_KEY
)

# ChromaDB Setup
db_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
collection = db_client.get_or_create_collection(
    name="pdf_knowledge", 
    embedding_function=emb_fn
)

# --- ROUTES ---

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
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Convert PDF to Markdown (OCR disabled for Render compatibility)
        md_content = pymupdf4llm.to_markdown(filepath, force_ocr=False) 
        
        if not md_content.strip():
            return jsonify({"error": "Could not extract text from this PDF."}), 400

        # Generate Topic Tags
        tag_prompt = f"List 5 main topics from this text as a comma-separated list:\n\n{md_content[:4000]}"
        tag_resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=tag_prompt)
        tags = [t.strip() for t in tag_resp.text.split(",")]

        # Split and Store
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(md_content)
        
        collection.add(
            documents=chunks,
            ids=[f"{filename}-{uuid.uuid4()}" for _ in chunks],
            metadatas=[{"source": filename} for _ in chunks]
        )

        return jsonify({"message": "Ready!", "tags": tags})
    
    except Exception as e:
        print(f"DEPLOYMENT_ERROR_UPLOAD: {str(e)}") # This prints to Render Logs
        return jsonify({"error": "Internal server error during upload"}), 500

@app.route("/chat", methods=["POST"])
def chat_logic():
    try:
        data = request.json
        user_query = data.get("message")
        
        if not user_query:
            return jsonify({"error": "No message provided"}), 400

        # Retrieve relevant chunks
        results = collection.query(query_texts=[user_query], n_results=5)
        
        if not results['documents'] or not results['documents'][0]:
            return jsonify({"response": "I couldn't find any information about that in the uploaded documents."})

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
    
    except Exception as e:
        print(f"DEPLOYMENT_ERROR_CHAT: {str(e)}")
        return jsonify({"error": "Internal server error during chat"}), 500

# --- DYNAMIC PORT FOR CLOUD ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
