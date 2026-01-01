import os
import uuid
import sys
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

import pymupdf4llm
from google import genai

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# --------------------------------------------------
# 1. APP INITIALIZATION
# --------------------------------------------------
load_dotenv()
app = Flask(__name__)

# 🔒 Upload size limit (Railway safe)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# --------------------------------------------------
# 2. DIRECTORY SETUP (Railway compatible)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# --------------------------------------------------
# 3. GEMINI CLIENT
# --------------------------------------------------
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ GEMINI_API_KEY missing!", file=sys.stderr)

client = genai.Client(api_key=API_KEY)

# --------------------------------------------------
# 4. LOCAL EMBEDDINGS (NO QUOTA, CPU ONLY)
# --------------------------------------------------
try:
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
except Exception as e:
    print(f"❌ Embedding load error: {e}", file=sys.stderr)
    raise

# --------------------------------------------------
# 5. VECTOR DATABASE
# --------------------------------------------------
db_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = db_client.get_or_create_collection(
    name="pdf_knowledge",
    embedding_function=emb_fn
)

# --------------------------------------------------
# 6. ROUTES
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("chat.html")

# --------------------------------------------------
# 7. PDF UPLOAD & INDEX
# --------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # 🔹 Extract text
        md_content = pymupdf4llm.to_markdown(filepath, force_ocr=False)

        if not md_content or not md_content.strip():
            return jsonify({"error": "PDF contains no readable text"}), 400

        # 🔹 Topic tagging (Gemini)
        try:
            tag_prompt = (
                "List 5 short main topics from this document "
                "as a comma-separated list:\n\n"
                f"{md_content[:3000]}"
            )
            tag_resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=tag_prompt
            )
            tags = [t.strip() for t in tag_resp.text.split(",")]
        except Exception as e:
            print(f"⚠️ Tagging failed: {e}", file=sys.stderr)
            tags = ["Document Indexed"]

        # 🔹 Chunking & vector storage
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(md_content)

        for chunk in chunks:
            collection.add(
                documents=[chunk],
                ids=[f"{filename}-{uuid.uuid4()}"],
                metadatas=[{"source": filename}]
            )

        print(f"✅ Indexed {len(chunks)} chunks", file=sys.stderr)

        return jsonify({
            "message": "Document indexed successfully",
            "tags": tags
        })

    except Exception as e:
        print(f"❌ UPLOAD ERROR: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# 8. CHAT ENDPOINT
# --------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat_logic():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 415

        user_query = data.get("message", "").strip()
        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        # 🔹 Vector search
        results = collection.query(
            query_texts=[user_query],
            n_results=5
        )
        context = "\n\n".join(results["documents"][0])

        # 🔹 Gemini answer
        prompt = (
            "Answer the question using ONLY the context below.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{user_query}"
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return jsonify({"response": response.text})

    except Exception as e:
        print(f"❌ CHAT ERROR: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# 9. FILE SIZE ERROR HANDLER
# --------------------------------------------------
@app.errorhandler(RequestEntityTooLarge)
def file_too_large(e):
    return jsonify({
        "error": "File too large. Maximum allowed size is 16MB."
    }), 413

# --------------------------------------------------
# 10. RAILWAY ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
