import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from openai import OpenAI, OpenAIError
from tqdm import tqdm
from numpy.linalg import norm

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set")
    raise ValueError("OPENAI_API_KEY not set")
else:
    print("OPENAI_API_KEY found and loaded")
client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDINGS_FILE = "embeddings.pkl"
CONTENT_FILE = "content.txt"  # Use relative path for deployment

# --- Utility Functions ---

def load_content():
    """Load course content and split into chunks."""
    print(f"Looking for content file at: {CONTENT_FILE}")
    if not os.path.exists(CONTENT_FILE):
        print(f"ERROR: {CONTENT_FILE} not found in {os.getcwd()}")
        raise FileNotFoundError(f"{CONTENT_FILE} not found.")
    with open(CONTENT_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    print(f"Loaded {len(chunks)} content chunks")
    return chunks

def get_embedding(text):
    """Get embedding for text using OpenAI."""
    print(f"Getting embedding for text (first 30 chars): {text[:30]}...")
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise

def build_embeddings():
    """Build and save embeddings for all content chunks."""
    print("Starting embedding building process...")
    chunks = load_content()
    embeddings = []
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding content")):
        print(f"Embedding chunk {i+1}/{len(chunks)}")
        emb = get_embedding(chunk)
        embeddings.append(emb)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (norm(a) * norm(b))

def find_relevant_chunks(question, top_k=3):
    """Find the most relevant content chunks for a question."""
    print(f"Finding relevant chunks for question: {question}")
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"ERROR: {EMBEDDINGS_FILE} not found in {os.getcwd()}")
        raise FileNotFoundError(f"{EMBEDDINGS_FILE} not found. Run 'python app.py embed' to generate it.")
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    embeddings = data["embeddings"]
    q_emb = get_embedding(question)
    sims = [cosine_similarity(q_emb, emb) for emb in embeddings]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    print(f"Top relevant chunk indices: {top_indices}")
    return [chunks[i] for i in top_indices]

# --- Flask App ---

app = Flask(__name__)

@app.route('/')
def home():
    """Serve the chatbot UI."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests from the frontend."""
    try:
        user_message = request.json.get('message', '').strip()
        print(f"Received chat message: {user_message}")
        if not user_message:
            return jsonify({'response': "Please enter a question."}), 400

        relevant_chunks = find_relevant_chunks(user_message)
        context = "\n\n".join(relevant_chunks)
        prompt = (
            "Use the following course materials to answer the question.\n\n"
            f"Materials:\n{context}\n\n"
            f"Question: {user_message}\nAnswer:"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        print(f"Chatbot answer: {answer[:60]}...")
        return jsonify({'response': answer})

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return jsonify({'response': str(e)}), 500
    except OpenAIError as e:
        print(f"OpenAIError: {e}")
        return jsonify({'response': f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        import traceback
        print("Exception:", e)
        traceback.print_exc()
        return jsonify({'response': f"An error occurred: {str(e)}"}), 500

@app.route('/health')
def health():
    """Health check endpoint for Render and monitoring."""
    return "ok", 200

if __name__ == "__main__":
    import sys
    print(f"Running app.py with arguments: {sys.argv}")
    if len(sys.argv) > 1 and sys.argv[1] == "embed":
        print("Generating embeddings...")
        build_embeddings()
        print("Embeddings saved.")
    else:
        port = int(os.environ.get("PORT", 10000))
        print(f"Starting Flask app on port {port}")
        app.run(host="0.0.0.0", port=port)
