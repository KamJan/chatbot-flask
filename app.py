import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from openai import OpenAI
from tqdm import tqdm
from numpy.linalg import norm

# Use environment variable for API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")
client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDINGS_FILE = "embeddings.pkl"
CONTENT_FILE = r"content.txt"

def load_content():
    with open(CONTENT_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def build_embeddings():
    chunks = load_content()
    embeddings = []
    for chunk in tqdm(chunks, desc="Embedding content"):
        emb = get_embedding(chunk)
        embeddings.append(emb)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (norm(a) * norm(b))

def find_relevant_chunks(question, top_k=3):
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    embeddings = data["embeddings"]
    q_emb = get_embedding(question)
    sims = [cosine_similarity(q_emb, emb) for emb in embeddings]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    relevant_chunks = find_relevant_chunks(user_message)
    context = "\n\n".join(relevant_chunks)
    prompt = f"Use the following course materials to answer the question.\n\nMaterials:\n{context}\n\nQuestion: {user_message}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()
    return jsonify({'response': answer})

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "embed":
        print("Generating embeddings...")
        build_embeddings()
        print("Embeddings saved.")
    else:
        app.run(port=5000)
