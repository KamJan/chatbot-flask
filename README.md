# Chatbot-flask

A conversational AI chatbot for physiotherapy use, built with Flask and OpenAI.

## Purpose

This project provides a simple, deployable chatbot that leverages OpenAI's language and embedding models to answer physiotherapy-related questions. It is designed to assist users—such as patients or clinicians—by providing relevant, context-aware responses using course or clinical materials supplied by the user.

## Technology Stack

- **Python 3**: Core programming language
- **Flask**: Lightweight backend web framework for serving the chatbot UI and API endpoints
- **OpenAI API**: Used for both text embeddings (semantic search) and chat completions
- **NumPy**: Efficient vector math for similarity calculations
- **tqdm**: Progress bar for generating embeddings
- **Pickle**: Storage of precomputed embeddings and content chunks

## How It Works

1. **Content Preparation**: 
   - The `content.txt` file contains the course or clinical material, divided into paragraphs or logical chunks.
   - The script generates vector embeddings for each chunk using OpenAI's embedding model and stores them in `embeddings.pkl`.

2. **Question Answering**:
   - When a user asks a question, the system embeds the question and compares it to all content chunks using cosine similarity.
   - The most relevant chunks are selected as context.
   - These chunks, together with the user’s question, are sent to OpenAI’s chat model to generate a tailored answer.

3. **Frontend**:
   - A simple web UI (served from Flask) allows users to interact with the chatbot in real time.

## Use Cases

- Provide physiotherapy patients with information and guidance based on curated materials.
- Help clinicians quickly reference relevant clinical content.
- Educational tool for physiotherapy students.

## Getting Started

1. **Clone the repository**
2. **Install requirements**:  
   `pip install -r requirements.txt`
3. **Set your OpenAI API key**:  
   Export `OPENAI_API_KEY` as an environment variable.
4. **Prepare your `content.txt`**:  
   Place your physiotherapy material or course content in the file.
5. **Generate embeddings**:  
   `python app.py embed`
6. **Run the Flask app**:  
   `python app.py`

## Security & Privacy

- All content and embeddings are processed and stored locally.
- OpenAI API is used for embeddings and chat completions; ensure no sensitive patient data is included unless permitted by your data policy.

## Extending & Customizing

- Replace `content.txt` with your own materials to tailor the chatbot for different clinical or educational contexts.
- Adjust the number of top relevant chunks (`top_k`) for broader or narrower context selection.
- Integrate with authentication or other frameworks for production use.

## License


---

**Contact:**  
For questions or suggestions, please open an issue or reach out via GitHub.
