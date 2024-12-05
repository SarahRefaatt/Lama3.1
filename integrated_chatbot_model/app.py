import os
import fitz  # PyMuPDF
import logging
import re
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import faiss
from langchain_ollama import OllamaLLM
nltk.data.path.append(r'C:\Users\sarah.hanafy\AppData\Roaming\nltk_data\tokenize')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torchvision

# Disable the beta transforms warning
torchvision.disable_beta_transforms_warning()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize LLM and embeddings model
llm = OllamaLLM(base_url="http://localhost:11434", model="llama3.1")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
index = None  # FAISS index will be initialized dynamically
chunk_texts = []  # Store chunks' text
chunk_metadata = []  # Store metadata (file names and offsets) for each chunk


def get_pdf_files(directory):
    return [os.path.join(root, file) 
            for root, _, files in os.walk(directory) 
            for file in files if file.endswith('.pdf')]


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


# def preprocess_text(text):
#     """Basic text preprocessing: lowercase."""
#     return text.lower()

def preprocess_text(text):
    text = text.lower()#for not separating btw Apple apple
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))# remove  is ,the ,are ....
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer() # remove ing 
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def create_chunks(text, chunk_size=512):
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def embed_chunks(chunks):
    """Generate embeddings for text chunks using SentenceTransformers."""
    return embedding_model.encode(chunks, convert_to_tensor=False)


@app.route('/upload', methods=['POST'])
def process_pdfs():
    logging.info("Received a request to process PDFs.")

    # # Validate input
    # if not request.json or 'directory' not in request.json:
    #     return jsonify({'error': 'Invalid request format.'}), 400

    pdf_directory = "uploads"
    logging.info(f"Processing directory: {pdf_directory}")

    if not os.path.exists(pdf_directory):
        return jsonify({'error': 'Directory does not exist.'}), 404

    pdf_files = get_pdf_files(pdf_directory)
    if not pdf_files:
        return jsonify({'error': 'No PDF files found.'}), 404

    global index, chunk_texts, chunk_metadata
    chunk_texts = []
    chunk_metadata = []
    index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())

    for pdf in pdf_files:
        try:
            text = extract_text_from_pdf(pdf)
            preprocessed_text = preprocess_text(text)
            chunks = create_chunks(preprocessed_text)
            embeddings = embed_chunks(chunks)

            # Add embeddings to the index
            index.add(embeddings)
            chunk_texts.extend(chunks)
            chunk_metadata.extend([(os.path.basename(pdf), i) for i in range(len(chunks))])

            logging.info(f"Processed file: {pdf}")
        except Exception as e:
            logging.error(f"Error processing file {pdf}: {e}")

    # return jsonify({
    #     'message': 'PDFs processed successfully',
    #     'num_files': len(pdf_files),
    #     'num_chunks': len(chunk_texts),
    # })
@app.route('/ask', methods=['POST'])
def ask_question():
    process_pdfs()
    logging.info("Received a request to ask a question.")

    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided.'}), 400

    if not index or not chunk_metadata or not chunk_texts:
        return jsonify({'error': 'No preprocessed texts found. Please process PDFs first.'}), 400

    # Query the FAISS index
    try:
        question_embedding = embedding_model.encode([question], convert_to_tensor=False)
        top_k = 3  # Number of top relevant chunks to retrieve
        D, I = index.search(question_embedding, top_k)
        retrieved_chunks = [(chunk_metadata[idx], chunk_texts[idx]) for idx in I[0]]

        # Combine metadata and chunk text for the context
        formatted_context = "\n".join([
            f"File: {meta[0]}, Chunk Index: {meta[1]}\nText: {text}"
            # f"File: {meta[0]}\nText: {text}"
            for meta, text in retrieved_chunks
        ])



        # Combine into the prompt for the LLM
        full_prompt = f"Context:\n{formatted_context}\n\nQuestion: {question}"
        response = llm.invoke(full_prompt)

        # Return the LLM's response and retrieved context
        return jsonify({
            'answer': response
            # 'retrieved_context': formatted_context
        })

    except Exception as e:
        logging.error(f"Error during question answering: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/test', methods=['GET'])
def test_route():
    return jsonify({"message": "Server is running!"})


if __name__ == '__main__':
    app.run(debug=True, port=8080)

#only one api :topic,