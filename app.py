import streamlit as st
import sqlite3
from cryptography.fernet import Fernet, InvalidToken, InvalidSignature
from PyPDF2 import PdfReader
import docx
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, BertTokenizer
from torch.nn.functional import cosine_similarity
import torch

# Downloading NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Encryption key
key = b'rzmLBk2cIJMcIGlKbEcRzGik7u-XQMp5g4k8pnIID7E='  # Ensure this key is consistent
cipher = Fernet(key)

# Connect to the database
conn = sqlite3.connect('user_data.db')
c = conn.cursor()

# Create table for user history if not exists
c.execute('''
CREATE TABLE IF NOT EXISTS user_history (
          user_id TEXT,
          query TEXT,
          response TEXT, 
          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
          )
''')
conn.commit()

# Streamlit App
st.title("Document Query Application 📰")

# User authentication
user_id = st.text_input("Enter your user ID:", value="guest")

# Sidebar for Uploading documents
st.sidebar.header("Upload Documents") 
uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True, type=["pdf", "docx", "txt"])

# Sidebar for download history option
st.sidebar.header("Options")
download_history = st.sidebar.button("Download Chat History")

st.write(f"Welcome, {user_id}")

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to decrypt data with error handling
def decrypt_data(encrypted_data):
    try:
        decrypted_data = cipher.decrypt(encrypted_data.encode()).decode()
        return decrypted_data
    except (InvalidToken, InvalidSignature):
        st.error("Failed to decrypt data. Possible causes: key mismatch, corrupted data, or encoding issues.")
        return None

# Function to preprocess text
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    return ' '.join(tokens)

# Function to query PDF
def query_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return preprocess_text(text)

# Function to query DOCX
def query_docx(file):
    doc = docx.document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return preprocess_text(text)

# Function to query TXT
def query_txt(file):
    text = file.read().decode('utf-8')
    return preprocess_text(text)

# Function to encode text using BERT
def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim = 1)

# Function to search documents using BERT and cosine similarity
def search_documents_bert(documents, query):
    query_embedding = encode_text(query)
    doc_embeddings = [encode_text(doc) for doc in documents]
    similarities = [cosine_similarity(query_embedding, doc_emb).item() for doc_emb in doc_embeddings]
    most_relevant_idx = torch.argmax(torch.tensor(similarities)).item()
    return documents[most_relevant_idx], similarities[most_relevant_idx]

# Handling file uploads and querying
if uploaded_files:
    st.write("Files uploaded successfully!")
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            documents.append(query_pdf(uploaded_file))
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            documents.append(query_docx(uploaded_file))
        elif uploaded_file.type == "text/plain":
            documents.append(query_txt(uploaded_file))

    query = st.text_input("Enter your query:")

    if st.button("Search"):
        processed_query = preprocess_text(query)
        response, score = search_documents_bert(documents, processed_query)
        st.write(f"Most relevant document content: {response}")
        st.write(f"Relevance score: {score:.2f}")
        
        # Save the query and response to the user history
        encrypted_query = cipher.encrypt(query.encode()).decode()
        encrypted_response = cipher.encrypt(response.encode()).decode()
        conn.execute('INSERT INTO user_history (user_id, query, response) VALUES (?, ?, ?)', 
                     (user_id, encrypted_query, encrypted_response))
        conn.commit()

# Download history 
import io

# if download_history:
#     history = conn.execute('SELECT * FROM user_history WHERE user_id = ?', (user_id,)).fetchall()
#     if history:
#         with open(f"{user_id}_chat_history.txt", "w", encoding="utf-8") as f:
#             for record in history:
#                 decrypted_query = decrypt_data(record[1])
#                 decrypted_response = decrypt_data(record[2])
                
#                 if decrypted_query and decrypted_response:
#                     f.write(f"Query: {decrypted_query}\nResponse: {decrypted_response}\n\n")
#         st.write(f"Chat history downloaded as {user_id}_chat_history.txt!")
#     else:
#         st.write("No history available to download.")

import io

if download_history:
    history = conn.execute('SELECT * FROM user_history WHERE user_id = ?', (user_id,)).fetchall()
    if history:
        # Use an in-memory buffer to avoid duplicate writes
        buffer = io.StringIO()
        for record in history:
            decrypted_query = decrypt_data(record[1])
            decrypted_response = decrypt_data(record[2])
            
            if decrypted_query and decrypted_response:
                buffer.write(f"Query: {decrypted_query}\nResponse: {decrypted_response}\n\n")
        
        # Get the content of the buffer
        file_content = buffer.getvalue()
        
        # Create a download button
        st.download_button(
            label="Download Chat History",
            data=file_content,
            file_name=f"{user_id}_chat_history.txt",
            mime="text/plain"
        )
    else:
        st.write("No history available to download.")

# Close the database connection
conn.close()