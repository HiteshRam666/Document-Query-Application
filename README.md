# Document Query Application 📰

Welcome to the **Document Query Application**! This Streamlit app allows you to upload various document types, perform text queries, and securely store and download your query history.

## Features

- **Upload Documents**: Supports PDF, DOCX, and TXT file formats.
- **Text Querying**: Utilize BERT embeddings to find the most relevant content in your documents.
- **Secure Storage**: Encrypted storage of user queries and responses.
- **Download History**: Download your chat history securely.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [How It Works](#how-it-works)
4. [Dependencies](#dependencies)
5. [Conclusion](#conclusion)

## Installation

To get started, clone this repository and install the required dependencies.

```bash
git clone https://github.com/your-username/document-query-app.git
cd document-query-app
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage

1. **Run the App**: Start the Streamlit application with the following command:

    ```bash
    streamlit run app.py
    ```

2. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files.

3. **Search Documents**: Enter your query in the provided text input field and click "Search" to find the most relevant content.

4. **View and Download History**: Click "Download Chat History" to download your past queries and responses.

## How It Works

1. **Document Upload**: Users can upload PDF, DOCX, or TXT files through the sidebar.
2. **Text Extraction**: The application extracts text from the uploaded documents and preprocesses it.
3. **Query Processing**: User queries are processed and compared with document content using BERT embeddings.
4. **Encrypted Storage**: Queries and responses are encrypted and stored in an SQLite database.
5. **History Download**: Users can download their query history as a text file.

## Dependencies

- **Streamlit**: For creating the web interface.
- **SQLite3**: For database operations.
- **Cryptography**: For data encryption and decryption.
- **PyPDF2**: For extracting text from PDF files.
- **python-docx**: For extracting text from DOCX files.
- **nltk**: For text preprocessing.
- **transformers**: For BERT model and tokenizer.
- **torch**: For handling tensor operations and similarity computations.

Install dependencies using:

```bash
pip install streamlit sqlite3 cryptography PyPDF2 python-docx nltk transformers torch
```

## Conclusion

The Document Query Application is designed to simplify the process of extracting and querying information from various document formats. By leveraging advanced NLP techniques and secure data storage, it provides a powerful tool for managing and retrieving document content efficiently. Whether you're handling a large collection of documents or just need a secure way to track your queries and responses, this application offers a robust and user-friendly solution.

Thank you for checking out the Document Query Application. We hope it meets your needs and encourages you to explore and contribute to its development!
