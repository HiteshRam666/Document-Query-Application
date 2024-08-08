# Document Query Application 📰

Welcome to the **Document Query Application**! This Streamlit app allows users to upload documents, perform text queries, and securely store and download query history.

👉 [Click link for Demo](https://document-query-application-ul92uaezfnmdmdktdhqh8y.streamlit.app/)

## Table of Contents

1. [System Architecture and Design](#system-architecture-and-design)
2. [Database Schema](#database-schema)
3. [Instructions for Adding New Documents](#instructions-for-adding-new-documents)
4. [Security Measures Implemented](#security-measures-implemented)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Dependencies](#dependencies)
8. [Conclusion](#conclusion)

## System Architecture and Design

The Document Query Application is designed with a modular architecture to ensure maintainability and scalability. The system is divided into the following components:

1. **Frontend (Streamlit App)**:
    - Provides an interactive web interface for users to upload documents, enter queries, and view results.
    - Handles user authentication, document uploads, and query processing.

2. **Backend (Data Processing and Storage)**:
    - **Document Processing**: Extracts text from uploaded PDF, DOCX, and TXT files.
    - **Text Preprocessing**: Tokenizes and removes stop words from the text.
    - **Query Processing**: Uses BERT embeddings to encode and compare queries with document content.
    - **Database Management**: Stores user queries and responses securely.

3. **Security**:
    - **Encryption**: Uses Fernet encryption to securely store user queries and responses.
    - **Data Integrity**: Handles errors related to data decryption and encryption.

4. **Dependencies**:
    - Utilizes libraries like PyPDF2, python-docx, NLTK, and transformers for text processing and analysis.

## Database Schema

The application uses an SQLite database to store user history. The schema is defined as follows:

### `user_history` Table

| Column    | Type    | Description                           |
|-----------|---------|---------------------------------------|
| `user_id` | TEXT    | Unique identifier for the user.        |
| `query`   | TEXT    | Encrypted query text submitted by the user. |
| `response`| TEXT    | Encrypted response text from the documents. |
| `timestamp` | DATETIME | Timestamp of when the query was made.   |

**SQL Statement to Create Table:**

```sql
CREATE TABLE IF NOT EXISTS user_history (
    user_id TEXT,
    query TEXT,
    response TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Instructions for Adding New Documents

1. **File Upload**:
    - Navigate to the sidebar on the Streamlit app and click on "Upload Documents".
    - Select and upload PDF, DOCX, or TXT files. Multiple files can be uploaded at once.

2. **Processing**:
    - Uploaded files are processed to extract text.
    - The extracted text is then preprocessed (tokenized, stop words removed) for better search results.

3. **Querying**:
    - After uploading documents, enter a query in the provided input field and click "Search".
    - The app will process the query and compare it against the uploaded documents using BERT embeddings.

## Security Measures Implemented

1. **Data Encryption**:
    - User queries and responses are encrypted using the Fernet symmetric encryption algorithm.
    - Encryption key is stored securely and must be consistent across sessions.

2. **Error Handling**:
    - The application includes error handling for encryption and decryption failures.
    - Users are notified if data decryption fails due to key mismatch or data corruption.

3. **Data Integrity**:
    - The application ensures that encrypted data is correctly handled and stored.
    - Attempts to decrypt data are managed to prevent unauthorized access.

4. **File Handling**:
    - Uploaded files are processed and stored temporarily to avoid data leakage.
    - Secure practices are followed in handling and processing user files.

## Installation

To set up the application, follow these steps:

1. Clone the repository and navigate to the project directory:

    ```bash
    git clone https://github.com/your-username/document-query-app.git
    cd document-query-app
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the App**:

    ```bash
    streamlit run app.py
    ```

2. **Upload Documents**:
    - Use the sidebar to upload PDF, DOCX, or TXT files.

3. **Search Documents**:
    - Enter your query in the text input field and click "Search" to find the most relevant document content.

4. **View and Download History**:
    - Click "Download Chat History" to download your past queries and responses.

## Dependencies

- **Streamlit**: For creating the web interface.
- **SQLite3**: For database operations.
- **Cryptography**: For data encryption and decryption.
- **PyPDF2**: For extracting text from PDF files.
- **python-docx**: For extracting text from DOCX files.
- **nltk**: For text preprocessing.
- **transformers**: For BERT model and tokenizer.
- **torch**: For tensor operations and similarity computations.

Install dependencies using:

```bash
pip install streamlit sqlite3 cryptography PyPDF2 python-docx nltk transformers torch
```

## Conclusion

The Document Query Application provides a robust solution for managing and querying document content. Its design incorporates modern NLP techniques and secure data handling to ensure an efficient and safe user experience. Whether you're working with a large number of documents or need a secure method to track queries and responses, this application offers the tools and features to meet your needs.

Thank you for exploring the Document Query Application. We hope it helps you efficiently manage and retrieve document information!
