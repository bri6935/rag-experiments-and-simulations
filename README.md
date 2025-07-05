# Generalized RAG: A Portable RAG Solution for Local Document Analysis

This project provides a powerful and portable Retrieval-Augmented Generation (RAG) script, `generalized_rag.py`, designed to work with your local files and a local Large Language Model (LLM). You can simply place this script in any project directory, and it will automatically build a searchable knowledge base from all `.txt` and `.pdf` files within that folder and its subdirectories.

The script is built with three core principles in mind: efficiency, contextual awareness, and ease of use. It intelligently processes and caches your documents, retrieves not just the direct answer but also the surrounding context, and communicates directly with your local LLM server, offering a completely private and self-contained RAG solution.

## Key Features

*   **Recursive Document Processing:** Automatically scans the current directory and all subdirectories for `.txt` and `.pdf` files to build its knowledge base.
*   **Efficient & Incremental Updates:** Utilizes MD5 hashing of document chunks to prevent re-processing of unchanged files. Only new or modified content is added to the vector store, making subsequent runs fast and efficient.
*   **Context-Aware Retrieval:** Implements a specialized "left/right" retriever. Instead of just fetching the single most relevant text chunk, it also retrieves neighboring chunks, providing the LLM with a richer, more complete context to formulate its answers.
*   **Direct LLM Communication:** Interacts directly with your local LLM server (e.g., Ollama) using the `requests` library for both embedding generation and question-answering. This avoids complex dependencies and gives you full control over your models.
*   **Dynamic & Safe Naming:** Automatically creates a unique and valid ChromaDB collection name based on the name of the root folder, preventing conflicts and making it easy to manage multiple projects.
*   **Flexible ChromaDB Connection:** Attempts to connect to a running ChromaDB server but gracefully falls back to a local, file-based persistent database if a server is not available.

## How It Works

The `generalized_rag.py` script follows a three-step process:

1.  **Indexing:**

    *   It scans the target directory for supported documents (`.txt`, `.pdf`).
    *   Each document is parsed and split into smaller, overlapping text chunks.
    *   A unique ID is generated for each chunk by hashing its content.
    *   If a chunk's ID is not already in the vector database, the script generates an embedding for it using the specified embedding model and stores the chunk, its embedding, and metadata (source file, chunk index) in a local ChromaDB collection.

2.  **Retrieval:**

    *   When you ask a question, the script generates an embedding for your query.
    *   It queries the ChromaDB collection to find the most relevant document chunks (based on semantic similarity).
    *   The "Enriched Retriever" then identifies the source documents of these initial hits and fetches the chunks immediately preceding and succeeding them (the "left" and "right" neighbors). This provides crucial context that might be missed with a standard retrieval approach.

3.  **Generation:**

    *   The retrieved context chunks are combined and formatted into a comprehensive prompt, along with your original question.
    *   This final prompt is sent to the specified local LLM.
    *   The LLM generates an answer based on the provided context, which is then streamed back to you in real-time.

## Prerequisites

Before you begin, ensure you have the following installed and configured:

1.  **Python 3.x**
2.  **Required Python Libraries:**
    ```bash
    pip install requests chromadb langchain pypdf
    ```
3.  **A Local LLM Server:** This script is configured to work with an [Ollama](https://ollama.ai/) server. Make sure Ollama is running and you have downloaded the necessary models.
    *   An embedding model (e.g., `nomic-embed-text`)
    *   A generative language model (e.g., `gemma3:12b-it-qat`)

## Getting Started

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Place the Script:** Ensure the `generalized_rag.py` script is in the root directory of the project or folder you wish to analyze.

3.  **Configure the Script:** Open `generalized_rag.py` in a text editor and update the `CONFIGURATION` section to match your setup:

    ```python
    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    # --- Server and Model Configuration ---
    BASE_URL = "http://localhost:11434"  # IMPORTANT: Update with your Ollama server URL
    LLM_MODEL_NAME = "gemma3:12b-it-qat"  # Model for generating answers
    EMBEDDING_MODEL_NAME = "nomic-embed-text:latest" # Model for creating embeddings
    CONTEXT_WINDOW_SIZE = 128000      # Set the context window for the LLM

    # --- Retriever Configuration ---
    K_SEARCH_RESULTS = 4        # Number of initial search results to fetch
    NEIGHBORS_TO_FETCH = 1      # Number of text chunks to fetch to the left and right

    # --- Text Splitting Configuration ---
    CHUNK_SIZE = 3000
    CHUNK_OVERLAP = 100

    # --- Chroma Server Connection Details ---
    CHROMA_SERVER_HOST = "localhost"
    CHROMA_SERVER_PORT = 8000
    ```

4.  **Run the Script:** Execute the script from your terminal:

    ```bash
    python generalized_rag.py
    ```

    The script will first scan your documents and build the database. This may take some time on the first run. Subsequent runs will be much faster.

5.  **Ask Questions:** Once the indexing is complete, you can start asking questions about your documents directly in the terminal.

    ```
    Please enter your question: [Your question here]
    ```

    To exit the application, type `exit` or `quit`, or press `Ctrl+C`.

## Database Storage

The script will create a subdirectory named `rag_chromadb` in the location where you run it. This folder contains the vector database. You can safely delete this folder to rebuild the database from scratch. It is recommended to add this directory to your `.gitignore` file to avoid committing the database to your repository.
