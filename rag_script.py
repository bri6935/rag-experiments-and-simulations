# =============================================================================
# generalized_rag.py
#
# Description:
# This script performs Retrieval Augmented Generation (RAG) on a directory of
# documents. It is designed to be portable and reusable. Simply place it
# in any folder, and it will recursively find all .txt and .pdf files in that
# folder and its subdirectories to build a searchable knowledge base.
#
# It combines three key pieces of logic:
# 1.  Efficient document embedding with hashing to prevent re-processing of
#     unchanged files.
# 2.  A specialized "left/right" retriever that fetches neighboring chunks
#     of text around a search hit to provide richer context.
# 3.  Direct API calls to an LLM server using the `requests` library for
#     both embeddings and generation.
#
# How to run:
# 1. Ensure you have the required libraries:
#    pip install requests chromadb langchain pypdf
# 2. Make sure your local LLM server (e.g., Ollama) is running.
# 3. Update the CONFIGURATION section below with your server URL and model names.
# 4. Place this script in the root directory of the project you want to query.
# 5. Run the script from your terminal: python generalized_rag.py
#
# =============================================================================

import os
import re
import requests
import json
import hashlib
import chromadb
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================
# --- Server and Model Configuration ---
BASE_URL = "http://localhost:11434"  # IMPORTANT: Update with your server URL
LLM_MODEL_NAME = "gemma3:12b-it-qat"       # Model for generating answers
EMBEDDING_MODEL_NAME = "nomic-embed-text:latest" # Model for creating embeddings
CONTEXT_WINDOW_SIZE = 128000      # Set the context window for the LLM

# --- Directory and Database Configuration ---
# The script will now use the current working directory as the target.
# The database will be stored in a subfolder named 'rag_chromadb' within this directory.
CHROMA_DB_SUBDIR = "rag_chromadb"

# --- Retriever Configuration ---
K_SEARCH_RESULTS = 4        # Number of initial search results to fetch
NEIGHBORS_TO_FETCH = 1      # Number of text chunks to fetch to the left and right

# --- Text Splitting Configuration ---
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 100

# --- Chroma Server Connection Details ---
CHROMA_SERVER_HOST = "localhost"
CHROMA_SERVER_PORT = 8000


# =============================================================================
# CHROMA DATABASE MANAGER
# =============================================================================

class ChromaManager:
    """
    Handles the creation, management, and updating of the ChromaDB vector store.
    It processes text and PDF files, splits them into chunks, generates embeddings
    for new or modified chunks, and stores them in the database.
    """
    def __init__(self, db_path: str, collection_name: str, embedding_model: str, base_url: str,
                 chroma_server_host: str = "localhost", chroma_server_port: int = 8000):
        """
        Initializes the ChromaDB client and the text splitter.

        Args:
            db_path (str): Path to the ChromaDB persistent storage.
            collection_name (str): Name of the collection to use.
            embedding_model (str): The name of the model to use for embeddings.
            base_url (str): The base URL of the LLM server.
            chroma_server_host (str): The host of the ChromaDB server (for HTTP client).
            chroma_server_port (int): The port of the ChromaDB server (for HTTP client).
        """
        print("Initializing ChromaDB Manager...")
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.chroma_server_host = chroma_server_host
        self.chroma_server_port = chroma_server_port

        self.client = self._get_chroma_client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        print(f"ChromaDB collection '{collection_name}' loaded/created successfully.")

    def _get_chroma_client(self):
        """
        Attempts to connect to a ChromaDB server, falling back to a local
        persistent client if the server is unreachable.
        """
        print(f"Attempting to connect to ChromaDB server at http://{self.chroma_server_host}:{self.chroma_server_port}...")
        try:
            http_client = chromadb.HttpClient(host=self.chroma_server_host, port=self.chroma_server_port)
            http_client.list_collections()
            print("Successfully connected to ChromaDB server.")
            return http_client
        except requests.exceptions.ConnectionError:
            print("ChromaDB server not found. Falling back to local PersistentClient.")
            return chromadb.PersistentClient(path=self.db_path)
        except Exception as e:
            print(f"An error occurred connecting to ChromaDB server: {e}. Falling back to PersistentClient.")
            return chromadb.PersistentClient(path=self.db_path)

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates an embedding for a given text using a direct API call.

        Args:
            text (str): The text to embed.

        Returns:
            Optional[List[float]]: The embedding vector, or None if an error occurs.
        """
        try:
            url = f"{self.base_url}/api/embeddings"
            payload = {"model": self.embedding_model, "prompt": text}
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("embedding")
        except requests.RequestException as e:
            print(f"Error generating embedding: {e}")
            return None

    def process_directory(self, target_dir: str):
        """
        Scans the target directory recursively, processes .txt and .pdf files, and updates the
        vector store with any new or modified content.

        Args:
            target_dir (str): The root directory to scan for documents.
        """
        print(f"\nScanning '{os.path.abspath(target_dir)}' recursively for .txt and .pdf files...")
        if not os.path.isdir(target_dir):
            print(f"Error: Directory '{target_dir}' not found.")
            return

        total_files_found = 0
        added_chunks = 0

        for root, _, files in os.walk(target_dir):
            # Skip the ChromaDB directory itself to avoid processing its internal files
            if os.path.abspath(root).startswith(os.path.abspath(self.db_path)):
                continue

            for filename in files:
                file_path = os.path.join(root, filename)
                content = ""

                if filename.lower().endswith('.txt'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Could not read text file {filename}: {e}")
                        continue
                elif filename.lower().endswith('.pdf'):
                    try:
                        reader = pypdf.PdfReader(file_path)
                        content = "".join(page.extract_text() for page in reader.pages if page.extract_text())
                    except Exception as e:
                        print(f"Could not process PDF file {filename}: {e}")
                        continue
                else:
                    continue # Skip unsupported file types

                if not content.strip():
                    continue

                total_files_found += 1
                
                # Get the folder path relative to the main target directory
                folder_source = os.path.relpath(root, target_dir)
                if folder_source == ".":
                    folder_source = "" # Use empty string for root, cleaner metadata

                chunks = self.text_splitter.split_text(content)

                for i, chunk_text in enumerate(chunks):
                    # Use a hash of the content as a unique ID to prevent re-processing
                    chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()

                    # Check if the chunk already exists in the database
                    if self.collection.get(ids=[chunk_id])['ids']:
                        continue  # Skip if chunk is already present

                    print(f"  -> Found new chunk in '{file_path}'. Generating embedding...")
                    embedding = self._generate_embedding(chunk_text)
                    if embedding:
                        self.collection.add(
                            ids=[chunk_id],
                            embeddings=[embedding],
                            documents=[chunk_text],
                            metadatas=[{
                                "source": filename,
                                "chunk_index": i,
                                "folder_source": folder_source
                            }]
                        )
                        added_chunks += 1

        print(f"\nScan complete. Found {total_files_found} supported files (.txt, .pdf).")
        print(f"Processing complete. Added {added_chunks} new chunks to the vector store.")
        print(f"Total documents in collection '{self.collection_name}': {self.collection.count()}")


# =============================================================================
# ENRICHED RAG RETRIEVER
# =============================================================================

class EnrichedRetriever:
    """
    Performs RAG by first finding relevant chunks and then "enriching" the
    context by fetching neighboring chunks (left and right).
    """
    def __init__(self, chroma_manager: ChromaManager, num_neighbors: int):
        """
        Args:
            chroma_manager (ChromaManager): An instance of the ChromaManager.
            num_neighbors (int): The number of chunks to retrieve on each side.
        """
        print("\nInitializing Enriched Retriever...")
        self.chroma_manager = chroma_manager
        self.num_neighbors = num_neighbors

    def get_relevant_documents(self, query: str, k: int) -> Dict[str, Any]:
        """
        Retrieves relevant documents and their neighbors for a given query.

        Args:
            query (str): The user's question.
            k (int): The number of initial relevant documents to find.

        Returns:
            Dict[str, Any]: A dictionary containing the combined context string
                            and a set of unique source file paths.
        """
        print(f"Generating embedding for query: '{query}'")
        query_embedding = self.chroma_manager._generate_embedding(query)
        if not query_embedding:
            return {"context": "Error: Could not generate query embedding.", "sources": set()}

        # 1. Get initial search results
        results = self.chroma_manager.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        initial_docs_metadata = results.get('metadatas', [[]])[0]
        if not initial_docs_metadata:
            return {"context": "No relevant documents found.", "sources": set()}
        
        print(f"Found {len(initial_docs_metadata)} initial relevant document(s).")
        
        # 2. Identify unique source files (folder + filename) to fetch all their chunks
        unique_source_identifiers = []
        seen_sources = set()
        for meta in initial_docs_metadata:
            source_id = (meta.get('folder_source', ""), meta['source'])
            if source_id not in seen_sources:
                seen_sources.add(source_id)
                unique_source_identifiers.append({
                    "$and": [
                        {"folder_source": {"$eq": source_id[0]}},
                        {"source": {"$eq": source_id[1]}}
                    ]
                })
        
        where_filter = {"$or": unique_source_identifiers} if len(unique_source_identifiers) > 1 else unique_source_identifiers[0]
        all_relevant_source_chunks = self.chroma_manager.collection.get(where=where_filter)
        
        # 3. Organize chunks by source and sort by index for efficient neighbor lookup
        sorted_chunks_by_source = {}
        for i, meta in enumerate(all_relevant_source_chunks['metadatas']):
            source_id = (meta.get('folder_source', ""), meta['source'])
            if source_id not in sorted_chunks_by_source:
                sorted_chunks_by_source[source_id] = []
            
            sorted_chunks_by_source[source_id].append({
                'content': all_relevant_source_chunks['documents'][i],
                'metadata': meta
            })
        
        for source in sorted_chunks_by_source:
            sorted_chunks_by_source[source].sort(key=lambda x: x['metadata']['chunk_index'])

        # 4. Get neighboring chunks for each initial result, avoiding duplicates
        final_context_chunks = []
        processed_chunk_keys = set() # Use (folder, file, index) to track added chunks

        print("Fetching neighboring chunks for context...")
        for doc_meta in initial_docs_metadata:
            source_id_tuple = (doc_meta.get('folder_source', ""), doc_meta['source'])
            chunk_index = doc_meta['chunk_index']
            
            source_chunks = sorted_chunks_by_source.get(source_id_tuple)
            if not source_chunks: continue

            current_position = next((i for i, chunk in enumerate(source_chunks) if chunk['metadata']['chunk_index'] == chunk_index), -1)
            if current_position == -1: continue

            start_index = max(0, current_position - self.num_neighbors)
            end_index = min(len(source_chunks), current_position + self.num_neighbors + 1)

            for i in range(start_index, end_index):
                neighbor_chunk = source_chunks[i]
                neighbor_meta = neighbor_chunk['metadata']
                neighbor_key = (neighbor_meta.get('folder_source', ""), neighbor_meta['source'], neighbor_meta['chunk_index'])

                if neighbor_key not in processed_chunk_keys:
                    final_context_chunks.append(neighbor_chunk)
                    processed_chunk_keys.add(neighbor_key)
        
        # 5. Sort final chunks for logical flow and build final context
        final_context_chunks.sort(key=lambda x: (x['metadata'].get('folder_source', ""), x['metadata']['source'], x['metadata']['chunk_index']))
        
        combined_content = "\n---\n".join([chunk['content'] for chunk in final_context_chunks])
        unique_sources = {os.path.join(c['metadata'].get('folder_source', ""), c['metadata']['source']).replace('\\', '/') for c in final_context_chunks}

        print(f"Retrieved {len(final_context_chunks)} total chunks for context from sources: {unique_sources}")
        
        return {"context": combined_content, "sources": unique_sources}


# =============================================================================
# LLM INTERACTION
# =============================================================================

def get_llm_response(prompt: str) -> str:
    """
    Sends a prompt to the LLM and streams the response.
    """
    print("\nSending prompt to LLM...")
    url = f"{BASE_URL}/api/chat"
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "num_ctx": CONTEXT_WINDOW_SIZE
        }
    }
    full_response = ""
    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            print("Streaming response from LLM:")
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = json.loads(line.decode('utf-8'))
                        content = json_line.get("message", {}).get("content", "")
                        print(content, end="", flush=True)
                        full_response += content
                    except json.JSONDecodeError:
                        continue
            print("\n--- End of Stream ---")
    except requests.RequestException as e:
        error_message = f"Error calling LLM API: {e}"
        print(error_message)
        return error_message
    return full_response

def format_prompt(query: str, context: str) -> str:
    """
    Creates the final prompt to be sent to the language model.
    """
    return f"""
You are an AI assistant that was build specifically for question and answer rag. When you recieve the user's input, it follows the rag format of a question, and then context that will pulled from a vectorestore after embeddings.
Given this, the process that is happening is you are atempting to see if the recieved "chunks" from the vector store can asnwer the user's questions. 

If the chunk contains the answer to the user question, answer it and highlight the reference/where it came from.
If it looks like the chunk is not related to the question of the user (signifying that the vector retrieval from RAG wasnt very good), let the user know that you cannot find any answer from the source documents, or the retrieved chunks seems unrelated to the question and more documents may be needed for reference.

Here is the user question for the RAG process: {query}

--- and here are the "chunks" retrieved from the vectorstore. ---
{context}
--- END OF CONTEXT ---

"""

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def create_safe_collection_name(name: str) -> str:
    """
    Creates a valid ChromaDB collection name from a directory name.
    Follows ChromaDB naming conventions.
    """
    name = name.lower()
    # Replace invalid characters with underscores
    name = re.sub(r'[^a-z0-9_-]', '_', name)
    # Ensure it doesn't start or end with invalid characters
    name = re.sub(r'^[^a-z0-9]+', '', name)
    name = re.sub(r'[^a-z0-9]+$', '', name)
    # Enforce length constraints
    if len(name) < 3:
        name = f"{name}pad"
    if len(name) > 63:
        name = name[:63]
    # Handle edge case of empty or invalid name
    if not name or not name[0].isalnum():
        name = f"collection_{name}"
    return name

def main():
    """
    Main function to run the RAG application.
    """
    # Use the current working directory as the target for RAG
    target_directory = "."
    current_dir_name = os.path.basename(os.getcwd())
    
    # Dynamically create a collection name based on the folder name
    collection_name = create_safe_collection_name(f"{current_dir_name}_rag")
    db_path = os.path.join(target_directory, CHROMA_DB_SUBDIR)

    print("==========================================================")
    print("===        General Purpose RAG Command-Line Tool       ===")
    print("==========================================================")
    print(f"Target Directory: {os.path.abspath(target_directory)}")
    print(f"Database Path:    {os.path.abspath(db_path)}")
    print(f"Collection Name:  {collection_name}")
    print(f"LLM Model:        {LLM_MODEL_NAME}")
    print(f"Context Window:   {CONTEXT_WINDOW_SIZE}")
    print("----------------------------------------------------------")

    # 1. Initialize the DB manager and process the files
    chroma_manager = ChromaManager(
        db_path=db_path,
        collection_name=collection_name,
        embedding_model=EMBEDDING_MODEL_NAME,
        base_url=BASE_URL,
        chroma_server_host=CHROMA_SERVER_HOST,
        chroma_server_port=CHROMA_SERVER_PORT
    )
    chroma_manager.process_directory(target_directory)

    # 2. Initialize the retriever
    retriever = EnrichedRetriever(
        chroma_manager=chroma_manager,
        num_neighbors=NEIGHBORS_TO_FETCH
    )
    
    # 3. Start interactive loop
    print("\n==========================================================")
    print("Ready to answer questions. Type 'exit' or 'quit' to end.")
    print("==========================================================")
    while True:
        try:
            query = input("\nPlease enter your question: ")
            if query.lower() in ['exit', 'quit']:
                print("Exiting application. Goodbye!")
                break

            retrieval_result = retriever.get_relevant_documents(query, k=K_SEARCH_RESULTS)
            context = retrieval_result.get("context", "")
            sources = retrieval_result.get("sources", set())

            if not context or "No relevant documents found" in context:
                print("\nI could not find any relevant information in the documents to answer your question.")
                continue

            final_prompt = format_prompt(query, context)
            
            get_llm_response(final_prompt)

            if sources:
                print(f"\n\n---\nSources consulted: {', '.join(sorted(list(sources)))}\n---")

        except KeyboardInterrupt:
            print("\nExiting application. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            continue

if __name__ == "__main__":
    main()