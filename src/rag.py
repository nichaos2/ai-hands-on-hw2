import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent

env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_VERSION = "models/gemini-embedding-001"
LOAD_DIR = os.path.join(BASE_DIR, "data", "documents")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data", "vector_store")


def create_vector_store():
    print("Loading documents from the 'data' folder...")
    loader = DirectoryLoader(LOAD_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    print("Chunking documents...")
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} individual chunks.")

    print("Initializing Gemini Embeddings...")
    # embedding-001 is Google's optimized model for vector representations
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_VERSION,
        google_api_key=API_KEY,
    )

    print("Creating empty Chroma Vector Store...")
    vector_store = Chroma(
        # documents=chunks,
        # embedding=gemini_embeddings,
        embedding_function=gemini_embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    # We will send 50 chunks at a time to stay safely under the 100 RPM limit
    BATCH_SIZE = 50

    print(f"Adding {len(chunks)} chunks to the database in batches of {BATCH_SIZE}...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(f" -> Embedding chunks {i + 1} to {min(i + BATCH_SIZE, len(chunks))}...")

        # Add this specific batch to the database
        vector_store.add_documents(batch)

        # If there are still more chunks left to process, we must pause for the API quota to reset
        if i + BATCH_SIZE < len(chunks):
            print(" -> Sleeping for 60 seconds to respect Google's free tier limits...")
            time.sleep(60)

    print(f"\nThe vector database has been saved to the '{PERSIST_DIRECTORY}' folder.")
    print("You do not need to run this script again unless you add new text documents.")


if __name__ == "__main__":
    create_vector_store()
