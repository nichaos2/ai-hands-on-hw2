import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import API_KEY, EMBEDDING_VERSION, LOAD_DIR, PERSIST_DIRECTORY


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


def retrieve_chess_context(query: str) -> str:
    """
    Searches the Chroma database for the 3 most relevant chunks based on
    the user's query and returns them as a single formatted string.
    """
    # Use similarity_search to find the top 3 chunks (k=3)
    # This automatically converts the text query to a vector and calculates the distance

    # TODO DRY in this part
    # same code to create gemini embeddings and vector_store instance
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_VERSION,
        google_api_key=API_KEY,
    )

    vector_store = Chroma(
        # documents=chunks,
        # embedding=gemini_embeddings,
        embedding_function=gemini_embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    retrieved_docs = vector_store.similarity_search(query, k=3)

    # Check if anything was returned
    if not retrieved_docs:
        return "No relevant information found in the database."

    # Format and concatenate the chunks into a single string
    context_pieces = []
    for i, doc in enumerate(retrieved_docs):
        # Adding a visual separator helps the LLM distinguish between different chunks
        chunk_text = f"--- Document Chunk {i + 1} ---\n{doc.page_content}\n"
        context_pieces.append(chunk_text)

    final_context_string = "\n".join(context_pieces)

    return final_context_string


if __name__ == "__main__":
    create_vector_store()
