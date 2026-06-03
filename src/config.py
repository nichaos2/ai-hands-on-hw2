import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

# rag
API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_VERSION = "models/gemini-embedding-001"
LOAD_DIR = os.path.join(BASE_DIR, "data", "documents")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data", "vector_store")

# tools
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "target_encoder.pkl")

# dataset
DATASET_PATH = os.path.join(BASE_DIR, "data", "datasets", "lichess_dataset.csv")

if __name__ == "__main__":
    print(API_KEY)
    print(EMBEDDING_VERSION)
    print(LOAD_DIR)
    print(PERSIST_DIRECTORY)
    print(MODEL_PATH)
    print(SCALER_PATH)
    print(ENCODER_PATH)
    print(DATASET_PATH)
