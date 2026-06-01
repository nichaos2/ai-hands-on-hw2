"""
Code to 'test' the retrieval function
"""

from src import rag

if __name__ == "__main__":
    query = "What is a chess opening?"
    answer = rag.retrieve_chess_context(query)
    print(f"Query:\n{query}")
    print(f"Answer:\n{answer}")
