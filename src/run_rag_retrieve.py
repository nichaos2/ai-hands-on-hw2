"""
Code to 'test' the retrieval function
"""

from rag import retrieve_chess_context

if __name__ == "__main__":
    query = "What is a chess opening?"
    answer = retrieve_chess_context(query)
    print(f"Query:\n{query}")
    print(f"Answer:\n{answer}")
