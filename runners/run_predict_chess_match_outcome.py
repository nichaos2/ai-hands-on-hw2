"""
Code to 'test' the retrieval function
"""

from src import tools

if __name__ == "__main__":
    print("Testing the LangChain Prediction Tool...\n")

    # 1. Create a dummy payload exactly like the agent will generate
    # (Using the values from your Swagger UI example)
    agent_simulated_input = {
        "rated": True,
        "created_at": 1504210000000.0,
        "last_move_at": 1504210000000.0,
        "turns": 13,
        "white_rating": 1500,
        "black_rating": 1191,
        "opening_ply": 5,
        "increment_code": "15+2",
        "opening_eco": "D10",
        "opening_name": "Slav Defense",
    }

    print("--- Tool Input (Simulated from Agent) ---")
    print(agent_simulated_input)
    print("\n--- Executing Tool... ---")

    # 2. Call the tool using .invoke()
    try:
        tool_output = tools.predict_chess_match_outcome.invoke(agent_simulated_input)

        print("\n--- Tool Output (To Agent) ---")
        print(tool_output)

    except Exception as e:
        print(f"\nTest Failed: {e}")
