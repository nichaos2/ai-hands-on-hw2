# Chess AI Assistant: Prediction & Domain Knowledge Agent

## 1. System Overview

This project implements a conversational AI agent designed specifically for the chess domain. The assistant provides a seamless chat interface where users can ask general questions about chess concepts or request predictions for specific match setups. By combining a Retrieval-Augmented Generation (RAG) pipeline for domain knowledge with an optimized XGBoost machine learning model for match forecasting, the agent bridges the gap between historical data analysis and natural language education. 

## 2. Architecture

The system is built using the **LangChain** framework (specifically `langchain_classic` agents) and is powered by Google's `gemini-3.1-flash-lite` LLM. 

The agent has access to exactly two tools:
- `retrieve_chess_context`: A RAG-based tool that queries a vector database for domain knowledge.
- `predict_chess_match_outcome`: A predictive tool that processes match parameters through a trained machine learning pipeline.

### Decision Routing & State Management:

The agent uses a ReAct (Reasoning and Acting) prompt structure. It autonomously reads the user's natural language input, evaluates the descriptions of its available tools, and decides whether to route the query to the RAG tool, the ML tool, or simply answer conversationally. 

Conversation memory is maintained across turns using `RunnableWithMessageHistory` and an `InMemoryChatMessageHistory` store. To handle Gemini 3.1's specific output formatting (which returns list blocks instead of flat strings), the architecture includes a custom `RunnableLambda` function that intercepts, flattens, and cleans the agent's internal outputs before they hit the final output parser, preventing infinite loop crashes during conversation.

_Note_: While originally scoped for LangGraph, the architecture was adapted to use LangChain's classic `AgentExecutor` pipeline to ensure compatibility with Gemini's tool-calling structures and output coercions.

## 3. Knowledge Base

The RAG system is backed by a vector store containing detailed documents covering core chess terminology, mathematics, and systems. 

### Collected Documents Include

- Explanations of the **Elo Rating System**, including its history, mathematical probability expectations, and point-exchange mechanics.
- The **Encyclopaedia of Chess Openings (ECO)**, detailing how openings are categorized (A through E) and standard opening names.
- General tournament rules and time control formats (e.g., increments).

**Why they were chosen:**
These documents were explicitly selected to explain the specific features the machine learning model uses to make predictions. Users often do not know what "ECO" or "opening_ply" means; the RAG system allows the agent to answer questions like *"What is an ECO code?"* or *"How do ratings affect the outcome?"* contextually.

## 4. HW1 Model Integration

The predictive engine carries forward the best-performing model from HW1: an __XGBoost Classifier__ saved as `best_model.pkl`. 

### The Prediction Tool Pipeline

When the agent invokes the `predict_chess_match_outcome` tool, it passes a JSON payload of the match parameters. The tool then mirrors the exact preprocessing pipeline from HW1:

1. Feature Engineering: Calculates domain-specific features like `rating_advantage` and `game_duration_mins`.
2. Encoding: Uses the saved `target_encoder.pkl` to convert high-cardinality categorical strings (`opening_eco`, `increment_code`, `opening_name`) into historical probability averages.
3. Scaling: Applies the saved `scaler.pkl` to normalize numerical fields (turns, ratings, timestamps).
4. Prediction: Feeds the structured dataframe into the XGBoost model, returning both the discrete predicted class (White, Black, or Draw) and the exact confidence probabilities.

#### Expected Inputs

The tool requires the following inputs extracted from the user's prompt: `rated` (bool), `created_at` (timestamp), `last_move_at` (timestamp), `turns` (int), `white_rating` (int), `black_rating` (int), `opening_ply` (int), `increment_code` (str), `opening_eco` (str), and `opening_name` (str).

## 5. Example Conversations

### Example A: Prediction Request

_User_: Can you predict a match for me? White is rated 1500, Black is 1200, they play a C20 opening, 15+2 increment, and it lasts 40 turns. The game is rated, started at 1504210000000, ended at 1504210000000, and opening ply was 5. The opening name is 'King's Pawn'.

_Agent Internal Action:_ `Invoking: predict_chess_match_outcome` with the extracted parameters.

_AI Response_: Based on the parameters provided, here is the prediction for the match:

```
### Match Prediction Analysis
*   **Predicted Outcome**: **White**
*   **Model Probabilities**:
    *   **White Win**: 78.68%
    *   **Black Win**: 18.96%
    *   **Draw**: 2.36%

Given the significant rating difference (300 points) and the nature of the opening, the model strongly favors White to win the game.
```


---

### Example B: Domain Knowledge (RAG) Request

_User_: What is an ELO code and how does it work?

_Agent Internal Action_: `Invoking: retrieve_chess_context` with query "What is an ELO code and how does it work?"

_AI Response_:

```
> Finished chain.

AI: The **Elo rating system** (named after its creator, Arpad Elo) is a method used to calculate the relative skill levels of players in zero-sum games, most famously chess.

Here is a breakdown of how it works:

### 1. The Core Concept
An Elo rating is a number that represents a player's current strength. It is not a fixed score like a high score in a video game; rather, it is a **probabilistic estimate**. The difference in ratings between two players predicts the outcome of a match:
*   **Equal ratings:** Both players are expected to score equally (50% each).
*   **100-point difference:** The higher-rated player is expected to score about 64%.
*   **200-point difference:** The higher-rated player is expected to score about 76%.

### 2. How Ratings Change
After every rated game, the ratings are updated based on the result:
*   **Winning/Losing Points:** The winner takes points from the loser.
*   **The "Upset" Factor:** The system is self-correcting.
    *   If a **high-rated player beats a low-rated player**, the system expected that result, so the rating change is small.
    *   If a **low-rated player beats a high-rated player** (an upset), the system realizes the low-rated player is likely better than their current rating suggests, so they gain a significant number of points, and the high-rated player loses a significant number.
*   **Draws:** If a lower-rated player draws against a higher-rated player, the lower-rated player will gain a small number of points because they performed better than the system expected.

### 3. Why it is "Self-Correcting"
Because the system adjusts based on expectations, it naturally pushes players toward a rating that accurately reflects their true playing strength. If a player is underrated, they will consistently win more points than they lose until their rating reaches a point where their expected score matches their actual performance.

***Note:** You mentioned "ELO code" in your question. In the context of chess, "Elo" is a name (not an acronym, so it is usually written as "Elo" rather than "ELO"), and "ECO" (Encyclopedia of Chess Openings) is the term used for the codes that classify specific opening sequences (like the C20 you mentioned earlier).*
```

**NOTE** if the questions change order there is going to be a different answer.

### Why does changing the order of questions change the answers?

This happens because Large Language Models (LLMs) do not "think" about concepts in isolation; they generate text by predicting the next logical word based on **the entire context window**. 

When you use LangChain's memory feature, every time you ask a question, your entire previous conversation is secretly pasted into the prompt right above your new question. 
- If you ask Question A, then Question B, the LLM answers Question B while "looking" at the context of Question A. 
- If you reverse the order, the LLM answers Question A with zero prior context. 

Because the text it is reading is mathematically different, the probabilities of which words it should generate next shift entirely. This is why AI engineers spend so much time on "prompt engineering"—the exact sequence of words and history fundamentally alters the model's behavior.

### A Comment on `temperature=0.2`

In  `agent.py` the model is initialized with `temperature=0.2`. 

_Temperature_ is a setting between 0.0 and 1.0 (or sometimes up to 2.0) that controls the "creativity" or "randomness" of the LLM. 
* A high temperature (e.g., 0.8 - 1.0) makes the model output more diverse, creative, and unpredictable words. This is great for writing poetry or brainstorming ideas, but terrible for writing code or analyzing data.
* A low temperature (e.g., 0.0 - 0.2) makes the model highly deterministic, focused, and factual. It will consistently pick the most statistically likely next word.

Because the agent acts as a data-driven prediction tool and a factual domain expert for chess rules, we set the temperature to 0.2. This ensures the model sticks strictly to the exact numbers the XGBoost tool returns and the exact rules in the RAG documents, removing the risk of hallucination, e.g., fake stats or getting "creative" with the rules of chess.

## 6. Installation and execution 

### Installation

- Create a virtual environment: `python -m venv venv`
- Activate virtual environment (nix system): `source venv/bin/activate/`
- Install libraries/dependencies: `pip install -r requirements.txt`
- Create .env file with your `GEMINI_API_KEY`

### Execution

In the root folder run: `python -m src.agent`


### 7. REST API Integration (FastAPI)

The agent is exposed as a production-ready REST API using FastAPI, allowing external applications or front-end interfaces to interact with it seamlessly.

Architecture:
- `main.py`: The application entry point. It initializes the FastAPI server and connects the routing.
- `src/api.py`: Contains the `APIRouter` and the Pydantic schemas defining the API's contract.

Endpoint: `POST /chat`
Accepts a natural language message and routes it through the LangChain agent.

Request Schema (JSON):
```json
{
  "message": "Predict a match where White is 1500 and Black is 1200.",
  "session_id": "user-12345"
}
```

You can test it in swagger, postman or even Curl

```bash
curl -X 'POST' 'http://127.0.0.1:8000/chat' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"message": "Predict a match for me: White is rated 1650, Black is 1400, they play a C20 opening, 15+2 increment, and it lasts 35 turns. The game is rated, started at 1504210000000, ended at 1504210000000, opening ply was 4, opening name is Kings Pawn.", "session_id": "curl-test-session"}'
```