import time
import uuid

from langchain_classic.agents import AgentExecutor

# Import the core formatting and parsing pieces needed for tool agents
from langchain_classic.agents.format_scratchpad import format_to_tool_messages
from langchain_classic.agents.output_parsers import ToolsAgentOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# Import your LLM and Tools
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import API_KEY
from src.rag import retrieve_chess_context
from src.tools import predict_chess_match_outcome, recommend_strategic_opening

TOOLS = [
    predict_chess_match_outcome,
    retrieve_chess_context,
    recommend_strategic_opening,
]

LLM = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.2,
    google_api_key=API_KEY,
)

# LANGCHAIN PROMPT & AGENT
SYSTEM_PROMPT = """
You are an expert AI assistant specializing in chess match prediction and chess domain knowledge.
You have access to three specific tools:
1. 'predict_chess_match_outcome': Use this to predict the winner of a match based on stats.
2. 'retrieve_chess_context': Use this to answer general questions about chess concepts, rules, etc.
3. 'recommend_strategic_opening': Use this to advise a winning opening given the user's color (colour) and the players' rating (ELO).

Rules:
- Autonomously decide which tool to use based on the user's prompt.
- If the user asks a conversational follow-up, use the chat history to understand the context.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# MEMORY MANAGEMENT
session_store = {}


def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


# ---> NEW FIX: Flatten Gemini's text block list inside the agent loop <---
def fix_gemini_message(ai_message):
    if hasattr(ai_message, "content") and isinstance(ai_message.content, list):
        text_pieces = []
        for block in ai_message.content:
            if isinstance(block, dict) and "text" in block:
                text_pieces.append(block["text"])
            elif hasattr(block, "text"):
                text_pieces.append(block.text)
            else:
                text_pieces.append(str(block))
        ai_message.content = "".join(text_pieces).strip()
    return ai_message


def clean_agent_output(result):
    # This keeps our final terminal output clean and string-based
    raw_output = result.get("output", "")
    if isinstance(raw_output, list):
        text_pieces = [
            b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in raw_output
        ]
        result["output"] = "".join(text_pieces).strip()
    return result


# Create Conversational Agent
def create_conversational_agent():
    model_with_tools = LLM.bind_tools(TOOLS)

    # Rebuild the tool agent chain manually so we can insert 'fix_gemini_message'
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"])
        )
        | prompt
        | model_with_tools
        | RunnableLambda(fix_gemini_message)  # <-- FIXES THE HANG HERE
        | ToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)
    cleaned_executor = agent_executor | clean_agent_output

    conversational_agent = RunnableWithMessageHistory(
        cleaned_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return conversational_agent


# 4. INTERACTIVE CHAT LOOP
def run_conversational_agent():
    conversational_agent = create_conversational_agent()

    #
    print("\n" + "=" * 50)
    print("CHESS AI AGENT INITIALIZED (LANGCHAIN VERSION)")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 50 + "\n")
    #

    session_id = str(uuid.uuid4())

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Ending session. Goodbye!")
            break

        if not user_input.strip():
            continue

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = conversational_agent.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )

                print(f"\nAI: {response['output']}")
                break

            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "429" in error_msg:
                    print(
                        f"\n[Network] Google servers busy. Retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(5)
                else:
                    print(f"\n[Agent Error]: {error_msg}")
                    break


if __name__ == "__main__":
    run_conversational_agent()
