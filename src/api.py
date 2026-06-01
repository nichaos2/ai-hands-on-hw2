import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import the initialized agent directly from your src folder
from src.agent import create_conversational_agent

# Initialize the router
router = APIRouter()


# --- Define Pydantic Schemas ---
class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's natural language input.")
    session_id: str = Field(
        ..., description="A unique identifier to track conversation memory."
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="The agent's text response.")


# --- Define the Endpoint ---
@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Accepts a user message and a session ID, routes it through the LangChain agent,
    and returns the final response string.
    """
    conversational_agent = create_conversational_agent()
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Invoke the agent using the provided message and session ID
            result = conversational_agent.invoke(
                {"input": request.message},
                config={"configurable": {"session_id": request.session_id}},
            )

            return ChatResponse(response=result["output"])

        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue

            raise HTTPException(status_code=500, detail=error_msg)
