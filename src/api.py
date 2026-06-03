import asyncio
import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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


# --- Streaming Generator ---
async def stream_agent_response(message: str, session_id: str):
    """
    Asynchronously streams chunks from the agent using LangChain's astream_events,
    formatting them as Server-Sent Events (SSE).
    """
    conversational_agent = create_conversational_agent()
    try:
        # Use astream_events (v2) to capture granular execution steps
        # We target 'on_chat_model_stream' to intercept tokens right as the LLM emits them
        async for event in conversational_agent.astream_events(
            {"input": message},
            config={"configurable": {"session_id": session_id}},
            version="v2",
        ):
            # We filter for the final text generation steps of our chain
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    # In SSE protocol, every data block must start with "data: " and end with double newlines
                    # We wrap it in a JSON object for clean parsing on the client side
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"
                    await asyncio.sleep(0.01)  # Small yield back to the event loop

    except Exception as e:
        # If an error happens mid-stream, broadcast it to the client
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# --- Define the Endpoint ---
@router.post("/chat/stream")
async def chat_endpoint_stream(request: ChatRequest):
    """
    Accepts a user message and a session ID, and streams the agent's
    response token-by-token using Server-Sent Events (SSE).
    """
    # Return a StreamingResponse running our async generator
    return StreamingResponse(
        stream_agent_response(request.message, request.session_id),
        media_type="text/event-stream",
    )
