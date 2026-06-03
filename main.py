import uvicorn
from fastapi import FastAPI

# Import the router we just built in api.py
from src.api import router as chat_router

app = FastAPI(
    title="Chess AI Agent API",
    description="A conversational API for chess predictions and domain knowledge.",
    version="1.0.0",
)

# Connect the endpoints from src/api.py to this main application
app.include_router(chat_router)

if __name__ == "__main__":
    # Run the API server
    uvicorn.run("main:app", host="127.0.0.1", port=8008, reload=True)
