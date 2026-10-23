import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI(title="Shahajjo API Gateway", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Streamlit app er public URL ekhane bosbe (jemon: https://your-app.streamlit.app)
STREAMLIT_BACKEND_URL = os.environ.get("STREAMLIT_URL", "https://your-streamlit-app.streamlit.app")

class TextRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {
        "message": "Shahajjo API Gateway (Render) is running successfully!",
        "streamlit_backend": STREAMLIT_BACKEND_URL
    }

@app.get("/chat")
def chat_get(text: str = Query(..., description="Prompt for Shahajjo AI")):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    try:
        # Streamlit backend er query endpoint e request pathano
        target_url = f"{STREAMLIT_BACKEND_URL}/?text={text}"
        response = requests.get(target_url, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=500, detail=f"Streamlit backend error: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gateway connection error: {str(e)}")

@app.post("/chat")
def chat_post(request: TextRequest):
    return chat_get(request.prompt)
