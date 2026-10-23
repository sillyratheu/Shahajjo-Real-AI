import os
import json
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Shahajjo API Gateway", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STREAMLIT_BACKEND_URL = os.environ.get("STREAMLIT_URL", "https://shahajjo.streamlit.app")

class TextRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {
        "message": "Shahajjo API Gateway is active!",
        "backend": STREAMLIT_BACKEND_URL
    }

@app.get("/chat")
def chat_get(text: str = Query(..., description="Prompt for Shahajjo AI")):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    try:
        target_url = f"{STREAMLIT_BACKEND_URL}/?text={requests.utils.quote(text)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(target_url, headers=headers, timeout=25)
        
        if response.status_code == 200:
            # BeautifulSoup diye HTML parse kora
            soup = BeautifulSoup(response.text, 'html.parser')
            target_div = soup.find('div', id='real-json-output')
            
            if target_div:
                try:
                    return json.loads(target_div.text)
                except Exception as e:
                    return {"status": "error", "response": f"JSON decode error: {str(e)}"}
            
            return {
                "status": "error",
                "response": "Real JSON output tag not found in Streamlit HTML."
            }
        else:
            raise HTTPException(status_code=500, detail="Backend unreachable")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gateway error: {str(e)}")

@app.post("/chat")
def chat_post(request: TextRequest):
    return chat_get(request.prompt)
