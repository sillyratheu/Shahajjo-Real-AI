import os
import re
import json
import requests
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
            content = response.text
            
            # Direct JSON parse try
            try:
                return response.json()
            except Exception:
                pass

            # Extract JSON block from Streamlit HTML response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(1)
                return json.loads(extracted_json)
            
            # Alternative JSON extraction attempt
            json_match_raw = re.search(r'(\{[\s\S]*"response"[\s\S]*\})', content)
            if json_match_raw:
                try:
                    return json.loads(json_match_raw.group(1))
                except Exception:
                    pass

            return {
                "status": "success",
                "response": f"Streamlit processing complete for: {text}",
                "reply": f"Streamlit processing complete for: {text}",
                "message": f"Streamlit processing complete for: {text}",
                "text": f"Streamlit processing complete for: {text}"
            }
        else:
            raise HTTPException(status_code=500, detail="Backend unreachable")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gateway error: {str(e)}")

@app.post("/chat")
def chat_post(request: TextRequest):
    return chat_get(request.prompt)
