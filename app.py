from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import re
import json

app = FastAPI(title="Shahajjo Render Gateway")

# CORS Enable kora
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STREAMLIT_URL = "https://shahajjo.streamlit.app/"

@app.get("/")
def root():
    return {
        "message": "Shahajjo API Gateway is active!",
        "backend": STREAMLIT_URL
    }

@app.get("/chat")
def chat(text: str = Query(..., description="Enter your prompt text")):
    if not text:
        raise HTTPException(status_code=400, detail="Please provide text parameter.")
    
    try:
        # Streamlit backend-e request pathano
        resp = requests.get(STREAMLIT_URL, params={"text": text}, timeout=25)
        
        if resp.status_code != 200:
            return {
                "status": "error",
                "response": f"Backend returned status {resp.status_code}"
            }
        
        html_content = resp.text
        
        # Regex diye HTML-er vitor theke JSON object ({ ... "status" ... }) khuje ber kora
        json_match = re.search(r'(\{.*?"status"\s*:\s*"(?:success|error)".*?\})', html_content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            # Streamlit HTML entity gulo fix kora (jodi thake)
            json_str = json_str.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            return json.loads(json_str)
        else:
            return {
                "status": "error",
                "response": "Could not extract JSON from Streamlit response.",
                "debug_info": html_content[:300]
            }
            
    except Exception as e:
        return {
            "status": "error",
            "response": f"Gateway error: {str(e)}"
        }
