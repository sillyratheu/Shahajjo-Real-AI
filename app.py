from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
import json

app = FastAPI(title="Shahajjo Render Gateway")

# CORS Enable kora (Jate jekono front-end theke request nite pare)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STREAMLIT_URL = "https://shahajjo.streamlit.app/"

@app.get("/chat")
def chat(text: str = Query(..., description="Enter your prompt text")):
    if not text:
        raise HTTPException(status_code=400, detail="Please provide text parameter.")
    
    try:
        resp = requests.get(STREAMLittonUrl := STREAMLIT_URL, params={"text": text}, timeout=15)
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Backend status {resp.status_code}")
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        code_tag = soup.find('code')
        
        if code_tag:
            return json.loads(code_tag.get_text())
        else:
            raise HTTPException(status_code=500, detail="Could not parse Streamlit response.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gateway error: {str(e)}")
