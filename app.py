from fastapi import FastAPI, Query, HTTPException
import requests
from bs4 import BeautifulSoup
import json

app = FastAPI(title="Shahajjo Render Gateway")

# Tomar Streamlit backend URL
STREAMLIT_URL = "https://shahajjo.streamlit.app/"

@app.get("/chat")
def chat(text: str = Query(..., description="Enter your prompt text")):
    if not text:
        raise HTTPException(status_code=400, detail="Please provide text parameter.")
    
    try:
        # Streamlit backend-e request pathano
        resp = requests.get(STREAMLIT_URL, params={"text": text}, timeout=15)
        
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Backend returned status {resp.status_code}")
        
        # BeautifulSoup diye Streamlit-er HTML theke st.code er JSON tulhe ana
        soup = BeautifulSoup(resp.text, 'html.parser')
        code_tag = soup.find('code')
        
        if code_tag:
            data = json.loads(code_tag.get_text())
            return data
        else:
            raise HTTPException(status_code=500, detail="Could not parse response from Streamlit backend.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gateway error: {str(e)}")