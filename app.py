import requests
from bs4 import BeautifulSoup
import json

def get_shahajjo_response(user_text):
    # Apnar Streamlit app-er URL
    url = "https://shahajjo.streamlit.app/"
    
    # Query parameter hishebe text pathano
    params = {"text": user_text}
    
    try:
        # Request send kora
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return {"status": "error", "response": f"HTTP Error: {response.status_code}"}
        
        # BeautifulSoup diye HTML parse kora
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Streamlit-er st.code() shadharoto <code> tag-er vitor thake
        code_tag = soup.find('code')
        
        if code_tag:
            # Code tag-er vitorer text (JSON string) ke load kora
            json_data = json.loads(code_tag.get_text())
            return json_data
        else:
            return {"status": "error", "response": "JSON response block not found in HTML page."}
            
    except Exception as e:
        return {"status": "error", "response": f"Connection error: {str(e)}"}

# --- Test korar jonno ---
if __name__ == "__main__":
    result = get_shahajjo_response("hi")
    print("API Result:", result)
    print("AI Response text:", result.get("response"))