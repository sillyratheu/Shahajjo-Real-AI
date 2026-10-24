import requests
from bs4 import BeautifulSoup
import json

url = "https://shahajjo.streamlit.app/"
params = {"text": "hello"}

response = requests.get(url, params=params)

# Streamlit-er HTML theke st.code er vitorer JSON tulhe ana
soup = BeautifulSoup(response.text, 'html.parser')
code_block = soup.find('code')

if code_block:
    data = json.loads(code_block.get_text())
    print("Status:", data.get("status"))
    print("AI Response:", data.get("response"))
else:
    print("Could not parse JSON from Streamlit page.")
