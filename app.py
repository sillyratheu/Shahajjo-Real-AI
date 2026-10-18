import os
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from huggingface_hub import hf_hub_download
import requests
import urllib.parse
import re

app = FastAPI(title="Shahajjo AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "Vidclip/Shahajjo"

class ShahajjoTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        return self.fc_out(out)

model = None
vocab = {}
inv_vocab = {}

try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    model = ShahajjoTransformer(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except Exception:
    pass

class ChatRequest(BaseModel):
    message: Optional[str] = ""
    image: Optional[str] = None
    lang: Optional[str] = "bn"

def clean_text(text: str) -> str:
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', text).strip()

def search_and_remix_ai(query: str, lang: str = "bn", is_image: bool = False) -> str:
    msg_lower = query.lower().strip()

    if any(w in msg_lower for w in ["hi", "hello", "kemn", "kemon", "কেমন"]):
        if lang == "bn":
            return "Alhamdulillah, ami valo achi! Shahajjo AI hishebe apnake aj kibhabe sahajjo korte pari?"
        elif lang == "banglish":
            return "Ami valo achi! Apnake aj kibhabe sahajjo korte pari?"
        else:
            return "Hello! I am doing great. How can Shahajjo AI assist you today?"

    if any(w in msg_lower for w in ["nam", "name", "tumi ke", "কে তুমি", "নাম"]):
        return "Amar nam Shahajjo AI. Ami Bangladesher jonno toiri shompurno deshiyo technology er AI assistant."

    wiki_context = ""
    try:
        search_lang = "bn" if lang == "bn" else "en"
        encoded_query = urllib.parse.quote(query)
        wiki_url = f"https://{search_lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
        res = requests.get(wiki_url, timeout=4)
        if res.status_code == 200:
            data = res.json()
            if "extract" in data and data["extract"]:
                wiki_context = data["extract"]
    except Exception:
        pass

    try:
        system_instruction = (
            "You are 'Shahajjo AI' (সাহায্য), an intelligent indigenous AI assistant built for Bangladesh. "
            "Synthesize and remix live search context with your AI reasoning to give a comprehensive, friendly answer. "
            "Never show raw HTML tags or debug system text."
        )

        if lang == "bn":
            system_instruction += " Respond fluently in Bangla."
        elif lang == "banglish":
            system_instruction += " Respond in natural Banglish."
        else:
            system_instruction += " Respond in clear English."

        if is_image:
            system_instruction += " Note: The user has attached an image for analysis."

        user_prompt = f"User Question: {query}\n"
        if wiki_context:
            user_prompt += f"Context Data: {wiki_context}\n"
        user_prompt += "Shahajjo AI Response:"

        api_url = "https://text.pollinations.ai/" + urllib.parse.quote(f"{system_instruction}\n{user_prompt}")
        res = requests.get(api_url, timeout=12)

        if res.status_code == 200 and res.text.strip():
            return clean_text(res.text.strip())
    except Exception:
        pass

    if wiki_context:
        return clean_text(wiki_context)

    if lang == "bn":
        return f"Apnar proshno '{query}' analayse kore Shahajjo AI uttor toiri korche."
    else:
        return f"Shahajjo AI is processing your request regarding '{query}'."

@app.get("/")
def home():
    return {"status": "online", "assistant": "Shahajjo AI Engine"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        user_message = (req.message or "").strip()
        image_data = req.image
        lang = req.lang or "bn"

        if not user_message and not image_data:
            raise HTTPException(status_code=400, detail="Message or image is required")

        is_img = bool(image_data)
        prompt_text = user_message if user_message else "Analyze attached image"

        response_text = search_and_remix_ai(prompt_text, lang, is_img)
        return {"response": response_text}

    except Exception:
        return {"response": "Dukkito, tottho onushandane shomossha hoyeche. Abar chesta korun."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.main(["app:app", "--host", "0.0.0.0", "--port", str(port)])
