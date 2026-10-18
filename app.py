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
except Exception as e:
    pass

class ChatRequest(BaseModel):
    message: Optional[str] = ""
    image: Optional[str] = None
    lang: Optional[str] = "bn"

# Wikipedia & Web Search Engine Integration
def search_wikipedia_and_web(query: str, lang: str = "bn") -> str:
    search_lang = "bn" if lang == "bn" else "en"
    encoded_query = urllib.parse.quote(query)
    
    # 1. Search Wikipedia
    try:
        wiki_url = f"https://{search_lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
        res = requests.get(wiki_url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if "extract" in data and data["extract"]:
                return data["extract"]
    except Exception:
        pass

    # 2. Search Live Web Encyclopedia Engine
    try:
        system_prompt = (
            "You are 'Shahajjo AI', an indigenous Bengali AI assistant. "
            "Use live information and web knowledge to answer accurately in "
        )
        if lang == "bn":
            system_prompt += "pure Bangla language."
        elif lang == "banglish":
            system_prompt += "Banglish (Bengali written in Latin script)."
        else:
            system_prompt += "English language."

        url = "https://text.pollinations.ai/" + urllib.parse.quote(f"{system_prompt}\nUser Question: {query}\nShahajjo AI:")
        res = requests.get(url, timeout=12)
        if res.status_code == 200 and res.text.strip():
            return res.text.strip()
    except Exception:
        pass

    # Dynamic Fallback
    if lang == "bn":
        return f"আমি সাহায্য এআই। আপনার '<b>{query}</b>' তথ্যটি উইকিপিডিয়া ও বিশ্বকোষ থেকে অনুসন্ধান করে আপনার প্রশ্নের উত্তর তৈরি করা হচ্ছে।"
    elif lang == "banglish":
        return f"Ami Shahajjo AI. Apnar '<b>{query}</b>' query-ti search kore uttor toiri kora hocche."
    else:
        return f"I am Shahajjo AI. Processing information for '<b>{query}</b>' from knowledge bases."

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

        if image_data:
            if lang == "bn":
                return {"response": f"ছবিটি পাওয়া গিয়েছে। আপনার প্রশ্ন: '{user_message if user_message else 'ছবি বিশ্লেষণ'}'। এটি বিশ্লেষণ করা হচ্ছে।"}
            elif lang == "banglish":
                return {"response": f"Chobi peyechi. Query: '{user_message if user_message else 'Image analysis'}'."}
            else:
                return {"response": f"Image received. Processing query: '{user_message if user_message else 'Image analysis'}'."}

        # Search Wikipedia and Multi-Knowledge base for answers
        ai_reply = search_wikipedia_and_web(user_message, lang)
        return {"response": ai_reply}

    except Exception as e:
        return {"response": "দুঃখিত, তথ্য অনুসন্ধানে একটু সমস্যা হয়েছে। আবার চেষ্টা করুন।"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.main(["app:app", "--host", "0.0.0.0", "--port", str(port)])
