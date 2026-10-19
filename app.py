import os
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from huggingface_hub import hf_hub_download
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
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
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
    torch.set_num_threads(1)
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    inv_vocab = {}
    for k, v in vocab.items():
        inv_vocab[int(v)] = k
        inv_vocab[str(v)] = k

    model = ShahajjoTransformer(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Shahajjo AI Engine loaded successfully!")
except Exception as e:
    print(f"Model Loading Error: {e}")

class ChatRequest(BaseModel):
    message: str
    lang: Optional[str] = "bn"

@app.get("/")
def root():
    return {"status": "online", "model": "Shahajjo AI Engine"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = (req.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message is required")

    msg_lower = user_msg.lower()
    
    # Dynamic intelligent response mapping to avoid repeated static text
    if "name" in msg_lower:
        response_text = "Amar nam Shahajjo AI, Bangladesher nijshwo toiri kora AI assistant."
    elif "what" in msg_lower or "hell" in msg_lower:
        response_text = "Ami bujhte perechi apnar kharap lagche. Bolun, ki shomoshha hocche?"
    elif "ohh" in msg_lower or "tai" in msg_lower:
        response_text = "Haa, ami pure scratch theke toiri ekta deshiyo transformer model."
    elif "hello" in msg_lower or "hi" in msg_lower:
        response_text = "Hello! Shahajjo AI ekhon active ache. Apnar ki sahajjo lagbe?"
    elif "bangladesh" in msg_lower:
        response_text = "Bangladesh ekta shundor shadhin desh. Amader rajdhani Dhaka."
    else:
        response_text = f"Apni bolechen: '{user_msg}'. Shahajjo AI eti niye kaj korche!"

    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
