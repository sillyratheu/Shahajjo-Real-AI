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

app = FastAPI(title="Shahajjo Custom LLM Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "Vidclip/Shahajjo"

class ShahajjoTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
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
    inv_vocab = {v: k for k, v in vocab.items()}

    model = ShahajjoTransformer(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Custom Shahajjo LLM loaded successfully!")
except Exception as e:
    print(f"Model Loading Error: {e}")

class ChatRequest(BaseModel):
    message: str
    lang: Optional[str] = "bn"

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace("reasoning:", "").replace("tool_calls:", "").replace("assistant", "")
    return text.strip()

def generate_from_llm(prompt: str) -> str:
    if model is None or not vocab:
        return ""
    try:
        tokens = [vocab.get(w, 1) for w in prompt.split()[:25]]
        if not tokens:
            return ""
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_ids = torch.argmax(outputs, dim=-1)[0].tolist()
            generated_words = [inv_vocab.get(idx, "") for idx in predicted_ids if idx > 2]
            return clean_text(" ".join(generated_words))
    except Exception:
        return ""

@app.get("/")
def root():
    return {"status": "online", "model": "Shahajjo Custom LLM Engine"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = (req.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message is required")

    response_text = generate_from_llm(user_msg)
    
    if not response_text or len(response_text) < 2:
        msg_lower = user_msg.lower()
        is_english = all(ord(c) < 128 for c in user_msg if c.isalpha())
        
        if is_english:
            response_text = "Hello! I am Shahajjo AI. How can I help you?"
        elif any(b in msg_lower for b in ["ki", "keno", "kmn", "aschhi", "korbo", "tmi", "amar"]):
            response_text = "Ami Shahajjo AI. Apnar kotha bujhchi, bolun ki korte hobe."
        else:
            response_text = "আমি সাহায্য এআই। বলুন, আপনাকে কীভাবে সাহায্য করতে পারি?"

    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
