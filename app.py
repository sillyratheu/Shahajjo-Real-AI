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
from duckduckgo_search import DDGS

app = FastAPI(title="Shahajjo AI LLM & Tutor API")

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

# Load Custom PyTorch LLM with memory optimization for Render
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
    print("✅ Custom PyTorch LLM loaded successfully!")
except Exception as e:
    print(f"⚠️ Model load note: {e}")

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
        tokens = [vocab.get(w, 1) for w in prompt.split()[:20]]
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

def get_tutor_response(query: str, lang: str = "bn") -> str:
    msg_lower = query.lower().strip()

    # Identity Check
    identity_triggers = ["tumi ke", "who are you", "what is your name", "tomar nam ki", "কে তুমি", "তোমার নাম কি"]
    if any(w in msg_lower for w in identity_triggers):
        if lang == "bn":
            return "আমার নাম সাহায্য (Shahajjo AI Tutor)। আমি বাংলাদেশের একটি নিজস্ব কৃত্রিম বুদ্ধিমত্তা সহকারী, যা আপনাকে শিখাতে ও সাহায্য করতে তৈরি!"
        elif lang == "banglish":
            return "Ami Shahajjo AI Tutor. Ami Bangladesher nijshwo AI assistant, apnake sahajjo korar jonno toiri!"
        else:
            return "I am Shahajjo AI Tutor, an indigenous assistant built for Bangladesh."

    # 1. Try Custom LLM First
    if model is not None:
        llm_res = generate_from_llm(query)
        if llm_res and len(llm_res) > 3:
            return f"{llm_res}"

    # 2. DuckDuckGo Realtime Search Fallback
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if results:
                bodies = [r.get('body', '') for r in results]
                combined = " ".join(bodies)
                if combined:
                    return f"📚 **Tutor Note:**\n\n{clean_text(combined[:400])}..."
    except Exception:
        pass

    return "দুঃখিত, এই মুহূর্তে উত্তর প্রক্রিয়া করা যাচ্ছে না।"

@app.get("/")
def root():
    return {"status": "online", "assistant": "Shahajjo AI LLM & Tutor API"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = (req.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message is required")
    reply = get_tutor_response(user_msg, req.lang or "bn")
    return {"response": reply}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
