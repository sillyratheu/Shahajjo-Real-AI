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

app = FastAPI(title="Shahajjo AI Light Engine")

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

# Load Custom PyTorch LLM safely
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
    print(f"⚠️ LLM Note: {e}")

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

def get_smart_tutor_response(query: str, lang: str = "bn") -> str:
    msg_lower = query.lower().strip()

    # Identity Check (Bangla, Banglish, English)
    identity_triggers = ["tumi ke", "who are you", "what is your name", "tomar nam ki", "কে তুমি", "তোমার নাম কি"]
    if any(w in msg_lower for w in identity_triggers):
        if lang == "bn":
            return "আমার নাম সাহায্য (Shahajjo AI)। আমি বাংলাদেশের একটি নিজস্ব কৃত্রিম বুদ্ধিমত্তা সহকারী, যা বাংলা, বাংলিশ ও ইংরেজি ভাষায় প্রশ্নের উত্তর দিতে তৈরি।"
        elif lang == "banglish":
            return "Amar nam Shahajjo AI. Ami Bangladesher ekta nijshwo AI assistant!"
        else:
            return "My name is Shahajjo AI, an indigenous AI assistant built for Bangladesh."

    # 1. Custom PyTorch LLM (Nij theke uttor dibe)
    if model is not None:
        llm_res = generate_from_llm(query)
        if llm_res and len(llm_res) > 3:
            return llm_res

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

    # 3. Web Synthesis Fallback
    try:
        sys_prompt = "You are 'Shahajjo AI', a smart educational tutor for Bangladesh."
        api_url = "https://text.pollinations.ai/" + urllib.parse.quote(f"{sys_prompt}\nQuery: {query}\nAnswer:")
        res = requests.get(api_url, timeout=6)
        if res.status_code == 200 and res.text.strip():
            return clean_text(res.text.strip())
    except Exception:
        pass

    return "আমি সাহায্য এআই। আপনার প্রশ্নের উত্তর প্রক্রিয়া করা হচ্ছে।"

@app.get("/")
def root():
    return {"status": "online", "service": "Shahajjo AI Light API Engine"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = (req.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message is required")
    reply = get_smart_tutor_response(user_msg, req.lang or "bn")
    return {"response": reply}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
