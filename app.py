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

# Load Custom PyTorch LLM from Hugging Face safely
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
    message: Optional[str] = ""
    image: Optional[str] = None
    lang: Optional[str] = "bn"

def clean_output_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace("reasoning:", "").replace("tool_calls:", "").replace("assistant", "")
    
    sentences = text.split('.')
    unique_sentences = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in unique_sentences:
            unique_sentences.append(s_clean)
    return ". ".join(unique_sentences) + ("." if unique_sentences else "")

def generate_from_scratch_llm(prompt: str) -> str:
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
            res_text = " ".join(generated_words)
            return clean_output_text(res_text)
    except Exception:
        return ""

def get_smart_response(query: str, lang: str = "bn", is_image: bool = False) -> str:
    msg_lower = query.lower().strip()

    # Multi-lingual Identity Check
    identity_triggers = ["tumi ke", "who are you", "what is your name", "tomar nam ki", "কে তুমি", "তোমার নাম কি", "tumi konta desher", "where are you from", "kothay toiri"]
    if any(w in msg_lower for w in identity_triggers):
        if lang == "bn":
            return "আমার নাম সাহায্য (Shahajjo AI)। আমি গণপ্রজাতন্ত্রী বাংলাদেশের একটি গর্বিত নিজস্ব কৃত্রিম বুদ্ধিমত্তা (AI) সহকারী, যা বাংলা, বাংলিশ এবং ইংরেজি ভাষায় আপনাকে সাহায্য করার জন্য তৈরি করা হয়েছে।"
        elif lang == "banglish":
            return "Amar nam Shahajjo AI. Ami Bangladesher ekta nijshwo AI assistant, ja Bangla, Banglish ebong English-e sahajjo korar jonno toiri."
        else:
            return "My name is Shahajjo AI. I am an indigenous Artificial Intelligence assistant built for Bangladesh, supporting Bangla, Banglish, and English."

    # 1. Try Custom PyTorch LLM First
    if model is not None:
        llm_ans = generate_from_scratch_llm(query)
        if llm_ans and len(llm_ans) > 4:
            return llm_ans

    # 2. Live Web Search Fallback for Recent Events
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if results:
                combined = " ".join([r.get('body', '') for r in results])
                if combined:
                    return clean_output_text(combined[:450] + ".")
    except Exception:
        pass

    # 3. Synthesis Fallback
    try:
        sys_prompt = "You are 'Shahajjo AI' (সাহায্য), a powerful indigenous AI built for Bangladesh."
        api_url = "https://text.pollinations.ai/" + urllib.parse.quote(f"{sys_prompt}\nQuery: {query}\nAnswer:")
        res = requests.get(api_url, timeout=6)
        if res.status_code == 200 and res.text.strip():
            return clean_output_text(res.text.strip())
    except Exception:
        pass

    return "আমি সাহায্য এআই। আপনার প্রশ্নের উত্তর প্রক্রিয়া করা হচ্ছে।"

@app.get("/")
def home():
    return {"status": "online", "assistant": "Shahajjo AI Engine (Custom PyTorch LLM + Realtime Search)"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        user_message = (req.message or "").strip()
        image_data = req.image
        lang = req.lang or "bn"

        if not user_message and not image_data:
            raise HTTPException(status_code=400, detail="Message or image is required")

        prompt_text = user_message if user_message else "Analyze this image"
        response_text = get_smart_response(prompt_text, lang, bool(image_data))
        
        return {"response": response_text}
    except Exception:
        return {"response": "দুঃখিত, সিস্টেম প্রসেসিং এ একটু সমস্যা হয়েছে।"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.main(["app:app", "--host", "0.0.0.0", "--port", str(port)])
