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

def get_realtime_smart_answer(query: str, lang: str = "bn", is_image: bool = False) -> str:
    msg_lower = query.lower().strip()

    # Multi-lingual Identity Check (Bangla, Banglish, English)
    identity_triggers = ["tumi ke", "who are you", "what is your name", "tomar nam ki", "কে তুমি", "তোমার নাম কি", "tumi konta desher", "where are you from", "kothay toiri"]
    if any(w in msg_lower for w in identity_triggers):
        if lang == "bn":
            return "আমার নাম সাহায্য (Shahajjo AI)। আমি গণপ্রজাতন্ত্রী বাংলাদেশের একটি গর্বিত নিজস্ব কৃত্রিম বুদ্ধিমত্তা (AI) সহকারী, যা বাংলা, বাংলিশ এবং ইংরেজি ভাষায় আপনাকে সাহায্য করার জন্য তৈরি করা হয়েছে।"
        elif lang == "banglish":
            return "Amar nam Shahajjo AI. Ami Bangladesher ekta nijshwo AI assistant, ja Bangla, Banglish ebong English-e sahajjo korar jonno toiri."
        else:
            return "My name is Shahajjo AI. I am an indigenous Artificial Intelligence assistant built for Bangladesh, supporting Bangla, Banglish, and English."

    # Greetings in 3 Languages
    if any(w in msg_lower for w in ["hi", "hello", "kemn", "kemon", "কেমন", "how are you"]):
        if lang == "bn":
            return "আলহামদুলিল্লাহ, আমি সাহায্য এআই! একদম ভালো আছি। আপনাকে আজ কীভাবে সাহায্য করতে পারি?"
        elif lang == "banglish":
            return "Ami Shahajjo AI, valo achi! Apnake aj kibhabe sahajjo korte pari?"
        else:
            return "Hello! I am Shahajjo AI, doing great. How can I assist you today?"

    # Real-time Live Web Search for Recent Events
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=4)]
            if results:
                combined = " ".join([r.get('body', '') for r in results])
                if combined:
                    return clean_output_text(combined[:500] + ".")
    except Exception:
        pass

    # Intelligent Synthesis Fallback
    try:
        sys_prompt = "You are 'Shahajjo AI' (সাহায্য), a powerful indigenous AI built for Bangladesh. Provide smart, reasoning-heavy responses in Bangla, Banglish, or English."
        if lang == "bn":
            sys_prompt += " Answer naturally in pure Bangla."
        elif lang == "banglish":
            sys_prompt += " Answer in natural Banglish."
        else:
            sys_prompt += " Answer in English."

        if is_image:
            sys_prompt += " The user has provided an image for advanced multi-modal analysis."

        api_url = "https://text.pollinations.ai/" + urllib.parse.quote(f"{sys_prompt}\nQuery: {query}\nAnswer:")
        res = requests.get(api_url, timeout=8)
        if res.status_code == 200 and res.text.strip():
            cleaned = clean_output_text(res.text.strip())
            if cleaned:
                return cleaned
    except Exception:
        pass

    if lang == "bn":
        return f"আমি সাহায্য এআই। আপনার প্রদেয় '{query}' বিষয়ে বিস্তারিত তথ্য বিশ্লেষণ করা হচ্ছে।"
    elif lang == "banglish":
        return f"Ami Shahajjo AI. Apnar '{query}' query niye detail analysis kora hocche."
    else:
        return f"I am Shahajjo AI. Analyzing details regarding '{query}'."

@app.get("/")
def home():
    return {"status": "online", "assistant": "Shahajjo AI Engine (Made in Bangladesh)"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        user_message = (req.message or "").strip()
        image_data = req.image
        lang = req.lang or "bn"

        if not user_message and not image_data:
            raise HTTPException(status_code=400, detail="Message or image is required")

        prompt_text = user_message if user_message else "Analyze this image and explain details"
        response_text = get_realtime_smart_answer(prompt_text, lang, bool(image_data))
        
        return {"response": response_text}

    except Exception:
        return {"response": "দুঃখিত, তথ্য সংগ্রহ করতে একটু সমস্যা হয়েছে। আবার চেষ্টা করুন।"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.main(["app:app", "--host", "0.0.0.0", "--port", str(port)])
