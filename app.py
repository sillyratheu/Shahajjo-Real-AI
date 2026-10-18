import os
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from huggingface_hub import hf_hub_download

app = FastAPI(title="Shahajjo AI API")

# Enable CORS
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

@app.get("/")
def home():
    return {"status": "online", "assistant": "Shahajjo AI FastAPI Engine"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        user_message = (req.message or "").strip()
        image_data = req.image
        lang = req.lang or "bn"

        if not user_message and not image_data:
            raise HTTPException(status_code=400, detail="Message or image is required")

        if image_data:
            reply = "ছবিটি সফলভাবে পাওয়া গেছে। এটি নিয়ে কাজ চলছে!"
            return {"response": reply}

        msg_lower = user_message.lower()

        # Smart Response Logic
        if any(w in msg_lower for w in ["hi", "hello", "salam", "سلام", "assalamu"]):
            reply = "ওয়ালাইকুমুস সালাম / হ্যালো! আলহামদুলিল্লাহ, আমি ভালো আছি। আপনাকে আজ কীভাবে সাহায্য করতে পারি?"
        elif any(w in msg_lower for w in ["nam", "name", "tumi ke", "কে তুমি", "নাম কি"]):
            reply = "আমার নাম সাহায্য এআই (Shahajjo AI)। আমাকে বাংলাদেশের জন্য সম্পূর্ণ দেশীয় প্রযুক্তিতে তৈরি করা হয়েছে।"
        elif any(w in msg_lower for w in ["kemon", "valoo", "bhalo", "how are", "কেমন"]):
            reply = "আমি একদম ঠিক আছি! আপনার পড়ালেখা বা প্রজেক্টের কাজে আজ কীভাবে সাহায্য করতে পারি বলুন?"
        elif any(w in msg_lower for w in ["ke banise", "maker", "creator", "কে তৈরি"]):
            reply = "আমাকে বাংলাদেশের আইসিটি খাতের উপযোগী করে তৈরি করা হয়েছে।"
        else:
            if lang == "bn":
                reply = f"আপনার কথাটি বুঝতে পেরেছি: \"{user_message}\"। বিষয়টি চমৎকার, বলুন এ বিষয়ে আর কী জানতে চান?"
            elif lang == "banglish":
                reply = f"Apnar kotha bujhte perechi: \"{user_message}\". Eishoyb niye aro kotha bola jabe!"
            else:
                reply = f"I understand your query: \"{user_message}\". Let me know how you'd like to proceed!"

        return {"response": reply}

    except Exception as e:
        return {"response": "দুঃখিত, একটু টেকনিক্যাল সমস্যা হয়েছে। আবার চেষ্টা করুন।"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.main(["app:app", "--host", "0.0.0.0", "--port", str(port)])
