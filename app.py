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
import torch.nn.functional as F

app = FastAPI(title="Shahajjo AI Custom Engine")

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

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'^shahajjo\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^user\s*:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def generate_from_llm(prompt: str) -> str:
    if model is None or not vocab:
        return ""
    try:
        formatted_prompt = f"User: {prompt} Shahajjo:"
        tokens = [vocab.get(w, 1) for w in formatted_prompt.split()[:40]]
        if not tokens:
            return ""
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_ids = torch.argmax(outputs, dim=-1)[0].tolist()

            generated_words = []
            for idx in predicted_ids:
                if idx > 3:
                    word = inv_vocab.get(idx, inv_vocab.get(str(idx), ""))
                    if word and word not in ["User:", "Shahajjo:", "<pad>", "<unk>"]:
                        generated_words.append(word)

            filtered_words = []
            for w in generated_words:
                if not filtered_words or filtered_words[-1] != w:
                    filtered_words.append(w)

            res = clean_text(" ".join(filtered_words))
            return res
    except Exception as e:
        print(f"Generation Error: {e}")
    return ""

@app.get("/")
def root():
    return {"status": "online", "model": "Shahajjo AI Engine"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = (req.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message is required")

    response_text = generate_from_llm(user_msg)
    msg_lower = user_msg.lower()

    # Dynamic Intent Matcher
    if not response_text or len(response_text) < 2 or response_text.lower() == "why":
        if msg_lower in ["hi", "hello", "hey"]:
            response_text = "Hello! Ami Shahajjo AI. Apnake kibhabe sahajjo korte pari?"
        elif msg_lower in ["ki", "ki khobor"]:
            response_text = "Ami valo achi! Apnar ki dhoroner sahajjo lagbe?"
        elif msg_lower in ["keno", "why"]:
            response_text = "Apni ki bishoye janote chan? Sothik bhabe bolun, ami Uttar dicchi."
        elif "tmi" in msg_lower or "tumi" in msg_lower:
            response_text = "Ami Shahajjo AI, Bangladesher custom transformer LLM assistant."
        elif "ami" in msg_lower:
            response_text = "Apni Shahajjo AI-er sathe kotha bolchen. Ki lagbe bolun?"
        else:
            response_text = f"Shahajjo AI response: '{user_msg}' niye processing-e ache."

    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
