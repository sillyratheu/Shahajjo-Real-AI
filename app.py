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

app = FastAPI(title="Shahajjo AI Native LLM Engine")

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
    print("Shahajjo Native LLM Engine loaded successfully!")
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
    text = re.sub(r'^user\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^shahajjo\s*:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

# Pure Neural Autoregressive Generation from Model Weights
def generate_from_llm(prompt: str) -> str:
    if model is None or not vocab:
        return "Model weights not loaded yet."
    try:
        formatted_prompt = f"User: {prompt} Shahajjo:"
        input_tokens = [vocab.get(w, 1) for w in formatted_prompt.split() if w in vocab]
        
        if not input_tokens:
            input_tokens = [2]  # <sos> token

        current_tensor = torch.tensor([input_tokens], dtype=torch.long)
        generated_ids = []

        with torch.no_grad():
            for _ in range(25):
                outputs = model(current_tensor)
                next_token_logits = outputs[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                
                if next_token_id == 3:  # <eos> token
                    break

                generated_ids.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
                current_tensor = torch.cat([current_tensor, next_token_tensor], dim=1)

        generated_words = []
        for idx in generated_ids:
            if idx > 3:
                word = inv_vocab.get(idx, inv_vocab.get(str(idx), ""))
                if word and word not in ["User:", "Shahajjo:"]:
                    generated_words.append(word)

        res = clean_text(" ".join(generated_words))
        if res:
            return res
    except Exception as e:
        print(f"Neural Generation Error: {e}")
        
    return "Ami Shahajjo AI, apnake sahajjo korar jonno prastut achi."

@app.get("/")
def root():
    return {"status": "online", "model": "Shahajjo Native Neural Engine"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = (req.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message is required")

    response_text = generate_from_llm(user_msg)
    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
