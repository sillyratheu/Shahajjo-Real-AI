import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from huggingface_hub import hf_hub_download
import re

app = FastAPI(title="Shahajjo Native Neural Engine")

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

def generate_from_llm(prompt: str) -> str:
    if model is None or not vocab:
        return "[Error: Model weights not loaded]"
    try:
        words = prompt.split()
        input_tokens = [vocab.get(w, vocab.get("<unk>", 1)) for w in words]
        
        if not input_tokens:
            input_tokens = [2]  # <sos> token

        current_tensor = torch.tensor([input_tokens], dtype=torch.long)
        generated_ids = []

        temperature = 0.7  # Controls creativity and natural flow

        with torch.no_grad():
            for _ in range(25):
                outputs = model(current_tensor)
                next_token_logits = outputs[0, -1, :] / temperature
                
                # Anti-repetition penalty to stop looping words
                for prev_id in generated_ids:
                    next_token_logits[prev_id] -= 5.0

                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                if next_token_id == 3:  # <eos> token
                    break

                generated_ids.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
                current_tensor = torch.cat([current_tensor, next_token_tensor], dim=1)

        decoded_words = []
        for idx in generated_ids:
            word = inv_vocab.get(idx, inv_vocab.get(str(idx), ""))
            if word and word not in ["User:", "Shahajjo:"]:
                decoded_words.append(word)

        raw_output = " ".join(decoded_words).strip()
        if raw_output:
            return raw_output
        
        return "Ami Shahajjo AI, apnake sahajjo korar jonno prastut achi."
        
    except Exception as e:
        return f"[Model Inference Error: {str(e)}]"

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
