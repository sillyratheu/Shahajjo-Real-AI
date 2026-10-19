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

app = FastAPI(title="Shahajjo Qwen Engine")

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
        inv_vocab[k] = k

    model = ShahajjoTransformer(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Shahajjo Qwen Engine loaded successfully!")
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
    text = re.sub(r'^shahajjo\s*:\s*', '', text, flags=re.IGNORECASE)
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
            logits = outputs[0, -1, :] / 0.7
            probs = F.softmax(logits, dim=-1)
            predicted_id = torch.multinomial(probs, num_samples=1).item()
            
            generated_ids = [predicted_id]
            current_tensor = torch.cat([input_tensor, torch.tensor([[predicted_id]], dtype=torch.long)], dim=1)
            
            for _ in range(25):
                out = model(current_tensor)
                next_logits = out[0, -1, :] / 0.7
                next_probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(next_probs, num_samples=1).item()
                if next_id == 3:  # EOS Token
                    break
                generated_ids.append(next_id)
                current_tensor = torch.cat([current_tensor, torch.tensor([[next_id]], dtype=torch.long)], dim=1)

            generated_words = []
            for idx in generated_ids:
                if idx > 3:
                    word = inv_vocab.get(idx, inv_vocab.get(str(idx), ""))
                    if word:
                        generated_words.append(word)

            filtered_words = []
            for w in generated_words:
                if w and w not in filtered_words:
                    filtered_words.append(w)

            res = clean_text(" ".join(filtered_words))
            if res.lower().startswith("shahajjo"):
                res = res[8:].strip()

            if len(res) > 2 and res.lower() != "why":
                return res
    except Exception as e:
        print(f"Generation Error: {e}")
    return ""

@app.get("/")
def root():
    return {"status": "online", "model": "Shahajjo Qwen-Style Custom LLM Engine"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = (req.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message is required")

    response_text = generate_from_llm(user_msg)

    if not response_text or len(response_text) < 2 or response_text.lower() == "why":
        msg_lower = user_msg.lower()
        if "hello" in msg_lower or "hi" in msg_lower:
            response_text = "Hello! I am Shahajjo AI, built for Bangladesh. How can I help you?"
        elif "bangladesh" in msg_lower:
            response_text = "Bangladesh ekta shundor shadhin desh jar rajdhani Dhaka."
        elif any(b in msg_lower for b in ["ki", "keno", "kmn", "kothay", "tmi"]):
            response_text = "Ami apnar kotha bujhchi. Bolun, kibhabe sahajjo korbo?"
        else:
            response_text = f"Apni jigeshechen: '{user_msg}'. Shahajjo AI ekhon Qwen style instruction mode-e response dicche!"

    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
