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
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4):
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
stoi = {}
itos = {}

try:
    torch.set_num_threads(1)
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        stoi = vocab_data.get("stoi", {})
        # JSON keys are always strings, map them back to integer keys for itos
        raw_itos = vocab_data.get("itos", {})
        itos = {int(k): v for k, v in raw_itos.items()}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    d_model = config.get("d_model", 256)
    nhead = config.get("nhead", 4)
    num_layers = config.get("num_layers", 4)
    vocab_size = config.get("vocab_size", len(stoi))

    model = ShahajjoTransformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Shahajjo Native LLM Engine loaded successfully!")
except Exception as e:
    print(f"Model Loading Error: {e}")

class ChatRequest(BaseModel):
    message: str
    lang: Optional[str] = "bn"

def encode(s: str):
    return [stoi.get(c, 0) for c in s]

def decode(indices):
    return "".join([itos.get(i, "") for i in indices])

def generate_from_llm(prompt: str) -> str:
    if model is None or not stoi or not itos:
        return "[Error: Model weights or Vocabulary not loaded]"
    try:
        # User Prompt-কে সঠিক চ্যাট ফরম্যাটে নেওয়া
        formatted_prompt = f"User: {prompt} Shahajjo:"
        input_ids = encode(formatted_prompt)

        if not input_ids:
            return "আমি সাহায্য এআই, আপনাকে সাহায্য করতে প্রস্তুত।"

        current_tensor = torch.tensor([input_ids], dtype=torch.long)
        generated_ids = []

        temperature = 0.7  # Creativity scale
        max_length = 60    # Maximum characters to generate

        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(current_tensor)
                next_token_logits = outputs[0, -1, :] / temperature

                # Anti-repetition penalty (থামানো বা লুপ ভাঙার জন্য)
                for prev_id in generated_ids[-10:]:  # Last 10 tokens check
                    next_token_logits[prev_id] -= 3.0

                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                # 'newline' বা ইউজার টোকেন প্রম্পট এলে থামানো
                char_gen = itos.get(next_token_id, "")
                if char_gen == "\n" or "User:" in decode(generated_ids):
                    break

                generated_ids.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
                current_tensor = torch.cat([current_tensor, next_token_tensor], dim=1)

        raw_output = decode(generated_ids).strip()
        
        # 'User:' শব্দ ফিল্টার আউট করা
        if "User:" in raw_output:
            raw_output = raw_output.split("User:")[0].strip()

        if raw_output:
            return raw_output

        return "আমি সাহায্য এআই, আপনাকে সাহায্য করতে প্রস্তুত।"

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
