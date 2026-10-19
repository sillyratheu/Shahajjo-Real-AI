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
load_error_msg = ""

def load_shahajjo_model():
    global model, stoi, itos, load_error_msg
    try:
        torch.set_num_threads(1)
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
        vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            if "stoi" in vocab_data:
                stoi = vocab_data["stoi"]
                itos = {int(k): v for k, v in vocab_data["itos"].items()}
            else:
                # Fallback if raw vocab dictionary saved directly
                stoi = vocab_data
                itos = {int(v): k for k, v in vocab_data.items()}

        # Try downloading config if exists
        d_model, nhead, num_layers = 256, 4, 4
        try:
            config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                d_model = config.get("d_model", 256)
                nhead = config.get("nhead", 4)
                num_layers = config.get("num_layers", 4)
        except Exception:
            pass

        vocab_size = len(stoi)
        if vocab_size == 0:
            vocab_size = 1000  # Fallback safety

        model = ShahajjoTransformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        load_error_msg = ""
        print("🎉 Shahajjo Native LLM Engine loaded successfully!")
    except Exception as e:
        load_error_msg = str(e)
        print(f"❌ Model Loading Error: {e}")

# Initial load attempt
load_shahajjo_model()

class ChatRequest(BaseModel):
    message: str
    lang: Optional[str] = "bn"

def encode(s: str):
    return [stoi.get(c, 0) for c in s]

def decode(indices):
    return "".join([itos.get(i, "") for i in indices])

def generate_from_llm(prompt: str) -> str:
    global model, stoi, itos
    
    # Retry loading if model failed earlier
    if model is None or not stoi:
        load_shahajjo_model()

    if model is None or not stoi:
        return f"[Error loading model weights/vocab: {load_error_msg}]"

    try:
        formatted_prompt = f"User: {prompt} Shahajjo:"
        input_ids = encode(formatted_prompt)

        if not input_ids:
            return "আমি সাহায্য এআই, আপনাকে সাহায্য করতে প্রস্তুত।"

        current_tensor = torch.tensor([input_ids], dtype=torch.long)
        generated_ids = []

        temperature = 0.7
        max_length = 60

        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(current_tensor)
                next_token_logits = outputs[0, -1, :] / temperature

                # Anti-repetition penalty
                for prev_id in generated_ids[-8:]:
                    if prev_id < next_token_logits.size(0):
                        next_token_logits[prev_id] -= 3.0

                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                char_gen = itos.get(next_token_id, "")
                if char_gen == "\n" or "User:" in decode(generated_ids):
                    break

                generated_ids.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
                current_tensor = torch.cat([current_tensor, next_token_tensor], dim=1)

        raw_output = decode(generated_ids).strip()
        
        if "User:" in raw_output:
            raw_output = raw_output.split("User:")[0].strip()

        if raw_output:
            return raw_output

        return "আমি সাহায্য এআই, আপনাকে সাহায্য করতে প্রস্তুত।"

    except Exception as e:
        return f"[Model Inference Error: {str(e)}]"

@app.get("/")
def root():
    return {"status": "online", "model": "Shahajjo Native Neural Engine", "loaded": model is not None}

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
