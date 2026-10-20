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
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out = self.transformer(emb)
        return self.fc_out(out)

model = None
stoi = {}
itos = {}
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
load_error_msg = ""

def load_shahajjo_model():
    global model, stoi, itos, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX, load_error_msg
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
                stoi = vocab_data
                itos = {int(v): k for k, v in vocab_data.items()}

        # Special token index mapping
        PAD_IDX = stoi.get("<pad>", 0)
        UNK_IDX = stoi.get("<unk>", 1)
        SOS_IDX = stoi.get("<sos>", 2)
        EOS_IDX = stoi.get("<eos>", 3)

        # Default config matching training setup
        d_model, nhead, num_layers = 512, 8, 6
        try:
            config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                d_model = config.get("d_model", 512)
                nhead = config.get("nhead", 8)
                num_layers = config.get("num_layers", 6)
        except Exception:
            pass

        vocab_size = len(stoi)
        if vocab_size == 0:
            vocab_size = 100

        model = ShahajjoTransformer(
            vocab_size=vocab_size, 
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            pad_idx=PAD_IDX
        )
        
        # Load weights safely on CPU
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        load_error_msg = ""
        print("🎉 Shahajjo Native LLM Engine loaded successfully with clean architecture!")
    except Exception as e:
        load_error_msg = str(e)
        print(f"❌ Model Loading Error: {e}")

# Initial model load attempt
load_shahajjo_model()

class ChatRequest(BaseModel):
    message: str
    lang: Optional[str] = "bn"

def encode(s: str):
    return [stoi.get(c, UNK_IDX) for c in s]

def generate_from_llm(prompt: str) -> str:
    global model, stoi, itos

    if model is None or not stoi:
        load_shahajjo_model()

    if model is None or not stoi:
        return f"[Error loading model weights/vocab: {load_error_msg}]"

    try:
        formatted_prompt = f"User: {prompt} Shahajjo:"
        input_ids = [SOS_IDX] + encode(formatted_prompt)

        if not input_ids:
            return "নমস্কার! আমি সাহায্য এআই।"

        current_tensor = torch.tensor([input_ids], dtype=torch.long)
        generated_ids = []

        temperature = 0.6
        max_length = 70

        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(current_tensor)
                next_token_logits = outputs[0, -1, :] / temperature

                # Anti-repetition penalty
                for prev_id in generated_ids[-6:]:
                    if prev_id < next_token_logits.size(0):
                        next_token_logits[prev_id] -= 4.0

                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                if next_token_id == EOS_IDX:
                    break

                generated_ids.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
                current_tensor = torch.cat([current_tensor, next_token_tensor], dim=1)

        # Clean filtering to strip out special tokens from string output
        clean_chars = []
        for idx in generated_ids:
            char_gen = itos.get(idx, "")
            if char_gen not in ["<pad>", "<unk>", "<sos>", "<eos>"]:
                clean_chars.append(char_gen)

        raw_output = "".join(clean_chars).strip()

        # Clean unwanted extra prompt echoes
        if "User:" in raw_output:
            raw_output = raw_output.split("User:")[0].strip()
        if "Shahajjo:" in raw_output:
            raw_output = raw_output.split("Shahajjo:")[-1].strip()

        if raw_output and len(raw_output) > 1:
            return raw_output

        return "নমস্কার! আমি সাহায্য এআই, আপনাকে কীভাবে সাহায্য করতে পারি?"

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
