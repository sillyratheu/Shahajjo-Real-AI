import os
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from huggingface_hub import hf_hub_download

# Extended Timeout for Hugging Face Downloads (32MB+ support)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_HTTP_TIMEOUT"] = "300"

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
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True)
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

# Word Tokenizer Pattern (Regex based for Bangla & English words)
def word_tokenize(text: str):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def load_shahajjo_model():
    global model, stoi, itos, PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX, load_error_msg
    try:
        torch.set_num_threads(1)
        print("📥 Downloading Model Weights & Vocab from Hugging Face...", flush=True)
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

        PAD_IDX = stoi.get("<pad>", 0)
        UNK_IDX = stoi.get("<unk>", 1)
        SOS_IDX = stoi.get("<sos>", 2)
        EOS_IDX = stoi.get("<eos>", 3)

        d_model, nhead, num_layers = 128, 4, 2
        try:
            config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                d_model = config.get("d_model", 128)
                nhead = config.get("nhead", 4)
                num_layers = config.get("num_layers", 2)
        except Exception:
            pass

        vocab_size = len(stoi)
        if vocab_size == 0:
            vocab_size = 1000

        model = ShahajjoTransformer(
            vocab_size=vocab_size, 
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            pad_idx=PAD_IDX
        )

        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        load_error_msg = ""
        print("🎉 Shahajjo Word-Level Engine Loaded Successfully!")
    except Exception as e:
        load_error_msg = str(e)
        print(f"❌ Model Loading Error: {e}")

load_shahajjo_model()

class ChatRequest(BaseModel):
    message: str
    lang: Optional[str] = "bn"

def encode(s: str):
    tokens = word_tokenize(s)
    return [stoi.get(t, UNK_IDX) for t in tokens]

def generate_from_llm(prompt: str) -> str:
    global model, stoi, itos

    if model is None or not stoi:
        load_shahajjo_model()

    if model is None or not stoi:
        return "নমস্কার! আমি সাহায্য এআই। বলুন আপনাকে কীভাবে সাহায্য করতে পারি?"

    try:
        formatted_prompt = f"User: {prompt} Shahajjo:"
        input_ids = [SOS_IDX] + encode(formatted_prompt)

        if not input_ids:
            return "নমস্কার! আমি সাহায্য এআই।"

        current_tensor = torch.tensor([input_ids], dtype=torch.long)
        generated_words = []

        temperature = 0.5
        max_length = 35

        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(current_tensor)
                next_token_logits = outputs[0, -1, :] / temperature

                # Anti-repetition penalty
                for prev_id in generated_words[-5:]:
                    if prev_id < next_token_logits.size(0):
                        next_token_logits[prev_id] -= 3.5

                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                if next_token_id == EOS_IDX:
                    break

                generated_words.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
                current_tensor = torch.cat([current_tensor, next_token_tensor], dim=1)

        clean_tokens = []
        for idx in generated_words:
            word_str = itos.get(idx, "")
            if word_str not in ["<pad>", "<unk>", "<sos>", "<eos>"]:
                clean_tokens.append(word_str)

        response_text = ""
        for token in clean_tokens:
            if token in [".", ",", "!", "?", "।"]:
                response_text = response_text.strip() + token + " "
            else:
                response_text += token + " "

        raw_output = response_text.strip()

        if "User:" in raw_output:
            raw_output = raw_output.split("User:")[0].strip()
        if "Shahajjo:" in raw_output:
            raw_output = raw_output.split("Shahajjo:")[-1].strip()

        if raw_output and len(raw_output) > 1:
            return raw_output

        return "নমস্কার! আমি সাহায্য এআই, আপনাকে কীভাবে সাহায্য করতে পারি?"

    except Exception as e:
        return "নমস্কার! আমি সাহায্য এআই। বলুন আপনাকে কীভাবে সাহায্য করতে পারি?"

@app.get("/")
def root():
    return {"status": "online", "model": "Shahajjo Word-Level Engine", "loaded": model is not None}

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
