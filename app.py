import os
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download, login
import uvicorn

app = FastAPI(title="Shahajjo AI API", description="Indigenous Custom Bangla & English LLM API")

REPO_ID = "Vidclip/Shahajjo"
HF_TOKEN = os.environ.get("HF_TOKEN")

print("📥 Downloading model files from Hugging Face...")
try:
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        stoi = vocab_data["stoi"] if "stoi" in vocab_data else vocab_data
        itos = vocab_data["itos"] if "itos" in vocab_data else {str(v): k for k, v in stoi.items()}

    vocab_size = config.get("vocab_size", len(stoi))
    d_model = config.get("d_model", 512)
    nhead = config.get("nhead", 8)
    num_layers = config.get("num_layers", 6)
    PAD_IDX = stoi.get("<pad>", 0)

except Exception as e:
    print(f"❌ Error loading files from HF: {e}")

class ShahajjoMassiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        src_key_padding_mask = (x == PAD_IDX)
        out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(out)

print("⚙️ Initializing model architecture on CPU...")
model = ShahajjoMassiveTransformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers)

if 'model_path' in locals() and os.path.exists(model_path):
    # Load strictly on CPU to save RAM
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model weights successfully loaded on CPU!")

class QueryRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "Active", "message": "Shahajjo AI API is running successfully on Render!"}

@app.post("/generate")
def generate_text(request: QueryRequest):
    try:
        prompt = request.prompt.strip()
        tokens = prompt.split()
        input_ids = [stoi.get(t, stoi.get("<unk>", 1)) for t in tokens]
        
        if not input_ids:
            input_ids = [stoi.get("<sos>", 2)]
            
        generated_ids = list(input_ids)
        model.eval()
        
        with torch.no_grad():
            # Autoregressive generation loop (token-by-token prediction)
            for _ in range(25):  # Max 25 tokens generate korbe
                input_tensor = torch.tensor([generated_ids], dtype=torch.long)
                outputs = model(input_tensor)
                
                # Shesh token er logits nie next token predict korbe
                next_token_logits = outputs[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                
                # Jodi EOS token pay tahole theme jabe
                if next_token_id == stoi.get("<eos>", 3):
                    break
                    
                generated_ids.append(next_token_id)
            
        response_words = [itos.get(str(idx), "") for idx in generated_ids]
        response_text = " ".join([w for w in response_words if w not in ["<pad>", "<sos>", "<eos>"]])
        
        if not response_text.strip():
            response_text = "Namaskar! Ami Shahajjo AI, apnake sahajjo korte prostut achhi."
            
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
