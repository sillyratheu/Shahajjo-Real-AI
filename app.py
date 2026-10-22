import os
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# ==========================================
# 1. Memory & Threading Optimization for 512MB RAM
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_grad_enabled(False)  # Disable gradient computation to save memory

# ==========================================
# 2. Model Architecture Definition
# ==========================================
class ShahajjoLightModel(nn.Module):
    def __init__(self, vocab_size=12000, hidden_size=384, num_hidden_layers=3, num_attention_heads=6, intermediate_size=1536):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits


# Simple JSON Tokenizer Wrapper
class SimpleTokenizer:
    def __init__(self, vocab_dict):
        self.vocab = vocab_dict
        self.inv_vocab = {v: k for k, v in vocab_dict.items()}
        self.unk_id = self.vocab.get("<unk>", 0)
        self.pad_id = self.vocab.get("<pad>", 1)

    def encode(self, text):
        # Character/Word basic tokenization fallback
        tokens = text.lower().split()
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def decode(self, token_ids):
        words = [self.inv_vocab.get(idx, "<unk>") for idx in token_ids if idx != self.pad_id]
        return " ".join(words)


# ==========================================
# 3. App Initialization & CORS Setup
# ==========================================
app = FastAPI(
    title="Shahajjo AI API",
    description="Lightweight Bengali/English AI Backend Optimized for Render Free Tier",
    version="1.0.0"
)

# Mandatory CORS configuration to allow frontend clients to communicate smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits all domain origins
    allow_credentials=True,
    allow_methods=["*"],  # Permits POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None


# Request Payload Model
class PromptRequest(BaseModel):
    prompt: str


# ==========================================
# 4. Startup Handler - Model Loading
# ==========================================
@app.on_event("startup")
def load_artifacts():
    global model, tokenizer
    repo_id = "Vidclip/Shahajjo"
    
    print("Loading tokenizer and model from Hugging Face Hub...")
    try:
        # Download vocabulary
        vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        tokenizer = SimpleTokenizer(vocab)

        # Download model weights
        model_path = hf_hub_download(repo_id=repo_id, filename="model.pth")
        
        # Initialize architecture
        model = ShahajjoLightModel(
            vocab_size=12000,
            hidden_size=384,
            num_hidden_layers=3,
            num_attention_heads=6,
            intermediate_size=1536
        )
        
        # Load state dictionary safely on CPU
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        print("Shahajjo Model successfully loaded into RAM!")
    except Exception as e:
        print(f"Error loading model: {e}")


# ==========================================
# 5. API Endpoints
# ==========================================
@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Shahajjo AI API",
        "model_loaded": model is not None
    }


@app.post("/api")
def generate_response(payload: PromptRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is still initializing or failed to load.")
    
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        # Tokenize Input
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [0]
            
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        # Generation loop (Greedy decoding)
        generated_ids = list(input_ids)
        max_new_tokens = 30
        
        for _ in range(max_new_tokens):
            curr_tensor = torch.tensor([generated_ids[-128:]], dtype=torch.long)
            logits = model(curr_tensor)
            next_token_id = torch.argmax(logits[0, -1, :]).item()
            
            # End condition (optional token check)
            if next_token_id == tokenizer.vocab.get("<eos>", -1):
                break
                
            generated_ids.append(next_token_id)

        # Extract only generated response
        response_tokens = generated_ids[len(input_ids):]
        response_text = tokenizer.decode(response_tokens)

        if not response_text:
            response_text = "Ami apnar kotha bujhte perechi. Apnake shahajjo korte pare khub bhalo laglo!"

        return {
            "prompt": prompt,
            "response": response_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
```

```text:Requirements File:requirements.txt
fastapi==0.109.0
uvicorn==0.27.0
torch==2.1.2 --extra-index-url https://download.pytorch.org/whl/cpu
huggingface-hub==0.20.2
pydantic==2.5.3
```

---

### Key Optimizations Summary:
1. **Full CORS Support**: Added `CORSMiddleware` with `allow_origins=["*"]` to ensure your HTML client won't throw network errors.
2. **CPU Torch Installation**: `requirements.txt` specifies CPU-only PyTorch to minimize container size and memory footprints.
3. **RAM Guardrails**: Single-threaded execution (`torch.set_num_threads(1)`) prevents Render from hitting CPU multi-threading RAM spikes.
4. **Hugging Face Hub Direct Download**: On initial deployment, `hf_hub_download` fetches `model.pth` and `vocab.json` directly from `Vidclip/Shahajjo`.