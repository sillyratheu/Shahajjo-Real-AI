from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from typing import List
import os

# 1. Define Model Architecture
class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size=8000, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048):
        super(CustomTransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        out = self.embedding(src)
        out = self.transformer_encoder(out)
        out = self.fc(out)
        return out

# Initialize FastAPI App
app = FastAPI(title="Transformer Model API")

# Enable CORS (Frontend/App connection-er jonno khubi dorkari)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  // Shob source theke request allow korar jonno
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Weights from Hugging Face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomTransformerModel().to(device)

try:
    print("Downloading model weights from Hugging Face...")
    hf_token = os.getenv("HF_TOKEN")
    
    weights_path = hf_hub_download(
        repo_id="Vidclip/Shahajjo", 
        filename="model.safetensors",
        token=hf_token
    )
    
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model weights successfully loaded from Hugging Face!")
except Exception as e:
    print(f"Error loading model weights: {e}")

# Define Input Request Schema
class InputData(BaseModel):
    input_ids: List[int]

@app.get("/")
def home():
    return {"message": "Transformer API is running!"}

# Prediction Endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        if not data.input_ids:
            raise HTTPException(status_code=400, detail="input_ids list empty hote parbe na.")
            
        tensor_input = torch.tensor([data.input_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output = model(tensor_input)
            predictions = torch.argmax(output, dim=-1).cpu().numpy().tolist()[0]
            
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
