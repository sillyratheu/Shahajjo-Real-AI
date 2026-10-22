from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from safetensors.torch import load_file
from typing import List

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

# Load Model Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomTransformerModel().to(device)

weights_path = "model.safetensors"
try:
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model weights successfully loaded!")
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
