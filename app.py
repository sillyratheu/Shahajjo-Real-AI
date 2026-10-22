# -*- coding: utf-8 -*-
import os
import torch

# Render Free Tier (512MB RAM) er jonno memory optimize korar baire
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

app = FastAPI(title="Shahajjo AI API", version="1.0")

REPO_ID = "Vidclip/Shahajjo"

# Lightweight Model Config & Architecture for Render
class ShahajjoLightConfig(PretrainedConfig):
    model_type = "shahajjo_light"
    def __init__(self, vocab_size=12000, hidden_size=384, num_hidden_layers=3, num_attention_heads=6, intermediate_size=1536, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

class ShahajjoLightModel(PreTrainedModel):
    config_class = ShahajjoLightConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads, 
            dim_feedforward=config.intermediate_size,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        out = self.transformer_encoder(x)
        logits = self.fc(out)
        return {"logits": logits}

model = None
tokenizer = None

@app.on_event("startup")
def load_model_and_tokenizer():
    global model, tokenizer
    print("Hugging Face theke model ebong tokenizer download o load kora hocche...")
    
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=vocab_path)
    tokenizer.pad_token = "<pad>"
    
    config = ShahajjoLightConfig()
    model = ShahajjoLightModel(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("Model successfully loaded!")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/api")
def generate_text(request: PromptRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model ekhono load hoyni!")
    
    inputs = tokenizer(request.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs["logits"]
        predicted_ids = torch.argmax(logits, dim=-1)
        
    generated_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return {"prompt": request.prompt, "response": generated_text}

@app.get("/")
def home():
    return {"status": "Shahajjo API is active and running successfully!"}
