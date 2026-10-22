from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from tokenizers import Tokenizer

app = FastAPI(title="Banglish Text Classifier API", version="1.0")

# 🟢 FIX FOR CORS ERROR: Allows requests from any web page / frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define PyTorch Model Architecture (VOCAB_SIZE matches tokenizer_3.json)
VOCAB_SIZE = 2324
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_CLASSES = 2

class BanglishLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(BanglishLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        pooled, _ = torch.max(lstm_out, dim=1)
        x = self.relu(self.fc1(pooled))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

# Initialize Model and Tokenizer
model = BanglishLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
model.eval()

try:
    tokenizer = Tokenizer.from_file("tokenizer_3.json")
except Exception as e:
    tokenizer = None

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Banglish Text Classifier API is running successfully!"}

@app.post("/predict")
def predict_text(request: TextRequest):
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer file (tokenizer_3.json) not found on server.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    encoded = tokenizer.encode(request.text)
    tokens = encoded.tokens
    token_ids = encoded.ids

    pad_id = 1
    max_len = 128
    padded_ids = token_ids[:max_len] + [pad_id] * max(0, max_len - len(token_ids))
    input_tensor = torch.tensor([padded_ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    labels = ["Negative / Non-Standard", "Positive / Standard"]
    pred_class = labels[probs.argmax()]
    confidence = float(probs.max() * 100)

    return {
        "text": request.text,
        "tokens": tokens,
        "token_ids": token_ids,
        "predicted_label": pred_class,
        "confidence_percentage": round(confidence, 2),
        "probabilities": {
            labels[0]: round(float(probs[0]), 4),
            labels[1]: round(float(probs[1]), 4)
        }
    }
