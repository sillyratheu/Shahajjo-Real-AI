import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

app = FastAPI(title="Shahajjo AI Chat API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "vidclip/Shahajjo"

def download_model_files():
    files = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]
    for file in files:
        if not os.path.exists(file):
            try:
                print(f"Downloading {file} from Hugging Face ({REPO_ID})...")
                hf_hub_download(repo_id=REPO_ID, filename=file, local_dir=".")
            except Exception as e:
                print(f"Could not download {file}: {e}")

download_model_files()

tokenizer = None
if os.path.exists("tokenizer.json"):
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
        print("Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Tokenizer load error: {e}")

class TextRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {
        "message": "Shahajjo AI Chat API is running successfully!",
        "tokenizer_loaded": tokenizer is not None,
        "model_file_exists": os.path.exists("model.safetensors")
    }

@app.get("/chat")
def chat_get(text: str = Query(..., description="Prompt for Shahajjo AI")):
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer not loaded.")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    encoded = tokenizer.encode(text)
    token_ids = encoded.ids
    decoded_response = tokenizer.decode(token_ids)

    # Frontend-er jate kono key mismatch na hoy, tai shob popular keys eksathe dewa holo
    return {
        "status": "success",
        "response": decoded_response,
        "reply": decoded_response,
        "message": decoded_response,
        "text": decoded_response
    }

@app.post("/chat")
def chat_post(request: TextRequest):
    return chat_get(request.prompt)
