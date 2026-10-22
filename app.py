import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Shahajjo AI Chat API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Hugging Face Causal LM and Tokenizer from local directory files
MODEL_PATH = "." 

tokenizer = None
model = None

try:
    if os.path.exists("config.json") and os.path.exists("tokenizer.json"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        model.eval()
        print("Shahajjo AI LLM Model loaded successfully!")
    else:
        print("Model configuration or tokenizer files missing.")
except Exception as e:
    print(f"Error loading LLM model: {e}")

class TextRequest(BaseModel):
    prompt: str
    max_length: int = 100

def generate_ai_response(prompt: str, max_length: int = 100):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=500, 
            detail="LLM model files (model.safetensors, config.json, tokenizer.json) not found on server."
        )
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Optionally remove the prompt from the response if the model repeats it
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()

        return response_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/")
def home():
    return {
        "message": "Shahajjo AI LLM API is running successfully!",
        "model_loaded": model is not None
    }

# GET Method (Pollinations er moto direct link theke use korar jonno)
@app.get("/predict")
def predict_get(text: str = Query(..., description="Prompt for Shahajjo AI"), max_length: int = 100):
    answer = generate_ai_response(text, max_length)
    return {
        "prompt": text,
        "response": answer
    }

# POST Method
@app.post("/predict")
def predict_post(request: TextRequest):
    answer = generate_ai_response(request.prompt, request.max_length)
    return {
        "prompt": request.prompt,
        "response": answer
    }
