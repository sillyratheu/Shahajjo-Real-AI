from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# আপনার Hugging Face মডেলের নাম
MODEL_NAME = "Vidclip/Shahajjo"

print("Model load hocche, ektu shomoy lagte pare...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 250MB model-er jonno CPU-te float32 ba torch_dtype auto use kora jabe
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Model successfully loaded!")

@app.get("/")
def home():
    return {"status": "Shahajjo LLM API is live 24/7 on Render!"}

@app.post("/generate")
def generate_text(data: dict):
    prompt = data.get("prompt", "")
    if not prompt:
        return {"error": "Prompt dewa hoyni!"}
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"response": response_text}
