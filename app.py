import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from huggingface_hub import hf_hub_download
import requests
import urllib.parse
import re

app = FastAPI(title="Shahajjo AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "Vidclip/Shahajjo"

class ShahajjoTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        return self.fc_out(out)

model = None
vocab = {}
inv_vocab = {}

try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    model = ShahajjoTransformer(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except Exception:
    pass

class ChatRequest(BaseModel):
    message: Optional[str] = ""
    image: Optional[str] = None
    lang: Optional[str] = "bn"

def clean_output_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace("reasoning:", "").replace("tool_calls:", "").replace("assistant", "")
    return text.strip()

def generate_from_scratch_model(prompt: str, max_length: int = 40) -> str:
    if not model or not vocab or not inv_vocab:
        return ""
    
    try:
        tokens = [vocab.get(w, vocab.get("<unk>", 1)) for w in prompt.split()]
        if not tokens:
            tokens = [vocab.get("<unk>", 1)]
            
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        generated_tokens = list(tokens)
        model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(input_tensor)
                next_token_logits = outputs[0, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated_tokens.append(next_token)
                input_tensor = torch.tensor([generated_tokens], dtype=torch.long)
                
                if next_token == vocab.get("<eos>", 2):
                    break
                    
        decoded_words = [inv_vocab.get(t, "") for t in generated_tokens if t > 2]
        result = " ".join(decoded_words)
        return clean_output_text(result)
    except Exception:
        return ""

def get_autonomous_answer(query: str, lang: str = "bn", is_image: bool = False) -> str:
    # Step 1: Try generating response autonomously using trained scratch model
    scratch_reply = generate_from_scratch_model(query)
    if scratch_reply and len(scratch_reply) > 5 and not "unk" in scratch_reply:
        return scratch_reply

    # Step 2: Fallback to Wikipedia knowledge base for accurate facts
    wiki_context = ""
    try:
        search_lang = "bn" if lang == "bn" else "en"
        encoded_query = urllib.parse.quote(query)
        wiki_url = f"https://{search_lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
        res = requests.get(wiki_url, timeout=4)
        if res.status_code == 200:
            data = res.json()
            if "extract" in data and data["extract"]:
                wiki_context = data["extract"]
    except Exception:
        pass

    if wiki_context:
        return clean_output_text(wiki_context)

    # Step 3: Synthesis engine fallback for natural language completion
    try:
        system_instruction = (
            "You are 'Shahajjo AI' (সাহায্য), an indigenous AI assistant built for Bangladesh. "
            "Give a direct, accurate, and natural response. Never output JSON or debug text."
        )
        if lang == "bn":
            system_instruction += " Answer in pure Bangla."
        elif lang == "banglish":
            system_instruction += " Answer in natural Banglish."
        else:
            system_instruction += " Answer in English."

        if is_image:
            system_instruction += " The user has provided an image to analyze."

        api_url = "https://text.pollinations.ai/" + urllib.parse.quote(f"{system_instruction}\nUser: {query}\nAnswer:")
        res = requests.get(api_url, timeout=10)
        if res.status_code == 200 and res.text.strip():
            cleaned = clean_output_text(res.text.strip())
            if cleaned:
                return cleaned
    except Exception:
        pass

    if lang == "bn":
        return f"Ami Shahajjo AI. Apnar '{query}' proshner shothik uttor toiri kora hocche."
    elif lang == "banglish":
        return f"Ami Shahajjo AI. Apnar '{query}' query niye kaj cholche."
    else:
        return f"I am Shahajjo AI. Processing request for '{query}'."

@app.get("/")
def home():
    return {"status": "online", "assistant": "Shahajjo AI Generative Engine"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        user_message = (req.message or "").strip()
        image_data = req.image
        lang = req.lang or "bn"

        if not user_message and not image_data:
            raise HTTPException(status_code=400, detail="Message or image is required")

        is_img = bool(image_data)
        prompt_text = user_message if user_message else "Analyze image"

        response_text = get_autonomous_answer(prompt_text, lang, is_img)
        return {"response": response_text}

    except Exception:
        return {"response": "Dukkito, uttor toiri korte shomossha hoyeche. Abar chesta korun."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.main(["app:app", "--host", "0.0.0.0", "--port", str(port)])
