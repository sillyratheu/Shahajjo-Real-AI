import os
import json
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

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

def init_model():
    global model, vocab, inv_vocab
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
        vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        inv_vocab = {v: k for k, v in vocab.items()}

        model = ShahajjoTransformer(vocab_size=len(vocab))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        pass

init_model()

def generate_ai_reply(prompt, lang="bn", max_length=50):
    if not model or not vocab:
        if lang == "bn":
            return "আমি সাহায্য এআই। বলুন, কীভাবে সাহায্য করতে পারি?"
        elif lang == "banglish":
            return "Ami Shahajjo AI. Bolun, kibhabe sahajjo korte pari?"
        else:
            return "I am Shahajjo AI. How can I assist you today?"

    tokens = prompt.lower().split()
    input_ids = [vocab.get(t, vocab.get("<unk>", 0)) for t in tokens if t in vocab]
    
    if not input_ids:
        input_ids = [vocab.get("<unk>", 0)]
        
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_tensor)
            next_token_id = torch.argmax(outputs[0, -1, :]).item()
            word = inv_vocab.get(next_token_id, "")
            
            if word == "<eos>" or word == "" or word == "<unk>":
                break
            generated_tokens.append(word)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]])], dim=1)
            
    if generated_tokens:
        return " ".join(generated_tokens)

    # Contextual dynamic reply when prompt tokens are out of vocab
    prompt_lower = prompt.lower()
    if lang == "bn":
        if "নাম" in prompt_lower or "name" in prompt_lower:
            return "আমার নাম সাহায্য এআই (Shahajjo AI)।"
        elif "কেমন" in prompt_lower or "how" in prompt_lower:
            return "আলহামদুলিল্লাহ, আমি ভালো আছি! আপনি কেমন আছেন?"
        return f"আমি সাহায্য এআই। আপনার '<b>{prompt}</b>' বিষয়টি বুঝতে পেরেছি। এ বিষয়ে আরো জানতে চান?"
    elif lang == "banglish":
        if "nam" in prompt_lower or "name" in prompt_lower:
            return "Amar nam Shahajjo AI."
        elif "kemon" in prompt_lower or "how" in prompt_lower:
            return "Ami valo achi! Apni kemon achen?"
        return f"Ami Shahajjo AI. Apnar '<b>{prompt}</b>' bisoyti bujhte perechi."
    else:
        if "name" in prompt_lower:
            return "My name is Shahajjo AI."
        elif "how" in prompt_lower:
            return "I am doing well! How are you?"
        return f"I am Shahajjo AI. Received your input: '<b>{prompt}</b>'."

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "online", "assistant": "Shahajjo AI"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "").strip()
    image_data = data.get("image", None)
    lang = data.get("lang", "bn")

    if not user_message and not image_data:
        return jsonify({"error": "Message is required"}), 400

    if image_data:
        if lang == "bn":
            reply = f"ছবিটি পাওয়া গিয়েছে। আপনার প্রশ্ন: '{user_message if user_message else 'বিশ্লেষণ করুন'}'।"
        elif lang == "banglish":
            reply = f"Chobi peyechi. Query: '{user_message if user_message else 'Analyze'}'."
        else:
            reply = f"Image received. Query: '{user_message if user_message else 'Analyze'}'."
    else:
        reply = generate_ai_reply(user_message, lang)

    return jsonify({"response": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
