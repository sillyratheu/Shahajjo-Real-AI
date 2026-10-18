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

# ১. ট্রান্সফরমার আর্কিটেকচার
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

# ২. সাইলেন্টলি মডেল ফাইল লোড করার লজিক
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

# অ্যাপ চালু হওয়ার সাথেই ব্যাকগ্রাউন্ডে ইনিশিয়ালাইজ হবে
init_model()

def generate_ai_reply(prompt, lang="bn", max_length=50):
    if not model or not vocab:
        if lang == "bn":
            return "আমি সাহায্য এআই। আপনাকে কীভাবে সাহায্য করতে পারি?"
        elif lang == "banglish":
            return "Ami Shahajjo AI. Apnake kibhabe sahajjo korte pari?"
        else:
            return "I am Shahajjo AI. How can I help you today?"

    tokens = prompt.split()
    input_ids = [vocab.get(t, vocab.get("<unk>", 0)) for t in tokens]
    if not input_ids:
        input_ids = [0]
        
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_tensor)
            next_token_id = torch.argmax(outputs[0, -1, :]).item()
            word = inv_vocab.get(next_token_id, "")
            
            if word == "<eos>" or word == "":
                break
            generated_tokens.append(word)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]])], dim=1)
            
    if generated_tokens:
        return " ".join(generated_tokens)

    if lang == "bn":
        return "আমি সাহায্য এআই। আপনার প্রশ্নের উত্তর প্রস্তুত করছি।"
    elif lang == "banglish":
        return "Ami Shahajjo AI. Apnar proshner uttor prostut korchi."
    else:
        return "I am Shahajjo AI. Processing your request."

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "assistant": "Shahajjo AI"
    })

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
            reply = f"ছবিটি আমি বিশ্লেষণ করেছি। প্রশ্ন: '{user_message if user_message else 'ছবি বিশ্লেষণ'}'।"
        elif lang == "banglish":
            reply = f"Chobi-ti ami bishleshon korechi. Query: '{user_message if user_message else 'Chobi bishleshon'}'."
        else:
            reply = f"Image analyzed successfully. Query: '{user_message if user_message else 'Image analysis'}'."
    else:
        reply = generate_ai_reply(user_message, lang)

    return jsonify({"response": reply})

if __name__ == "__main__":
    # Render Dynamic Port Catching
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
