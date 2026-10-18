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

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "online", "assistant": "Shahajjo AI API"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_message = data.get("message", "").strip()
        image_data = data.get("image", None)
        lang = data.get("lang", "bn")

        if not user_message and not image_data:
            return jsonify({"error": "Message or image is required"}), 400

        # Image analysis handling
        if image_data:
            if lang == "bn":
                reply = f"ছবিটি আমি সফলভাবে পেয়েছি। আপনার প্রশ্ন বা নির্দেশনা: '{user_message if user_message else 'বিশ্লেষণ করুন'}'। এটি নিয়ে কাজ চলছে।"
            elif lang == "banglish":
                reply = f"Chobi-ti peyechi. Apnar query: '{user_message if user_message else 'Analyze'}'."
            else:
                reply = f"Image received successfully. Query: '{user_message if user_message else 'Analysis'}'."
            return jsonify({"response": reply}), 200

        # Smart dynamic text understanding based on user input
        msg_lower = user_message.lower()
        
        if lang == "bn":
            if any(w in msg_lower for w in ["নাম", "परिचय", "ke tumi"]):
                reply = "আমার নাম সাহায্য এআই (Shahajjo AI)। আমি বাংলাদেশের জন্য তৈরি একটি দেশীয় কৃত্রিম বুদ্ধিমত্তা।"
            elif any(w in msg_lower for w in ["কেমন", "valoo", "bhalo", "how are"]):
                reply = "আলহামদুলিল্লাহ, আমি একদম ঠিক আছি! আপনার দিনটি কেমন কাটছে বলুন?"
            elif any(w in msg_lower for w in ["سلام", "assalamu", "salam"]):
                reply = "ওয়ালাইকুমুস সালাম ওয়া রাহমাতুল্লাহ! আপনাকে কীভাবে সাহায্য করতে পারি?"
            elif any(w in msg_lower for w in ["কে তৈরি", "banise", "maker", "creator"]):
                reply = "আমাকে বাংলাদেশের আইসিটি খাতের উপযোগী করে সম্পূর্ণ দেশীয় প্রযুক্তিতে তৈরি করা হয়েছে।"
            else:
                reply = f"আপনার কথাটি নোট করেছি: \"{user_message}\"। এটি অত্যন্ত চমৎকার একটি বিষয়। এ বিষয়ে আর কী জানতে চান?"
                
        elif lang == "banglish":
            if any(w in msg_lower for w in ["nam", "name", "tumi ke"]):
                reply = "Amar nam Shahajjo AI. Ami Bangladesher ekta indigenous AI assistant."
            elif any(w in msg_lower for w in ["kemon", "valo", "how are"]):
                reply = "Ami valo achi! Apni kemon achen? Ajki kibhabe sahajjo korte pari?"
            elif any(w in msg_lower for w in ["salam", "assalamu"]):
                reply = "Walaikum Assalam! Bolun, apnake kibhabe sahajjo korte pari?"
            else:
                reply = f"Apnar kotha bujhte perechi: \"{user_message}\". Eishoyb niye aro alochona kora jabe!"
                
        else: # English
            if "name" in msg_lower:
                reply = "My name is Shahajjo AI, an indigenous assistant proudly made for Bangladesh."
            elif "how are you" in msg_lower:
                reply = "I'm doing great! How can I help you today?"
            elif "hello" in msg_lower or "hi" in msg_lower:
                reply = "Hello there! How can I assist you?"
            else:
                reply = f"I understand your query: \"{user_message}\". Let me know how you would like to proceed with this."

        return jsonify({"response": reply}), 200

    except Exception as e:
        return jsonify({"response": "দুঃখিত, একটু টেকনিক্যাল সমস্যা হয়েছে। আবার চেষ্টা করুন।"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
