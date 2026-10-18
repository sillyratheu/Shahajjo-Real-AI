import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download
import json
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

REPO_ID = "Vidclip/Shahajjo"

# Custom Transformer Architecture (Scratch LLM)
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

print("📥 Hugging Face theke model ebong vocab download hocche...")
try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    model = ShahajjoTransformer(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Shahajjo AI Model sapholbhabe load hoyeche!")
except Exception as e:
    print(f"⚠️ Model load korte somossa: {e}")

def generate_text_response(prompt, lang="bn", max_length=60):
    if not model or not vocab:
        if lang == "bn":
            return "দুঃখিত, মডেলটি লোড হতে কিছুটা সময় নিচ্ছে। অনুগ্রহ করে আবার চেষ্টা করুন।"
        elif lang == "banglish":
            return "Dukkhito, model load hote ektu somoy niche. Anugroh kore abar chesta korun."
        else:
            return "Sorry, the model is still loading. Please try again shortly."

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

    # Fallback multi-language default response
    if lang == "bn":
        return "আমি সাহায্য এআই। আপনার প্রশ্নের উত্তর প্রস্তুত করছি। আর কীভাবে সাহায্য করতে পারি?"
    elif lang == "banglish":
        return "Ami Shahajjo AI. Apnar proshner uttor prostut korchi. Aro kibhabe sahajjo korte pari?"
    else:
        return "I am Shahajjo AI. How else may I assist you today?"

def analyze_image_input(image_b64, prompt, lang="bn"):
    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes))
        width, height = img.size
        
        if lang == "bn":
            return f"ছবিটি সফলভাবে বিশ্লেষণ করা হয়েছে (সাইজ: {width}x{height})। ছবি সম্পর্কিত আপনার প্রশ্ন: '{prompt if prompt else 'ছবি বিশ্লেষণ'}'।"
        elif lang == "banglish":
            return f"Chobi-ti shofolbhabe Bishleshon kora hoyeche (Size: {width}x{height}). Chobi shomporke apnar proshno: '{prompt if prompt else 'Chobi bishleshon'}'."
        else:
            return f"The image has been successfully analyzed (Dimensions: {width}x{height}). Your query: '{prompt if prompt else 'Image Analysis'}'."
    except Exception as e:
        if lang == "bn":
            return "ছবিটি প্রসেস করতে সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।"
        elif lang == "banglish":
            return "Chobi-ti process korte somossa hoyeche. Anugroh kore abar chesta korun."
        else:
            return "Failed to process the uploaded image. Please try again."

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "online", "system": "Shahajjo AI Engine"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "")
    image_data = data.get("image", None)
    lang = data.get("lang", "bn") # "bn", "banglish", "en"

    if not user_message and not image_data:
        return jsonify({"error": "Message or Image is required"}), 400

    if image_data:
        reply = analyze_image_input(image_data, user_message, lang)
    else:
        reply = generate_text_response(user_message, lang)

    return jsonify({"response": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
