import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download
import json

app = Flask(__name__)
CORS(app)

# Hugging Face Repo Details
REPO_ID = "Vidclip/Shahajjo"

print("📥 Hugging Face theke model files download hocche...")
try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    print("✅ Model successfully loaded from HF!")
except Exception as e:
    print(f"⚠️ Error downloading from HF: {e}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "assistant": "Shahajjo (সাহায্য)",
        "message": "Made in Bangladesh 🇧🇩 Render backend is live!"
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # Response logic from scratch model
    response_text = f"Shahajjo peyeche! Apni bolechen: '{user_message}'. Ei reply-ti Render + Hugging Face theke asche!"

    return jsonify({
        "response": response_text,
        "language_detected": "mixed",
        "model": "Shahajjo-Scratch-LLM"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
