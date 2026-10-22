import streamlit as st
import json
import torch
import torch.nn as nn
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import pandas as pd

# Define the PyTorch Model Architecture matching vocabulary size from tokenizer_2.json (2323 tokens)
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
        # Global max pooling
        pooled, _ = torch.max(lstm_out, dim=1)
        x = self.relu(self.fc1(pooled))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

@st.cache_resource
def load_tokenizer_and_model():
    """Load and initialize the tokenizer and model."""
    # Load custom BPE Tokenizer from tokenizer_2.json
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
    except Exception as e:
        st.error(f"Error loading tokenizer.json: {e}")
        tokenizer = None

    model = BanglishLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
    model.eval()
    return tokenizer, model

def tokenize_text(tokenizer, text):
    """Tokenize text using tokenizer_2.json configuration."""
    if tokenizer is None:
        return []
    encoded = tokenizer.encode(text)
    return encoded

# Streamlit Page Config
st.set_page_config(
    page_title="Banglish Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Configuration & Info")
st.sidebar.info(
    "This app tokenizes and classifies Banglish/Bengali text using the `tokenizer_2.json` BPE vocabulary configuration."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Tokenizer Stats")
st.sidebar.write(f"**Vocab Size:** {VOCAB_SIZE} tokens")
st.sidebar.write("**Max Sequence Length:** 128")
st.sidebar.write("**Padding Token:** `<pad>` (ID: 1)")
st.sidebar.write("**Unknown Token:** `<unk>` (ID: 0)")

# Main Layout
st.title("🇧🇩 Banglish Text Analysis & Classification")
st.markdown("Enter Banglish or Bengali text below to inspect tokenization and predicted sentiment/category probabilities.")

tokenizer, model = load_tokenizer_and_model()

user_input = st.text_area(
    "Input Text:",
    value="Amar bangla khub bhalo lage!",
    height=100,
    placeholder="Type here..."
)

col1, col2 = st.columns([1, 1])

if st.button("Analyze Text", type="primary"):
    if user_input.strip() and tokenizer:
        encoded = tokenize_text(tokenizer, user_input)
        tokens = encoded.tokens
        token_ids = encoded.ids
        
        with col1:
            st.subheader("🔍 Tokenization Breakdown")
            st.write(f"**Total Tokens Generated:** {len(tokens)}")
            
            # Display Tokens Table
            df_tokens = pd.DataFrame({
                "Token": tokens,
                "Token ID": token_ids
            })
            st.dataframe(df_tokens, use_container_width=True)

        with col2:
            st.subheader("📊 Classification Results")
            
            # Prepare tensor input (pad/truncate to 128)
            pad_id = 1
            max_len = 128
            padded_ids = token_ids[:max_len] + [pad_id] * max(0, max_len - len(token_ids))
            input_tensor = torch.tensor([padded_ids], dtype=torch.long)
            
            # Model Inference
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]
            
            labels = ["Negative / Non-Standard", "Positive / Standard"]
            
            # Display Prediction
            pred_class = labels[probs.argmax()]
            confidence = probs.max() * 100
            
            st.metric("Predicted Label", pred_class, f"{confidence:.2f}% confidence")
            
            # Confidence Chart
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(labels, probs, color=['#ff6b6b', '#51cf66'])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)
            
    elif not tokenizer:
        st.error("Tokenizer file (`tokenizer_2.json`) not found or failed to load. Please ensure it is present in the working directory.")
    else:
        st.warning("Please enter some text to analyze.")