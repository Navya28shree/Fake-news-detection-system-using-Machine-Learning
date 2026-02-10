import os
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import joblib
import re
import nltk
from nltk.corpus import stopwords
import requests
from flask_cors import CORS

# Download stopwords once at startup
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return " ".join([w for w in tokens if w not in stop_words])

# PyTorch MLP definition
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.output(self.hidden(x))

# Load vectorizer and model
vectorizer = joblib.load("manual_tfidf_vectorizer.pkl")
input_dim = len(vectorizer.get_feature_names_out())

model = MLP(input_dim)
model.load_state_dict(torch.load("manual_fake_news_model.pth", map_location='cpu'))
model.eval()

# NewsAPI configuration
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "a6dcb59058044cd7953289380c286180")
TRUSTED_DOMAINS = ",".join([
    "bbc.com","www.bbc.com","bbc.co.uk",
    "cnn.com","reuters.com",
    "apnews.com","nytimes.com","theguardian.com",
    "macrumors.com","9to5mac.com","macobserver.com",
    "iphoneincanada.ca","macworld.com","yahoo.com"
])
NEWSAPI_URL = "https://newsapi.org/v2/everything"

def check_newsapi_real(text):
    params = {
        "q": text,
        "searchIn": "title,description,content",
        "domains": TRUSTED_DOMAINS,
        "apiKey": NEWSAPI_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
    }
    try:
        resp = requests.get(NEWSAPI_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            return False
        return data.get("totalResults", 0) > 0
    except requests.RequestException:
        return False

# Flask app setup
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Fake News Detection API running."

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload or "text" not in payload:
        return jsonify({"error": "Please provide 'text' in JSON"}), 400

    raw_text = payload["text"].strip()

    # 1. Try NewsAPI first
    if check_newsapi_real(raw_text):
        return jsonify({
            "prediction": 1,
            "label": "Real"
        })

    # 2. Otherwise, use ML model
    cleaned = clean_text(raw_text)
    features = vectorizer.transform([cleaned]).toarray()
    tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        score = model(tensor).item()

    label = 1 if score >= 0.5 else 0
    return jsonify({
        "prediction": label,
        "label": "Real" if label else "Fake"
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
