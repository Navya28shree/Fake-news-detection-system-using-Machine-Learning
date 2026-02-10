import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import joblib

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------------------------
# 1. Text Preprocessing
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return " ".join([word for word in tokens if word not in stop_words])

# ---------------------------
# 2. Load and Prepare Dataset
# ---------------------------
def load_data():
    true_df = pd.read_csv('True.csv')
    fake_df = pd.read_csv('Fake.csv')

    true_df.columns = true_df.columns.str.strip()
    fake_df.columns = fake_df.columns.str.strip()

    true_df['label'] = 1
    fake_df['label'] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df['cleaned'] = (df['title'].astype(str) + " " + df['text'].astype(str)).apply(clean_text)

    print("Label distribution:")
    print(df['label'].value_counts())

    return df['cleaned'], df['label']

# ---------------------------
# 3. Neural Network Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)

# ---------------------------
# 4. Main Training Function
# ---------------------------
def train_model():
    print("Loading and preprocessing data...")
    texts, labels = load_data()

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts).toarray()
    y = labels.values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    input_dim = X_train.shape[1]
    model = MLP(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    epoch = 0
    best_loss = float('inf')
    patience = 30
    patience_counter = 0

    while True:
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        epoch += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        # Early stopping condition
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if loss.item() < 0.001:
            print("Loss threshold reached, stopping training.")
            break

    # Evaluation on test data
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
    predicted_classes = (predictions >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Test Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), 'manual_fake_news_model.pth')
    joblib.dump(vectorizer, 'manual_tfidf_vectorizer.pkl')
    print("✅ Training complete. Model and vectorizer saved.")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    train_model()
