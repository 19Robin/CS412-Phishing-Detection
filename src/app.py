from flask import Flask, request, jsonify
from flask_cors import CORS
from src.preprocessing import preprocess_single_email
from src.classifiers import classify_email
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os
import torch
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)
device = torch.device("cpu")  # Force CPU to avoid CUDA issues
print(f"Using device: {device}")

# Paths
model_dir = "C:/Users/slade/Downloads/CS412/Week 4/CS412-Phishing-Detection/models/"

# Load models
print("Loading models...")
rf_model = joblib.load(os.path.join(model_dir, "rf_model.joblib"))
xgb_model = joblib.load(os.path.join(model_dir, "xgb_model.joblib"))
tfidf_text = joblib.load(os.path.join(model_dir, "tfidf_text.joblib"))

# Define and load LSTM
from src.classifiers import LSTMClassifier
lstm_model = LSTMClassifier(input_size=868, hidden_size=128, num_layers=2).to(device)  # Adjust input_size
lstm_model.load_state_dict(torch.load(os.path.join(model_dir, "lstm_model.pth"), map_location=device))
lstm_model.eval()
print("Models loaded successfully.")

# Gmail API setup
def get_gmail_service():
    print("Setting up Gmail API service...")
    creds = None
    credentials_path = os.path.join(os.path.dirname(__file__), '..', 'credentials.json')
    token_path = os.path.join(os.path.dirname(__file__), '..', 'token.json')
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/gmail.readonly'])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, ['https://www.googleapis.com/auth/gmail.readonly'])
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def fetch_email_snippet():
    print("Fetching email snippet...")
    service = get_gmail_service()
    results = service.users().messages().list(userId='me').execute()
    messages = results.get('messages', [])
    if messages:
        msg = service.users().messages().get(userId='me', id=messages[0]['id']).execute()
        return msg['snippet']
    return "No email found"

@app.route('/classify', methods=['POST'])
def classify():
    if not request.json or 'email' not in request.json:
        email = fetch_email_snippet()
    else:
        email = request.json['email']
    if not email:
        return jsonify({"error": "No email provided"}), 400
    X = preprocess_single_email(email, tfidf_text, None)
    preds = classify_email(X, rf_model, xgb_model, lstm_model, device)
    prediction = "Phishing" if max(preds.values()) > 0.5 else "Safe"
    return jsonify({"prediction": prediction, "accuracies": preds})

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)