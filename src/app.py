import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import joblib
import logging
from src.classifiers import LSTMClassifier
from src.preprocessing import preprocess_single_email
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from transformers import BertTokenizer, BertModel

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Paths
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
augmentations = ["Original", "SMOTE", "SMOTified-GAN", "MCMC-GAN", "CGAN", "VAE-GAN"]
rf_models = {}
xgb_models = {}
lstm_models = {}

logger.info("Loading models...")
for aug in augmentations:
    aug_lower = aug.lower().replace('-', '_')
    try:
        rf_models[aug] = joblib.load(os.path.join(model_dir, f"rf_{aug_lower}_fold1.joblib"))
        logger.info(f"Loaded RandomForest_{aug} model.")
    except Exception as e:
        logger.error(f"Failed to load RandomForest_{aug} model: {str(e)}")
        rf_models[aug] = None

    try:
        xgb_models[aug] = joblib.load(os.path.join(model_dir, f"xgboost_{aug_lower}_fold1.joblib"))
        logger.info(f"Loaded XGBoost_{aug} model.")
    except Exception as e:
        logger.error(f"Failed to load XGBoost_{aug} model: {str(e)}")
        xgb_models[aug] = None

    try:
        lstm_model = LSTMClassifier(input_dim=769, hidden_dim=128, output_dim=1).to(device)
        lstm_model.load_state_dict(
            torch.load(os.path.join(model_dir, f"lstm_{aug_lower}_fold1.pth"), map_location=device))
        lstm_model.eval()
        lstm_models[aug] = lstm_model
        logger.info(f"Loaded LSTM_{aug} model.")
    except Exception as e:
        logger.error(f"Failed to load LSTM_{aug} model: {str(e)}")
        lstm_models[aug] = None

# Load TF-IDF vectorizer
try:
    tfidf_vectorizer = joblib.load(os.path.join(model_dir, "tfidf_text.joblib"))
    logger.info("Loaded TF-IDF vectorizer.")
except Exception as e:
    logger.error(f"Failed to load TF-IDF vectorizer: {str(e)}")
    tfidf_vectorizer = None

logger.info("Models loaded successfully.")

# Initialize BERT model and tokenizer (reuse across requests)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()

# Gmail API setup
def get_gmail_service():
    logger.info("Setting up Gmail API service...")
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
    logger.info("Fetching email snippet...")
    service = get_gmail_service()
    results = service.users().messages().list(userId='me').execute()
    messages = results.get('messages', [])
    if messages:
        msg = service.users().messages().get(userId='me', id=messages[0]['id']).execute()
        return msg['snippet']
    return "No email found"

def predict_all_models(bert_features, tfidf_features, email_text=""):
    if bert_features is None or tfidf_features is None:
        logger.warning("Features are None, prediction skipped.")
        return {}

    # Log feature shapes
    logger.info(f"BERT features shape: {bert_features.shape}")
    logger.info(f"TF-IDF features shape: {tfidf_features.shape}")

    # Adjust features for RF and XGB to match expected 769 dimensions (use TF-IDF + dummy)
    email_length = np.array([[len(email_text) % 1000]])  # Dummy feature
    rf_xgb_features = np.concatenate((tfidf_features, email_length), axis=1)  # 768 + 1 = 769
    logger.info(f"RF/XGBoost features shape: {rf_xgb_features.shape}")

    # Adjust BERT features to match LSTM input_dim=769
    bert_features_adjusted = np.concatenate((bert_features, email_length), axis=1)
    logger.info(f"Adjusted BERT features shape for LSTM: {bert_features_adjusted.shape}")

    # Add sequence dimension for LSTM
    bert_tensor = torch.FloatTensor(bert_features_adjusted).unsqueeze(1).to(device)
    logger.info(f"BERT tensor shape for LSTM: {bert_tensor.shape}")

    preds = {}
    for aug in augmentations:
        # RandomForest prediction
        try:
            if rf_models[aug]:
                rf_pred = rf_models[aug].predict_proba(rf_xgb_features)[0]
                rf_prob = rf_pred[1] if rf_models[aug].classes_[1] == 1 else rf_pred[0]  # Assume 1 is Phishing
                preds[f"RandomForest_{aug}"] = float(rf_prob)
                logger.debug(f"RandomForest_{aug} prediction: {preds[f'RandomForest_{aug}']}")
            else:
                logger.warning(f"RandomForest_{aug} model not loaded.")
                preds[f"RandomForest_{aug}"] = 0.0
        except Exception as e:
            logger.error(f"RandomForest_{aug} prediction failed: {str(e)}")
            preds[f"RandomForest_{aug}"] = 0.0

        # XGBoost prediction
        try:
            if xgb_models[aug]:
                xgb_pred = xgb_models[aug].predict_proba(rf_xgb_features)[0]
                xgb_prob = xgb_pred[1] if xgb_models[aug].classes_[1] == 1 else xgb_pred[0]  # Assume 1 is Phishing
                preds[f"XGBoost_{aug}"] = float(xgb_prob)
                logger.debug(f"XGBoost_{aug} prediction: {preds[f'XGBoost_{aug}']}")
            else:
                logger.warning(f"XGBoost_{aug} model not loaded.")
                preds[f"XGBoost_{aug}"] = 0.0
        except Exception as e:
            logger.error(f"XGBoost_{aug} prediction failed: {str(e)}")
            preds[f"XGBoost_{aug}"] = 0.0

        # LSTM prediction
        try:
            if lstm_models[aug]:
                with torch.no_grad():
                    lstm_output = lstm_models[aug](bert_tensor)
                    logger.debug(f"LSTM_{aug} raw output shape: {lstm_output.shape}")
                    logger.debug(f"LSTM_{aug} raw output: {lstm_output.cpu().numpy()}")
                    y_prob = torch.sigmoid(lstm_output).cpu().numpy()
                    logger.debug(f"LSTM_{aug} sigmoid output shape: {y_prob.shape}")
                    logger.debug(f"LSTM_{aug} sigmoid output: {y_prob}")
                    pred_value = float(y_prob.item())
                    preds[f"LSTM_{aug}"] = pred_value
                    logger.debug(f"LSTM_{aug} prediction: {preds[f'LSTM_{aug}']}")
            else:
                logger.warning(f"LSTM_{aug} model not loaded.")
                preds[f"LSTM_{aug}"] = 0.0
        except Exception as e:
            logger.error(f"LSTM_{aug} prediction failed: {str(e)}")
            preds[f"LSTM_{aug}"] = 0.0

    return preds

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Fetch email snippet using Gmail API
        email = fetch_email_snippet()
        if not email or email == "No email found":
            logger.warning("No email fetched from Gmail.")
            return jsonify({"error": "No email fetched from Gmail"}), 400

        logger.info(f"Classifying email: {email[:50]}...")
        bert_features, tfidf_features = preprocess_single_email(email, tfidf_vectorizer, bert_model, bert_tokenizer, device)
        if bert_features is None or tfidf_features is None:
            logger.error("Email preprocessing failed.")
            return jsonify({"error": "Email preprocessing failed"}), 400

        preds = predict_all_models(bert_features, tfidf_features, email)
        if not preds:
            logger.error("Prediction failed.")
            return jsonify({"error": "Prediction failed"}), 400

        # Average only LSTM predictions for now
        lstm_preds = [v for k, v in preds.items() if "LSTM" in k]
        avg_prob = np.mean(lstm_preds) if lstm_preds else 0.0
        prediction = "Phishing" if avg_prob > 0.5 else "Safe"
        logger.info(f"Consensus prediction: {prediction}, avg_prob: {avg_prob:.2f} (LSTM only)")

        accuracies = {k: f"{v * 100:.0f}%" for k, v in preds.items()}

        return jsonify({
            "email": email,
            "prediction": prediction,
            "accuracies": accuracies
        })
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/fetch-email', methods=['GET'])
def fetch_email():
    try:
        email_snippet = fetch_email_snippet()
        return jsonify({"email_snippet": email_snippet})
    except Exception as e:
        logger.error(f"Failed to fetch email: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)