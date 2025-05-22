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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

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
        lstm_model = LSTMClassifier(input_dim=768, hidden_dim=128, output_dim=1).to(device)
        lstm_model.load_state_dict(
            torch.load(os.path.join(model_dir, f"lstm_{aug_lower}_fold1.pth"), map_location=device))
        lstm_model.eval()
        lstm_models[aug] = lstm_model
        logger.info(f"Loaded LSTM_{aug} model.")
    except Exception as e:
        logger.error(f"Failed to load LSTM_{aug} model: {str(e)}")
        lstm_models[aug] = None
logger.info("Models loaded successfully.")


def predict_all_models(bert_features):
    if bert_features is None:
        logger.warning("Features are None, prediction skipped.")
        return {}

    bert_tensor = torch.FloatTensor(bert_features).unsqueeze(1).to(device)
    bert_array = bert_features
    preds = {}

    for aug in augmentations:
        try:
            if rf_models[aug]:
                preds[f"RandomForest_{aug}"] = rf_models[aug].predict_proba(bert_array)[0][1]
                logger.debug(f"RandomForest_{aug} prediction: {preds[f'RandomForest_{aug}']}")
            else:
                preds[f"RandomForest_{aug}"] = 0.0
        except Exception as e:
            logger.error(f"RandomForest_{aug} prediction failed: {str(e)}")
            preds[f"RandomForest_{aug}"] = 0.0

        try:
            if xgb_models[aug]:
                preds[f"XGBoost_{aug}"] = xgb_models[aug].predict_proba(bert_array)[0][1]
                logger.debug(f"XGBoost_{aug} prediction: {preds[f'XGBoost_{aug}']}")
            else:
                preds[f"XGBoost_{aug}"] = 0.0
        except Exception as e:
            logger.error(f"XGBoost_{aug} prediction failed: {str(e)}")
            preds[f"XGBoost_{aug}"] = 0.0

        try:
            if lstm_models[aug]:
                with torch.no_grad():
                    y_prob = lstm_models[aug](bert_tensor).squeeze().cpu().numpy()
                    preds[f"LSTM_{aug}"] = y_prob[0] if y_prob.ndim > 0 else y_prob.item()
                    logger.debug(f"LSTM_{aug} prediction: {preds[f'LSTM_{aug}']}")
            else:
                preds[f"LSTM_{aug}"] = 0.0
        except Exception as e:
            logger.error(f"LSTM_{aug} prediction failed: {str(e)}")
            preds[f"LSTM_{aug}"] = 0.0

    return preds


@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    email = data.get('email', '')
    if not email:
        logger.warning("No email provided in request.")
        return jsonify({"error": "No email provided"}), 400

    logger.info(f"Classifying email: {email[:50]}...")
    bert_features, _ = preprocess_single_email(email)
    if bert_features is None:
        logger.error("Email preprocessing failed.")
        return jsonify({"error": "Email preprocessing failed"}), 400

    preds = predict_all_models(bert_features)
    if not preds:
        logger.error("Prediction failed.")
        return jsonify({"error": "Prediction failed"}), 400

    avg_prob = np.mean(list(preds.values()))
    prediction = "Phishing" if avg_prob > 0.5 else "Safe"
    logger.info(f"Consensus prediction: {prediction}, avg_prob: {avg_prob:.2f}")

    accuracies = {k: f"{v * 100:.0f}%" for k, v in preds.items()}

    return jsonify({
        "prediction": prediction,
        "accuracies": accuracies
    })


if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)