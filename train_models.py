import os
import torch
import pandas as pd
import numpy as np
import joblib
from src.preprocessing import preprocess_email_data, preprocess_single_email
from src.classifiers import train_random_forest, train_xgboost, train_lstm, classify_email
from imblearn.over_sampling import SMOTE

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
dataset_path = "data/Phishing_Email.csv"
model_dir = "models/"
os.makedirs(model_dir, exist_ok=True)


# Training function
def train_and_save_models():
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please upload Phishing_Email.csv to data/.")

    print("Dataset found, starting preprocessing...")
    X, y, tfidf_text, tfidf_pos = preprocess_email_data(dataset_path)
    print("Preprocessing completed. Shape:", X.shape)
    print("Label distribution:", pd.Series(y).value_counts())

    # Apply SMOTE if imbalanced
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    print("Resampled label distribution:", pd.Series(y).value_counts())

    # Train and save models
    print("Training Random Forest...")
    rf_model = train_random_forest(X, y)
    joblib.dump(rf_model, os.path.join(model_dir, "rf_model.joblib"))

    print("Training XGBoost...")
    xgb_model = train_xgboost(X, y)
    joblib.dump(xgb_model, os.path.join(model_dir, "xgb_model.joblib"))

    print("Training LSTM...")
    lstm_model = train_lstm(X, y).to(device)
    torch.save(lstm_model.state_dict(), os.path.join(model_dir, "lstm_model.pth"))

    # Save TF-IDF vectorizer
    joblib.dump(tfidf_text, os.path.join(model_dir, "tfidf_text.joblib"))
    print("All models and vectorizers saved to models/.")


# Run training
try:
    train_and_save_models()
except Exception as e:
    print(f"Training failed: {str(e)}")