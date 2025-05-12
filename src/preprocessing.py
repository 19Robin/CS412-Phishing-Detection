# src/preprocessing.py
import pandas as pd
import numpy as np
import nltk
import spacy
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import os

# Download NLTK resources
nltk.download('punkt')

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')


def preprocess_email_data(file_path):
    """
    Preprocess Kaggle Phishing Email Dataset.
    Returns features (TF-IDF, URL count, POS tags), labels, and vectorizers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    # Load and rename columns to match dataset
    data = pd.read_csv(file_path)
    data = data.rename(columns={"Email Text": "email_text", "Email Type": "label"})

    # Validate and map labels
    valid_labels = ["Phishing Email", "Safe Email"]
    data = data[data["label"].isin(valid_labels)]
    data["label"] = data["label"].map({"Phishing Email": 1, "Safe Email": 0})
    data["email_text"] = data["email_text"].fillna("")

    def extract_features(text):
        """Extract tokens, URL count, and POS tags from email text."""
        # Tokenize
        tokens = word_tokenize(text.lower())

        # Extract URLs using regex
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                          text.lower())

        # Extract POS tags
        doc = nlp(text[:1000000])  # Limit text length for SpaCy
        pos_tags = [token.pos_ for token in doc]

        return " ".join(tokens), len(urls), " ".join(pos_tags)

    # Apply feature extraction
    data[["tokens", "url_count", "pos_tags"]] = pd.DataFrame(
        data["email_text"].apply(extract_features).tolist(), index=data.index
    )

    # Vectorize tokens and POS tags
    tfidf_text = TfidfVectorizer(max_features=5000)
    tfidf_pos = TfidfVectorizer(max_features=1000)
    text_features = tfidf_text.fit_transform(data["tokens"])
    pos_features = tfidf_pos.fit_transform(data["pos_tags"])

    # Combine features (sparse matrix)
    features = hstack([text_features, pos_features, data[["url_count"]].values])

    return features, data["label"].values, tfidf_text, tfidf_pos


if __name__ == "__main__":
    # Example usage
    dataset_path = "../data/Phishing_Email.csv"
    X, y, tfidf_text, tfidf_pos = preprocess_email_data(dataset_path)
    print("Feature shape:", X.shape)
    print("Label distribution:", pd.Series(y).value_counts())