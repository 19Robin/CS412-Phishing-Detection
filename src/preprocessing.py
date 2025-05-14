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
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    data = pd.read_csv(file_path)
    data = data.rename(columns={"Email Text": "email_text", "Email Type": "label"})
    print(f"Dataset shape before preprocessing: {data.shape}")  # Add print here
    print(f"Label distribution before preprocessing: {data['label'].value_counts()}")

    valid_labels = ["Phishing Email", "Safe Email"]
    data = data[data["label"].isin(valid_labels)]
    data["label"] = data["label"].map({"Phishing Email": 1, "Safe Email": 0})
    data["email_text"] = data["email_text"].fillna("")

    def extract_features(text):
        tokens = word_tokenize(text.lower())
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text.lower())
        doc = nlp(text[:1000000])
        pos_tags = [token.pos_ for token in doc]
        return " ".join(tokens), len(urls), " ".join(pos_tags)

    data[["tokens", "url_count", "pos_tags"]] = pd.DataFrame(
        data["email_text"].apply(extract_features).tolist(), index=data.index
    )

    tfidf_text = TfidfVectorizer(max_features=1000)
    tfidf_pos = TfidfVectorizer(max_features=200)
    text_features = tfidf_text.fit_transform(data["tokens"])
    pos_features = tfidf_pos.fit_transform(data["pos_tags"])
    features = hstack([text_features, pos_features, data[["url_count"]].values])

    print(f"Feature shape after preprocessing: {features.shape}")  # Add print here
    print(f"Label distribution after preprocessing: {pd.Series(data['label']).value_counts()}")

    return features, data["label"].values, tfidf_text, tfidf_pos


if __name__ == "__main__":
    # Example usage
    dataset_path = "../data/Phishing_Email.csv"
    X, y, tfidf_text, tfidf_pos = preprocess_email_data(dataset_path)
    print("Feature shape:", X.shape)
    print("Label distribution:", pd.Series(y).value_counts())