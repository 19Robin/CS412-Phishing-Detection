# src/preprocessing.py
import pandas as pd
import numpy as np
import nltk
import spacy
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os

nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')


def preprocess_email_data(file_path):
    """Preprocess Kaggle Phishing Email Dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    data = pd.read_csv(file_path)
    data['email_text'] = data['email_text'].fillna('')

    def extract_features(text):
        tokens = word_tokenize(text.lower())
        doc = nlp(text)
        pos_tags = [token.pos_ for token in doc]
        urls = [ent.text for ent in doc.ents if 'http' in ent.text.lower()]
        return ' '.join(tokens), len(urls), ' '.join(pos_tags)

    data['tokens'], data['url_count'], data['pos_tags'] = zip(*data['email_text'].apply(extract_features))

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_features = tfidf.fit_transform(data['tokens']).toarray()

    features = np.hstack([tfidf_features, data[['url_count']].values])

    return features, data['label'].values, tfidf