import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from transformers import BertTokenizer, BertModel
import torch
import joblib

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


def get_bert_embeddings(texts, batch_size=16):
    """Generate BERT embeddings for a list of texts."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token embedding (first token) as the sentence representation
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


def preprocess_email_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"Dataset shape before preprocessing: {df.shape}")
    print(f"Label distribution before preprocessing: {df['Email Type'].value_counts()}")

    # Map labels: Safe Email -> 0, Phishing Email -> 1
    df['Label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    df = df.dropna(subset=['Label'])  # Drop rows with invalid labels
    X_raw = df['Email Text'].fillna('').astype(str).tolist()
    y = df['Label'].astype(int).values

    # Clean text
    X_cleaned = [clean_text(text) for text in X_raw]

    # Generate BERT embeddings
    X = get_bert_embeddings(X_cleaned)

    # TF-IDF vectorizer for compatibility with RF/XGB models
    tfidf_vectorizer = TfidfVectorizer(max_features=768)  # Match BERT dim for consistency
    X_tfidf = tfidf_vectorizer.fit_transform(X_cleaned).toarray()

    # Save the TF-IDF vectorizer for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf_vectorizer, 'models/tfidf_text.joblib')

    print(f"BERT feature shape after preprocessing: {X.shape}")
    print(f"TF-IDF feature shape after preprocessing: {X_tfidf.shape}")
    print(f"Label distribution after preprocessing: {pd.Series(y).value_counts()}")
    return X, y, X_tfidf, tfidf_vectorizer


def preprocess_single_email(email_text, tfidf_vectorizer=None, bert_model=None, tokenizer=None):
    """Preprocess a single email for classification."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Clean the email text
    cleaned_text = clean_text(email_text)

    # BERT embeddings
    if bert_model is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    bert_model.eval()
    inputs = tokenizer(cleaned_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    bert_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # TF-IDF features
    if tfidf_vectorizer is None:
        tfidf_vectorizer = joblib.load('models/tfidf_text.joblib')
    tfidf_features = tfidf_vectorizer.transform([cleaned_text]).toarray()

    return bert_embeddings, tfidf_features


if __name__ == "__main__":
    # Test preprocessing
    file_path = "/content/CS412-Phishing-Detection/data/Phishing_Emails.csv"
    X, y, X_tfidf, tfidf_vectorizer = preprocess_email_data(file_path)

    # Test single email preprocessing
    sample_email = "This is a test email with a link http://example.com"
    bert_emb, tfidf_emb = preprocess_single_email(sample_email)
    print(f"Single email BERT embedding shape: {bert_emb.shape}")
    print(f"Single email TF-IDF embedding shape: {tfidf_emb.shape}")