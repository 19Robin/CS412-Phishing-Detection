import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import re
from transformers import BertTokenizer, BertModel
import torch

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load SpaCy and BERT models globally
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to('cuda')

def count_urls(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return len(url_pattern.findall(text))

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

def get_pos_tags(text):
    doc = nlp(text, disable=["ner", "parser"])  # Disable unused components for speed
    return ' '.join([token.pos_ for token in doc])

def get_bert_embeddings(texts, max_length=128):
    bert_model.eval()
    embeddings = []
    for text in texts:
        inputs = bert_tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding=True).to('cuda')
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return np.array(embeddings)

def preprocess_email_data(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)
    # Use correct column names
    texts = df['Email Text'].astype(str).values  # Email text column
    labels = (df['Email Type'] == 'Phishing Email').astype(int).values  # 0 (Safe Email), 1 (Phishing Email)

    # Handle missing or invalid text entries
    texts = ["" if pd.isna(text) else text for text in texts]

    # Preprocess text
    processed_texts = [preprocess_text(text) for text in texts]
    pos_tags = [get_pos_tags(text) for text in texts]
    url_counts = [count_urls(text) for text in texts]

    # TF-IDF features
    tfidf_text = TfidfVectorizer(max_features=500)
    tfidf_pos = TfidfVectorizer(max_features=100)
    text_features = tfidf_text.fit_transform(processed_texts)
    pos_features = tfidf_pos.fit_transform(pos_tags)

    # BERT embeddings
    bert_features = get_bert_embeddings(texts)

    # Combine features
    X = np.hstack([
        text_features.toarray(),
        pos_features.toarray(),
        bert_features,
        np.array(url_counts).reshape(-1, 1)
    ])

    # Debug prints
    print(f"Dataset shape before preprocessing: {df.shape}")
    print(f"Label distribution before preprocessing: {df['Email Type'].value_counts()}")
    print(f"Feature shape after preprocessing: {X.shape}")
    print(f"Label distribution after preprocessing: {pd.Series(labels).value_counts()}")

    return X, labels, tfidf_text, tfidf_pos

if __name__ == "__main__":
    dataset_path = "C:/Users/slade/Downloads/CS412/Week 4/CS412-Phishing-Detection/data/Phishing_Email.csv"
    X, y, tfidf_text, tfidf_pos = preprocess_email_data(dataset_path)