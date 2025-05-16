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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


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
    doc = nlp(text)
    return ' '.join([token.pos_ for token in doc])


def get_bert_embeddings(texts, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)


def preprocess_email_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df['text'] = df['text'].astype(str)

    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    df['pos_tags'] = df['text'].apply(get_pos_tags)
    df['url_count'] = df['text'].apply(count_urls)

    # TF-IDF features
    tfidf_text = TfidfVectorizer(max_features=500)
    tfidf_pos = TfidfVectorizer(max_features=100)
    text_features = tfidf_text.fit_transform(df['processed_text'])
    pos_features = tfidf_pos.fit_transform(df['pos_tags'])

    # BERT embeddings
    bert_features = get_bert_embeddings(df['processed_text'].tolist())

    # Combine features
    X = np.hstack([
        text_features.toarray(),
        pos_features.toarray(),
        bert_features,
        df['url_count'].values.reshape(-1, 1)
    ])
    y = df['label'].map({'Safe Email': 0, 'Phishing Email': 1}).values

    print(f"Dataset shape before preprocessing: {df.shape}")
    print(f"Label distribution before preprocessing: {df['label'].value_counts()}")
    print(f"Feature shape after preprocessing: {X.shape}")
    print(f"Label distribution after preprocessing: {pd.Series(y).value_counts()}")

    return X, y, tfidf_text, tfidf_pos


if __name__ == "__main__":
    dataset_path = "C:/Users/slade/Downloads/CS412/Week 4/CS412-Phishing-Detection/data/Phishing_Email.csv"
    X, y, tfidf_text, tfidf_pos = preprocess_email_data(dataset_path)