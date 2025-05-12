# src/classifiers.py
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import issparse


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out)


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train.toarray() if issparse(X_train) else X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train.toarray() if issparse(X_train) else X_train, y_train)
    return model


def train_lstm(X_train, y_train, epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = 1

    model = LSTMClassifier(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.FloatTensor(X_train.toarray() if issparse(X_train) else X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).to(device)

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            inputs = X_tensor[i:i + batch_size].unsqueeze(1)  # Add sequence dimension
            labels = y_tensor[i:i + batch_size]

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

    return model


def predict_lstm(model, X, device):
    model.eval()
    X_tensor = torch.FloatTensor(X.toarray() if issparse(X) else X).to(device)
    with torch.no_grad():
        outputs = model(X_tensor.unsqueeze(1)).squeeze()
    return (outputs.cpu().numpy() > 0.5).astype(int), outputs.cpu().numpy()