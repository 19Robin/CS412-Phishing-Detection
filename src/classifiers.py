# src/classifiers.py
import torch
import torch.nn as nn
import scipy.sparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
from sklearn.model_selection import cross_val_score


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)  # Increase dropout
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        c0 = torch.zeros(1, batch_size, self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)  # Tighten constraints
    scores = cross_val_score(model, X_train.toarray() if issparse(X_train) else X_train, y_train, cv=5, scoring='f1')
    print(f"Random Forest CV F1: {scores.mean():.2f} ± {scores.std():.2f}")
    model.fit(X_train.toarray() if issparse(X_train) else X_train, y_train)
    return model

def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBClassifier(max_depth=3, reg_lambda=2.0, reg_alpha=1.0, random_state=42, use_label_encoder=False, eval_metric='logloss')  # Increase regularization
    scores = cross_val_score(model, X_train.toarray() if issparse(X_train) else X_train, y_train, cv=5, scoring='f1')
    print(f"XGBoost CV F1: {scores.mean():.2f} ± {scores.std():.2f}")
    model.fit(X_train.toarray() if issparse(X_train) else X_train, y_train)
    return model


def train_lstm(X, y, batch_size=32, epochs=10, hidden_dim=128):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = X_train.toarray() if scipy.sparse.issparse(X_train) else X_train
    X_test = X_test.toarray() if scipy.sparse.issparse(X_test) else X_test

    input_size = X_train.shape[1]
    model = LSTMClassifier(input_size, hidden_dim, 1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for i in range(0, X_train.shape[0], batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            batch_X = torch.FloatTensor(batch_X).unsqueeze(1)  # [batch_size, 1, features]
            batch_y = torch.FloatTensor(batch_y).reshape(-1, 1)  # [batch_size, 1]
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test = torch.FloatTensor(X_test).unsqueeze(1)
        y_pred = model(X_test).numpy().flatten()
    return model


def predict_lstm(model, X, device='cpu'):
    model.eval()
    # Convert sparse matrix to dense if necessary
    if issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X

    X_tensor = torch.FloatTensor(X_dense).unsqueeze(1).to(device)  # [batch_size, 1, features]
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()  # [batch_size]
    return (outputs.cpu().numpy() > 0.5).astype(int), outputs.cpu().numpy()
