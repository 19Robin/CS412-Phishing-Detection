import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

def train_random_forest(X, y):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty dataset provided for training Random Forest")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"Random Forest CV F1: {scores.mean():.2f} ± {scores.std():.2f}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_prob)
    print(f"Test F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, AUC-ROC: {auc_roc:.2f}")
    return model, {'f1': f1, 'precision': precision, 'recall': recall, 'auc_roc': auc_roc}

def train_xgboost(X, y):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty dataset provided for training XGBoost")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
    model = XGBClassifier(max_depth=3, reg_lambda=2.0, reg_alpha=1.0, scale_pos_weight=scale_pos_weight,
                          random_state=42, eval_metric='logloss')
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"XGBoost CV F1: {scores.mean():.2f} ± {scores.std():.2f}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_prob)
    print(f"Test F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, AUC-ROC: {auc_roc:.2f}")
    return model, {'f1': f1, 'precision': precision, 'recall': recall, 'auc_roc': auc_roc}

def train_lstm(X, y, batch_size=32, epochs=10, hidden_dim=128):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty dataset provided for training LSTM")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Remove URL count feature for LSTM input (last column)
    X_train_lstm = X_train[:, :-1]
    X_test_lstm = X_test[:, :-1]
    input_size = X_train_lstm.shape[1]  # Should be 768 (BERT features)
    model = LSTMClassifier(input_size, hidden_dim, 1).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train_lstm), batch_size):
            batch_X = X_train_lstm[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            batch_X = torch.FloatTensor(batch_X).unsqueeze(1).to(device)
            batch_y = torch.FloatTensor(batch_y).reshape(-1, 1).to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_lstm).unsqueeze(1).to(device)
        y_prob = model(X_test_tensor).squeeze().cpu().numpy()
        y_pred = (y_prob > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_prob)
    print(f"Test F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, AUC-ROC: {auc_roc:.2f}")
    return model, {'f1': f1, 'precision': precision, 'recall': recall, 'auc_roc': auc_roc}

def predict_lstm(model, X, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    model.to(device)
    # Remove URL count feature for LSTM input
    X_lstm = X[:, :-1] if X.shape[1] > 768 else X
    X_tensor = torch.FloatTensor(X_lstm).unsqueeze(1).to(device)
    with torch.no_grad():
        outputs = model(X_tensor).squeeze().cpu().numpy()
    y_pred = (outputs > 0.5).astype(int)
    y_prob = outputs
    return y_pred, y_prob

if __name__ == "__main__":
    # Test the classifiers
    X = np.random.rand(100, 769)  # 768 BERT + 1 URL count
    y = np.random.randint(0, 2, 100)

    rf_model, rf_metrics = train_random_forest(X, y)
    print("Random Forest Metrics:", rf_metrics)

    xgb_model, xgb_metrics = train_xgboost(X, y)
    print("XGBoost Metrics:", xgb_metrics)

    lstm_model, lstm_metrics = train_lstm(X, y, batch_size=32, epochs=2, hidden_dim=128)
    print("LSTM Metrics:", lstm_metrics)