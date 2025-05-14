from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import torch
from scipy.sparse import issparse

def predict_lstm(model, X_test, device='cpu'):
    model.eval()
    model.to(device)
    # Convert sparse matrix to dense if necessary
    if issparse(X_test):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test
    X_test_tensor = torch.FloatTensor(X_test_dense).unsqueeze(1)  # [batch_size, 1, features]
    with torch.no_grad():
        y_prob = model(X_test_tensor).cpu().numpy()  # [batch_size, 1]
    y_pred = (y_prob > 0.5).astype(int)  # Binary classification threshold
    return y_pred.flatten(), y_prob.flatten()

def evaluate_model(model, X, y, X_test, y_test, clf_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if clf_type.lower() == "lstm":  # Case-insensitive check
        y_pred, y_prob = predict_lstm(model, X_test, device)
    else:  # scikit-learn models
        model.fit(X.toarray() if issparse(X) else X, y)
        y_pred = model.predict(X_test.toarray() if issparse(X_test) else X_test)
        y_prob = model.predict_proba(X_test.toarray() if issparse(X_test) else X_test)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_roc = auc(fpr, tpr)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "fpr": fpr[0] if len(fpr) > 0 else 0.0
    }

if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from classifiers import LSTMClassifier

    # Test scikit-learn model
    X_train_rf = np.random.rand(100, 5000)
    y_train_rf = np.array([0] * 90 + [1] * 10)
    X_test_rf = np.random.rand(20, 5000)
    y_test_rf = np.array([0] * 18 + [1] * 2)
    model_rf = RandomForestClassifier().fit(X_train_rf, y_train_rf)
    metrics_rf = evaluate_model(model_rf, X_train_rf, y_train_rf, X_test_rf, y_test_rf, "rf")
    print("Random Forest Metrics:", metrics_rf)

    # Test LSTM model (dummy data and model)
    X_train_lstm = np.random.rand(100, 5000)
    y_train_lstm = np.array([0] * 90 + [1] * 10)
    X_test_lstm = np.random.rand(20, 5000)
    y_test_lstm = np.array([0] * 18 + [1] * 2)
    model_lstm = LSTMClassifier(input_dim=5000, hidden_dim=128, num_layers=2, output_dim=1)
    metrics_lstm = evaluate_model(model_lstm, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, "lstm")
    print("LSTM Metrics:", metrics_lstm)