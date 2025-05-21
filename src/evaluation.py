from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score
import torch
from scipy.sparse import issparse
import numpy as np

def predict_lstm(model, X_test, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Predict using LSTM model with specified device.
    Returns binary predictions and probabilities.
    """
    model.eval()
    model.to(device)
    if issparse(X_test):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test
    X_test_tensor = torch.FloatTensor(X_test_dense).unsqueeze(1).to(device)  # [batch_size, 1, features]
    with torch.no_grad():
        y_prob = model(X_test_tensor).squeeze().cpu().numpy()  # [batch_size]
    y_pred = (y_prob > 0.5).astype(int)
    return y_pred, y_prob

def evaluate_model(model, X_train, y_train, X_test, y_test, clf_type):
    """
    Evaluate model and return metrics including precision-recall curve data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if clf_type.lower() == "lstm":
        # Train LSTM (assuming it's not pre-trained for this evaluation)
        model.to(device)
        y_pred, y_prob = predict_lstm(model, X_test, device)
    else:  # scikit-learn models
        X_train_dense = X_train.toarray() if issparse(X_train) else X_train
        X_test_dense = X_test.toarray() if issparse(X_test) else X_test
        model.fit(X_train_dense, y_train)
        y_pred = model.predict(X_test_dense)
        y_prob = model.predict_proba(X_test_dense)[:, 1]

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)

    # Compute single-point metrics at threshold 0.5
    precision_score_val = precision_score(y_test, y_pred, zero_division=0)
    recall_score_val = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Compute ROC and AUC-ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_roc = auc(fpr, tpr)
    fpr_val = fpr[np.argmax(tpr >= recall_score_val)] if len(fpr) > 0 else 0.0  # Match FPR to recall threshold

    return {
        "precision": precision_score_val,
        "recall": recall_score_val,
        "f1": f1,
        "auc_roc": auc_roc,
        "fpr": fpr_val,
        "pr_curve": (precision, recall, auc_pr)  # Added for plotting
    }

if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from classifiers import LSTMClassifier, train_lstm

    # Test scikit-learn model
    X_train_rf = np.random.rand(100, 5000)
    y_train_rf = np.array([0] * 90 + [1] * 10)
    X_test_rf = np.random.rand(20, 5000)
    y_test_rf = np.array([0] * 18 + [1] * 2)
    model_rf = RandomForestClassifier(max_depth=5, random_state=42)
    metrics_rf = evaluate_model(model_rf, X_train_rf, y_train_rf, X_test_rf, y_test_rf, "rf")
    print("Random Forest Metrics:", metrics_rf)

    # Test LSTM model (using train_lstm for proper initialization)
    X_train_lstm = np.random.rand(100, 5000)
    y_train_lstm = np.array([0] * 90 + [1] * 10)
    X_test_lstm = np.random.rand(20, 5000)
    y_test_lstm = np.array([0] * 18 + [1] * 2)
    model_lstm = train_lstm(X_train_lstm, y_train_lstm, batch_size=32, epochs=2, hidden_dim=128)  # Train briefly
    metrics_lstm = evaluate_model(model_lstm, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, "lstm")
    print("LSTM Metrics:", metrics_lstm)