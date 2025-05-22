from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score
import torch
import numpy as np


def predict_lstm(model, X_test, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    model.to(device)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)  # Add sequence dimension
    with torch.no_grad():
        y_prob = model(X_test_tensor).squeeze().cpu().numpy()
    y_pred = (y_prob > 0.5).astype(int)
    return y_pred, y_prob


def evaluate_model(model, X_train, y_train, X_test, y_test, clf_type, X_train_tfidf=None, X_test_tfidf=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select appropriate features based on classifier type
    if clf_type.lower() == "lstm":
        # LSTM uses BERT embeddings
        model.to(device)
        y_pred, y_prob = predict_lstm(model, X_test, device)
    else:
        # Random Forest and XGBoost use TF-IDF features (or BERT if retrained)
        X_train_features = X_train_tfidf if X_train_tfidf is not None else X_train
        X_test_features = X_test_tfidf if X_test_tfidf is not None else X_test
        model.fit(X_train_features, y_train)
        y_pred = model.predict(X_test_features)
        y_prob = model.predict_proba(X_test_features)[:, 1]

    # Compute metrics
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    precision_score_val = precision_score(y_test, y_pred, zero_division=0)
    recall_score_val = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_roc = auc(fpr, tpr)
    fpr_val = fpr[np.argmax(tpr >= recall_score_val)] if len(fpr) > 0 else 0.0

    return {
        "precision": precision_score_val,
        "recall": recall_score_val,
        "f1": f1,
        "auc_roc": auc_roc,
        "fpr": fpr_val,
        "pr_curve": (precision, recall, auc_pr)
    }


if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from classifiers import LSTMClassifier, train_lstm

    # Test Random Forest with TF-IDF features
    X_train_rf = np.random.rand(100, 768)
    X_train_rf_tfidf = np.random.rand(100, 768)  # Simulate TF-IDF
    y_train_rf = np.array([0] * 90 + [1] * 10)
    X_test_rf = np.random.rand(20, 768)
    X_test_rf_tfidf = np.random.rand(20, 768)  # Simulate TF-IDF
    y_test_rf = np.array([0] * 18 + [1] * 2)
    model_rf = RandomForestClassifier(max_depth=5, random_state=42)
    metrics_rf = evaluate_model(model_rf, X_train_rf, y_train_rf, X_test_rf, y_test_rf, "rf", X_train_rf_tfidf,
                                X_test_rf_tfidf)
    print("Random Forest Metrics:", metrics_rf)

    # Test LSTM with BERT embeddings
    X_train_lstm = np.random.rand(100, 768)
    y_train_lstm = np.array([0] * 90 + [1] * 10)
    X_test_lstm = np.random.rand(20, 768)
    y_test_lstm = np.array([0] * 18 + [1] * 2)
    model_lstm = train_lstm(X_train_lstm, y_train_lstm, batch_size=32, epochs=2, hidden_dim=128)
    metrics_lstm = evaluate_model(model_lstm, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, "lstm")
    print("LSTM Metrics:", metrics_lstm)