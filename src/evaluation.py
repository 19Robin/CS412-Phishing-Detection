# src/evaluation.py
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy.sparse import issparse
import torch


def evaluate_model(model, X_train, y_train, X_test, y_test, model_type='sklearn'):
    """
    Evaluate a model with precision, recall, F1, AUC-ROC, and FPR.
    Args:
        model: Trained model (sklearn or PyTorch)
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_type: 'sklearn' or 'pytorch'
    Returns:
        precision, recall, f1, auc_roc, fpr
    """
    if issparse(X_train):
        X_train = X_train.toarray()
    if issparse(X_test):
        X_test = X_test.toarray()

    if model_type == 'sklearn':
        # Cross-validation F1-score
        cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()

        # Test set predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:  # PyTorch LSTM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y_pred, y_prob = predict_lstm(model, X_test, device)
        cv_f1 = 0.0  # Simplified, as cross-validation is complex for LSTM

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc_roc = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return precision, recall, f1, auc_roc, fpr


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    X_train = np.random.rand(100, 5000)
    y_train = np.array([0] * 90 + [1] * 10)
    X_test = np.random.rand(20, 5000)
    y_test = np.array([0] * 18 + [1] * 2)
    model = RandomForestClassifier().fit(X_train, y_train)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    print("Precision, Recall, F1, AUC-ROC, FPR:", metrics)