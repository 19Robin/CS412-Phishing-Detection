import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc

def evaluate_model(X_train, y_train, X_test, y_test):
    """
    Trains a Random Forest classifier and evaluates it using Precision-Recall AUC.
    Returns precision, recall, and AUC-PR for plotting.
    """
    print("Training data shape:", X_train.shape, "Labels shape:", y_train.shape)
    print("Starting Random Forest training...")

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    print("Random Forest training completed.")

    # Predict probabilities for the test set
    y_scores = clf.predict_proba(X_test)[:, 1]

    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    # Calculate AUC-PR
    auc_pr = auc(recall, precision)

    print(f"AUC-PR: {auc_pr:.2f}")
    return precision, recall, auc_pr