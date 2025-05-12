# src/main.py
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from preprocessing import preprocess_email_data
from smotified_gan import smotified_gan
from mcmc_gan import mcmc_gan
from cgan import cgan
from vae_gan import vae_gan
from classifiers import train_random_forest, train_xgboost, train_lstm, predict_lstm
from evaluation import evaluate_model
from imblearn.over_sampling import SMOTE


def plot_precision_recall(metrics, labels, title="Precision-Recall Curve Comparison"):
    plt.figure(figsize=(10, 8))
    for (precision, recall, _, auc_roc, _), label in zip(metrics, labels):
        plt.plot(recall, precision, label=f"{label} (AUC-ROC = {auc_roc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid()
    plt.savefig("precision_recall.png")
    plt.show()


if __name__ == "__main__":
    # Load and preprocess data
    dataset_path = "../data/Phishing_Email.csv"
    X, y, tfidf_text, tfidf_pos = preprocess_email_data(dataset_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define augmentation methods
    augmentations = [
        ("Original", X_train, y_train),
        ("SMOTE", *SMOTE(k_neighbors=5, random_state=42).fit_resample(X_train, y_train)),
        ("SMOTified-GAN", *smotified_gan(X_train, y_train, epochs=100)),
        ("MCMC-GAN", *mcmc_gan(X_train, y_train, epochs=100)),
        ("CGAN", *cgan(X_train, y_train, epochs=100)),
        ("VAE-GAN", *vae_gan(X_train, y_train, epochs=100))
    ]

    # Define classifiers
    classifiers = [
        ("Random Forest", train_random_forest, 'sklearn'),
        ("XGBoost", train_xgboost, 'sklearn'),
        ("LSTM", train_lstm, 'pytorch')
    ]

    # Evaluate each augmentation-classifier combination
    results = []
    for aug_name, X_aug, y_aug in augmentations:
        aug_results = []
        for clf_name, clf_fn, clf_type in classifiers:
            print(f"Evaluating {clf_name} with {aug_name}...")
            model = clf_fn(X_aug, y_aug)
            metrics = evaluate_model(model, X_aug, y_aug, X_test, y_test, clf_type)
            print(f"{clf_name} - {aug_name}: Precision={metrics[0]:.2f}, Recall={metrics[1]:.2f}, "
                  f"F1={metrics[2]:.2f}, AUC-ROC={metrics[3]:.2f}, FPR={metrics[4]:.2f}")
            aug_results.append(metrics)
        results.append((aug_name, aug_results))

    # Plot results for Random Forest (example)
    rf_metrics = [(metrics[0], clf_name) for aug_name, metrics in results for clf_name, m, t in classifiers if
                  clf_name == "Random Forest"]
    metrics, labels = zip(
        *[(m, aug_name) for aug_name, ms in results for clf_name, m, t in classifiers if clf_name == "Random Forest"])
    plot_precision_recall(metrics, labels, "Random Forest Precision-Recall Curves")

    # Print summary
    print("\n--- Summary ---")
    for aug_name, metrics in results:
        for clf_name, m, _ in classifiers:
            precision, recall, f1, auc_roc, fpr = m
            print(f"{clf_name} - {aug_name}: AUC-ROC={auc_roc:.2f}, F1={f1:.2f}")