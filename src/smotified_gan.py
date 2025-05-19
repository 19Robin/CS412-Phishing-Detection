import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from smotified_gan_function import smotified_gan
from mcmc_gan import mcmc_gan
from evaluation import evaluate_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load dataset
    dataset_path = "C:/Users/skath/PycharmProjects/CS412-Phishing-Detection/data/Phishing_Email.csv"
    data = pd.read_csv(dataset_path)

    # Preprocess dataset
    data = data.rename(columns={"Email Text": "email_text", "Email Type": "label"})
    data["label"] = data["label"].map({"Phishing Email": 1, "Safe Email": 0})
    data["email_text"] = data["email_text"].fillna("")  # Replace NaN with empty strings
    X = data["email_text"].values
    y = data["label"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
    X_test_vectorized = vectorizer.transform(X_test).toarray()

    # Evaluate without augmentation
    print("Evaluating model without augmentation...")
    precision_original, recall_original, auc_pr_original = evaluate_model(X_train_vectorized, y_train, X_test_vectorized, y_test)

    # Run SMOTified-GAN
    X_augmented, y_augmented = smotified_gan(X_train_vectorized, y_train, epochs=1000, batch_size=32)
    print("SMOTified-GAN completed successfully!")
    precision_smotified, recall_smotified, auc_pr_smotified = evaluate_model(X_augmented, y_augmented, X_test_vectorized, y_test)

    # Run MCMC-GAN
    X_augmented_mcmc, y_augmented_mcmc = mcmc_gan(X_train_vectorized, y_train, epochs=1000, batch_size=32)
    print("MCMC-GAN completed successfully!")
    precision_mcmc, recall_mcmc, auc_pr_mcmc = evaluate_model(X_augmented_mcmc, y_augmented_mcmc, X_test_vectorized, y_test)

    # Plot Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    plt.plot(recall_original, precision_original, label=f"Original (AUC = {auc_pr_original:.2f})", color="red")
    plt.plot(recall_smotified, precision_smotified, label=f"SMOTified-GAN (AUC = {auc_pr_smotified:.2f})", color="blue")
    plt.plot(recall_mcmc, precision_mcmc, label=f"MCMC-GAN (AUC = {auc_pr_mcmc:.2f})", color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()

    # Compare results
    print("\n--- Comparison of Precision-Recall AUC Scores ---")
    print(f"Original Data: {auc_pr_original:.2f}")
    print(f"SMOTified-GAN: {auc_pr_smotified:.2f}")
    print(f"MCMC-GAN: {auc_pr_mcmc:.2f}")