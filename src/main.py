import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from preprocessing import preprocess_email_data
from smotified_gan import smotified_gan
from mcmc_gan import mcmc_gan
from cgan import cgan
from vae_gan import vae_gan
from classifiers import train_random_forest, train_xgboost, train_lstm, predict_lstm
from evaluation import evaluate_model
from imblearn.over_sampling import SMOTE
import psutil

def plot_precision_recall(metrics, labels, title="Precision-Recall Curve Comparison"):
    plt.figure(figsize=(10, 8))
    for metric_dict, label in zip(metrics, labels):
        plt.plot([0, 1], [metric_dict["precision"], metric_dict["precision"]], label=f"{label} (AUC-ROC = {metric_dict['auc_roc']:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid()
    plt.savefig("precision_recall.png")
    plt.show()

if __name__ == "__main__":
    try:
        # Load and preprocess data (updated path for Colab or local)
        ##dataset_path = "/content/drive/MyDrive/CS412-Phishing-Detection/data/Phishing_validation_emails.csv"  # For Colab
        ##dataset_path = "C:/Users/slade/Downloads/CS412/Week 4/CS412 - Phishing - Detection/data/Phishing_Email.csv"
        dataset_path = "C:/Users/slade/Downloads/CS412/Week 4/CS412-Phishing-Detection/data/Phishing_Email.csv"
        X, y, tfidf_text, tfidf_pos = preprocess_email_data(dataset_path)
        print(f"Loaded dataset shape: {X.shape}")
        print(f"Loaded label distribution: {pd.Series(y).value_counts()}")

        # Use a subset for faster experimentation
        subset_indices = np.random.choice(len(X), 500, replace=False)
        X_subset = X[subset_indices]
        y_subset = y[subset_indices]

        # 2-fold cross-validation
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        fold_results = []
        print("Starting 2-fold CV...")
        for fold, (train_index, test_index) in enumerate(kf.split(X_subset)):
            print(f"\n--- Fold {fold + 1} ---")
            X_train, X_test = X_subset[train_index], X_subset[test_index]
            y_train, y_test = y_subset[train_index], y_subset[test_index]
            print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
            print(f"Memory usage: {psutil.virtual_memory().percent}%")

            # Check for overlap
            X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
            X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
            overlap = np.any([np.array_equal(X_train_dense[i], X_test_dense[j]) for i in range(min(10, X_train_dense.shape[0])) for j in range(min(10, X_test_dense.shape[0]))])
            print(f"Overlap between X_train and X_test: {overlap}")

            # Define augmentation methods with reduced epochs
            augmentations = [
                ("Original", X_train, y_train),
                ("SMOTE", *SMOTE(k_neighbors=5, random_state=42).fit_resample(X_train, y_train)),
                ("SMOTified-GAN", *smotified_gan(X_train, y_train, epochs=20)),
                ("MCMC-GAN", *mcmc_gan(X_train, y_train, epochs=20)),
                ("CGAN", *cgan(X_train, y_train, epochs=20)),
                ("VAE-GAN", *vae_gan(X_train, y_train, epochs=20))
            ]

            # Print augmentation shapes
            for aug_name, X_aug, y_aug in augmentations:
                print(f"Augmented {aug_name} shape: {X_aug.shape}")

            # Define classifiers
            classifiers = [
                ("Random Forest", train_random_forest, "rf"),
                ("XGBoost", train_xgboost, "xgboost"),
                ("LSTM", train_lstm, "lstm")
            ]

            # Evaluate each augmentation-classifier combination for this fold
            fold_metrics = []
            for aug_name, X_aug, y_aug in augmentations:
                for clf_name, clf_fn, clf_type in classifiers:
                    print(f"Evaluating {clf_name} with {aug_name} in fold {fold + 1}...")
                    try:
                        model = clf_fn(X_aug, y_aug)
                        metrics = evaluate_model(model, X_aug, y_aug, X_test, y_test, clf_type)
                        print(f"{clf_name} - {aug_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, "
                              f"F1={metrics['f1']:.2f}, AUC-ROC={metrics['auc_roc']:.2f}, FPR={metrics['fpr']:.2f}")
                        fold_metrics.append((aug_name, clf_name, metrics))
                    except Exception as e:
                        print(f"Error evaluating {clf_name} with {aug_name} in fold {fold + 1}: {str(e)}")

            fold_results.extend(fold_metrics)

        # Aggregate results across folds
        results_dict = {}
        for aug_name, clf_name, metrics in fold_results:
            key = (aug_name, clf_name)
            if key not in results_dict:
                results_dict[key] = {"precision": [], "recall": [], "f1": [], "auc_roc": [], "fpr": []}
            for metric_name in metrics:
                results_dict[key][metric_name].append(metrics[metric_name])

        # Compute average metrics
        avg_results = []
        for (aug_name, clf_name), metrics_dict in results_dict.items():
            avg_metrics = {
                "precision": np.mean(metrics_dict["precision"]),
                "recall": np.mean(metrics_dict["recall"]),
                "f1": np.mean(metrics_dict["f1"]),
                "auc_roc": np.mean(metrics_dict["auc_roc"]),
                "fpr": np.mean(metrics_dict["fpr"])
            }
            avg_results.append((aug_name, clf_name, avg_metrics))
            print(f"Average for {clf_name} - {aug_name}: Precision={avg_metrics['precision']:.2f}, "
                  f"Recall={avg_metrics['recall']:.2f}, F1={avg_metrics['f1']:.2f}, "
                  f"AUC-ROC={avg_metrics['auc_roc']:.2f}, FPR={avg_metrics['fpr']:.2f}")

        # Plot results for Random Forest
        rf_metrics = [metrics for aug_name, clf_name, metrics in avg_results if clf_name == "Random Forest"]
        labels = [aug_name for aug_name, clf_name, metrics in avg_results if clf_name == "Random Forest"]
        if rf_metrics:
            plot_precision_recall(rf_metrics, labels, "Random Forest Precision-Recall Curves")

        # Print summary
        print("\n--- Summary ---")
        for aug_name, clf_name, metrics in avg_results:
            print(f"{clf_name} - {aug_name}: AUC-ROC={metrics['auc_roc']:.2f}, F1={metrics['f1']:.2f}")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")