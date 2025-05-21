import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy import interpolate
from preprocessing import preprocess_email_data
from smotified_gan import smotified_gan
from mcmc_gan import mcmc_gan
from cgan import cgan
from vae_gan import vae_gan
from classifiers import train_random_forest, train_xgboost, train_lstm, predict_lstm
from evaluation import evaluate_model
from imblearn.over_sampling import SMOTE
import psutil
import os
import scipy

def plot_precision_recall(metrics_data, labels, title, classifier_name):
    plt.figure(figsize=(10, 8))
    for (precision, recall, _), label in zip(metrics_data, labels):
        plt.plot(recall, precision, label=f"{label} (AUC-PR = {auc(recall, precision):.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid()
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"precision_recall_{classifier_name.lower()}.png"))
    plt.show()

def interpolate_curves(precision_list, recall_list, num_points=100):
    """Interpolate precision-recall curves to a common length."""
    interpolated_precisions = []
    interpolated_recalls = []
    for precision, recall in zip(precision_list, recall_list):
        if len(recall) > 1:  # Ensure there are points to interpolate
            f = interpolate.interp1d(recall, precision, kind='linear', bounds_error=False, fill_value=(precision[0], precision[-1]))
            new_recall = np.linspace(0, 1, num_points)
            new_precision = f(new_recall)
            interpolated_precisions.append(new_precision)
            interpolated_recalls.append(new_recall)
        else:
            # Handle edge case with minimal data
            interpolated_precisions.append(np.full(num_points, precision[0]))
            interpolated_recalls.append(np.linspace(0, 1, num_points))
    return np.array(interpolated_precisions), np.array(interpolated_recalls)

if __name__ == "__main__":
    try:
        # Load and preprocess data
        dataset_path = "/content/CS412-Phishing-Detection/data/Phishing_Emails.csv"
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
                        fold_metrics.append((aug_name, clf_name, metrics["pr_curve"]))
                    except Exception as e:
                        print(f"Error evaluating {clf_name} with {aug_name} in fold {fold + 1}: {str(e)}")

            fold_results.extend(fold_metrics)

        # Aggregate results across folds
        results_dict = {}
        for aug_name, clf_name, metrics in fold_results:
            key = (aug_name, clf_name)
            if key not in results_dict:
                results_dict[key] = []
            results_dict[key].append(metrics)

        # Compute average AUC-PR and interpolate curves
        avg_results = []
        for (aug_name, clf_name), metrics_list in results_dict.items():
            auc_pr_values = [m[2] for m in metrics_list]
            precision_list = [m[0] for m in metrics_list]
            recall_list = [m[1] for m in metrics_list]
            avg_auc_pr = np.mean(auc_pr_values)
            interpolated_precision, interpolated_recall = interpolate_curves(precision_list, recall_list)
            avg_precision = np.mean(interpolated_precision, axis=0)
            avg_recall = np.mean(interpolated_recall, axis=0)
            avg_results.append((aug_name, clf_name, (avg_precision, avg_recall, avg_auc_pr)))
            print(f"Average for {clf_name} - {aug_name}: AUC-PR={avg_auc_pr:.2f}")

        # Plot results for each classifier
        for clf_name in ["Random Forest", "XGBoost", "LSTM"]:
            clf_metrics = [metrics for aug_name, c_name, metrics in avg_results if c_name == clf_name]
            labels = [f"{aug_name} (AUC-PR = {metrics[2]:.2f})" for aug_name, c_name, metrics in avg_results if c_name == clf_name]
            if clf_metrics:
                plot_precision_recall(clf_metrics, labels, f"{clf_name} Precision-Recall Curves", clf_name)

        # Print summary
        print("\n--- Summary ---")
        for aug_name, clf_name, metrics in avg_results:
            print(f"{clf_name} - {aug_name}: AUC-PR={metrics[2]:.2f}")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")