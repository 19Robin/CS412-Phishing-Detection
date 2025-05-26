import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import auc
from scipy import interpolate
from preprocessing import preprocess_email_data, preprocess_single_email
from smotified_gan import smotified_gan
from mcmc_gan import mcmc_gan
from cgan import cgan
from vae_gan import vae_gan
from classifiers import train_random_forest, train_xgboost, train_lstm, LSTMClassifier
from evaluation import evaluate_model
from imblearn.over_sampling import SMOTE
import psutil
import torch
import joblib
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

def plot_precision_recall(metrics_data, labels, title, classifier_name):
    # Extract AUC-PR and F1 scores
    auc_pr_data = {}
    f1_data = {}
    for (aug_name, clf_name, metrics), label in zip(metrics_data, labels):
        if clf_name == classifier_name:
            key = label.split(' (')[0]  # Extract the base label (e.g., "Original")
            auc_pr_data[key] = metrics[2]  # AUC-PR
            f1_data[key] = metrics[3]      # F1

    # Bar chart configuration
    chart_config = {
        "type": "bar",
        "data": {
            "labels": list(auc_pr_data.keys()),
            "datasets": [
                {
                    "label": "AUC-PR Score",
                    "data": list(auc_pr_data.values()),
                    "backgroundColor": "#36A2EB",
                    "borderColor": "#FFFFFF",
                    "borderWidth": 1
                },
                {
                    "label": "F1 Score",
                    "data": list(f1_data.values()),
                    "backgroundColor": "#FF6384",
                    "borderColor": "#FFFFFF",
                    "borderWidth": 1
                }
            ]
        },
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {"display": True, "text": "Score"},
                    "max": 1.0
                },
                "x": {
                    "title": {"display": True, "text": "Model-Augmentation Pair"}
                }
            },
            "plugins": {
                "legend": {"position": "top"},
                "title": {"display": True, "text": title}
            }
        }
    }
    return chart_config

def interpolate_curves(precision_list, recall_list, num_points=100):
    interpolated_precisions = []
    interpolated_recalls = []
    for precision, recall in zip(precision_list, recall_list):
        if len(recall) > 1:
            f = interpolate.interp1d(recall, precision, kind='linear', bounds_error=False,
                                     fill_value=(precision[0], precision[-1]))
            new_recall = np.linspace(0, 1, num_points)
            new_precision = f(new_recall)
            interpolated_precisions.append(new_precision)
            interpolated_recalls.append(new_recall)
        else:
            interpolated_precisions.append(np.full(num_points, precision[0]))
            interpolated_recalls.append(np.linspace(0, 1, num_points))
    return np.array(interpolated_precisions), np.array(interpolated_recalls)

def get_gmail_service():
    creds = None
    credentials_path = '/content/CS412-Phishing-Detection/credentials.json'
    token_path = '/content/CS412-Phishing-Detection/token.json'
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/gmail.readonly'])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, ['https://www.googleapis.com/auth/gmail.readonly'])
                creds = flow.run_local_server(port=0)
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            except Exception as e:
                print(f"Gmail authentication failed: {str(e)}")
                return None
    return build('gmail', 'v1', credentials=creds)

def fetch_latest_email_snippet():
    try:
        service = get_gmail_service()
        if service is None:
            raise Exception("Gmail service unavailable")
        results = service.users().messages().list(userId='me').execute()
        messages = results.get('messages', [])
        if messages:
            msg = service.users().messages().get(userId='me', id=messages[0]['id']).execute()
            return msg['snippet']
        return "No email found"
    except Exception as e:
        print(f"Error fetching email: {str(e)}")
        # Mock email snippet for demo
        return "This is a mock email for demo purposes with a link http://example.com"

def save_models(clf_type, model, aug_name, fold):
    model_dir = '/content/CS412-Phishing-Detection/models'
    os.makedirs(model_dir, exist_ok=True)
    if clf_type == "lstm":
        torch.save(model.state_dict(),
                   os.path.join(model_dir, f"{clf_type}_{aug_name.lower().replace('-', '_')}_fold{fold}.pth"))
    else:
        joblib.dump(model,
                    os.path.join(model_dir, f"{clf_type}_{aug_name.lower().replace('-', '_')}_fold{fold}.joblib"))

def classify_email(email_text, rf_models, xgb_models, lstm_models):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = preprocess_single_email(email_text)
    if X is None or X.size == 0:
        return {}
    bert_feature = X[:, :-1]
    url_count = X[:, -1].reshape(1, -1)
    X_processed = np.hstack([bert_feature, url_count])
    preds = {}
    for aug in ["Original", "SMOTE", "SMOTified-GAN", "MCMC-GAN", "CGAN", "VAE-GAN"]:
        aug_lower = aug.lower().replace('-', '_')
        preds[f"RandomForest_{aug}"] = rf_models[aug].predict_proba(X_processed)[0][1]
        preds[f"XGBoost_{aug}"] = xgb_models[aug].predict_proba(X_processed)[0][1]
        lstm_model = lstm_models[aug]
        with torch.no_grad():
            X_tensor = torch.FloatTensor(bert_feature).unsqueeze(1).to(device)
            y_prob = lstm_model(X_tensor).squeeze().cpu().numpy()
            preds[f"LSTM_{aug}"] = y_prob[0] if y_prob.ndim > 0 else y_prob.item()
    return preds

if __name__ == "__main__":
    try:
        dataset_path = "/content/CS412-Phishing-Detection/data/Phishing_Emails.csv"
        X, y = preprocess_email_data(dataset_path)
        print(f"Loaded dataset shape: {X.shape}")
        print(f"Loaded label distribution: {pd.Series(y).value_counts()}")

        subset_indices = np.random.choice(len(X), 1000, replace=False)
        X_subset = X[subset_indices]
        y_subset = y[subset_indices]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        print("Starting 5-fold CV...")
        for fold, (train_index, test_index) in enumerate(kf.split(X_subset)):
            print(f"\n--- Fold {fold + 1} ---")
            X_train, X_test = X_subset[train_index], X_subset[test_index]
            y_train, y_test = y_subset[train_index], y_subset[test_index]
            print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
            print(f"Memory usage: {psutil.virtual_memory().percent}%")

            X_train_dense = X_train
            X_test_dense = X_test
            overlap = np.any(
                [np.array_equal(X_train_dense[i], X_test_dense[j]) for i in range(min(10, X_train_dense.shape[0])) for j
                 in range(min(10, X_test_dense.shape[0]))])
            print(f"Overlap between X_train and X_test: {overlap}")

            augmentations = [
                ("Original", X_train, y_train),
                ("SMOTE", *SMOTE(k_neighbors=5, random_state=42).fit_resample(X_train, y_train)),
                ("SMOTified-GAN", *smotified_gan(X_train, y_train, epochs=50)),
                ("MCMC-GAN", *mcmc_gan(X_train, y_train, epochs=50)),
                ("CGAN", *cgan(X_train, y_train, epochs=50)),
                ("VAE-GAN", *vae_gan(X_train, y_train, epochs=50))
            ]

            for aug_name, X_aug, y_aug in augmentations:
                print(f"Augmented {aug_name} shape: {X_aug.shape}")

            classifiers = [
                ("Random Forest", train_random_forest, "rf"),
                ("XGBoost", train_xgboost, "xgboost"),
                ("LSTM", train_lstm, "lstm")
            ]

            rf_models = {aug[0]: None for aug in augmentations}
            xgb_models = {aug[0]: None for aug in augmentations}
            lstm_models = {aug[0]: None for aug in augmentations}

            fold_metrics = []
            for aug_name, X_aug, y_aug in augmentations:
                for clf_name, clf_fn, clf_type in classifiers:
                    print(f"Evaluating {clf_name} with {aug_name} in fold {fold + 1}...")
                    try:
                        if clf_type == "lstm":
                            model_tuple = clf_fn(X_aug, y_aug, batch_size=32, epochs=10, hidden_dim=128)
                            model = model_tuple[0] if isinstance(model_tuple, tuple) else model_tuple
                            lstm_models[aug_name] = model
                        else:
                            model_tuple = clf_fn(X_aug, y_aug)
                            model = model_tuple[0] if isinstance(model_tuple, tuple) else model_tuple
                            if clf_type == "rf":
                                rf_models[aug_name] = model
                            elif clf_type == "xgboost":
                                xgb_models[aug_name] = model
                        save_models(clf_type, model, aug_name, fold + 1)
                        metrics = evaluate_model(model, X_aug, y_aug, X_test, y_test, clf_type)
                        print(
                            f"{clf_name} - {aug_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, "
                            f"F1={metrics['f1']:.2f}, AUC-ROC={metrics['auc_roc']:.2f}, FPR={metrics['fpr']:.2f}")
                        fold_metrics.append((aug_name, clf_name, (metrics["pr_curve"][0], metrics["pr_curve"][1], metrics["pr_curve"][2], metrics["f1"])))
                    except Exception as e:
                        print(f"Error evaluating {clf_name} with {aug_name} in fold {fold + 1}: {str(e)}")

            fold_results.extend(fold_metrics)

        results_dict = {}
        for aug_name, clf_name, metrics in fold_results:
            key = (aug_name, clf_name)
            if key not in results_dict:
                results_dict[key] = []
            results_dict[key].append(metrics)

        avg_results = []
        for (aug_name, clf_name), metrics_list in results_dict.items():
            auc_pr_values = [m[2] for m in metrics_list]
            f1_values = [m[3] for m in metrics_list]
            precision_list = [m[0] for m in metrics_list]
            recall_list = [m[1] for m in metrics_list]
            avg_auc_pr = np.mean(auc_pr_values)
            avg_f1 = np.mean(f1_values)
            interpolated_precision, interpolated_recall = interpolate_curves(precision_list, recall_list)
            avg_precision = np.mean(interpolated_precision, axis=0)
            avg_recall = np.mean(interpolated_recall, axis=0)
            avg_results.append((aug_name, clf_name, (avg_precision, avg_recall, avg_auc_pr, avg_f1)))
            print(f"Average for {clf_name} - {aug_name}: AUC-PR={avg_auc_pr:.2f}, F1={avg_f1:.2f}")

        for clf_name in ["Random Forest", "XGBoost", "LSTM"]:
            clf_metrics = [metrics for aug_name, c_name, metrics in avg_results if c_name == clf_name]
            labels = [f"{aug_name} (AUC-PR = {metrics[2]:.2f}, F1 = {metrics[3]:.2f})" for aug_name, c_name, metrics in avg_results if c_name == clf_name]
            if clf_metrics:
                chart = plot_precision_recall(clf_metrics, labels, f"{clf_name} Performance Scores", clf_name)
                print(f"Bar chart for {clf_name} generated. Visualize in canvas panel:")
                print(chart)

        print("\n--- Summary ---")
        for aug_name, clf_name, metrics in avg_results:
            print(f"{clf_name} - {aug_name}: AUC-PR={metrics[2]:.2f}, F1={metrics[3]:.2f}")

        # Classify latest email
        email_text = fetch_latest_email_snippet()
        print(f"Classifying email: {email_text[:50]}...")
        preds = classify_email(email_text, rf_models, xgb_models, lstm_models)
        avg_prob = np.mean(list(preds.values()))
        prediction = "Phishing" if avg_prob > 0.5 else "Safe"
        accuracies = {k: f"{v * 100:.0f}%" for k, v in preds.items()}
        print(f"Email Prediction: {prediction}")
        print(f"Accuracies: {accuracies}")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")