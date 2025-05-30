CS412: Artificial Intelligence - Phishing Email Detection
Welcome to Our Project!
We are Group 10 from the University of the South Pacific, working on our CS412: Artificial Intelligence project (Semester 1, 2025). Our project, Bio-Inspired GAN-Augmented Phishing Email Detection with Multi-Model Classification, aims to detect phishing emails in imbalanced datasets using advanced generative techniques. Supervised by Dr. Anuraganand Sharma, we’ve built a system inspired by the human immune system to tackle this cybersecurity challenge. This README will guide you through our project, setup, and usage.


Group Members

Ranjan Naidu (S11201181)  
Salote Katia (S11196202)  
Samuela Robin (S11199961)  
Bulou Vitukawalu (S11210019)


What Is This Project About? 
Phishing emails are a major cybersecurity threat, but they’re rare—making up less than 1% of email traffic. This rarity creates imbalanced datasets, which confuse traditional detection models. Our goal was to improve phishing detection by:

Generating Synthetic Samples: We used SMOTified-GAN, MCMC-GAN, VAE-GAN, and CGAN to create realistic phishing email samples, balancing the dataset.  
Classifying Emails: We trained Random Forest, XGBoost, and LSTM models to classify emails as phishing or legitimate.  
Real-Time Detection: We integrated our system with a Gmail browser extension for real-time classification.

SMOTE: Generates synthetic phishing samples using k-nearest neighbors interpolation.  
SMOTified-GAN: Combines SMOTE with a Wasserstein GAN (with gradient penalty) to create diverse, email-specific samples (e.g., URLs, linguistic patterns).  
MCMC-GAN: Uses Markov Chain Monte Carlo methods to improve sample diversity and stability for small phishing datasets.  
VAE-GAN: Combines Variational Autoencoders with GANs, using BERT-based embeddings for realistic phishing samples.  
CGAN: Conditions sample generation on class labels to target phishing emails.  
Classifiers: Random Forest, XGBoost, and LSTM, evaluated on F1-score, recall, and false positive rate.


Features 
Real-time phishing detection via a Gmail browser extension.  
Flask-based server with a weighted ensemble of models.  
Advanced class imbalance mitigation using SMOTified-GAN, MCMC-GAN, VAE-GAN, and CGAN.  
Detailed logging for debugging predictions and feature extraction.


Setup
What You’ll Need

Python: Version 3.8 or higher (we used 3.2).  
Git: For cloning the repository.  
Virtual Environment: We recommend virtualenv.  
Hardware: A computer with internet and decent processing power.  
Dataset: Kaggle Phishing Email Dataset (includes Enron, Ling, CEAS, Nazario, Nigerian, and SpamAssassin emails).

Installation Steps

Clone the Repository  
git clone https://github.com/your-username/CS412-Phishing-Detection.git
cd CS412-Phishing-Detection


Set Up a Virtual Environment  
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Install Dependencies  
pip install -r requirements.txt


Download NLTK Data (for preprocessing)  
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"



Running the Project

Start the Flask Server  
python src/app.py

The server will run on http://127.0.0.1:5000.

Load the Browser Extension in Chrome  

Go to chrome://extensions/.  
Enable "Developer mode."  
Click "Load unpacked" and select the extension directory.


Test It OutOpen Gmail, select an email, and click "Classify Email" in the extension to see the prediction.



Usage 

Classify Emails: Use the browser extension in Gmail to classify emails as "Phishing" or "Safe." The extension sends the email to our Flask server, which returns a prediction with model accuracies.  
Check Logs: Open your terminal to view server logs, which include raw email content, feature samples, and model predictions for debugging.  
Customize: Edit src/app.py to adjust model weights, thresholds, or the safe domain list to improve performance on your dataset.



Acknowledgments

A big thanks to Dr. Anuraganand Sharma for guiding us throughout this project.  
We’re grateful to our CS412 instructors and peers at USP for their support.  
Thanks to Hugging Face for BERT resources, and the open-source community for tools like NLTK and Flask.


