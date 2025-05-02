import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_dataset(file_path):
    """
    Load the phishing email dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def clean_data(data):
    """
    Perform basic data cleaning, such as handling missing values.
    """
    # Drop rows with missing values
    data = data.dropna()
    print(f"Data cleaned. Remaining rows: {data.shape[0]}")
    return data

def encode_labels(data, target_column):
    """
    Encode the target labels into numerical format.
    """
    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])
    print(f"Labels encoded. Classes: {list(label_encoder.classes_)}")
    return data, label_encoder

def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    dataset_path = "../data/phishing_email_dataset.csv"  # Adjust path as needed
    target_column = "label"  # Replace with the actual target column name

    # Load and preprocess the dataset
    data = load_dataset(dataset_path)
    if data is not None:
        data = clean_data(data)
        data, label_encoder = encode_labels(data, target_column)
        X_train, X_test, y_train, y_test = split_data(data, target_column)