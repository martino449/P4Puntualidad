#-------------------------------
# Name:        P4puntualidad
# Purpose:     Train a decision tree classifier and make predictions
#
# Author:      Martin449
#
# Created on:  18/07/2024
# dependencies:
#    python = ">=3.10.0,<3.12"
#    pandas = "^2.2.2"
#    scikit-learn = "^1.5.1"
#    cryptography = "^42.0.8"
#
# License:     GNU GENERAL PUBLIC LICENSE v.3, June 29, 2007
#-------------------------------



import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import logging
import json
from datetime import datetime
import os
import re
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(level=logging.INFO)

def generate_key():
    """Generate a key for encryption and save it to a file if it doesn't already exist."""
    key_path = 'secret.key'
    if not os.path.isfile(key_path):
        key = Fernet.generate_key()
        with open(key_path, 'wb') as key_file:
            key_file.write(key)
        logging.info("Encryption key generated and saved to %s", key_path)
    else:
        with open(key_path, 'rb') as key_file:
            key = key_file.read()
        logging.info("Encryption key loaded from %s", key_path)
    return key

# Load or generate encryption key
key = generate_key()
cipher_suite = Fernet(key)

def log_interaction(action, details):
    """Log interactions to a file named log.json with file protection and encryption."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details
    }

    log_file = 'log.json'

    try:
        # Encrypt the log entry
        log_entry_str = json.dumps(log_entry)
        encrypted_entry = cipher_suite.encrypt(log_entry_str.encode())

        with open(log_file, 'ab') as log_file:
            log_file.write(encrypted_entry + b'\n')  # Write a new line for each entry
    except Exception as e:
        logging.error("Failed to write to log file: %s", e)

def load_data(file_path):
    """Load the dataset from the given file path with error handling and security checks."""
    if not os.path.isfile(file_path):
        log_interaction("load_data", {"file_path": file_path, "status": "error", "message": "File not found"})
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    try:
        data = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")
        log_interaction("load_data", {"file_path": file_path, "status": "success"})
        return data
    except pd.errors.EmptyDataError:
        log_interaction("load_data", {"file_path": file_path, "status": "error", "message": "File is empty"})
        raise
    except pd.errors.ParserError:
        log_interaction("load_data", {"file_path": file_path, "status": "error", "message": "File parsing error"})
        raise

def preprocess_data(data):
    """Preprocess the data with validation and encoding."""
    if 'risultato' not in data.columns:
        raise ValueError("'risultato' column is missing from the dataset.")

    # Separate features and target
    X = data.drop(columns=["risultato"])
    y = data["risultato"]

    # Check for missing values
    if X.isnull().any().any():
        raise ValueError("Features contain missing values. Please handle missing values before training.")

    # Encode categorical variables
    label_encoders = {}
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le
            logging.info("Encoded column: %s", column)
            log_interaction("preprocess_data", {"action": "encode", "column": column})

    return X, y, label_encoders

def train_model(X, y):
    """Train a decision tree classifier with the given features and target."""
    model = DecisionTreeClassifier()
    model.fit(X, y)
    logging.info("Model trained successfully.")
    log_interaction("train_model", {"status": "success"})
    return model

def sanitize_input(user_input):
    """Sanitize user input to prevent code injection and ensure valid format."""
    pattern = re.compile(r'^[a-zA-Z0-9, ]+$')
    if not pattern.match(user_input):
        log_interaction("sanitize_input", {"user_input": user_input, "status": "error", "message": "Invalid characters in input"})
        raise ValueError("Invalid characters in input.")
    return user_input

def predict_match(model, input_data, label_encoders):
    """Make predictions for the given input data with validation."""
    input_df = pd.DataFrame([input_data], columns=['squadra_casa', 'squadra_trasferta'])

    # Encode the input data
    for column in input_df.columns:
        if column in label_encoders:
            if not input_df[column].isin(label_encoders[column].classes_).all():
                log_interaction("predict_match", {"input_data": input_data, "status": "error", "message": "Invalid team names"})
                raise ValueError("The entered teams are not present in the dataset. Check the team names.")
            input_df[column] = label_encoders[column].transform(input_df[column])
        else:
            raise ValueError(f"The column '{column}' is not present in the training dataset.")

    # Make predictions
    try:
        prediction = model.predict(input_df)
        logging.info("Prediction made successfully.")
        log_interaction("predict_match", {"input_data": input_data, "prediction": prediction[0]})
        return prediction[0]
    except ValueError as e:
        logging.error("Prediction error: %s", e)
        log_interaction("predict_match", {"input_data": input_data, "status": "error", "message": str(e)})
        raise

def decrypt_log():
    """Decrypt and display the log entries."""
    log_file_path = 'log.json'
    if not os.path.isfile(log_file_path):
        print(f"The log file {log_file_path} does not exist.")
        return

    key = generate_key()
    cipher_suite = Fernet(key)

    try:
        with open(log_file_path, 'rb') as log_file:
            for line in log_file:
                decrypted_entry = cipher_suite.decrypt(line.strip())
                log_entry = json.loads(decrypted_entry.decode())
                print(json.dumps(log_entry, indent=4))
    except Exception as e:
        print(f"An error occurred while decrypting the log file: {e}")

def main():
    # Load and preprocess data
    data = load_data("partite.csv")
    X, y, label_encoders = preprocess_data(data)

    # Train the model
    model = train_model(X, y)

    # Request user input with validation
    user_input = input("Enter the teams you want to analyze (first the home team, then the away team, separated by a comma, e.g., inter, milan): \n")

    # Sanitize and validate user input
    user_input = sanitize_input(user_input)

    # Basic validation
    if not user_input or ',' not in user_input:
        log_interaction("user_input", {"teams": user_input, "status": "error", "message": "Invalid input format"})
        raise ValueError("Input must contain exactly two teams separated by a comma.")

    teams = [team.strip() for team in user_input.split(',')]

    if len(teams) != 2:
        log_interaction("user_input", {"teams": user_input, "status": "error", "message": "Input must contain exactly two teams"})
        raise ValueError("Input must contain exactly two teams separated by a comma.")

    # Log user input
    log_interaction("user_input", {"teams": teams})

    # Predict and display results
    prediction = predict_match(model, teams, label_encoders)
    print("Prediction:")
    print(prediction)

if __name__ == "__main__":
    while True:
        choice = input("Choose an option:\n1. Make a prediction\n2. View logs\n3. Exit\n")
        if choice == '1':
            main()
        elif choice == '2':
            decrypt_log()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please choose 1, 2, or 3.")
