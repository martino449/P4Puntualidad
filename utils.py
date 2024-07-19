import json
import logging
import os
import pandas as pd
from datetime import datetime
from cryptography.fernet import Fernet
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from config import LOG_FILE, KEY_FILE

# Configura il logging
logging.basicConfig(level=logging.INFO)

class Encryption:
    def __init__(self):
        self.key = self._load_or_generate_key()
        self.cipher_suite = Fernet(self.key)

    def _load_or_generate_key(self):
        if not os.path.isfile(KEY_FILE):
            key = Fernet.generate_key()
            with open(KEY_FILE, 'wb') as key_file:
                key_file.write(key)
            logging.info("Encryption key generated and saved to %s", KEY_FILE)
        else:
            with open(KEY_FILE, 'rb') as key_file:
                key = key_file.read()
            logging.info("Encryption key loaded from %s", KEY_FILE)
        return key

    def encrypt(self, data: str) -> bytes:
        return self.cipher_suite.encrypt(data.encode())

    def decrypt(self, data: bytes) -> str:
        return self.cipher_suite.decrypt(data).decode()


class Logger:
    def __init__(self):
        self.encryption = Encryption()

    def log_interaction(self, action: str, details: dict):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        try:
            with open(LOG_FILE, 'ab') as log_file:
                log_entry_str = json.dumps(log_entry)
                encrypted_entry = self.encryption.encrypt(log_entry_str)
                log_file.write(encrypted_entry + b'\n')
        except Exception as e:
            logging.error("Failed to write to log file: %s", e)

    def decrypt_log(self):
        if not os.path.isfile(LOG_FILE):
            print(f"The log file {LOG_FILE} does not exist.")
            return

        try:
            with open(LOG_FILE, 'rb') as log_file:
                for line in log_file:
                    decrypted_entry = self.encryption.decrypt(line.strip())
                    log_entry = json.loads(decrypted_entry)
                    print(json.dumps(log_entry, indent=4))
        except Exception as e:
            print(f"An error occurred while decrypting the log file: {e}")

class DataProcessor:
    def __init__(self):
        self.logger = Logger()

    def load_data(self, file_path: str):
        if not os.path.isfile(file_path):
            self.logger.log_interaction("load_data", {"file_path": file_path, "status": "error", "message": "File not found"})
            raise FileNotFoundError(f"The file '{file_path}' was not found.")

        try:
            data = pd.read_csv(file_path)
            logging.info("Dataset loaded successfully.")
            self.logger.log_interaction("load_data", {"file_path": file_path, "status": "success"})
            return data
        except pd.errors.EmptyDataError:
            self.logger.log_interaction("load_data", {"file_path": file_path, "status": "error", "message": "File is empty"})
            raise
        except pd.errors.ParserError:
            self.logger.log_interaction("load_data", {"file_path": file_path, "status": "error", "message": "File parsing error"})
            raise

    def preprocess_data(self, data):
        if 'risultato' not in data.columns:
            raise ValueError("'risultato' column is missing from the dataset.")

        X = data.drop(columns=["risultato"])
        y = data["risultato"]

        if X.isnull().any().any():
            raise ValueError("Features contain missing values. Please handle missing values before training.")

        label_encoders = {}
        for column in X.columns:
            if X[column].dtype == 'object':
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column])
                label_encoders[column] = le
                logging.info("Encoded column: %s", column)
                self.logger.log_interaction("preprocess_data", {"action": "encode", "column": column})

        return X, y, label_encoders

class UserInput:
    def __init__(self):
        self.logger = Logger()

    def sanitize_input(self, user_input: str) -> str:
        pattern = re.compile(r'^[a-zA-Z0-9, ]+$')
        if not pattern.match(user_input):
            self.logger.log_interaction("sanitize_input", {"user_input": user_input, "status": "error", "message": "Invalid characters in input"})
            raise ValueError("Invalid characters in input.")
        return user_input

    def parse_teams(self, user_input: str):
        if not user_input or ',' not in user_input:
            self.logger.log_interaction("user_input", {"teams": user_input, "status": "error", "message": "Invalid input format"})
            raise ValueError("Input must contain exactly two teams separated by a comma.")

        teams = [team.strip() for team in user_input.split(',')]

        if len(teams) != 2:
            self.logger.log_interaction("user_input", {"teams": user_input, "status": "error", "message": "Input must contain exactly two teams"})
            raise ValueError("Input must contain exactly two teams separated by a comma.")

        self.logger.log_interaction("user_input", {"teams": teams})
        return teams

class ModelTrainer:
    def __init__(self):
        self.logger = Logger()

    def train_model(self, X, y):
        model = RandomForestClassifier()
        model.fit(X, y)
        logging.info("Random Forest model trained successfully.")
        self.logger.log_interaction("train_model", {"status": "success"})
        return model

class Predictor:
    def __init__(self):
        self.logger = Logger()

    def predict_match(self, model, input_data, label_encoders):
        input_df = pd.DataFrame([input_data], columns=['squadra_casa', 'squadra_trasferta'])

        for column in input_df.columns:
            if column in label_encoders:
                if not input_df[column].isin(label_encoders[column].classes_).all():
                    self.logger.log_interaction("predict_match", {"input_data": input_data, "status": "error", "message": "Invalid team names"})
                    raise ValueError("The entered teams are not present in the dataset. Check the team names.")
                input_df[column] = label_encoders[column].transform(input_df[column])
            else:
                raise ValueError(f"The column '{column}' is not present in the training dataset.")

        try:
            prediction = model.predict(input_df)
            logging.info("Prediction made successfully.")
            self.logger.log_interaction("predict_match", {"input_data": input_data, "prediction": prediction[0]})
            return prediction[0]
        except ValueError as e:
            logging.error("Prediction error: %s", e)
            self.logger.log_interaction("predict_match", {"input_data": input_data, "status": "error", "message": str(e)})
            raise
