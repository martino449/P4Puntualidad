#-------------------------------
# Name:        P4puntualidad
# Purpose:     Train a decision tree classifier and make predictions
#
# Author:      Martin449
#
# Created on:  18/07/2024
# dependencies:
# python = ">=3.10.0,<3.12"
# pandas = "^2.2.2"
# scikit-learn = "^1.5.1"
#
# License:     GNU GENERAL PUBLIC LICENSE v.3, June 29, 2007
#-------------------------------


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load the dataset from the given file path."""
    try:
        data = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error("The file '%s' was not found.", file_path)
        raise
    except pd.errors.EmptyDataError:
        logging.error("The file '%s' is empty.", file_path)
        raise
    except pd.errors.ParserError:
        logging.error("Error parsing the file '%s'. Check the file format.", file_path)
        raise

def preprocess_data(data):
    """Preprocess the data: split into features and target, handle missing values, and encode categorical variables."""
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
    
    return X, y, label_encoders

def train_model(X, y):
    """Train a decision tree classifier with the given features and target."""
    model = DecisionTreeClassifier()
    model.fit(X, y)
    logging.info("Model trained successfully.")
    return model

def predict_match(model, input_data, label_encoders):
    """Make predictions for the given input data."""
    input_df = pd.DataFrame([input_data], columns=['squadra_casa', 'squadra_trasferta'])
    
    # Encode the input data
    for column in input_df.columns:
        if column in label_encoders:
            if not input_df[column].isin(label_encoders[column].classes_).all():
                raise ValueError("The entered teams are not present in the dataset. Check the team names.")
            input_df[column] = label_encoders[column].transform(input_df[column])
        else:
            raise ValueError(f"The column '{column}' is not present in the training dataset.")
    
    # Make predictions
    try:
        prediction = model.predict(input_df)
        logging.info("Prediction made successfully.")
        return prediction[0]
    except ValueError as e:
        logging.error("Prediction error: %s", e)
        raise

def main():
    # Load and preprocess data
    data = load_data("partite.csv")
    X, y, label_encoders = preprocess_data(data)
    
    # Train the model
    model = train_model(X, y)
    
    # Request user input
    user_input = input("Enter the teams you want to analyze (first the home team, then the away team, separated by a comma, e.g., inter, milan): \n")
    teams = user_input.split(', ')
    
    if len(teams) != 2:
        raise ValueError("Input must contain exactly two teams separated by a comma.")
    
    # Predict and display results
    prediction = predict_match(model, teams, label_encoders)
    print("Prediction:")
    print(prediction)

if __name__ == "__main__":
    main()
