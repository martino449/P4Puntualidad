from utils import DataProcessor, ModelTrainer, UserInput, Predictor, Logger
from config import DATA_FILE

def main():
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    user_input_handler = UserInput()
    predictor = Predictor()

    # Load and preprocess data
    data = data_processor.load_data(DATA_FILE)
    X, y, label_encoders = data_processor.preprocess_data(data)

    # Train the model
    model = model_trainer.train_model(X, y)

    # Request user input with validation
    user_input = input("Enter the teams you want to analyze (first the home team, then the away team, separated by a comma, e.g., inter, milan): \n")

    # Sanitize and validate user input
    sanitized_input = user_input_handler.sanitize_input(user_input)
    teams = user_input_handler.parse_teams(sanitized_input)

    # Predict and display results
    prediction = predictor.predict_match(model, teams, label_encoders)
    print("Prediction:")
    print(prediction)

if __name__ == "__main__":
    while True:
        choice = input("Choose an option:\n1. Make a prediction\n2. View logs\n3. Exit\n")
        if choice == '1':
            main()
        elif choice == '2':
            Logger().decrypt_log()  # Ensure Logger is imported correctly
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please choose 1, 2, or 3.")
