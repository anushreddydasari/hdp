import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

class HeartDiseasePredictor:
    def __init__(self, model_path, scaler_path):
        # Load the trained model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def get_user_input(self, input_data):
        """
        Collects the user input in the form of a dictionary and returns it as a pandas DataFrame.
        """
        try:
            # Convert the input data to a DataFrame
            data = {
                "age": float(input_data["Age"]),
                "sex": int(input_data["Sex"]),
                "cp": int(input_data["Chest Pain Type"]),
                "trestbps": float(input_data["Resting BP"]),
                "chol": float(input_data["Cholesterol"]),
                "fbs": int(input_data["Fasting Blood Sugar"]),
                "restecg": int(input_data["Resting ECG"]),
                "thalach": float(input_data["Max Heart Rate"]),
                "exang": int(input_data["Exercise Induced Angina"]),
                "oldpeak": float(input_data["Oldpeak"]),
                "slope": int(input_data["Slope"]),
                "ca": int(input_data["CA"]),
                "thal": int(input_data["Thal"])
            }

            # Convert to DataFrame for prediction
            return pd.DataFrame([data])

        except ValueError:
            # Handle invalid input (e.g., non-numeric values)
            print("Error: Please enter valid numeric values.")
            return None

    def preprocess_data(self, new_data):
        """
        Scales the user input data using the pre-fitted scaler.
        """
        return self.scaler.transform(new_data)

    def predict(self, input_data):
        """
        Makes the prediction based on user input and returns the result.
        """
        new_data = self.get_user_input(input_data)

        if new_data is not None:
            # Preprocess the input data (scaling)
            scaled_data = self.preprocess_data(new_data)

            # Make prediction
            prediction = self.model.predict(scaled_data)[0]

            # Return the result
            if prediction == 1:
                return "The model predicts: Heart Disease (1)"
            else:
                return "The model predicts: No Heart Disease (0)"
        else:
            return "Invalid input data."

# Main function to run the predictor
def main():
    # Load paths to your model and scaler
    model_path = 'random_forest_model.pkl'
    scaler_path = 'scaler.pkl'

    # Create the HeartDiseasePredictor instance
    predictor = HeartDiseasePredictor(model_path, scaler_path)

    # Example input data (replace this with user input in practice)
    input_data = {
        "Age": 63,
        "Sex": 1,
        "Chest Pain Type": 2,
        "Resting BP": 166,
        "Cholesterol": 290,
        "Fasting Blood Sugar": 0,
        "Resting ECG": 2,
        "Max Heart Rate": 160,
        "Exercise Induced Angina": 2,
        "Oldpeak": 1.3,
        "Slope": 0,
        "CA": 0,
        "Thal": 1
    }

    # Make the prediction
    result = predictor.predict(input_data)

    # Output the result
    print(result)

if __name__ == "__main__":
    main()
