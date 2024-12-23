import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

class HeartDiseasePredictor:
    def __init__(self, model_path, scaler_path):
        # Load the trained model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def get_user_input(self):
        """
        Collects the user input interactively and returns it as a pandas DataFrame.
        """
        try:
            # Get user input for each parameter
            age = float(input("Enter Age: "))
            sex = int(input("Enter Sex (1 for Male, 0 for Female): "))
            cp = int(input("Enter Chest Pain Type (0-3): "))
            trestbps = float(input("Enter Resting Blood Pressure (mm Hg): "))
            chol = float(input("Enter Cholesterol (mg/dl): "))
            fbs = int(input("Enter Fasting Blood Sugar (1 if > 120 mg/dl, else 0): "))
            restecg = int(input("Enter Resting ECG (0-2): "))
            thalach = float(input("Enter Max Heart Rate (bpm): "))
            exang = int(input("Enter Exercise Induced Angina (1 for Yes, 0 for No): "))
            oldpeak = float(input("Enter Oldpeak (Depression in ST segment): "))
            slope = int(input("Enter Slope (0-2): "))
            ca = int(input("Enter Number of Major Vessels (0-3): "))
            thal = int(input("Enter Thalassemia (3 for Normal, 6 or 7 for Abnormal): "))

            # Convert the user input into a dictionary
            data = {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
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

    def predict(self):
        """
        Makes the prediction based on user input and returns the result.
        """
        new_data = self.get_user_input()

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

    # Make the prediction
    result = predictor.predict()

    # Output the result
    print(result)

if __name__ == "__main__":
    main()
