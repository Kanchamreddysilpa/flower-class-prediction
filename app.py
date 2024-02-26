import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)

# Load the model
try:
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    print("Model file not found. Make sure you have the 'model.pkl' file in the same directory as this script.")
    # Exit or handle the missing file appropriately
    exit()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extracting input features from the form
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        
        # Make prediction
        prediction = model.predict(features)
        
        # Convert numpy objects to native Python types for compatibility with jsonify/render_template
        prediction = prediction.tolist()

        # Display the result
        return render_template("index.html", prediction_text=f"The flower species is {prediction}")
    except Exception as e:
        # Handle errors in processing or in prediction
        print(f"An error occurred: {e}")
        return render_template("index.html", prediction_text="An error occurred during prediction.")

if __name__ == "__main__":
    app.run(debug=True)
