import tensorflow as tf
import tensorflow_hub as hub
import warnings
import numpy as np
import os
import cv2
from flask import Flask, request, render_template

warnings.filterwarnings('ignore')

# Load the trained model with error handling
try:
    model = tf.keras.models.load_model(filepath='rice.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    print("Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None  # Prevent crashes if the model fails to load

# Initialize Flask app
app = Flask(__name__)

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        f = request.files['image']
        if not f:
            return render_template('results.html', prediction_text="No image uploaded")

        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(filepath)

        # Process the image
        img = cv2.imread(filepath)  # Read image
        if img is None:
            return render_template('results.html', prediction_text="Invalid image format")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (224, 224))  # Resize to match model input
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Ensure model is loaded before prediction
        if model is None:
            return render_template('results.html', prediction_text="Model not loaded properly")

        # Predict
        pred = model.predict(img)
        pred_class = np.argmax(pred)

        # Label mapping
        labels = {0: 'Arborio', 1: 'Basmati', 2: 'Ipsala', 3: 'Jasmine', 4: 'Karacadag'}
        prediction = labels.get(pred_class, "Unknown")

        return render_template('results.html', prediction_text=f"Predicted Rice Type: {prediction}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
