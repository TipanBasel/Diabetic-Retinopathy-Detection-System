from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define class labels (Modify based on your dataset)
CLASS_LABELS = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]

# Load ResNet-101 model
RESNET_MODEL_PATH = "models/resnet101_optimized.h5"

try:
    resnet_model = load_model(RESNET_MODEL_PATH)
    print("ResNet-101 model loaded successfully!")
except Exception as e:
    print(f"Error loading ResNet-101 model: {e}")
    resnet_model = None

# Set up upload folder
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the uploaded image
def preprocess_image(image_path, img_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))  # Resize based on model input size
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to classify an image using ResNet-101
def classify_image(image_path):
    resnet_prediction = "Error"

    # ResNet Prediction (224x224)
    if resnet_model:
        try:
            resnet_image = preprocess_image(image_path, 224)
            resnet_predictions = resnet_model.predict(resnet_image)
            resnet_predicted_class = np.argmax(resnet_predictions, axis=1)[0]
            resnet_prediction = CLASS_LABELS[resnet_predicted_class]
        except Exception as e:
            print(f"ResNet Prediction Error: {e}")

    return {
        "resnet_prediction": resnet_prediction
    }

@app.route("/")
def index():
    return render_template("index.html")  # Create an index.html file for the UI

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Get prediction from ResNet model
        predictions = classify_image(file_path)

        return jsonify({
            "resnet_prediction": predictions["resnet_prediction"],
            "image_url": f"/{file_path}"
        })

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    app.run(debug=True)
