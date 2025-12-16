from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
cnn_model = load_model("cnn_disaster_damage_resnet50.keras")
lstm_model = load_model("lstm_risk_model.keras")

class_names = ['High', 'Low', 'Medium']
risk_map = {0: 1.0, 1: 0.2, 2: 0.6}

def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def allocate_resources(predicted_risk):
    if predicted_risk >= 0.8:
        return "Severe: Deploy NDRF, Medical Teams, Emergency Shelters"
    elif predicted_risk >= 0.5:
        return "Moderate: Deploy Medical Teams & Relief Supplies"
    else:
        return "Low: Monitoring & Early Warning"

# Landing Page
@app.route("/")
def index():
    return render_template("index.html")

# Prediction Page
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    processed_img = process_image(filepath)
    probs = cnn_model.predict(processed_img)
    pred_class = np.argmax(probs)
    risk_score = risk_map[pred_class]

    seq = np.array([[risk_score]*3]).reshape(1, 3, 1)
    future_risk = lstm_model.predict(seq)[0][0]

    result = {
        "current_class": class_names[pred_class],
        "current_risk": round(risk_score, 2),
        "future_risk": round(float(future_risk), 2),
        "decision": allocate_resources(future_risk)
    }

    return render_template("home.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
