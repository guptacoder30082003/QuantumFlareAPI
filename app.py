from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import requests
import json
from datetime import datetime
import cv2
import os
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        fl = weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    return focal_loss_fixed

image_model = None

def load_models():
    global image_model
    try:
        if os.path.exists('models/best_transfer_model.h5'):
            image_model = tf.keras.models.load_model('models/best_transfer_model.h5', 
                                                   custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.25)})
            print("Image model loaded successfully")
        else:
            print("Image model file not found")
            image_model = None
    except Exception as e:
        print(f"Error loading image model: {e}")
        image_model = None

def count_microplastics_opencv(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 50
    max_area = 5000
    
    particles = []
    microplastic_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(contour, True)
            
            x, y, w, h = cv2.boundingRect(contour)
            width = w
            height = h
            
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            
            aspect_ratio = float(w) / h if h != 0 else 0
            extent = float(area) / (w * h) if w * h != 0 else 0
            solidity = float(area) / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) != 0 else 0
            
            particle_data = {
                "area": float(area),
                "perimeter": float(perimeter),
                "width": float(width),
                "height": float(height),
                "equivalent_diameter": float(equivalent_diameter),
                "aspect_ratio": float(aspect_ratio),
                "extent": float(extent),
                "solidity": float(solidity)
            }
            
            particles.append(particle_data)
            microplastic_count += 1
    
    return microplastic_count, particles

def preprocess_image_with_opencv(base64_image):
    image_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Invalid image data")
    
    image = cv2.bilateralFilter(image, 9, 75, 75)
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    microplastic_count, particles = count_microplastics_opencv(original_rgb)
    
    image_resized = cv2.resize(image, (128, 128))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_gray = np.expand_dims(image_gray, axis=-1)
    image_rgb_converted = np.repeat(image_gray, 3, axis=-1)
    
    image_normalized = image_rgb_converted / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch, microplastic_count, particles

def generate_mock_photodiode_prediction(photodiode_data):
    data_array = np.array(photodiode_data)
    mean_value = np.mean(data_array)
    std_value = np.std(data_array)
    
    if mean_value > 0.5:
        mock_probs = [0.2, 0.3, 0.5]
    elif mean_value > 0.3:
        mock_probs = [0.4, 0.4, 0.2]
    else:
        mock_probs = [0.6, 0.3, 0.1]
    
    return np.array([mock_probs])

@app.route('/classify', methods=['POST'])
def classify_microplastic():
    try:
        if image_model is None:
            return jsonify({
                "status": "error", 
                "message": "Image model not loaded. Service unavailable."
            }), 503
        
        if not request.is_json:
            return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        required_fields = ['device_id', 'timestamp', 'photodiode_data', 'image_data']
        for field in required_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing required field: {field}"}), 400
        
        device_id = data['device_id']
        timestamp = data['timestamp']
        photodiode_data = data['photodiode_data']
        image_data = data['image_data']
        
        if not isinstance(photodiode_data, list) or len(photodiode_data) == 0:
            return jsonify({"status": "error", "message": "photodiode_data must be a non-empty list"}), 400
        
        processed_image, microplastic_count, particles = preprocess_image_with_opencv(image_data)
        image_prediction = image_model.predict(processed_image, verbose=0)
        
        photodiode_prediction = generate_mock_photodiode_prediction(photodiode_data)
        
        combined_prediction = (0.7 * image_prediction) + (0.3 * photodiode_prediction)
        
        predicted_class = int(np.argmax(combined_prediction[0]))
        confidence_score = float(np.max(combined_prediction[0]))
        
        result = {
            "device_id": device_id,
            "timestamp": timestamp,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z",
            "microplastic_count": microplastic_count,
            "particles": particles,
            "image_prediction": {
                "predicted_class": int(np.argmax(image_prediction[0])),
                "confidence": float(np.max(image_prediction[0])),
                "probabilities": image_prediction[0].tolist()
            },
            "photodiode_prediction": {
                "predicted_class": int(np.argmax(photodiode_prediction[0])),
                "confidence": float(np.max(photodiode_prediction[0])),
                "probabilities": photodiode_prediction[0].tolist()
            },
            "combined_prediction": {
                "predicted_class": predicted_class,
                "confidence": confidence_score,
                "probabilities": combined_prediction[0].tolist()
            }
        }
        
        try:
            response = requests.post('http://www.example.com/api/results', 
                                   json=result,
                                   headers={'Content-Type': 'application/json'},
                                   timeout=10)
            
            if response.status_code == 200:
                return jsonify({"status": "success", "message": "Classification completed and sent"})
            else:
                return jsonify({"status": "partial_success", 
                              "message": "Classification completed but failed to send to example.com",
                              "result": result})
        except requests.exceptions.RequestException:
            return jsonify({"status": "partial_success", 
                          "message": "Classification completed but failed to send to example.com",
                          "result": result})
        
    except ValueError as e:
        return jsonify({"status": "error", "message": f"Invalid data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    models_loaded = image_model is not None
    return jsonify({
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "service": "microplastic-classifier"
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "service": "Microplastic Classification API",
        "version": "1.0.0",
        "endpoints": {
            "POST /classify": "Main classification endpoint",
            "GET /health": "Health check endpoint",
            "GET /": "API information"
        }
    })

if __name__ == '__main__':
    load_models()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    load_models()