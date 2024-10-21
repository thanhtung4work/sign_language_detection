import base64
from collections import deque
import functools
from io import BytesIO
import json
import os
import shutil

import cv2
from flask import (
    Blueprint, request, jsonify, render_template
)
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import train_model, process_images, evaluate_model, load_data, save_model

bp = Blueprint('model', __name__, url_prefix='/')

# CONSTANTS
DATA_DIR = './data'

# Load model and mediapipe configuration
model_dict = pickle.load(open('./outputs/model.pickle', 'rb'))
model = model_dict['model']

# Mediapipe hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8, max_num_hands=1)

# Label
labels_dict = {}
try:
    with open('data/labels.json', 'r') as file:
        labels_dict = json.load(file)
except:
    labels_dict = {}

# To store predictions and smooth the results over multiple frames
prediction_history = deque(maxlen=20)  # Store last 20 predictions
threshold = 10  # Display prediction if it appears this many times consecutively

sentence = []
last_appended_char = None


def decode_base64_image(base64_string):
    """Decodes a base64 string and converts it into an OpenCV image."""
    img_data = base64.b64decode(base64_string.split(',')[1])  # Remove the data URL prefix
    img = Image.open(BytesIO(img_data))
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def create_directory(directory):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_frame(frame):
    """Process the image frame and predict hand gestures. Also return landmarks."""
    data_aux = []
    x_ = []
    y_ = []
    hand_landmarks_list = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_points = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                hand_points.append({'x': x, 'y': y})  # Collect x, y coordinates for frontend

            hand_landmarks_list.append(hand_points)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        if len(data_aux) == 42:  # Ensure correct number of landmarks
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[str(prediction[0])]
            return predicted_character, hand_landmarks_list

    return None, []


@bp.route('/', methods=['GET', 'POST'])
def predict():
    global prediction_history
    global threshold
    global sentence
    global last_appended_char
    global model
    global labels_dict


    # GET Requests
    if request.method == 'GET':
        # Load model and mediapipe configuration
        model_dict = pickle.load(open('./outputs/model.pickle', 'rb'))
        
        model = model_dict['model']

        sentence = []
        prediction_history.clear()
        last_appended_char = None
        
        try:
            with open('data/labels.json', 'r') as file:
                labels_dict = json.load(file)
        except:
            labels_dict = {}
        
        return render_template('predict/index.html')
    
    # POST Requests
    data = request.json
    image_data = data['image']
    
    # Convert base64 image to OpenCV image
    frame = decode_base64_image(image_data)

    # Process the frame to get the hand gesture prediction and landmarks
    predicted_character, hand_landmarks_list = process_frame(frame)

    # Build sentence
    if predicted_character:
        prediction_history.append(predicted_character)
        most_common_prediction = max(set(prediction_history), key=prediction_history.count)
        if most_common_prediction != last_appended_char and prediction_history.count(most_common_prediction) >= threshold:
            sentence.append(most_common_prediction)
            last_appended_char = most_common_prediction


    if predicted_character:
        return jsonify({
            'prediction': predicted_character, 'landmarks': hand_landmarks_list,
            'sentence': ''.join(sentence)
        })
    else:
        return jsonify({'prediction': 'No hand detected', 'landmarks': []})
    

@bp.route('/collect', methods=['GET'])
def collect():
    if request.method == 'GET':
        return render_template('collect/index.html')


@bp.route('/clear-data', methods=['POST'])
def clear_data():
    """Clear the old data from the data directory."""
    try:
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)  # Remove the entire data directory
        create_directory(DATA_DIR)  # Recreate an empty data directory
        return jsonify({'message': 'Old data cleared successfully.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/start-collection', methods=['POST'])
def start_collection():
    """Create directories for the specified class."""
    class_id = request.json['class_id']
    class_dir = os.path.join(DATA_DIR, str(class_id))
    create_directory(class_dir)
    return jsonify({'message': f'Directory created for class {class_id}'})


@bp.route('/upload-image', methods=['POST'])
def upload_image():
    """Save captured images to the specified class directory."""
    data = request.json
    class_id = data['class_id']
    img_index = data['img_index']
    img_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64 prefix
    class_dir = os.path.join(DATA_DIR, str(class_id))

    # Decode the image from base64 and save it
    image_bytes = base64.b64decode(img_data)
    img_path = os.path.join(class_dir, f'{img_index}.jpg')
    with open(img_path, 'wb') as f:
        f.write(image_bytes)

    return jsonify({'message': f'Image {img_index} saved to class {class_id}'})


@bp.route('/save-labels', methods=['POST'])
def save_labels():
    """Save class labels to a JSON file."""
    labels = request.json['labels']
    with open(os.path.join(DATA_DIR, 'labels.json'), 'w') as f:
        json.dump(labels, f)
    return jsonify({'message': 'Labels saved to labels.json'})


@bp.route('/start-preprocessing', methods=['POST'])
def start_preprocessing():
    """Preprocess on collected data."""
    data_dir = DATA_DIR
    output_file = "data.pickle"

    process_images(data_dir, output_file)
    
    return jsonify({"message": f"Data preprocessing complete. Processed data saved to {output_file}."})


@bp.route("/start-training", methods=['POST'])
def start_training():
    data_path = "./outputs/data.pickle"
    test_size = .2
    num_tree = 400
    output_file = "model.pickle"
    data, labels = load_data(data_path)
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, shuffle=True, stratify=labels
    )

    # Train the model
    model = train_model(x_train, y_train, num_tree)

    # Evaluate the model
    accuracy = evaluate_model(model, x_test, y_test)
    print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

    # Save the model
    save_model(model, output_file)

    return jsonify({"message": f'{accuracy * 100:.2f}% of samples were classified correctly!'})