import base64
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

bp = Blueprint('model', __name__, url_prefix='/')

# Base directory to store images
BASE_IMAGE_PATH = './user_data'


os.makedirs(BASE_IMAGE_PATH, exist_ok=True)

# Load model and mediapipe configuration
model_dict = pickle.load(open('./outputs/model.pickle', 'rb'))
model = model_dict['model']

# Mediapipe hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8, max_num_hands=1)

labels_dict = {}
with open('data/labels.json', 'r') as file:
    labels_dict = json.load(file)

def decode_base64_image(base64_string):
    """Decodes a base64 string and converts it into an OpenCV image."""
    img_data = base64.b64decode(base64_string.split(',')[1])  # Remove the data URL prefix
    img = Image.open(BytesIO(img_data))
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

DATA_DIR = './data'

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
    if request.method == 'GET':
        # Load model and mediapipe configuration
        model_dict = pickle.load(open('./outputs/model.pickle', 'rb'))
        model = model_dict['model']
        return render_template('predict/index.html')
    
    data = request.json
    image_data = data['image']
    
    # Convert base64 image to OpenCV image
    frame = decode_base64_image(image_data)

    # Process the frame to get the hand gesture prediction and landmarks
    predicted_character, hand_landmarks_list = process_frame(frame)

    if predicted_character:
        return jsonify({'prediction': predicted_character, 'landmarks': hand_landmarks_list})
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

    data = []
    labels = []

    for dir_ in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_)
        if not os.path.isdir(dir_path):
            continue
        
        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            try:
                img = cv2.imread(os.path.join(data_dir, dir_, img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as err:
                print("error loading image")
                continue

            # Process the image using MediaPipe
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    # Normalize landmarks by min(x_) and min(y_)
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Append valid data and corresponding label
                data.append(data_aux)
                labels.append(dir_)
            else:
                # Skip this image if no hand landmarks are detected
                print(f"Skipping image {img_path} in class {dir_}: No hand landmarks detected.")

    # Save the data and labels to a pickle file
    if not os.path.exists("./outputs"):
        os.mkdir("outputs")
    output_path = os.path.join("outputs", output_file)
    with open(output_path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    
    return jsonify({"message": f"Data preprocessing complete. Processed data saved to {output_file}."})


@bp.route("/start-training", methods=['POST'])
def start_training():
    def load_data(data_path):
        """Load dataset from a pickle file.

        Args:
            data_path (str): Path to the pickle file containing the data and labels.

        Returns:
            tuple: A tuple containing data and labels as numpy arrays.
        """
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])
        return data, labels
    
    def train_model(x_train, y_train, num_tree):
        """Train a` Random Forest classifier.

        Args:
            x_train (numpy.ndarray): Training data.
            y_train (numpy.ndarray): Training labels.

        Returns:
            RandomForestClassifier: The trained model.
        """
        model = RandomForestClassifier(n_estimators=num_tree)
        model.fit(x_train, y_train)
        return model

    def evaluate_model(model, x_test, y_test):
        """Evaluate the trained model using accuracy score.

        Args:
            model (RandomForestClassifier): The trained model.
            x_test (numpy.ndarray): Test data.
            y_test (numpy.ndarray): Test labels.

        Returns:
            float: The accuracy score of the model on the test data.
        """
        y_predict = model.predict(x_test)
        score = accuracy_score(y_predict, y_test)
        return score

    def save_model(model, output_file):
        """Save the trained model to a pickle file.

        Args:
            model (RandomForestClassifier): The trained model.
            output_file (str): Path to save the model pickle file.
        """
        if not os.path.exists("./outputs"):
            os.mkdir("outputs")
        output_path = os.path.join("outputs", output_file)
        with open(output_path, 'wb') as f:
            pickle.dump({'model': model}, f)
    
    
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