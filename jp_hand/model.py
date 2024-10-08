import base64
import functools
from io import BytesIO

import cv2
from flask import (
    Blueprint, request, jsonify, render_template
)
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image

bp = Blueprint('model', __name__, url_prefix='/')

# Load model and mediapipe configuration
model_dict = pickle.load(open('./outputs/model.pickle', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)

labels_dict = {0: 'A', 1: 'I', 2: 'U', 3: 'E', 4: 'O'}

def decode_base64_image(base64_string):
    """Decodes a base64 string and converts it into an OpenCV image."""
    img_data = base64.b64decode(base64_string.split(',')[1])  # Remove the data URL prefix
    img = Image.open(BytesIO(img_data))
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def process_frame(frame):
    """Process the image frame and predict hand gestures."""
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        if len(data_aux) == 42:  # Ensure correct number of landmarks
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            return predicted_character

    return None


@bp.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('base.html')
    
    data = request.json
    image_data = data['image']
    
    # Convert base64 image to OpenCV image
    frame = decode_base64_image(image_data)

    # Process the frame to get the hand gesture prediction
    predicted_character = process_frame(frame)

    if predicted_character:
        return jsonify({'prediction': predicted_character})
    else:
        return jsonify({'prediction': 'No hand detected'})