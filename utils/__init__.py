import os

import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def train_model(x_train, y_train, num_tree):
    """Train a Random Forest classifier.

    Args:
        x_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.

    Returns:
        RandomForestClassifier: The trained model.
    """
    model = RandomForestClassifier(n_estimators=num_tree)
    model.fit(x_train, y_train)
    return model


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


def load_model(model_path):
    """Load the trained model from a pickle file.

    Args:
        model_path (str): Path to the model pickle file.

    Returns:
        object: The loaded model object.
    """
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict['model']


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


def process_images(data_dir, output_file):
    """Process images using MediaPipe Hands and save the processed data and labels to a pickle file.

    Args:
        data_dir (str): Directory where the images are stored.
        output_file (str): Path to the output pickle file.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

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

            img = cv2.imread(os.path.join(data_dir, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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