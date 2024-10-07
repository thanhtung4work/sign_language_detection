import os
import cv2
import mediapipe as mp
import pickle
import argparse

def process_image(img_path, hands):
    """Process an image to extract hand landmarks.

    Args:
        img_path (str): Path to the image file.
        hands (mp.solutions.hands.Hands): Mediapipe Hands object for hand detection.

    Returns:
        list: List of hand landmarks relative to the minimum x and y coordinates, or an empty list if no hands detected.
    """
    data_aux = []
    x_ = []
    y_ = []

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
    return data_aux

def collect_hand_landmarks(data_dir, hands):
    """Collect hand landmarks from all images in the dataset.

    Args:
        data_dir (str): The directory containing images classified into subdirectories.
        hands (mp.solutions.hands.Hands): Mediapipe Hands object for hand detection.

    Returns:
        tuple: A tuple containing the list of extracted data and labels for all images.
    """
    data = []
    labels = []

    for dir_ in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_)
        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            data_aux = process_image(img_full_path, hands)

            if data_aux:
                data.append(data_aux)
                labels.append(dir_)

    return data, labels

def save_to_pickle(data, labels, output_file):
    """Save data and labels to a pickle file.

    Args:
        data (list): The list of data points to save.
        labels (list): The list of labels corresponding to the data points.
        output_file (str): Path to the output pickle file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

def main(data_dir, output_file):
    """Main function to extract hand landmarks from images and save them to a file.

    Args:
        data_dir (str): Directory where the dataset is stored.
        output_file (str): Path to the output pickle file.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)

    data, labels = collect_hand_landmarks(data_dir, hands)

    save_to_pickle(data, labels, output_file)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract hand landmarks from images using Mediapipe and save to a pickle file.")
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory where the dataset is stored.')
    parser.add_argument('--output_file', type=str, default='data.pickle', help='Path to save the output pickle file.')

    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args.data_dir, args.output_file)
