import os
import cv2
import mediapipe as mp
import pickle

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
    
    print(f"Data preprocessing complete. Processed data saved to {output_file}.")

if __name__ == '__main__':
    DATA_DIR = './data'  # Directory containing the images
    OUTPUT_FILE = 'data.pickle'  # Output pickle file

    process_images(DATA_DIR, OUTPUT_FILE)
