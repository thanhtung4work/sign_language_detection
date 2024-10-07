import pickle
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


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

def process_frame(frame, hands, model, labels_dict):
    """Process a video frame to detect hand landmarks and predict the corresponding character.

    Args:
        frame (numpy.ndarray): The current frame from the video feed.
        hands (mp.solutions.hands.Hands): Mediapipe Hands object for hand detection.
        model (object): The trained classifier model for predictions.
        labels_dict (dict): Dictionary mapping label indices to characters.

    Returns:
        numpy.ndarray: The processed frame with predictions drawn on it.
    """
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if len(data_aux) == 42:  # Ensure correct number of landmarks
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Draw the rectangle and prediction on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    return frame

def main():
    """Main function to run the real-time hand gesture recognition."""
    model_path = './model.pickle'
    model = load_model(model_path)

    labels_dict = {0: 'A', 1: 'I', 2: 'U', 3: 'E', 4: 'O'}

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for hand landmarks and predictions
        processed_frame = process_frame(frame, hands, model, labels_dict)

        # Display the frame
        cv2.imshow('Hand Gesture Recognition', processed_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
