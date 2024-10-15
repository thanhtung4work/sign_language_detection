import argparse
import json
import os

import cv2

def create_directory(directory):
    """Create a directory if it does not exist.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def collect_data_for_class(class_id, dataset_size, data_dir, cap):
    """Collects a dataset of images for a specific class.

    Args:
        class_id (int): The ID of the class for which data is being collected.
        dataset_size (int): The number of images to collect.
        data_dir (str): The base directory where the dataset is stored.
        cap (cv2.VideoCapture): The video capture object for accessing the webcam.
    """
    class_dir = os.path.join(data_dir, str(class_id))
    create_directory(class_dir)

    print(f'Collecting data for class {class_id}')

    # Wait for user to press 'q' before starting collection
    while True:
        ret, frame = cap.read()
        cv2.putText(
            frame, 'Ready? Press "Q" !', 
            (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA
        )
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

def main(data_dir, number_of_classes, dataset_size):
    """Main function to handle data collection for multiple classes.

    Args:
        data_dir (str): Directory where the dataset will be stored.
        number_of_classes (int): The number of classes to collect data for.
        dataset_size (int): The number of images to collect per class.
    """
    create_directory(data_dir)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    for class_id in range(number_of_classes):
        collect_data_for_class(class_id, dataset_size, data_dir, cap)

    cap.release()
    cv2.destroyAllWindows()

    vocab = {}
    for i in range(number_of_classes):
        label = input(f"Label for class {i}: ")
        vocab[i] = label
    
    print(vocab)

    with open("data/labels.json", "w") as outfile:
        json.dump(vocab, outfile)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Collect dataset for multiple classes using webcam.")
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save the dataset.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes to collect data for.')
    parser.add_argument('--dataset_size', type=int, default=200, help='Number of images to collect per class.')
    
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args.data_dir, args.num_classes, args.dataset_size)
