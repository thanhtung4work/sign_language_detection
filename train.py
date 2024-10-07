import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import argparse

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

def train_model(x_train, y_train):
    """Train a Random Forest classifier.

    Args:
        x_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.

    Returns:
        RandomForestClassifier: The trained model.
    """
    model = RandomForestClassifier()
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
    with open(output_file, 'wb') as f:
        pickle.dump({'model': model}, f)

def main(data_path, output_file, test_size):
    """Main function to load data, train a model, evaluate, and save the model.

    Args:
        data_path (str): Path to the input data pickle file.
        output_file (str): Path to save the trained model.
        test_size (float): Proportion of the dataset to include in the test split.
    """
    data, labels = load_data(data_path)
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, shuffle=True, stratify=labels
    )

    # Train the model
    model = train_model(x_train, y_train)

    # Evaluate the model
    accuracy = evaluate_model(model, x_test, y_test)
    print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

    # Save the model
    save_model(model, output_file)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a Random Forest classifier on hand landmark data.")
    parser.add_argument('--data_path', type=str, default='./data.pickle', help='Path to the input data pickle file.')
    parser.add_argument('--output_file', type=str, default='model.p', help='Path to save the trained model.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing (default: 0.2).')

    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args.data_path, args.output_file, args.test_size)
