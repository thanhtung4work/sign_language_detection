import argparse
import os

from sklearn.model_selection import train_test_split

from utils import train_model, evaluate_model, load_data, save_model

def main(data_path, output_file, test_size, num_tree):
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
    model = train_model(x_train, y_train, num_tree)

    # Evaluate the model
    accuracy = evaluate_model(model, x_test, y_test)
    print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

    # Save the model
    save_model(model, output_file)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a Random Forest classifier on hand landmark data.")
    parser.add_argument('--data_path', type=str, default='./outputs/data.pickle', help='Path to the input data pickle file.')
    parser.add_argument('--output_file', type=str, default='model.pickle', help='Path to save the trained model.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing (default: 0.2).')
    parser.add_argument('--num_tree', type=int, default=100, help='Number of tree in the forest.')

    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args.data_path, args.output_file, args.test_size, args.num_tree)
