import os
import cv2
import mediapipe as mp
import pickle

from utils import process_images


if __name__ == '__main__':
    DATA_DIR = './data'  # Directory containing the images
    OUTPUT_FILE = 'data.pickle'  # Output pickle file

    process_images(DATA_DIR, OUTPUT_FILE)
