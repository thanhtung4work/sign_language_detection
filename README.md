# Japanese Hand Gesture Recognition with Flask and Mediapipe

This project implements a hand gesture recognition system using Flask for the backend and Mediapipe for hand landmark detection. The frontend captures video from the user's webcam, sends frames to the backend for processing, and receives and visualizes hand landmarks and predictions based on a trained model.

## Table of contents

1. Project Structure
2. Installation
3. Usage

## Project Structure

```text
|- collect_data.py
|- inference.py
|- preprocessing.py
|- train.py
|- jp_hands/
|   |- __init__.py
|   |- model.py
|   |- templates/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
flask --app jp_hands run
```