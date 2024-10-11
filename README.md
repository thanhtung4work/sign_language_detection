# Japanese Hand Gesture Recognition with Flask and Mediapipe

This project implements a hand gesture recognition system using Flask for the backend and Mediapipe for hand landmark detection. The frontend captures video from the user's webcam, sends frames to the backend for processing, and receives and visualizes hand landmarks and predictions based on a trained model.

## Table of contents

1. Project Structure
2. Installation
3. Usage

## Project Structure

```text
|- jp_hands/
|   |- __init__.py
|   |- model.py
|   |- templates/
|- utils/
|-  |- collect_data.py
|-  |- inference.py
|-  |- preprocess.py
|-  |- train.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run web app

```bash
flask --app jp_hands run
```

2. Collect data

```bash
python utils/collect_data.py --data_dir "save image path goes here" --num_classes "num classes goes here" --dataset_size "dataset size goes here"
```

3. Preprocess data

```bash
python utils/preprocessing.py
```

4. Train model

```bash
python utils/train.py  --data_path "data_path goes here" --output_path "output path goes here" --test_size "test size goes here"
```
