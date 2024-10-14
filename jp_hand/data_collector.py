import base64
import os
import glob
import shutil

from flask import (
    Blueprint, request, jsonify, render_template
)

bp = Blueprint('collect', __name__, url_prefix='/')

# Base directory to store images
BASE_IMAGE_PATH = './data'
os.makedirs(BASE_IMAGE_PATH, exist_ok=True)


@bp.route('/collect', methods=['POST', 'GET'])
def collect_data():
    """Endpoint to receive image data from frontend."""
    if request.method == 'GET':
        return render_template('collect/index.html')
    
    # for d in os.listdir(BASE_IMAGE_PATH):
    #     shutil.rmtree(os.path.join(BASE_IMAGE_PATH, d))

    data = request.json
    image_data = data['image']
    class_name = data['className']
    index = data['index']

    # Create a directory for the class if it doesn't exist
    class_path = os.path.join(BASE_IMAGE_PATH, class_name)
    os.makedirs(class_path, exist_ok=True)

    # Decode the image from base64 and save it
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image_path = os.path.join(class_path, f'image_{index}.jpg')

    with open(image_path, 'wb') as f:
        f.write(image_bytes)

    return jsonify({'status': 'success', 'message': f'Image {index} saved for class {class_name}.'})