"""
Performs continuous object detection with the camera.

Simply run the script and it will draw boxes around detected objects along 
with the predicted labels:

"""

from aiymakerkit import vision
from aiymakerkit import utils
from pycoral.utils.dataset import read_label_file

import os.path

def path(name):
    root = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(root, 'models', name)

# Model
ROAD_SIGN_DETECTION_MODEL = path('efficientdet-lite-fullmodelv2.tflite')
ROAD_SIGN_DETECTION_MODEL_EDGETPU = path('efficientdet-lite-fullmodelv2_edgetpu.tflite')

# Labels
ROAD_SIGN_DETECTION_LABELS = path('fullmodelv2-labels.txt')

detector = vision.Detector(ROAD_SIGN_DETECTION_MODEL)
labels = read_label_file(ROAD_SIGN_DETECTION_LABELS)
for frame in vision.get_frames():
    objects = detector.get_objects(frame, threshold=0.4)
    vision.draw_objects(frame, objects, labels)