from datetime import datetime
import os
import json
import sys
sys.path.append('.')

import cv2

from src import paths
from src.image_object_visualizer import ImageObjectVisualizer
from src.prediction_image import PredictionImage
from src.prediction_image_detection import PredictionImageDetection

if __name__ == "__main__":
    try:
        prediction = sys.argv[1]
        image_array = cv2.imread(os.path.join(paths.PREDICTIONS, prediction, "image.png"))
        if image_array is None:
            raise FileNotFoundError()
        with open(os.path.join(paths.PREDICTIONS, prediction, "detections.json"), 'r') as f:
            raw_detections = json.load(f)
    except:
        print("USAGE: pipenv run python scripts/visualize_prediction.py <timestamp>")
        exit(1)

    print("Visualisation beginning: " + datetime.now().isoformat())

    detections = [PredictionImageDetection(d["x1"], d["y1"], d["x2"], d["y2"], d["label"], d["score"]) for d in raw_detections]
    image = PredictionImage(image_array, image_array.shape[1], image_array.shape[0], detections)
    scores = [d.score for d in image.detections]

    with open(paths.LABEL_MAP, 'r') as f:
        label_map = json.load(f)

    visualizer = ImageObjectVisualizer([image], [detections], [scores], label_map)
    visualizer.visualize()

    print("Visualisation complete: " + datetime.now().isoformat())

