from datetime import datetime
import os
import json
import sys
sys.path.append('.')

from src import paths
from src.prediction_image_detection import PredictionImageDetection
from src.web_spike import WebSpike

if __name__ == "__main__":
    try:
        prediction = sys.argv[1]
        prediction_path = os.path.join(paths.PREDICTIONS, prediction)
        with open(os.path.join(prediction_path, "detections.json"), 'r') as f:
            raw_detections = json.load(f)
    except:
        print("USAGE: pipenv run python scripts/generate_prediction_web_spike.py <timestamp>")
        exit(1)

    print("Generation beginning: " + datetime.now().isoformat())

    detections = [PredictionImageDetection(d["x1"], d["y1"], d["x2"], d["y2"], d["label"], d["score"]) for d in raw_detections]

    generator = WebSpike(detections, prediction_path)
    generator.generate()

    print("Generation complete: " + datetime.now().isoformat())

