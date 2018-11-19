from datetime import datetime
import cv2
import json
import sys
sys.path.append('.')

from src import paths
from src.predicter import Predicter

if __name__ == "__main__":
    try:
        image_array = cv2.imread(sys.argv[1])
        if image_array is None:
            raise FileNotFoundError()
    except:
        print("USAGE: pipenv run python scripts/predict.py <path_to_image>")
        exit(1)

    print("Prediction beginning: " + datetime.now().isoformat())

    with open(paths.LABEL_MAP, 'r') as f:
        label_map = json.load(f)
    predicter = Predicter(image_array, label_map)
    predicter.predict()
    predicter.write()

    print("Prediction complete: " + datetime.now().isoformat())

