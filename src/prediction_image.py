from datetime import datetime
import os
import json

from PIL import Image

from src import paths

class PredictionImage:
    def __init__(self, image_array, width, height, detections):
        self.array = image_array
        self.detections = detections
        self.width = width
        self.height = height
        self.timestamp = str(int(datetime.utcnow().timestamp()))

    def write(self):
        def detection_data(detection):
            return {
                "x1": detection.x1,
                "y1": detection.y1,
                "x2": detection.x2,
                "y2": detection.y2,
                "label": detection.label,
                "score": detection.score
            }

        os.mkdir(os.path.join(paths.PREDICTIONS, self.timestamp))

        manifest = [detection_data(d) for d in self.detections]
        with open(self.manifest_filepath(), 'w') as f:
            json.dump(manifest, f, indent=2)

        with open(self.filepath(), 'wb') as f:
            Image.fromarray(self.array).save(f, format="PNG", optimize=True, dpi=[96,96])

    def filename(self):
        return "image.png"

    def filepath(self):
        return os.path.join(paths.PREDICTIONS, self.timestamp, self.filename())

    def manifest_filename(self):
        return "detections.json"

    def manifest_filepath(self):
        return os.path.join(paths.PREDICTIONS, self.timestamp, self.manifest_filename())

