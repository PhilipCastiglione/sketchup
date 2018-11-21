import os
import json

import cv2

from src import paths
from src.dataset_image_region import DatasetImageRegion
from src.dataset_image import DatasetImage

class Dataset:
    def __init__(self, images, label_map):
        self.images = images
        self.label_map = label_map

    def load_input():
        with open(paths.INPUT_DATASET, 'r') as f:
            dataset = json.load(f)

        images = []
        for datum in dataset:
            guid = datum["id"]
            width = datum["width"]
            height = datum["height"]
            regions = [DatasetImageRegion(r["x1"], r["y1"], r["x2"], r["y2"], r["label"]) for r in datum["regions"]]
            images.append(DatasetImage(guid, width, height, regions))

        with open(paths.LABEL_MAP, 'r') as f:
            label_map = json.load(f)

        return Dataset(images, label_map)

    def load_original():
        with open(paths.ORIGINAL_DATASET, 'r') as f:
            dataset = json.load(f)

        images = []
        for datum in dataset:
            regions = []
            for region in datum["regions"]:
                x1 = datum["width"] * region["left"]
                y1 = datum["height"] * region["top"]
                x2 = x1 + datum["width"] * region["width"]
                y2 = y1 + datum["height"] * region["height"]
                label = region["tagName"]
                regions.append(DatasetImageRegion(x1, y1, x2, y2, label))

            guid = datum["id"]
            width = datum["width"]
            height = datum["height"]
            image_array = cv2.imread(os.path.join(paths.ORIGINAL_IMAGES, guid + ".png"))
            images.append(DatasetImage(guid, width, height, regions, image_array))

        with open(paths.LABEL_MAP, 'r') as f:
            label_map = json.load(f)

        return Dataset(images[:5], label_map)

    def write(self, limit_labels=False):
        def image_data(image):
            def region_data(region):
                return {
                    "x1": region.x1,
                    "y1": region.y1,
                    "x2": region.x2,
                    "y2": region.y2,
                    "label": region.label
                }

            if limit_labels:
                regions = [r for r in image.regions if r.label == "Button"]
            else:
                regions = image.regions

            return {
                "id": image.guid,
                "width": image.width,
                "height": image.height,
                "regions": [region_data(r) for r in regions]
            }

        dataset = [image_data(image) for image in self.images]

        with open(paths.INPUT_DATASET, 'w') as f:
            json.dump(dataset, f, indent=2)

        for image in self.images:
            image.write()

