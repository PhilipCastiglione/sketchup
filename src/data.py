import json

from src import paths
from src.data_region import DataRegion
from src.data_image import DataImage

class Data:
    def __init__(self, images):
        self.images = images
        with open(paths.LABEL_MAP, 'r') as f:
            self.label_map = json.load(f)

    def load_input():
        with open(paths.INPUT_DATASET, 'r') as f:
            dataset = json.load(f)

        images = []
        for datum in dataset:
            guid = datum["id"]
            width = datum["width"]
            height = datum["height"]
            regions = [DataRegion(r["x1"], r["y1"], r["x2"], r["y2"], r["label"]) for r in datum["regions"]]
            images.append(DataImage(guid, width, height, regions))

        return Data(images)

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
                regions.append(DataRegion(x1, y1, x2, y2, label))

            guid = datum["id"]
            width = datum["width"]
            height = datum["height"]
            images.append(DataImage(guid, width, height, regions))

        return Data(images)

    def write(self):
        def image_data(image):
            return {
                "id": image.guid,
                "width": image.width,
                "height": image.height,
                "regions": [r.__dict__ for r in image.regions]
            }

        dataset = [image_data(image) for image in self.images]

        with open(paths.INPUT_DATASET, 'w') as f:
            json.dump(dataset, f, indent=2)
