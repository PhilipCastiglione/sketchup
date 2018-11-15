import os

import cv2
from PIL import Image

from src import paths

class DataImage:
    def __init__(self, guid, width, height, regions, image_array=None):
        self.guid = guid
        self.regions = regions
        self.width = width
        self.height = height
        if image_array is not None:
            self.array = image_array
        else:
            self.array = cv2.imread(self.filepath())

    def write(self):
        with open(self.filepath(), 'wb') as f:
            Image.fromarray(self.array).save(f, format="PNG", optimize=True, dpi=[96,96])

    def filename(self):
        return self.guid + ".png"

    def filepath(self):
        return os.path.join(paths.IMAGES, self.filename())
