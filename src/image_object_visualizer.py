import random

from matplotlib import colors
from PIL import Image
import cv2

class ImageObjectVisualizer:
    def __init__(self, images, objects, scores, label_map):
        self.images = images
        self.objects = objects
        self.scores = scores
        self.label_map = label_map

    def visualize(self):
        label_colors = self._map_labels_to_random_colors()
        line_thickness = 2
        text_thickness = 2

        for i, image in enumerate(self.images):
            image_array = image.array[:]
            cv2.putText(image_array, image.filepath(), (0, image.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), text_thickness)
            for j, obj in enumerate(self.objects[i]):
                color = label_colors[obj.label]
                cv2.rectangle(image_array, (obj.x1, obj.y1), (obj.x2, obj.y2), color, line_thickness)
                cv2.putText(image_array, obj.label, (obj.x1, obj.y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, text_thickness)
                cv2.putText(image_array, str(self.scores[i][j]), (obj.x1, obj.y2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, text_thickness)

            Image.fromarray(image_array).show()

    def _map_labels_to_random_colors(self):
        sampled_colors = random.sample(list(colors.CSS4_COLORS.values()), len(self.label_map))
        rgb_colors = [colors.to_rgb(c) for c in sampled_colors]
        scaled_rgb_colors = [tuple([int(c * 255) for c in colors]) for colors in rgb_colors]

        return dict(zip(self.label_map.keys(), scaled_rgb_colors))

