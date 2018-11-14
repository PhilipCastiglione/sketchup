"""
# PURPOSE

For visualising training images with bounds around elements.

# USAGE

pipenv run python visualise_bounds.py <num_images>

# NOTES

You will be presented num_images randomly sampled images
with boxes bounding labeled elements.
"""

import paths
import json
import sys
import random
import cv2
from PIL import Image
from matplotlib import colors
from datetime import datetime

def map_tags_to_random_colors():
    sampled_colors = random.sample(list(colors.CSS4_COLORS.values()), len(label_map))
    rgb_colors = [colors.to_rgb(c) for c in sampled_colors]
    scaled_rgb_colors = [tuple([int(c * 255) for c in colors]) for colors in rgb_colors]
    
    return dict(zip(label_map.keys(), scaled_rgb_colors))

def image_with_bounds(datum, tag_colors):
    file_path = paths.IMAGES + datum["id"] + ".png"
    image_array = cv2.imread(file_path)
    height = datum["height"]
    width = datum["width"]
    line_thickness = 3
    text_thickness = 2

    for r in datum["regions"]:
        x1 = int(width * r["left"])
        x2 = int(width * (r["left"] + r["width"]))
        y1 = int(height * r["top"])
        y2 = int(height * (r["top"] + r["height"]))
        color = tag_colors[r["tagName"]]

        cv2.rectangle(image_array, (x1, y1), (x2, y2), color, line_thickness)
        cv2.putText(image_array, r["tagName"], (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness)

    cv2.putText(image_array, file_path, (0, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), text_thickness)

    return Image.fromarray(image_array)


if __name__ == "__main__":
    print("Visualisation beginning: " + datetime.now().isoformat())

    with open(paths.LABEL_MAP, 'r') as f:
        label_map = json.load(f)
    with open(paths.INPUT_DATASET, 'r') as f:
        dataset = json.load(f)

    try:
        num_images = int(sys.argv[1])
    except:
        print("USAGE:")
        print("    pipenv run python visualise_bounds.py <num_images>")
        exit(0)

    tag_colors = map_tags_to_random_colors()

    for datum in random.sample(dataset, num_images):
        image_with_bounds(datum, tag_colors).show()

    print("Visualisation complete: " + datetime.now().isoformat())
