import random
from datetime import datetime
import sys
sys.path.append('.')

from src.dataset import Dataset
from src.detected_object_visualizer import DetectedObjectVisualizer

if __name__ == "__main__":
    print("Visualisation beginning: " + datetime.now().isoformat())

    dataset = Dataset.load_input()
    images = random.sample(dataset.images, 5)
    regions = [image.regions for image in images]
    scores = [[1.0 for r in image.regions] for image in images]
    label_map = dataset.label_map
    visualizer = DetectedObjectVisualizer(images, regions, scores, label_map)
    visualizer.visualize()

    print("Visualisation complete: " + datetime.now().isoformat())
