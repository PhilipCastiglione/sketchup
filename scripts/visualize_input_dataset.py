import random
from datetime import datetime
import sys
sys.path.append('.')

from src.dataset import Dataset
from src.image_object_visualizer import ImageObjectVisualizer

if __name__ == "__main__":
    try:
        num_images = int(sys.argv[1])
    except:
        print("USAGE: pipenv run python scripts/visualize_input_dataset.py <num_images>")
        exit(1)

    print("Visualisation beginning: " + datetime.now().isoformat())

    dataset = Dataset.load_input()
    images = random.sample(dataset.images, num_images)
    regions = [image.regions for image in images]
    scores = [[1.0 for r in image.regions] for image in images]
    label_map = dataset.label_map
    visualizer = ImageObjectVisualizer(images, regions, scores, label_map)
    visualizer.visualize()

    print("Visualisation complete: " + datetime.now().isoformat())
