from datetime import datetime
import sys
sys.path.append('.')

from src.dataset import Dataset
from src.dataset_augmenter import DatasetAugmenter

if __name__ == "__main__":
    print("Augmentation beginning: " + datetime.now().isoformat())

    dataset = Dataset.load_original()
    augmenter = DatasetAugmenter(dataset)
    augmenter.augment()
    augmenter.write()

    print("Augmentation complete: " + datetime.now().isoformat())
