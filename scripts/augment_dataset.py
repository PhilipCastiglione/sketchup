from datetime import datetime
import sys
sys.path.append('.')

from src.dataset import Dataset
from src.augmenter import Augmenter

if __name__ == "__main__":
    print("Augmentation beginning: " + datetime.now().isoformat())

    dataset = Dataset.load_original()
    augmenter = Augmenter(dataset)
    augmenter.augment()
    augmenter.write()

    print("Augmentation complete: " + datetime.now().isoformat())
