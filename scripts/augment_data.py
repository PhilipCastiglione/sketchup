from datetime import datetime
import sys
sys.path.append('.')

from src.data import Data
from src.data_augmenter import DataAugmenter

if __name__ == "__main__":
    print("Augmentation beginning: " + datetime.now().isoformat())

    data = Data.load_original()
    augmenter = DataAugmenter(data)
    augmenter.augment()
    augmenter.write()

    print("Augmentation complete: " + datetime.now().isoformat())
