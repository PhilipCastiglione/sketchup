from datetime import datetime
import sys
sys.path.append('.')

from src.dataset import Dataset
from src.dataset_tf_converter import DatasetTfConverter

if __name__ == "__main__":
    print("Conversion beginning: " + datetime.now().isoformat())

    dataset = Dataset.load_input()
    converter = DatasetTfConverter(dataset)
    converter.convert()
    converter.write()

    print("Conversion complete: " + datetime.now().isoformat())
