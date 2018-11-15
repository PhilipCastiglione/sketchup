from datetime import datetime
import sys
sys.path.append('.')

from src.data import Data
from src.data_tf_converter import DataTfConverter

if __name__ == "__main__":
    print("Conversion beginning: " + datetime.now().isoformat())

    data = Data.load_input()
    converter = DataTfConverter(data)
    converter.convert()
    converter.write()

    print("Conversion complete: " + datetime.now().isoformat())
