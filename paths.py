import os

data_path = "data"
model_path = os.path.join("models", "model")
export_path = os.path.join(model_path, "export", "Servo")

LABEL_MAP = os.path.join(data_path, "label_map.json")
OUTPUT_LABEL_MAP = os.path.join(data_path, "label_map.pbtxt")
ORIGINAL_DATASET = os.path.join(data_path, "original_dataset.json")
INPUT_DATASET = os.path.join(data_path, "dataset.json")
VALIDATION_DATASET = os.path.join(data_path, "validation.json")
OUTPUT_TRAIN = os.path.join(data_path, "train.record")
OUTPUT_TEST = os.path.join(data_path, "test.record")
IMAGES = os.path.join(data_path, "images")

def latest_frozen_graph():
    latest_dir = sorted(os.listdir(export_path))[-1]
    return os.path.join(export_path, latest_dir, "saved_model.pb")

