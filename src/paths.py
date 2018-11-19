import os

data_path = "data"

LABEL_MAP = os.path.join(data_path, "label_map.json")
OUTPUT_LABEL_MAP = os.path.join(data_path, "label_map.pbtxt")
ORIGINAL_DATASET = os.path.join(data_path, "original_dataset.json")
INPUT_DATASET = os.path.join(data_path, "dataset.json")
VALIDATION_DATASET = os.path.join(data_path, "validation.json")
TRAIN_RECORDS = os.path.join(data_path, "train.record")
TEST_RECORDS = os.path.join(data_path, "test.record")
IMAGES = os.path.join(data_path, "images")
MODELS = os.path.join("models", "model")
MODEL_CONFIG = os.path.join(MODELS, "sketchup.config")
FROZEN_GRAPH = os.path.join(MODELS, "frozen_inference_graph.pb")
EXTERNAL_EXPORT_SCRIPT = os.path.join("..", "object_detection", "export_inference_graph.py")
PREDICTIONS = os.path.join("predictions")

