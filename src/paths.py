import os

# NOTE: currently a number of these paths are hard coded in models/model/sketchup.config
# we could generate (or template) that file as a QOL/maintenance improvement

data_path = "data"

LABEL_MAP = os.path.join(data_path, "label_map.json")
OUTPUT_LABEL_MAP = os.path.join(data_path, "label_map.pbtxt")
ORIGINAL_DATASET = os.path.join(data_path, "original_dataset.json")
INPUT_DATASET = os.path.join(data_path, "dataset.json")
VALIDATION_IDS = os.path.join(data_path, "validation.txt")
TRAIN_RECORDS = os.path.join(data_path, "train.record")
TEST_RECORDS = os.path.join(data_path, "test.record")
ORIGINAL_IMAGES = os.path.join(data_path, "original_images")
IMAGES = os.path.join(data_path, "images")
MODELS = os.path.join("models", "model")
MODEL_CONFIG = os.path.join(MODELS, "sketchup.config")
MODEL_CONFIG_ALL_LABELS = os.path.join(MODELS, "sketchup_all_labels.config")
FROZEN_GRAPH = os.path.join(MODELS, "frozen_inference_graph.pb")
EXTERNAL_TRAIN_SCRIPT = os.path.join("..", "object_detection", "model_main.py")
EXTERNAL_EXPORT_SCRIPT = os.path.join("..", "object_detection", "export_inference_graph.py")
PREDICTIONS = os.path.join("predictions")

