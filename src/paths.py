import os

data_path = "data"
model_path = os.path.join("models", "model")

LABEL_MAP = os.path.join(data_path, "label_map.json")
OUTPUT_LABEL_MAP = os.path.join(data_path, "label_map.pbtxt")
ORIGINAL_DATASET = os.path.join(data_path, "original_dataset.json")
INPUT_DATASET = os.path.join(data_path, "dataset.json")
VALIDATION_DATASET = os.path.join(data_path, "validation.json")
TRAIN_RECORDS = os.path.join(data_path, "train.record")
TEST_RECORDS = os.path.join(data_path, "test.record")
IMAGES = os.path.join(data_path, "images")
FROZEN_GRAPH = os.path.join(model_path, "frozen_inference_graph.pb")

