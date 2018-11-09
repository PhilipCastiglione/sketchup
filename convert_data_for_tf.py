"""
Instructions on conversion to TensorFlow Record (note there are some errors):
    [conversion instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)
An example of usage:
    [raccoon dataset conversion](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py)

USAGE

pipenv run python convert_data_for_tf.py

Note: you probably don't need to do this because the TFRecords have been added
to the repo in:
    data/train.record
    data/test.record
"""

DATA_FOLDER_PATH = "./data/"
LABEL_MAP_PATH = DATA_FOLDER_PATH + "label_map.json"
OUTPUT_LABEL_MAP_PATH = DATA_FOLDER_PATH + "label_map.pbtxt"
INPUT_DATASET_PATH = DATA_FOLDER_PATH + "dataset.json"
OUTPUT_TRAIN_PATH = DATA_FOLDER_PATH + "train.record"
OUTPUT_TEST_PATH = DATA_FOLDER_PATH + "test.record"
IMAGES_PATH = DATA_FOLDER_PATH + "images/"

import sys
sys.path.append('..') # the object_detection module is in our parent dir and we rely on that

import tensorflow as tf
from object_detection.utils import dataset_util
from sklearn.model_selection import train_test_split
import json

def write_label_file(label_map):
    with open(OUTPUT_LABEL_MAP_PATH, 'w') as f:
        for name, id in label_map.items():
            f.write("item {\n")
            f.write("  id: " + id + "\n")
            f.write("  name: '" + name + "'\n")
            f.write("}\n\n")

def write_data_files(label_map, dataset):
    train_data, test_data = train_test_split(dataset, test_size=0.2)

    for data, path in [(train_data, OUTPUT_TRAIN_PATH), (test_data, OUTPUT_TEST_PATH)]:
        tf_examples = [build_example(d) for d in data]

        with tf.python_io.TFRecordWriter(path) as writer:
            for ex in tf_examples:
                writer.write(ex.SerializeToString())

# refer to the input dataset to see the structure of a data record
def build_example(datum):
    height = datum["height"]
    width = datum["width"]
    filename = datum["id"] + ".png"
    with open(IMAGES_PATH + filename, 'rb') as f:
        encoded_image_data = f.read()
    encoded_filename = filename.encode('utf8')
    image_format = b"png"

    regions = datum["regions"]
    classes_xmins = [r["left"] for r in regions]
    classes_xmaxs = [(r["left"] + r["width"]) for r in regions]
    classes_ymins = [r["top"] for r in regions]
    classes_ymaxs = [(r["top"] + r["height"]) for r in regions]
    classes_texts = [r["tagName"].encode('utf8') for r in regions]
    classes_labels = [int(label_map[r["tagName"]]) for r in regions]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(encoded_filename),
        'image/source_id': dataset_util.bytes_feature(encoded_filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(classes_xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(classes_xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(classes_ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(classes_ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_texts),
        'image/object/class/label': dataset_util.int64_list_feature(classes_labels),
        }))

    return tf_example


if __name__ == "__main__":
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)

    with open(INPUT_DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    write_label_file(label_map)
    write_data_files(label_map, dataset)
    print("Conversion complete.")
