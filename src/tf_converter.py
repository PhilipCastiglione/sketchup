import tensorflow as tf
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..') # the object_detection module is in our parent dir and we rely on that for relative imports
from object_detection.utils import dataset_util

from src import paths

class TfConverter:
    def __init__(self, dataset):
        self.dataset = dataset

    def convert(self):
        train_dataset_images, test_and_validation_dataset_images = train_test_split(self.dataset.images, test_size=0.3)
        test_dataset_images, validation_dataset_images = train_test_split(test_and_validation_dataset_images, test_size=(0.05/0.3))

        self.train_examples = [self._build_tf_example(di) for di in train_dataset_images]
        self.test_examples = [self._build_tf_example(di) for di in test_dataset_images]
        self.validation_ids = [di.guid for di in validation_dataset_images]

    def write(self):
        self._write_tf_label_file()
        self._write_tf_records()
        self._write_validation_ids_file()

    def _build_tf_example(self, image):
        with open(image.filepath(), 'rb') as f:
            encoded_image_data = f.read()

        classes_xmins = [r.x1 / image.width for r in image.regions]
        classes_xmaxs = [r.x2 / image.width for r in image.regions]
        classes_ymins = [r.y1 / image.height for r in image.regions]
        classes_ymaxs = [r.y2 / image.height for r in image.regions]
        classes_texts = [r.label.encode('utf8') for r in image.regions]
        classes_labels = [int(self.dataset.label_map[r.label]) for r in image.regions]

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(image.height),
            'image/width': dataset_util.int64_feature(image.width),
            'image/filename': dataset_util.bytes_feature(image.filename().encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(image.filename().encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(b"png"),
            'image/object/bbox/xmin': dataset_util.float_list_feature(classes_xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(classes_xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(classes_ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(classes_ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_texts),
            'image/object/class/label': dataset_util.int64_list_feature(classes_labels)
        }))

        return tf_example

    def _write_tf_label_file(self):
        with open(paths.OUTPUT_LABEL_MAP, 'w') as f:
            for name, id in self.dataset.label_map.items():
                f.write("item {\n")
                f.write("  id: " + id + "\n")
                f.write("  name: '" + name + "'\n")
                f.write("}\n\n")

    def _write_tf_records(self):
        with tf.python_io.TFRecordWriter(paths.TRAIN_RECORDS) as writer:
            for ex in self.train_examples:
                writer.write(ex.SerializeToString())

        with tf.python_io.TFRecordWriter(paths.TEST_RECORDS) as writer:
            for ex in self.test_examples:
                writer.write(ex.SerializeToString())

    def _write_validation_ids_file(self):
        with open(paths.VALIDATION_IDS, 'w') as f:
            for id in self.validation_ids:
                f.write(id + ".png\n")

