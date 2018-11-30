import tensorflow as tf
import numpy as np

from src import paths
from src.dataset_image_region import DatasetImageRegion
from src.prediction_image import PredictionImage
from src.prediction_image_detection import PredictionImageDetection

class Predicter:
    def __init__(self, image_array, label_map, confidence_bound=70):
        self.image_array = image_array
        self.image_height = image_array.shape[0]
        self.image_width = image_array.shape[1]
        self.label_map = label_map
        self.detection_graph = tf.Graph()
        self.confidence_bound = confidence_bound

    def predict(self):
        with self.detection_graph.as_default():
            obj_detection_graph_def = tf.GraphDef()
            with tf.gfile.GFile(paths.FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                obj_detection_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(obj_detection_graph_def, name='')
            
                output_dict = self._run_inference_for_single_image()

        detections = []
        for i in range(output_dict['num_detections']):
            x1, y1, x2, y2 = output_dict['detection_boxes'][i]
            x1 *= self.image_width
            x2 *= self.image_width
            y1 *= self.image_height
            y2 *= self.image_height
            label = self._label_from_class(output_dict['detection_classes'][i])
            score = output_dict['detection_scores'][i] * 100
            if score >= self.confidence_bound:
                detections.append(PredictionImageDetection(x1, y1, x2, y2, label, score))

        self.prediction = PredictionImage(self.image_array, self.image_width, self.image_height, detections)

    def write(self):
        self.prediction.write()

    def _label_from_class(self, detection_class):
        return [k for k,v in self.label_map.items() if v == str(detection_class)][0]

    def _run_inference_for_single_image(self):
        with self.detection_graph.as_default():
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
                # Get handles to input and output tensors
                operations = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in operations for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, self.image_height, self.image_width)
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
    
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = session.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(self.image_array, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

