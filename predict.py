"""
TODO

# PURPOSE

# USAGE

# NOTES
"""

import sys
sys.path.append('..') # the object_detection module is in our parent dir and we rely on that for relative imports

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import paths
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
import cv2

def load_image_array():
    image_path = 'data/images/94b81e83-315e-41d6-ad36-17ba9a565070.png'
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
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
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def image_with_bounds(image_array, predictions):
    image = Image.fromarray(image_array)
    (im_width, im_height) = image.size

    height = im_height
    width = im_width
    line_thickness = 3
    text_thickness = 2

    for i, x in enumerate(predictions['detection_boxes']):
        x1 = int(width * x[0])
        y1 = int(height * x[1])
        x2 = int(width * x[2])
        y2 = int(height * x[3])
        color = (255,0,0)

        cv2.rectangle(image_array, (x1, y1), (x2, y2), color, line_thickness)
        cv2.putText(image_array, str(predictions['detection_classes'][i]), (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness)
        cv2.putText(image_array, str(predictions['detection_scores'][i]), (x1, y2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness)

    return Image.fromarray(image_array)

def wip():
    image_np = load_image_array()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(paths.FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

            category_index = label_map_util.create_category_index_from_labelmap(paths.OUTPUT_LABEL_MAP, use_display_name=True)
            
            output_dict = run_inference_for_single_image(image_np, detection_graph)

    image_with_bounds(image_np, output_dict).show()
    """
    image_np = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8
    )
    Image.fromarray(image_np).show()
    """

if __name__ == "__main__":
    print("Prediction beginning: " + datetime.now().isoformat())

    detections = wip()

    print("Prediction complete: " + datetime.now().isoformat())
