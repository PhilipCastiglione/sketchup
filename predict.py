"""
TODO

# PURPOSE

# USAGE

# NOTES
"""

import paths
import tensorflow as tf

graph_path = paths.latest_frozen_graph()

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(graph_path, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
