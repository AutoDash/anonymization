#!/usr/bin/python
# Helper functions for working with the detection models used for anonymization.
#
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
import numpy as np
import tensorflow as tf

# Load and create a detection graph from a given path
def load_detection_graph(model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        # There is no idiomatic way to use a frozen graph in TensorFlow 2, so we use the compat library. 
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# Allow memory to grow as needed
def set_memory_growth(detection_graph):
    with detection_graph.as_default():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

# Get the bounding boxes from model prediction results
def get_bounding_boxes(boxes_det, scores_det, image, score_thres, enlarge_factor):
    boxes_det = np.squeeze(boxes_det)
    scores_det = np.squeeze(scores_det)
    h, w, _ = image.shape
    res = np.where(scores_det > score_thres)
    if not res[0].shape[0]:
        boxes_det = np.zeros((0, 4))
        scores_det = np.zeros((0, 1))
        return boxes_det, scores_det
    n = np.where(scores_det > score_thres)[0][-1] + 1

    # This creates an array with just enough rows for the objects with scores above the threshold
    # Format: absolute coordinates x, y, x, y
    boxes_det = np.array([boxes_det[:n, 1] * w, boxes_det[:n, 0] * h, boxes_det[:n, 3] * w, 
        boxes_det[:n, 2] * h]).T
    scores_det = scores_det[:n]

    # enlarge region of interest a bit to make the anonymization more effective
    for i in range(boxes_det.shape[0]):
        dx = int(enlarge_factor * (boxes_det[i, 2] - boxes_det[i, 0]))
        dy = int(enlarge_factor * (boxes_det[i, 3] - boxes_det[i, 1]))
        boxes_det[i, 0] = int(boxes_det[i, 0] - dx) if int(boxes_det[i, 0] - dx) > 0 else 0
        boxes_det[i, 1] = int(boxes_det[i, 1] - dy) if int(boxes_det[i, 1] - dy) > 0 else 0
        boxes_det[i, 2] = int(boxes_det[i, 2] + dx) if int(boxes_det[i, 2] + dx) < w else w
        boxes_det[i, 3] = int(boxes_det[i, 3] + dy) if int(boxes_det[i, 3] + dy) < h else h

    return boxes_det, scores_det