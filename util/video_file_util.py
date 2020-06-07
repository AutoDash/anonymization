#!/usr/bin/python
# Helper functions for validating, procesing and working with video files.
#
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

from pymediainfo import MediaInfo
import time
import cv2
import os
import numpy as np
import sys
from threading import Thread

import util.blurring_util as blurring_util
import util.detection_util as detection_util

# Import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue
# Otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue

# Buffered video stream adapted from
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/.
# A wrapper around an openCV VideoStream which processes frames of the stream using a FIFO buffer.
class BufferdVideoStream:

    def __init__(self, path, queueSize):
        # Initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.is_stream_stopped = False
        # Initialize the queue used to store frames read from
        # the video file
        self.buffer = Queue(maxsize=queueSize)
        
    # Start the stream
    def start(self):
        # Start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    # Stop the stream from running
    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True

    def release(self):
        self.stream.release()

    # Perform the reading and decoding of video frames
    def update(self):
        while True:
            if self.is_stream_stopped:
                return

            if not self.buffer.full():
                (has_frames, frame) = self.stream.read()
            
                if not has_frames:
                    self.stop()
                    return
                
                self.buffer.put(frame)

    # Return the next frame in the buffer
    def read(self):
        return self.buffer.get()

    def has_frames(self):
        return self.buffer.qsize() > 0
    
    # Get the width of the frames in the video stream
    def get_stream_width(self):
        return int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Get the height of the frames in the video stream
    def get_stream_height(self):
        return int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get input video length in s
def get_video_length(input_file):
    media_info = MediaInfo.parse(input_file)
    return media_info.tracks[0].duration/1000

# Calculate processing ratio (s processing time/s video)
def calculate_proc_ratio(elapsed_time, video_length):
    return elapsed_time/video_length

# Perform the object detection and blurring on a given video file
def process_video(input_file, output_path, sess, skip_frames, block_scaling_factor, 
    default_num_blocks, detection_graph, score_threshold, enlarge_factor, buffer_size,
        tensor_name='image_tensor:0', boxes_name='detection_boxes:0', scores_name='detection_scores:0', 
            classes_name='detection_classes:0'):
    print("Processing file: {} with confidence threshold {:3.2f}".format(input_file, score_threshold))
    split_name = os.path.splitext(os.path.basename(input_file))
    input_file_name = split_name[0]
    input_file_extension = split_name[1]
    output_file_name = os.path.join(output_path, '{}_blurred_auto_conf_{:3.2f}_skip{:d}{}'.format(input_file_name, 
        score_threshold, skip_frames, input_file_extension))

    start_time = time.time()

    # If output directory doesn't exist, create it
    if not os.path.isdir(output_path):
        print("Output directory doesn't exist, creating it now.")
        os.mkdir(output_path)
    # Start the buffered video stream   
    cap = BufferdVideoStream(input_file, buffer_size)
    cap.start()
    # 100 ms of sleep time, to allow the buffer to fill
    time.sleep(0.1)
    out = None
    # Specify codec to use when processing the video, eg. mp4v for a .mp4 file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    count = 0
    last_bounding_boxes = list()
    while cap.has_frames():
        frame = cap.read()
        count += 1

        if count % 50 == 0:
            print('Processing video {}, frame #{}...'.format(input_file_name, count))
        if out is None:
            [h, w] = frame.shape[:2]
            out = cv2.VideoWriter(output_file_name, fourcc, 25.0, (w, h))

        # Skip prediction on current frame if within skipping window and apply last calculated 
        # blurring masks (list of masks for each face)
        if count % (skip_frames + 1) != 0:
            for i in range(len(last_bounding_boxes)):
                bb = last_bounding_boxes[i]
                frame = blurring_util.create_and_merge_pixelated_blur_mask(frame, bb[0], bb[1], bb[2], bb[3], 
                    block_scaling_factor, default_num_blocks)
            out.write(frame)
            continue
        else:
            last_bounding_boxes.clear()

        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # The array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name(tensor_name)
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name(boxes_name)
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name(scores_name)
        classes = detection_graph.get_tensor_by_name(classes_name)

        # Actual detection
        (boxes, scores, classes) = sess.run(
            [boxes, scores, classes],
            feed_dict={image_tensor: image_np_expanded})

        # Get the final bounding boxes used for blurring
        boxes, scores = detection_util.get_bounding_boxes(boxes, scores, frame, 
            score_thres=score_threshold, enlarge_factor=enlarge_factor)

        for i in range(len(boxes)):
            x1 = int(boxes[i][0])
            y1 = int(boxes[i][1])
            x2 = int(boxes[i][2])
            y2 = int(boxes[i][3])
            
            last_bounding_boxes.append([x1, y1, x2, y2])
            frame = blurring_util.create_and_merge_pixelated_blur_mask(frame, x1, y1, x2, y2, 
                block_scaling_factor, default_num_blocks)

        out.write(frame)

    width = cap.get_stream_width()
    height = cap.get_stream_height()
    video_length = get_video_length(input_file)

    cap.release()
    out.release()

    elapsed_time = time.time() - start_time

    proc_ratio = calculate_proc_ratio(elapsed_time, video_length)

    print("Blurring video finished in {:.2f}s with processing ratio (s processing time/s video) {:2.4f} for video with dimensions {}x{}".format(elapsed_time, 
        proc_ratio, width, height))
    print("Output saved at {}".format(output_file_name))