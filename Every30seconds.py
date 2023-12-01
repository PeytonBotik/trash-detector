# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:24:05 2023

@author: jctzj
"""

import cv2
import numpy as np
import tensorflow as tf
from threading import Thread

# Load the TensorFlow Lite model
model_path = "waterbottle/waterbottle.tflite"  # Adjust the path
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input details and set input size
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# Class labels
CLASS_LABELS = [
    "Flexibles", "Bottle", "Jar", "Carton", "Sachets-&-Pouch", "Blister-pack",
    "Tray", "Tube", "Can", "Tub", "Cosmetic", "Box", "Clothes", "Bulb",
    "Cup-&-glass", "Book-&-magazine", "Bag", "Lid", "Clamshell", "Mirror",
    "Tangler", "Cutlery", "Cassette-&-tape", "Electronic-devices", "Battery",
    "Pen-&-pencil", "Paper-products", "Footwear", "Scissor", "Toys",
    "Brush", "Pipe", "Foil", "Hangers"
]

# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (input_width, input_height))
    image = np.expand_dims(image, axis=0)
    image = (image / 255.0).astype(np.float32)  # Normalization step (if required by the model)
    return image

# VideoStream class to handle webcam stream
class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Initialize the video stream
vs = VideoStream().start()


frame_counter = 0
process_every_n_frames = 30 * 30  # 30 seconds at 30 FPS

while True:
    frame = vs.read()
    frame_counter += 1

    if frame_counter % process_every_n_frames == 0:
        # Preprocess and run inference on the frame every 30 seconds
        frame_preprocessed = preprocess_image(frame)
        interpreter.set_tensor(input_details[0]['index'], frame_preprocessed)
        interpreter.invoke()

        # Retrieve the output tensors
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding boxes
        classes = interpreter.get_tensor(output_details[2]['index'])[0]  # Class indices
        scores = interpreter.get_tensor(output_details[3]['index'])[0]  # Confidence scores

        # Iterate over all detections
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            # Get the bounding box coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            # Draw the label
            label = f"{CLASS_LABELS[int(classes[i])]}: {int(scores[i]*100)}%"
            cv2.putText(frame, label, (int(left), int(top-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Live Detection', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
vs.stop()
cv2.destroyAllWindows()