# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from threading import Thread
import time
import logging

# Initialize logging
logging.basicConfig(filename='detected_objects.log', level=logging.INFO)

# Load the TensorFlow Lite model
model_path = "waterbottle\waterbottle.tflite"  # Adjust the path
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
    # Check if normalization is needed; if so, adjust accordingly
    if input_details[0]['dtype'] == np.uint8:
        image = np.uint8(image)  # Convert to uint8 without normalization
    else:
        image = (image / 255.0).astype(np.float32)  # Keep as it is for float32
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

# Initialize the video stream and FPS variables
vs = VideoStream().start()
fps = 0
start_time = time.time()
confidence_threshold = 0.5  # Adjustable confidence threshold

while True:
    frame = vs.read()
    frame_preprocessed = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], frame_preprocessed)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[2]['index'])[0]
    scores = interpreter.get_tensor(output_details[3]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f"{CLASS_LABELS[int(classes[i])]}: {int(scores[i]*100)}%"
            cv2.putText(frame, label, (int(left), int(top-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            logging.info(f"Detected {CLASS_LABELS[int(classes[i])]} with {scores[i]*100}% confidence")

    # Calculate and display FPS
    fps += 1
    end_time = time.time()
    if end_time - start_time >= 1:
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        fps = 0
        start_time = time.time()

    cv2.imshow('Live Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()