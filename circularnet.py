import cv2
import numpy as np
import tensorflow as tf
from threading import Thread

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="circularnet/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# # Example expected shape: [1, 224, 224, 3]
# height = 224
# width = 224
# channels = 3

# # Create an array of zeros or ones
# simple_input = np.ones((1, height, width, channels), dtype=np.float32)

# Prepare your input data. (This depends on your model)
input_data = np.array(np.random.random_sample(input_details[0]['shape']), dtype=np.float32)

# Set the tensor (input data)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
