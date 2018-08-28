import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()