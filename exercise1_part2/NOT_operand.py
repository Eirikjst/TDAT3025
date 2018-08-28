import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

x_train = np.mat([[0], [1]])
y_train = np.mat([[1], [0]])

def sigmoid(t):
    return 1 / (1 + np.exp(-t))
