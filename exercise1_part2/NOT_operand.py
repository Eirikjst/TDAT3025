import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

x_data = np.mat([[0.0], [1.0]])
y_data = np.mat([[1.0], [0.0]])

class SigmoidModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Logits
        logits = tf.matmul(self.x, self.W) + self.b

        # Predictor
        f = tf.sigmoid(logits)

        # Uses Cross Entropy
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logits)


model = SigmoidModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

#300000, W: -3.3970842, b: 1.449266
#400000, W: -3.9507854, b: 1.7375367
#500000, W: -4.3988886, b: 1.967955
for epoch in range(5000):
    session.run(minimize_operation, {model.x: x_data, model.y: y_data})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_data, model.y: y_data})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

W_init = np.mat([[-4.3988886]])
b_init = np.mat([[1.967955]])


def sigmoid(t):
    return 1 / (1 + np.exp(-t))

#Visulazation part
class SigmoidModel_visualize:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return sigmoid(x * self.W + self.b)

    # Uses Cross Entropy
    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))


model = SigmoidModel_visualize(W_init, b_init)

fig, ax = plt.subplots()

ax.plot(x_data,
        y_data,
        'o',
        label='$(\\hat x^{(i)},\\hat y^{(i)})$'
        )

x = np.linspace(0, 1, 10).reshape(-1, 1)

ax.plot(x,
        model.f(x),
        label='$y = f(x) = \sigma(xW+b)$'
        )

ax.legend()
plt.show()