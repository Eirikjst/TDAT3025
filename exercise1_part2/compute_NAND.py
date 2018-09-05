import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

x_train = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.mat([[1], [1], [1], [0]])

class SigmoidModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0], [0.0]])
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


#200000, W: -1.0210084, b: 1.9184799
#500000, W: -2.026981, b: 3.308494
for epoch in range(500000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {
                         model.x: x_train, model.y: y_train})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()