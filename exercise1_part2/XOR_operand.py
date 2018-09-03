import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import tensorflow as tf

x_train = np.transpose(np.mat([[0, 0, 1, 1], [0, 1, 0, 1]]))
y_train = np.transpose(np.mat([[0], [1], [1], [0]]))

class SigmoidModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[-1.0], [1.0]])
        self.b = tf.Variable([[0.5]])

        # Logits
        logits = tf.matmul(self.x, self.W) + self.b

        # Predictor
        f = tf.sigmoid(logits)

        # Uses Cross Entropy
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logits)


model = SigmoidModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.000001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(5000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {
                         model.x: x_train, model.y: y_train})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()

W1_init = np.mat([[W[0][0], W[1][0]], [W[0][0], W[1][0]]])
b1_init = np.mat([[b[0][0], b[0][0]]])
W2_init = np.mat([[W[0][0]], [W[1][0]]])
b2_init = np.mat([[b[0][0]]])


# Visulazation part
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

class SigmoidModel_visualize:
    def __init__(self, W1=W1_init.copy(), W2=W2_init.copy(), b1=b1_init.copy(), b2=b2_init.copy()):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2


    # First layer function
    def f1(self, x):
        return sigmoid(x * self.W1 + self.b1)

    # Second layer function
    def f2(self, h):
        return sigmoid(h * self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Uses Cross Entropy
    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))


model_visualize = SigmoidModel_visualize()

x_test = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test = np.mat([[0], [1], [1], [0]])


fig = plt.figure("Logistic regression: the logical NAND operator")

plot1 = fig.add_subplot(111, projection='3d')

plot1.plot(
    x_test[:, 0].A.squeeze(),
    x_test[:, 1].A.squeeze(),
    y_test[:, 0].A.squeeze(),
    'o',
    label="$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$",
    color="blue")

plot1_info = fig.text(0.01, 0.02, "")

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$y$")

x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
f_grid = np.empty([10, 10])
for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            f_grid[i, j] = model_visualize.f([[x1_grid[i, j], x2_grid[i, j]]])


plot_1f = plot1.plot_wireframe(x1_grid, x2_grid, f_grid, color='green', label='$y = f(x) = xW+b$')

plt.show()
