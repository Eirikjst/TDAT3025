import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

x_train = np.transpose(np.mat([[0, 0, 1, 1], [0, 1, 0, 1]]))
y_train = np.transpose(np.mat([[1], [1], [1], [0]]))

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

W_init = np.mat([[0.00062432], [0.00062432]])
b_init = np.mat([0.00124884])

# Visulazation part
def sigmoid(t):
    return 1 / (1 + np.exp(-t))
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

W_init = np.mat([[0.00062432], [0.00062432]])
b_init = np.mat([0.00124884])

# Visulazation part
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

class SigmoidModel_visualize:
    def __init__(self, W=W_init.copy(), b=b_init.copy()):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return sigmoid(x * self.W + self.b)

    # Uses Cross Entropy
    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))


model_visualize = SigmoidModel_visualize()

x_test = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test = np.mat([[1], [1], [1], [0]])


fig = plt.figure("Logistic regression: the logical NAND operator")

plot1 = fig.add_subplot(111, projection='3d')

#plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array(
#    [[]]), color="green", label="$y=f(x)=\\sigma(xW+b)$")

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
y_grid = np.empty([10, 10])
for i in range(0, y_grid.shape[0]):
    for j in range(0, y_grid.shape[1]):
        y_grid[i, j] = model_visualize.f([[x1_grid[i, j], x2_grid[i, j]]])

plot_1f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color='green', label='$y = f(x) = xW+b$')

plt.show()