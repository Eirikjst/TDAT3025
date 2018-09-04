import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import tensorflow as tf

"""
class SigmoidModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32, shape=[4,2])
        self.y = tf.placeholder(tf.float32, shape=[4,1])

        # Model variables
        self.W1 = tf.Variable(tf.random_uniform([2,2], -1, 1))
        self.b1 = tf.Variable(tf.random_uniform([2,1], -1, 1))
        self.W2 = tf.Variable(tf.zeros([2]))
        self.b2 = tf.Variable(tf.zeros([1]))
        #self.W1 = tf.Variable([[random.uniform(-1,1), random.uniform(-1,1)], [random.uniform(-1,1), random.uniform(-1,1)]])
        #self.b1 = tf.Variable([[random.uniform(-1,1), random.uniform(-1,1)]])
        #self.W2 = tf.Variable([[random.uniform(-1,1)], [random.uniform(-1,1)]])
        #self.b2 = tf.Variable([[random.uniform(-1,1)]])
        #print(self.W1.shape,'\n',self.b1.shape,'\n',self.W2.shape,'\n',self.b2.shape)
        #print(tf.random_uniform([2,2]).shape,'\n',tf.random_uniform([2,1]).shape,'\n',tf.zeros([2,1]).shape,'\n',tf.zeros([1,1]).shape)

        # Logits
        #logits = tf.matmul(self.x, self.W1) + self.b1
        #logits2 = tf.matmul(self.x, self.W2) + self.b2

        # Predictor
        f1 = tf.sigmoid(tf.matmul(self.x, self.W1) + self.b1)
        f2 = tf.sigmoid(tf.matmul(f1, self.W2) + self.b2)

        # Uses Cross Entropy
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, f2)
        #self.loss = tf.reduce_mean(( (self.y * tf.log(f2)) + ((1 - self.y) * tf.log(1.0 - f2)) ) * -1)


model = SigmoidModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())


x_train = np.mat([[0,0],[0,1],[1,0],[1,1]])
y_train = np.mat([[0],[1],[1],[0]])

for epoch in range(200000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W1, b1, W2, b2, loss = session.run([model.W1, model.b1, model.W2, model.b2, model.loss], {model.x: x_train, model.y: y_train})
print("W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" % (W1, b1, W2, b2, loss))

session.close()
"""
x_train = np.mat([[0,0],[0,1],[1,0],[1,1]])
y_train = np.mat([[0],[1],[1],[0]])

W1_init = np.mat([[-2.650115, -5.381952], [6.6170206, -7.18888]])
b1_init = np.mat([[1.1560602, 1.3110199]])
W2_init = np.mat([[-4.4486065], [-6.9377027]])
b2_init = np.mat([[4.45937]])

#W1_init = np.mat([[10.0, -10.0], [10.0, -10.0]])
#b1_init = np.mat([[-5.0, 15.0]])
#W2_init = np.mat([[10.0], [10.0]])
#b2_init = np.mat([[-15.0]])



# Visualize part
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


model = SigmoidModel_visualize()

fig = plt.figure("Logistic regression: the logical XOR operator")

plot1 = fig.add_subplot(121, projection='3d')

plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$h=$f1$(x)=\\sigma(x$W1$+$b1$)$")
plot1_h1 = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]))
plot1_h2 = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]))

plot1.plot(
    x_train[:, 0].A.squeeze(),
    x_train[:, 1].A.squeeze(),
    y_train[:, 0].A.squeeze(),
    'o',
    label="$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$",
    color="blue")

plot1_info = fig.text(0.01, 0.02, "")

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$h_1,h_2$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)

plot2 = fig.add_subplot(222, projection='3d')

plot2_f2 = plot2.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$y=$f2$(h)=\\sigma(h $W2$+$b2$)$")

plot2_info = fig.text(0.8, 0.9, "")

plot2.set_xlabel("$h_1$")
plot2.set_ylabel("$h_2$")
plot2.set_zlabel("$y$")
plot2.legend(loc="upper left")
plot2.set_xticks([0, 1])
plot2.set_yticks([0, 1])
plot2.set_zticks([0, 1])
plot2.set_xlim(-0.25, 1.25)
plot2.set_ylim(-0.25, 1.25)
plot2.set_zlim(-0.25, 1.25)

plot3 = fig.add_subplot(224, projection='3d')

plot3_f = plot3.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$y=f(x)=$f2$($f1$(x))$")

plot3_info = fig.text(0.3, 0.03, "")

plot3.set_xlabel("$x_1$")
plot3.set_ylabel("$x_2$")
plot3.set_zlabel("$y$")
plot3.legend(loc="upper left")
plot3.set_xticks([0, 1])
plot3.set_yticks([0, 1])
plot3.set_zticks([0, 1])
plot3.set_xlim(-0.25, 1.25)
plot3.set_ylim(-0.25, 1.25)
plot3.set_zlim(-0.25, 1.25)

table = plt.table(
    cellText=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], colWidths=[0.1] * 3, colLabels=["$x_1$", "$x_2$", "$f(x)$"], cellLoc="center", loc="lower right")

x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
h1_grid = np.empty([10, 10])
h2_grid = np.empty([10, 10])
f2_grid = np.empty([10, 10])
f_grid = np.empty([10, 10])
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        h = model.f1([[x1_grid[i, j], x2_grid[i, j]]])
        h1_grid[i, j] = h[0, 0]
        h2_grid[i, j] = h[0, 1]
        f2_grid[i, j] = model.f2([[x1_grid[i, j], x2_grid[i, j]]])
        f_grid[i, j] = model.f([[x1_grid[i, j], x2_grid[i, j]]])

plot1_h1 = plot1.plot_wireframe(x1_grid, x2_grid, h1_grid, color="lightgreen")
plot1_h2 = plot1.plot_wireframe(x1_grid, x2_grid, h2_grid, color="darkgreen")

plot1_info.set_text("W1$=\\left[\\stackrel{%.2f}{%.2f}\\/\\stackrel{%.2f}{%.2f}\\right]$\nb1$=[{%.2f}, {%.2f}]$" %
                    (model.W1[0, 0], model.W1[1, 0], model.W1[0, 1], model.W1[1, 1], model.b1[0, 0], model.b1[0, 1]))

plot2_f2 = plot2.plot_wireframe(x1_grid, x2_grid, f2_grid, color="green")

plot2_info.set_text("W2$=\\left[\\stackrel{%.2f}{%.2f}\\right]$\nb2$=[{%.2f}]$" % (model.W2[0, 0], model.W2[1, 0], model.b2[0, 0]))

plot3_f = plot3.plot_wireframe(x1_grid, x2_grid, f_grid, color="green")

plot3_info.set_text(
    "$loss = -\\frac{1}{n}\\sum_{i=1}^{n}\\left [ \\hat y^{(i)} \\log\\/f(\\hat x^{(i)}) + (1-\\hat y^{(i)}) \\log (1-f(\\hat x^{(i)})) \\right ] = %.2f$" %
    model.loss(x_train, y_train))

table._cells[(1, 2)]._text.set_text("${%.1f}$" % model.f([[0, 0]]))
table._cells[(2, 2)]._text.set_text("${%.1f}$" % model.f([[0, 1]]))
table._cells[(3, 2)]._text.set_text("${%.1f}$" % model.f([[1, 0]]))
table._cells[(4, 2)]._text.set_text("${%.1f}$" % model.f([[1, 1]]))

plt.show()


"""
x_test = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test = np.mat([[0], [1], [1], [0]])


fig = plt.figure("Logistic regression: the logical XOR operator")

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
"""