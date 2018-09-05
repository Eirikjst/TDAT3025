import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_train = np.mat([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.mat([[1], [1], [1], [0]])

W_init = np.mat([[-2.026981], [-2.026981]])
b_init = np.mat([3.308494])

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