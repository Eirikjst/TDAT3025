#
# Make an linear model that predicts the weight of a person given the length of the person.
#
# Observations: https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/length_weight.csv
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

path = 'C:/Users/eirik/workspace python/TDAT3025/Read_csv_files/Read_CSV.py'
url = 'https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/length_weight.csv'

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('Read_csv_files', path)
csv = loader.load_module('Read_csv_files')
data = csv.ReadCSV(url).read_csv_from_url()

#print(data.size)
x_data, y_data = np.mat(data[0:,[0]]), np.mat(data[0:,[1]])

class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = tf.matmul(self.x, self.W) + self.b

        # Uses Mean Squared Error, although instead of mean, sum is used.
        self.loss = tf.reduce_mean(tf.square(f - self.y))


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(50000):
    session.run(minimize_operation, {model.x: x_data, model.y: y_data})

# Evaluate training accuracy
#
# minimize: 0.0001, epoch: 500 = loss 4921.6963
# minimize: 0.0001, epoch: 5000 = loss 4554.1724
# minimize: 0.0001, epoch: 50000 = loss 2326.9766
#
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_data, model.y: y_data})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

compute_W, compute_b = W, b

session.close()

#Visulazation part
fig, ax = plt.subplots()

ax.plot(x_data,
        y_data,
        'o',
        label='$(\\hat x^{(i)},\\hat y^{(i)})$'
        )

ax.set_xlabel('length')
ax.set_ylabel('weight')


class LinearRegressionModel_visualize:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return x * self.W + self.b

    # Uses Mean Squared Error, although instead of mean, sum is used.
    def loss(self, x, y):
        return np.sum(np.square(self.f(x) - y))


model = LinearRegressionModel_visualize(np.mat(compute_W), np.mat(compute_b))

x = np.mat([[np.min(x_data)], [np.max(x_data)]])

ax.plot(x,
        model.f(x),
        label='$y = f(x) = xW+b$'
        )

print('loss(numpy):', model.loss(x_data, y_data))

ax.legend()
plt.show()