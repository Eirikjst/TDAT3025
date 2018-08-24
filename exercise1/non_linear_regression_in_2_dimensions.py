#
# Make an non-linear model that predicts head circumference given the persons age (in days)
#
# Observations: https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_head_circumference.csv
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math as math

path = 'E:\TDAT3025\Read_csv_files\Read_CSV.py'
url = 'https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_head_circumference.csv'

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
        a = 1 / (1+math.exp(0))
        f = (20*a*(tf.matmul(self.x, self.W) + self.b))+31

        # Uses Mean Squared Error, although instead of mean, sum is used.
        self.loss = tf.reduce_mean(tf.square(f - self.y))


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.000000001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(5000):
    session.run(minimize_operation, {model.x: x_data, model.y: y_data})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_data, model.y: y_data})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

compute_W, compute_b = W, b

session.close()