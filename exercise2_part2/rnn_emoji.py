import numpy as np
import tensorflow as tf


class LongShortTermMemoryModel:
    def __init__(self, encoding_size, label_size):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.contrib.rnn.BasicLSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32, [])  # Needed by cell.zero_state call, and is dependent on usage (training or generation)
        self.x = tf.placeholder(tf.float32, [None, None, encoding_size])  # Shape: [batch_size, max_time, encoding_size]
        self.y = tf.placeholder(tf.float32, [None, label_size])  # Shape: [batch_size, label_size]
        self.in_state = cell.zero_state(self.batch_size, tf.float32)  # Can be used as either an input or a way to get the zero state

        # Model variables
        W = tf.Variable(tf.random_normal([cell_state_size, label_size]))
        b = tf.Variable(tf.random_normal([label_size]))

        # Model operations
        lstm, self.out_state = tf.nn.dynamic_rnn(cell, self.x, initial_state=self.in_state)  # lstm has shape: [batch_size, max_time, cell_state_size]

        # Logits, where only the last time frame of lstm is used
        logits = tf.nn.bias_add(tf.matmul(lstm[:, -1, :], W), b)

        # Predictor
        self.f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)


class Dataset:
    def __init__(self, data):
        self.data = np.array(data)
        self.unique_chars = np.unique(list(''.join(self.data[:, 0])))
        self.char_to_index = dict()
        for element in self.unique_chars:
            self.char_to_index[element] = len(self.char_to_index)

    def encode_char(self, char):
        return np.eye(len(self.unique_chars))[self.char_to_index[char]]

    def encode_words(self):
        one_hots = []
        for word in self.data[:, 0]:
            one_hot = []
            for char in word:
                one_hot.append(self.encode_char(char))
            one_hots.append(one_hot)
        return one_hots

    def y_to_word(self, one_hot):
        return self.data[one_hot.argmax(), 1]


dataset = Dataset([['hat ', '🎩'], ['rat ', '🐀'], ['cat ', '🐈'], ['flat', '🏢'], ['matt', '👨'], ['cap ', '🧢'], ['son ', '👦']])

# Observed/training input and output
x_train = dataset.encode_words()
y_train = np.eye(np.shape(x_train)[0])

model = LongShortTermMemoryModel(len(dataset.unique_chars), np.shape(y_train)[1])

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())


def generate(text):
    state = session.run(model.in_state, {model.batch_size: 1})
    gen_x = []
    for char in text:
        gen_x.append(dataset.encode_char(char))
    gen_x = np.reshape(gen_x, (1, np.shape(gen_x)[0], -1))
    y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: gen_x, model.in_state: state})
    return dataset.y_to_word(y)


# Initialize model.in_state
zero_state = session.run(model.in_state, {model.batch_size: np.shape(x_train)[0]})

for epoch in range(500):
    session.run(minimize_operation, {model.batch_size: np.shape(x_train)[0], model.x: x_train, model.y: y_train, model.in_state: zero_state})

    if epoch % 10 == 9:
        print("epoch", epoch)
        print("loss", session.run(model.loss, {model.batch_size: np.shape(x_train)[0], model.x: x_train, model.y: y_train, model.in_state: zero_state}))

        print(generate('rat'))

session.close()
