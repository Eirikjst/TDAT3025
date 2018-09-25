import numpy as np
import tensorflow as tf


class LongShortTermMemoryModel:
    def __init__(self, encoding_size, label_size):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.contrib.rnn.BasicLSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32, [])  # Needed by cell.zero_state call, and can be dependent on usage (training or generation)
        self.x = tf.placeholder(tf.float32, [None, None, encoding_size])  # Shape: [batch_size, max_time, encoding_size]
        self.y = tf.placeholder(tf.float32, [None, label_size])  # Shape: [batch_size, max_time, encoding_size]
        self.in_state = cell.zero_state(self.batch_size, tf.float32)  # Can be used as either an input or a way to get the zero state

        # Model variables
        W = tf.Variable(tf.random_normal([cell_state_size, label_size]))
        b = tf.Variable(tf.random_normal([label_size]))

        # Model operations
        lstm, self.out_state = tf.nn.dynamic_rnn(cell, self.x, initial_state=self.in_state)  # lstm has shape: [batch_size, max_time, cell_state_size]

        # Logits, where tf.einsum multiplies a batch of txs matrices (lstm) with W
        #logits = tf.nn.bias_add(tf.einsum('bts,se->bte', lstm, W), b)  # b: batch, t: time, s: state, e: encoding

        # Logits, where only the last time frame of lstm is used
        logits = tf.nn.bias_add(tf.matmul(lstm[:, -1, :], W), b)

        # Predictor
        self.f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)


char_encodings = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # ' ' - 0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'a' - 1
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'c' - 2
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'f' - 3
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 'h' - 4
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 'l' - 5
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 'm' - 6
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 'n' - 7
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 'o' - 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 'p' - 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 'r' - 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # 's' - 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 't' - 12
]

index_to_char = [' ', 'a', 'c', 'f', 'h', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't']

x_train = [
    [
        char_encodings[4], char_encodings[1], char_encodings[12], char_encodings[0], # 'hat '
    ],
    [
        char_encodings[10], char_encodings[1], char_encodings[12], char_encodings[0], # 'rat '
    ],
    [
        char_encodings[2], char_encodings[1], char_encodings[12], char_encodings[0], # 'cat '
    ],
    [
        char_encodings[3], char_encodings[5], char_encodings[1], char_encodings[12], # 'flat'
    ],
    [
        char_encodings[6], char_encodings[1], char_encodings[12], char_encodings[12], # 'matt'
    ],
    [
        char_encodings[2], char_encodings[1], char_encodings[9], char_encodings[0], # 'cap '
    ],
    [
        char_encodings[11], char_encodings[8], char_encodings[7], char_encodings[0], # 'son '
    ],
]

y_train = np.eye(np.shape(x_train)[0])

model = LongShortTermMemoryModel(len(index_to_char), np.shape(y_train)[1])

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

# Initialize model.in_state
zero_state = session.run(model.in_state, {model.batch_size: np.shape(x_train)[0]})

#print(np.shape([char_encodings[10],char_encodings[2]]))

def generate_x(text):
    text_x = list(text)
    output = []
    for i in range(len(index_to_char)):
        for j in range(len(text_x)):
            if (index_to_char[i] == text_x[j]):
                output.append(char_encodings[i])

    return np.reshape(output, (1, np.shape(output)[0], -1))

for epoch in range(500):
    session.run(minimize_operation, {model.batch_size: np.shape(x_train)[0], model.x: x_train, model.y: y_train, model.in_state: zero_state})

    if epoch % 10 == 9:
        print("epoch", epoch)
        print("loss", session.run(model.loss, {model.batch_size: np.shape(x_train)[0], model.x: x_train, model.y: y_train, model.in_state: zero_state}))

        state = session.run(model.in_state, {model.batch_size: 1})
        # Generate characters from the initial characters in function argument
        x = generate_x('rt ')
        y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: x, model.in_state: state})
        result = ' '
        for c in range(len(x)):
            y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[y.argmax()]]], model.in_state: state})
            for i in range(len(x_train[y[0].argmax()])):
                for j in range(len(char_encodings)):
                    if (x_train[y[0].argmax()][i] == char_encodings[j]):
                        result += index_to_char[j]
            result += ' '

        print(result)

session.close()