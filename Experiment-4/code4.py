import tensorflow as tf
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# One-hot encoding
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# Hyperparameters (we will vary these)
hidden_units = 128
learning_rate = 0.01
epochs = 5
batch_size = 64
activation_function = "relu"  # try: "sigmoid", "tanh"

# Initialize weights
W1 = tf.Variable(tf.random.normal([784, hidden_units]))
b1 = tf.Variable(tf.zeros([hidden_units]))

W2 = tf.Variable(tf.random.normal([hidden_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

# Activation functions
def activation(x):
    if activation_function == "relu":
        return tf.nn.relu(x)
    elif activation_function == "sigmoid":
        return tf.nn.sigmoid(x)
    elif activation_function == "tanh":
        return tf.nn.tanh(x)

# Forward pass
def forward(x):
    z1 = tf.matmul(x, W1) + b1
    a1 = activation(z1)
    z2 = tf.matmul(a1, W2) + b2
    return z2

# Loss function
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# Optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# Training loop
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        with tf.GradientTape() as tape:
            logits = forward(x_batch)
            loss = compute_loss(logits, y_batch)

        gradients = tape.gradient(loss, [W1, b1, W2, b2])
        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))

    print("Epoch:", epoch+1, "Loss:", loss.numpy())

# Evaluate
test_logits = forward(x_test)
predictions = tf.argmax(test_logits, axis=1)
labels = tf.argmax(y_test, axis=1)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
print("Test Accuracy:", accuracy.numpy())