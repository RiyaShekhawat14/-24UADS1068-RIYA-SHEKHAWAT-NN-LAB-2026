import tensorflow as tf
import numpy as np

# Load MNIST dataset (low-level TF)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape data
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Convert labels to one-hot encoding
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

# Define parameters
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.1
epochs = 10
batch_size = 100

# Initialize weights and bias
W1 = tf.Variable(tf.random.normal([input_size, hidden_size]))
b1 = tf.Variable(tf.zeros([hidden_size]))

W2 = tf.Variable(tf.random.normal([hidden_size, output_size]))
b2 = tf.Variable(tf.zeros([output_size]))

# Training loop
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):

        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        with tf.GradientTape() as tape:
            # Feed Forward
            hidden_layer = tf.nn.relu(tf.matmul(x_batch, W1) + b1)
            output_layer = tf.matmul(hidden_layer, W2) + b2

            # Loss function
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=y_batch, logits=output_layer
                )
            )

        # Backpropagation
        gradients = tape.gradient(loss, [W1, b1, W2, b2])

        W1.assign_sub(learning_rate * gradients[0])
        b1.assign_sub(learning_rate * gradients[1])
        W2.assign_sub(learning_rate * gradients[2])
        b2.assign_sub(learning_rate * gradients[3])

    print(f"Epoch {epoch+1}, Loss = {loss.numpy():.4f}")

# Testing the model
correct = 0
total = len(x_test)

hidden_test = tf.nn.relu(tf.matmul(x_test, W1) + b1)
output_test = tf.matmul(hidden_test, W2) + b2
predicted = tf.argmax(output_test, axis=1)
actual = tf.argmax(y_test, axis=1)

correct = tf.reduce_sum(tf.cast(predicted == actual, tf.int32))
accuracy = (correct / total) * 100

print("Test Accuracy:", accuracy.numpy(), "%")