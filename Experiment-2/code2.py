import numpy as np


# Step 1: Sigmoid function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Step 2: XOR input and output

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])


# Step 3: Initialize weights

np.random.seed(1)

input_layer_neurons = 2
hidden_layer_neurons = 2
output_layer_neurons = 1

# Weights
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))

wo = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
bo = np.random.uniform(size=(1, output_layer_neurons))

learning_rate = 0.1


# Step 4: Training the MLP

for epoch in range(10000):

    # Forward Propagation
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wo) + bo
    final_output = sigmoid(final_input)

    # Error calculation
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(wo.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    wo += hidden_output.T.dot(d_output) * learning_rate
    bo += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    wh += X.T.dot(d_hidden) * learning_rate
    bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate


# Step 5: Testing the network

print("Input:")
print(X)

print("\nPredicted Output:")
print(np.round(final_output))

print("\nActual Output:")
print(y)