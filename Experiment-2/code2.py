import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([[0],[1],[1],[0]])

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# derivative
def sigmoid_derivative(x):
    return x*(1-x)

# network structure
input_size = 2
hidden_size = 2
output_size = 1

# weights initialization
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

lr = 0.1
epochs = 10000

losses = []

for epoch in range(epochs):

    # forward pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    y_pred = sigmoid(final_input)

    # loss
    loss = np.mean((y - y_pred)**2)
    losses.append(loss)

    # backpropagation
    d_output = (y - y_pred) * sigmoid_derivative(y_pred)

    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    # update weights
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr

    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr


# predictions
print("Predictions:")
print(y_pred)


# LOSS GRAPH 
plt.figure()
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


#  DECISION BOUNDARY 
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

grid = np.c_[xx.ravel(), yy.ravel()]

hidden = sigmoid(np.dot(grid, W1) + b1)
output = sigmoid(np.dot(hidden, W2) + b2)

Z = output.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.6)
plt.scatter(X[:,0], X[:,1], c=y.flatten(), s=100)
plt.title("XOR Decision Boundary")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()