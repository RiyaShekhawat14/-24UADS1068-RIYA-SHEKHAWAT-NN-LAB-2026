import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Sample time series data
data = np.sin(np.linspace(0, 20, 100))

# Prepare dataset
seq_length = 5
X = []
y = []

for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])
    y.append(data[i+seq_length])

X = np.array(X)
y = np.array(y)

# Convert to tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

# RNN Model
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=10, batch_first=True)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 100
losses = []

for epoch in range(epochs):
    output = model(X)
    loss = criterion(output.squeeze(), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

print("Training Complete")

# Prediction
predicted = model(X).detach().numpy()

# Plot graph
plt.figure()
plt.plot(y.numpy(), label='Actual')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.title("RNN Time Series Prediction")
plt.show()