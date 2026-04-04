import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Time Series Data

data = np.sin(np.linspace(0, 100, 500))  # sine wave


# 2. Create Sequences

def create_sequences(data, seq_len):
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 10
X, y = create_sequences(data, seq_len)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


# 3. Define RNN Model

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)           # (batch, seq_len, 1)
        out, _ = self.rnn(x)          # RNN output
        out = self.fc(out[:, -1, :])  # last time step
        return out.squeeze()

model = RNNModel()


# 4. Loss & Optimizer

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 5. Training

epochs = 50
losses = []

for epoch in range(epochs):
    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# 6. Prediction

predicted = model(X).detach().numpy()

# 7. Plot 1: Loss vs Epoch

plt.figure()
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()


# 8. Plot 2: Actual vs Predicted

plt.figure()
plt.plot(y.numpy(), label="Actual")
plt.plot(predicted, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.grid()
plt.show()