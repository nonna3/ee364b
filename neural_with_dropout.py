import torch
import pandas as pd
import numpy as np

dfU = pd.read_csv("UEC.csv")

D = dfU['Date'].values
O = dfU['Open'].values
H = dfU['High'].values
L = dfU['Low'].values
C = dfU['Close'].values
Adj = dfU['Adj Close'].values
V = dfU['Volume'].values /1000000

def one_block(data,k):
    return (data[k-1])

def three_block(data,k):
    return (data[k-4:k-1])

def five_block(data,k):
    return (data[k-6:k-1])

Y = C
print(np.shape(Y))
y_train = Y[6:3250+6].astype(np.float32)
y_test = Y[3250+6:].astype(np.float32)

x = []

for r in range(len(Y)-6):
    j = r
    v1 = one_block(V,j+6)
    triv3 = three_block(V,j+6)
    v3avg = np.average(triv3)
    v3slope = triv3[2] - triv3[0]
    triv5 = five_block(V, j + 6)
    v5 = np.average(triv5)
    v5slope = triv5[4] - triv5[0]

    h1 = one_block(H, j + 6)
    trih3 = three_block(H, j + 6)
    vhavg = np.average(trih3)
    vhslope = trih3[2] - trih3[0]
    trih5 = five_block(H, j + 6)
    h5 = np.average(trih5)
    h5slope = trih5[4] - trih5[0]

    l1 = one_block(L, j + 6)
    tril3 = three_block(L, j + 6)
    vlavg = np.average(tril3)
    vlslope = tril3[2] - tril3[0]
    tril5 = five_block(L, j + 6)
    l5 = np.average(tril5)
    l5slope = tril5[4] - tril5[0]

    open = O[j+6]
    close = C[j+5]
    open_close = O[j+5]-C[j+5]
    open_close3 = O[j+3] - C[j+5]
    open_close5 = O[j+1] - C[j+5]

    xi = [open,close,open_close,open_close3,open_close5,l1,vlavg,vlslope,l5,l5slope,h1,vhavg,vhslope,h5,h5slope,v1,v3avg,v3slope,v5,v5slope]
    x.append(xi)

X = np.array(x)
X_train = X[:3250,:].astype(np.float32)
X_test = X[3250:,:].astype(np.float32)

X_train_scaled = X_train
X_test_scaled = X_test

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test_scaled)
y_test_tensor = torch.tensor(y_test)

# Define the neural network architecture with dropout
class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(20, 64)
        self.dropout = torch.nn.Dropout(p=.05)  # Dropout layer with dropout rate of 0.05
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the neural network
model = NeuralNet()

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        targets = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

import matplotlib.pyplot as plt
plt.figure(1)
t = np.linspace(1,len(y_pred) + 1,num=len(y_pred))
plt.plot(t, y_pred, label = "pred")
plt.plot(t, y_test, label = "test")
plt.legend()
plt.savefig("nndropUEC.png")

O_interest = O[3256:]

prof_sum = 0
prof = []
naive = []
for j in range(len(y_pred)):
    i = j
    if y_pred[i] >= O_interest[i]:
        prof_sum += (y_test[i] - O_interest[i])
    elif y_pred[i] <= O_interest[i]:
        prof_sum += (O_interest[i] - y_test[i])
    else:
        prof_sum += (0)
    naive.append(y_test[i] - O_interest[0])
    prof.append(prof_sum)

plt.figure(2)
plt.plot(t,prof,label = "momentum_neural", color = "green")
plt.plot(t,naive,label="naive",color = "red")
plt.legend()
plt.savefig("profnndrop.png")