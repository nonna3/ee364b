import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

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
y_train = Y[6:3250+6]
y_test = Y[3250+6:]

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
    open1 = O[j+5]
    close = C[j+5]
    open3 = O[j+3]
    close3 = C[j+3]
    open5 = O[j+1]
    close5 = C[j+1]

    xi = [open,open1,close,open3,close3,open5,close5,l1,vlavg,vlslope,l5,l5slope,h1,vhavg,vhslope,h5,h5slope,v1,v3avg,v3slope,v5,v5slope]
    x.append(xi)

X = np.array(x)
X_train = X[:3250,:]
X_test = X[3250:,:]


# Define the mean squared error loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Scale the data (optional but recommended for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network architecture
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                     alpha=10, batch_size='auto', learning_rate='constant',
                     learning_rate_init=0.001, max_iter=400, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print("Train Score:", train_score)
print("Test Score:", test_score)

# Make predictions
predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
print(mse)

y_pred = predictions

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

import matplotlib.pyplot as plt
plt.figure(1)
t = np.linspace(1,len(y_pred) + 1,num=len(y_pred))
plt.plot(t, y_pred, label = "pred")
plt.plot(t, y_test, label = "test")
plt.legend()
plt.savefig("neuralnetUEC.png")

plt.figure(2)
plt.plot(t,prof,label = "momentum_neural", color = "green")
plt.plot(t,naive,label="naive",color = "red")
plt.legend()
plt.savefig("neuralnetUECprof.png")



x_tom = []

for r in range(1):
    j = 4301
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

    open = 7
    open1 = O[j + 5]
    close = C[j + 5]
    open3 = O[j + 3]
    close3 = C[j + 3]
    open5 = O[j + 1]
    close5 = C[j + 1]

    xi = [open,open1,close,open3,close3,open5,close5,l1,vlavg,vlslope,l5,l5slope,h1,vhavg,vhslope,h5,h5slope,v1,v3avg,v3slope,v5,v5slope]
    x_tom.append(xi)

tom_scale = scaler.transform(x_tom)
tom_pred = model.predict(tom_scale)
print(tom_pred)