from sklearn.kernel_ridge import KernelRidge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

dfU = pd.read_csv("UEC.csv")

D = dfU['Date'].values
O = dfU['Open'].values
H = dfU['High'].values
L = dfU['Low'].values
C = dfU['Close'].values
Adj = dfU['Adj Close'].values
V = dfU['Volume'].values /100000

def one_block(data,k):
    return (data[k-1])

def three_block(data,k):
    return (data[k-4:k-1])

def five_block(data,k):
    return (data[k-6:k-1])

Y = C
print(np.shape(Y))
Y_train = Y[6+3800:4250+6]
Y_test = Y[4250+6:]

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
X_train = X[3800:4250,:]
X_test = X[4250:,:]


model = KernelRidge(kernel='polynomial', gamma=0.1)
model.fit(X_train, Y_train)
options = {'cosine', 'laplacian', 'polynomial', 'linear', 'precomputed', 'poly', 'additive_chi2', 'chi2', 'sigmoid', 'rbf'}

# Make predictions on the test data
y_pred = model.predict(X_test)

import matplotlib.pyplot as plt

print(y_pred)
print(Y_test)
t = np.linspace(1,len(y_pred) + 1,num=len(y_pred))

plt.plot(t, y_pred, label = "pred")
plt.plot(t, Y_test, label = "test")
plt.legend()
plt.savefig("kernel.png")

O_interest = O[4256:]
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)

prof = 0
for j in range(len(y_pred)-1):
    i = j
    if y_pred[i] >= O_interest[i] + (mse):
        prof += Y_test[i] - O_interest[i]
    if y_pred[i] + (mse) <= O_interest[i]:
        prof += O_interest[i] - Y_test[i]
    else:
        prof += 0

print(prof)


# Evaluate the model

