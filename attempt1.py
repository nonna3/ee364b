import numpy as np
import cvxpy as cp
import pandas as pd

dfW = pd.read_csv('WMT.csv')
dfU = pd.read_csv("UEC.csv")

D = dfU['Date'].values
O = dfU['Open'].values
H = dfU['High'].values
L = dfU['Low'].values
C = dfU['Close'].values
Adj = dfU['Adj Close'].values
V = dfU['Volume'].values

def one_block(data,k):
    return (data[k-1])

def three_block(data,k):
    return (data[k-3:k])

def five_block(data,k):
    return (data[k-5:k])

y = C[6:4206]

print(np.shape(y))

x = []

for r in range(len(y)):
    j = 6 + r - 6
    v1 = one_block(V,j+5)
    triv3 = three_block(V,j+5)
    v3avg = np.average(triv3)
    v3slope = triv3[2] - triv3[0]
    triv5 = five_block(V, j + 5)
    v5 = np.average(triv5)
    v5slope = triv5[4] - triv5[0]

    h1 = one_block(H, j + 5)
    trih3 = three_block(H, j + 5)
    vhavg = np.average(trih3)
    vhslope = trih3[2] - trih3[0]
    trih5 = five_block(H, j + 5)
    h5 = np.average(trih5)
    h5slope = trih5[4] - trih5[0]

    l1 = one_block(L, j + 5)
    tril3 = three_block(L, j + 5)
    vlavg = np.average(tril3)
    vlslope = tril3[2] - tril3[0]
    tril5 = five_block(L, j + 5)
    l5 = np.average(tril5)
    l5slope = tril5[4] - tril5[0]

    open = O[j+6]
    close = C[j+5]
    open_close = O[j+5]-C[j+5]
    open_close3 = O[j+2] - C[j+5]
    open_close5 = O[j] - C[j+5]

    xi = [open,close,open_close,open_close3,open_close5,l1,vlavg,vlslope,l5,l5slope,h1,vhavg,vhslope,h5,h5slope,v1,v3avg,v3slope,v5,v5slope]
    x.append(xi)

X = np.array(x)

B = np.linalg.inv(X.T@X)@X.T@y

naives = []
momentum = []

gap = 1000
x_test = []
for z in range(gap):
    j = len(C) - gap - 6 + z
    v1 = one_block(V, j + 5)
    triv3 = three_block(V, j + 5)
    v3avg = np.average(triv3)
    v3slope = triv3[2] - triv3[0]
    triv5 = five_block(V, j + 5)
    v5 = np.average(triv5)
    v5slope = triv5[4] - triv5[0]

    h1 = one_block(H, j + 5)
    trih3 = three_block(H, j + 5)
    vhavg = np.average(trih3)
    vhslope = trih3[2] - trih3[0]
    trih5 = five_block(H, j + 5)
    h5 = np.average(trih5)
    h5slope = trih5[4] - trih5[0]

    l1 = one_block(L, j + 5)
    tril3 = three_block(L, j + 5)
    vlavg = np.average(tril3)
    vlslope = tril3[2] - tril3[0]
    tril5 = five_block(L, j + 5)
    l5 = np.average(tril5)
    l5slope = tril5[4] - tril5[0]

    open = O[j + 6]
    close = C[j + 5]
    open_close = O[j + 5] - C[j + 5]
    open_close3 = O[j + 2] - C[j + 5]
    open_close5 = O[j] - C[j + 5]

    xi = [open, close, open_close, open_close3, open_close5, l1, vlavg, vlslope, l5, l5slope, h1, vhavg, vhslope,
          h5, h5slope, v1, v3avg, v3slope, v5, v5slope]
    x_test.append(xi)

y_test = C[len(C) - gap:]
X_test = np.array(x_test)
decision = X_test @ B
mse = np.mean((y_test - decision) ** 2)
prof = 0

for i in range(len(decision)):
    if (X @ B)[i]-mse >= O[len(C) - gap + i]:
        dec = 1
    elif (X @ B)[i]+mse <= O[len(C) - gap + i]:
        dec = -1
    else:
        dec = 0
    if dec == 1:
        prof += y_test[i] - O[len(C) - gap + i]
    if dec == -1:
        prof += -y_test[i] + O[len(C) - gap + i]
    prof_naive = C[len(C) - gap + i - 1] - O[len(O) - gap - 1]
    momentum.append(prof)
    naives.append(prof_naive)


import matplotlib.pyplot as plt

plt.figure(1)
t = np.linspace(0, 1000, num=1000)
plt.plot(t, momentum, label="momentum")
plt.plot(t, naives, label="naive")

plt.legend()
plt.xlabel("gap")
plt.ylabel("profit over the gap")
plt.savefig("prof_UEC.png")

plt.figure(2)
t = np.linspace(0, 1000, num=1000)
plt.plot(t, y_test, label="y_test")
plt.plot(t, decision, label="y_pred")

plt.legend()
plt.xlabel("Day")
plt.ylabel("price over time")
plt.savefig("price_UEC.png")

x_tomorrow = []
for z in range(1):
    j = len(C)-6+z
    v1 = one_block(V,j+5)
    triv3 = three_block(V,j+5)
    v3avg = np.average(triv3)
    v3slope = triv3[2] - triv3[0]
    triv5 = five_block(V, j + 5)
    v5 = np.average(triv5)
    v5slope = triv5[4] - triv5[0]

    h1 = one_block(H, j + 5)
    trih3 = three_block(H, j + 5)
    vhavg = np.average(trih3)
    vhslope = trih3[2] - trih3[0]
    trih5 = five_block(H, j + 5)
    h5 = np.average(trih5)
    h5slope = trih5[4] - trih5[0]

    l1 = one_block(L, j + 5)
    tril3 = three_block(L, j + 5)
    vlavg = np.average(tril3)
    vlslope = tril3[2] - tril3[0]
    tril5 = five_block(L, j + 5)
    l5 = np.average(tril5)
    l5slope = tril5[4] - tril5[0]

    open = 6.92
    close = C[j+5]
    open_close = O[j+5]-C[j+5]
    open_close3 = O[j+2] - C[j+5]
    open_close5 = O[j] - C[j+5]

    xi = [open,close,open_close,open_close3,open_close5,l1,vlavg,vlslope,l5,l5slope,h1,vhavg,vhslope,h5,h5slope,v1,v3avg,v3slope,v5,v5slope]
    x_tomorrow.append(xi)

x_T = np.array(x_tomorrow)

print(x_T@B)




