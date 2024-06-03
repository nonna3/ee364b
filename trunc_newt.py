import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
import pandas as pd

STOCK_PATHS = ["./AAPL (20240528123000000 _ 20240429063000000).csv",
			   "./ABT (20240528123000000 _ 20240429063000000).csv",
			   "./AMC (20240528123000000 _ 20240429063000000).csv",
			   "./BA (20240528123000000 _ 20240429063000000).csv",
			   #"./BITC (20240528120000000 _ 20240429063000000).csv",
			   #"./CVX (20240528000000000 _ 20231128000000000).csv",
			   "./GME (20240528123000000 _ 20240429063000000).csv",
			   "./UCO (20240528123000000 _ 20240429063000000).csv",
			   "./UEC (20240528123000000 _ 20240429063000000).csv",
			   "./SPX (20240528123000000 _ 20240429063000000).csv"
			   ]
N = len(STOCK_PATHS) # number of stocks

def load_data(stock_path):
	stock_percent_change = []
	for i in range(len(stock_path)):
		path = stock_path[i]
		open = pd.read_csv(path,thousands=',')['Open'].to_numpy(dtype=float)
		close = pd.read_csv(path,thousands=',')['Close'].to_numpy(dtype=float)
		stock_percent_change.append(np.divide(close-open,open))
	combined_data = np.vstack(stock_percent_change)
	return combined_data


def recency_weights(beta,T):
	w = (1-beta)/(1-beta**T)*np.array([beta**i for i in range(0,T)])
	return w

def martingale_weights(T):
	w = np.zeros(T)
	w[-1] = 1
	return w

def weighted_expectation(X,w):
	'''
	X: data
	w : weights
	'''
	return np.average(X, axis=0,weights=w)
	
def truncated_newton_step(alpha,cov,exp,mu,rho):
	Hess = -alpha*cov
	dir = cp.Variable(len(rho))
	objective = cp.Minimize(cp.norm(Hess @ dir + (exp - alpha*cov@rho), 2))
	problem = cp.Problem(objective, [])
	problem.solve()
	dir = dir.value
	t = 1
	while exp.T @ (rho + t*dir) - 0.5*alpha*(rho + t*dir).T @ cov @ (rho + t*dir) > exp.T @ rho - 0.5 * alpha * rho.T @ cov @ rho + t*0.5*(exp - cov @ rho).T @ dir:
		t *= 0.9
	val = rho + mu*dir
	val = np.maximum(val, 0)
	val = np.round(val, 6)
	val/= np.linalg.norm(val, 1)
	return val
	

def truncated_newton(alpha,cov,exp):
	rho = (np.ones(exp.shape[0]))/np.linalg.norm(np.ones(exp.shape[0]),1)
	max_iter = 50
	for i in range(max_iter):
		rho = truncated_newton_step(alpha,cov,exp,1/np.sqrt(i + 1),rho)
		# print(f"iteration {i}")
	return rho

def find_opt_portfolio(alpha,cov,exp):
	'''
	alpha: risk aversion parameter
	'''
	p = truncated_newton(alpha, cov, exp)
	print(p)
	return p, exp.T@p - 0.5*alpha*(p.T @ cov @ p)

def run_martingale(X,T):
	alpha = 1
	(num_stocks,num_obs) = X.shape
	print(X.shape)
	if T > num_obs: raise ValueError("T is incompatible with observation")
	#num_obs = T+2 #testing
	returns = []
	for i in range(0,num_obs-T):
		period_data = X[:,i:i+T]
		martingale_exp = period_data[:,-1]
		w = np.linspace(0,1,T,dtype=np.float32) #tentative weights
		cov = np.cov(period_data,aweights=w)
		rho,opt = find_opt_portfolio(alpha,cov,martingale_exp)
		returns.append(rho.T@X[:,i+T])
	plt.plot(range(len(returns)),returns,label="portfolio returns")
	plt.plot(X[-1,T:],label="S&P")
	print(np.average(np.array(returns)))
	print("S&P avg", np.average(X[-1,T:]))
	plt.legend()
	#plt.savefig("Portfolio vs S&P weights equal")
	plt.show()
	return

def run_momentum():
	return

def test():
	A = np.array([[1, 1],
				  [2, 2],
				  [3, 3]])
	w = np.array([1,1,1])
	print(np.average(A, axis=0,weights=w))
	print(np.cov(A, rowvar=False,aweights=w))

def bond(num_obs,interest,freq):
	arr = ((1+interest)**(freq)-1)*np.ones((1,num_obs))
	print(arr)
	return arr
if __name__ == '__main__':
	INTEREST = 0.05
	FREQ = 1/(365*24*2)#fraction of a year
	T = 20
	data = load_data(STOCK_PATHS)
	num_obs = data.shape[1]
	data = np.vstack((data,bond(num_obs,INTEREST,FREQ)))
	print(data[0][0].dtype)
	run_martingale(data,T)
