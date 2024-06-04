import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
MARKET = -2
RISK_FREE = -1
INTEREST = 0.0538
FREQ = 1 / (365*24*2)  # fraction of a year
T = 20

def generate_paths():
    TICKERS = ["AAPL", "ABT", "AMC", "BA", "BITC", "GME", "CVX", "UCO", "UEC", "^GSPC"]

    START_DATE = "2023-05-28"
    END_DATE = "2024-05-28"
    path = []
    for ticker in TICKERS:
        path.append(f'daily {ticker} {START_DATE}-{END_DATE}.csv')
    return path
def load_percent_data(stock_path):
    stock_percent_change = []
    for i in range(len(stock_path)):
        path = stock_path[i]
        open = pd.read_csv(path,thousands=',')['Open'].to_numpy(dtype=float)
        close = pd.read_csv(path,thousands=',')['Close'].to_numpy(dtype=float)
        stock_percent_change.append(np.divide(close-open,open))
    combined_data = np.vstack(stock_percent_change)
    return combined_data

def load_open_data(stock_path):
    stock = []
    for i in range(len(stock_path)):
        path = stock_path[i]
        open = pd.read_csv(path,thousands=',')['Open'].to_numpy(dtype=float)
        stock.append(open)
    combined_data = np.vstack(stock)
    return combined_data

def load_close_data(stock_path):
    stock = []
    for i in range(len(stock_path)):
        path = stock_path[i]
        close = pd.read_csv(path,thousands=',')['Close'].to_numpy(dtype=float)
        stock.append(close)
    combined_data = np.vstack(stock)
    return combined_data

def find_opt_portfolio(alpha,cov,exp):
    '''
    alpha: risk aversion parameter
    '''
    num_stock = exp.shape[0]
    p = cp.Variable(num_stock)
    cons = [cp.sum(p)==1,p>=0]
    obj = cp.Maximize(exp.T@p-0.5*alpha*cp.quad_form(p,cov))
    prob = cp.Problem(obj,cons)
    opt = prob.solve()
    return p.value,opt

def geometric_weights(T,beta):
    return (1-beta)/(1-beta**T)*np.array([beta**(T-1-i) for i in range(T)])

def run_mean_reversion(open_data,close_data,X,T,alpha,display=True):
    (num_stocks,num_obs) = X.shape
    if T > num_obs: raise ValueError("T is incompatible with observation")
    returns = []
    for i in range(0,num_obs-T):
        period_data = X[:,i:i+T] # percent data
        period_open = open_data[:,i:i+T]
        period_close = close_data[:, i:i + T]
        w = geometric_weights(T, 0.8)
        expectation = np.zeros(period_data.shape[0])
        expectation[RISK_FREE]=period_data[RISK_FREE,-1]
        expectation[:RISK_FREE] = np.divide(0.5*np.average(period_open+period_close,axis=1,weights=w)-open_data[:,i+T],open_data[:,i+T])
        cov = np.cov(period_data,aweights=w)
        rho,opt = find_opt_portfolio(alpha,cov,expectation)
        returns.append(rho.T@X[:,i+T])
    if display == True:
        plt.plot(range(len(returns)), returns, label="portfolio returns")
        plt.plot(X[-1, T:], label="S&P")
        print(np.average(np.array(returns)))
        print("S&P avg", np.average(X[MARKET, T:]))
        plt.legend()
        plt.title(f'Portfolio of Mean Reversion Model,alpha{alpha}')
        plt.savefig(f'Mean Reversion,alpha{alpha}')
        plt.show()
    return np.average(np.array(returns)),np.var(np.array(returns))

def run_simple_momentum(X,T,alpha,display=True):
    (num_stocks,num_obs) = X.shape
    if T > num_obs: raise ValueError("T is incompatible with observation")
    returns = []
    for i in range(0,num_obs-T):
        period_data = X[:,i:i+T]
        expectation = period_data[:,-1]
        w = geometric_weights(T,0.8)
        cov = np.cov(period_data,aweights=w)
        rho,opt = find_opt_portfolio(alpha,cov,expectation)
        returns.append(rho.T@X[:,i+T])
    if display == True:
        plt.plot(range(len(returns)), returns, label="portfolio returns")
        plt.plot(X[-1, T:], label="S&P")
        print(np.average(np.array(returns)))
        print("S&P avg", np.average(X[MARKET, T:]))
        plt.legend()
        plt.title(f'Portfolio of Simple Momentum,alpha{alpha}')
        plt.savefig(f'Simple Momentum,alpha{alpha}')
        plt.show()
    return np.average(np.array(returns)),np.var(np.array(returns))
def run_weighted_momentum(X,T,alpha,display=True):
    (num_stocks,num_obs) = X.shape
    if T > num_obs: raise ValueError("T is incompatible with observation")
    returns = []
    for i in range(0,num_obs-T):
        period_data = X[:,i:i+T]
        w = geometric_weights(T, 0.8)
        w_momentum = geometric_weights(T, 0.5)
        expectation = np.average(period_data,axis=1,weights=w_momentum)
        cov = np.cov(period_data,aweights=w)
        rho,opt = find_opt_portfolio(alpha,cov,expectation)
        returns.append(rho.T@X[:,i+T])
    if display == True:
        plt.plot(range(len(returns)), returns, label="portfolio returns")
        plt.plot(X[-1, T:], label="S&P")
        print(np.average(np.array(returns)))
        print("S&P avg", np.average(X[MARKET, T:]))
        plt.legend()
        plt.title(f'Portfolio of Weighted Momentum,alpha{alpha}')
        plt.savefig(f'Weighted Momentum,alpha{alpha}')
        plt.show()
    return np.average(np.array(returns)),np.var(np.array(returns))
def run_CAPM(X,T,market_return,recency_weight,alpha,display=True):
    (num_stocks,num_obs) = X.shape
    if T > num_obs: raise ValueError("T is incompatible with observation")
    returns = []
    for i in range(0,num_obs-T):
        period_data = X[:,i:i+T]
        w = geometric_weights(T,recency_weight)
        #market_return = 0.0000053777927 #testing
        Em_minus_Rf = market_return - X[RISK_FREE,i+T]
        cov = np.cov(period_data,aweights=w)
        linear = Em_minus_Rf*cov[MARKET]/cov[MARKET,MARKET]
        rho,opt = find_opt_portfolio(alpha,cov,linear)
        returns.append(rho.T@X[:,i+T])
    if display == True:
        plt.plot(range(len(returns)), returns, label="portfolio returns")
        plt.plot(X[-1, T:], label="S&P")
        print("Portfolio avg", np.average(np.array(returns)))
        print("S&P avg", np.average(X[MARKET, T:]))
        print("Portfolio var", np.var(np.array(returns)))
        print("S&P var", np.var(X[MARKET, T:]))
        plt.legend()
        plt.title(f'Portfolio of CAPM, alpha{alpha}')
        plt.savefig(f'CAPM,alpha{alpha}.png')
    return np.average(np.array(returns)),np.var(np.array(returns))

def test():
    A = np.array([[1, 1],
                  [2, 2],
                  [3, 3]])
    w = np.array([1,1,1])
    print(np.average(A, axis=0,weights=w))
    print(np.cov(A, rowvar=False,aweights=w))

def bond(num_obs,interest,freq):
    arr = ((1+interest)**(freq)-1)*np.ones((1,num_obs))
    return arr

# for a fixed alpha
def run_trial(stock_path):
    recency_weight = 0.8
    alpha = 10
    percent_data = load_percent_data(stock_path)
    open_data = load_open_data(stock_path)
    close_data = load_close_data(stock_path)
    num_obs = percent_data.shape[1]
    data = np.vstack((percent_data, bond(num_obs, INTEREST, FREQ)))
    MARKET_RETURN = np.average(data[MARKET, T:])
    #run_simple_momentum(data, T,alpha)
    #run_weighted_momentum(data, T,alpha)
    #run_mean_reversion(open_data,close_data,data, T, alpha)
    #run_CAPM(percent_data, T, MARKET_RETURN, recency_weight,alpha)

def vary_alpha(stock_path):
    recency_weight = 0.8
    percent_data = load_percent_data(stock_path)
    open_data = load_open_data(stock_path)
    close_data = load_close_data(stock_path)
    num_obs = percent_data.shape[1]
    data = np.vstack((percent_data, bond(num_obs, INTEREST, FREQ)))
    MARKET_RETURN = np.average(data[MARKET, T:])

    mean_return=[]
    var_return = []
    alphas =[]
    for power in np.linspace(-6,6,12):
        alpha = 10 ** power
        avg,var= run_CAPM(percent_data, T, MARKET_RETURN, recency_weight,alpha,display=False)
        #avg,var=run_simple_momentum(data, T,alpha,display=False)
        #avg, var = run_weighted_momentum(data, T, alpha, display=False)
        #avg, var =run_mean_reversion(open_data, close_data, data, T, alpha,display=False)
        mean_return.append(avg)
        var_return.append(var)
        alphas.append(alpha)
    plt.semilogx(alphas,mean_return,label="mean")
    plt.semilogx(alphas, var_return,label="var")
    plt.axhline(y=np.var(percent_data[MARKET, T:]),color='y',label="Market Variance")
    plt.axhline(y=np.average(percent_data[MARKET, T:]), color='g',label="Market Average")
    #plt.axhline(y=0.023378, color='y', label="Historical Market Variance")
    #plt.axhline(y=0.0000053777, color='g', label="Historical Market Average")
    plt.legend()
    plt.title("alpha vs CAPM portfolio return/variance")
    #plt.savefig("vary_alpha_CAPM.png")
    plt.show()

def more_test():
    INTEREST = 0.0543
    FREQ = 1 / (365 * 24 * 2)  # fraction of a year
    T = 20
    percent_data = load_percent_data(STOCK_PATHS)
    open_data = load_open_data(STOCK_PATHS)
    close_data = load_close_data(STOCK_PATHS)
    (num_stock, num_obs) = percent_data.shape
    data = np.vstack((percent_data, bond(num_obs, INTEREST, FREQ)))
    for i in range(num_stock):
        print(np.average())
if __name__ == '__main__':
    yf_path = generate_paths()
    vary_alpha(STOCK_PATHS)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
