import yfinance as yf

TICKERS = ["AAPL","ABT","AMC","BA","BITC","GME","CVX","UCO","UEC","^GSPC"]

START_DATE = "2023-05-28"
END_DATE = "2024-05-28"
for ticker in TICKERS:
    data = yf.download(ticker, start=START_DATE, end=END_DATE)
    data.to_csv(f'daily {ticker} {START_DATE}-{END_DATE}.csv')