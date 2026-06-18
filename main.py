import yfinance as yf 

ticker = input("Enter the stock ticker: ").upper()

data = yf.download(ticker, period="1d", interval="1m", multi_level_index=False)

latest_close = data["Close"].iloc[-1]
day_high = data["High"].max()
day_low = data["Low"].min()
print("Ticker:", ticker)
print("Day High", day_high)
print("Day Low", day_low)
print("Latest Close", latest_close)

if latest_close > (day_high + day_low)/2: 
    print("The stock is currently trading above the midpoint of the day's range.")
else: 
    print("The stock is currently trading below the midpoint of the day's range.")

