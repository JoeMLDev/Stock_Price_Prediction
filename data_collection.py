# data_collection.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Download Stock Data
stock_data = yf.download('TSLA', start='2020-01-01', end='2023-10-01')

# Step 2: Data Exploration
# Plot the closing price over time
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='TSLA Close Price')
plt.title('Tesla Closing Price (2020-2023)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Optional: Plot Volume data (you can comment this if not needed)
plt.figure(figsize=(12, 6))
plt.bar(stock_data.index, stock_data['Volume'], color='blue', alpha=0.5)
plt.title('Tesla Trading Volume (2020-2023)')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()

# Step 3: Save data for future use
stock_data.to_csv('tesla_stock_data.csv')