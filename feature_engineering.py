import pandas as pd
from sklearn.preprocessing import MinMaxScaler
stock_data = pd.read_csv('tesla_stock_data.csv', index_col='Date', parse_dates=True)

# Step 2: Feature Engineering
# Adding lag features for the closing price
stock_data['Close_Lag1'] = stock_data['Close'].shift(1)
stock_data['Close_Lag2'] = stock_data['Close'].shift(2)
stock_data['Close_Lag3'] = stock_data['Close'].shift(3)

# Adding moving averages
stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

# Drop rows with NaN values caused by lagging or moving averages
stock_data.dropna(inplace=True)

# Step 3: Data Scaling using MinMaxScaler
scaler = MinMaxScaler()
features_to_scale = ['Close', 'Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Volume', 'MA50', 'MA200']
stock_data[features_to_scale] = scaler.fit_transform(stock_data[features_to_scale])

# Step 4: Save preprocessed data for future use
stock_data.to_csv('preprocessed_stock_data.csv')
