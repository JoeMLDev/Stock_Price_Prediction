# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ------------------ Step 1: Load Data and Models ------------------

# Load preprocessed stock data
stock_data = pd.read_csv('preprocessed_stock_data.csv', index_col='Date', parse_dates=True)

# Define features and target variable
target = 'Close'
features = ['Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Volume', 'MA50', 'MA200']

X = stock_data[features]
y = stock_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Load the Random Forest model saved in Day 3
forest_model = joblib.load('random_forest_model.pkl')

# ------------------ Step 2: Hyperparameter Tuning for Random Forest ------------------

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
rf = RandomForestRegressor()

# Perform grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)

# Predict using the tuned model
y_pred_tuned = best_model.predict(X_test)

# Evaluate the model
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
print(f"Optimized Random Forest MSE: {mse_tuned}")

# Plot actual vs predicted for the tuned Random Forest model
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_pred_tuned, label='Predicted (Tuned)', color='red')
plt.title('Tuned Random Forest: Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price (Scaled)')
plt.legend()
plt.show()

# Save the tuned model
joblib.dump(best_model, 'tuned_random_forest_model.pkl')

# ------------------ Step 3: LSTM Model for Time-Series Prediction ------------------

# Normalize the target data (Close price)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data[['Close']])

# Function to create sequences of lagged data
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        labels.append(data[i+sequence_length])
    return np.array(sequences), np.array(labels)

# Define sequence length (e.g., 60 days of data)
sequence_length = 60
X_lstm, y_lstm = create_sequences(scaled_data, sequence_length)

# Split into training and testing sets
train_size = int(X_lstm.shape[0] * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

# Reshape for LSTM (samples, time steps, features)
X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))  # Output layer for regression

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32)

# Predict on test set
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)

# Inverse transform the true test set values for comparison
y_test_lstm = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

# Plot actual vs predicted for LSTM model
plt.figure(figsize=(12, 6))
plt.plot(y_test_lstm, label='Actual', color='blue')
plt.plot(y_pred_lstm, label='Predicted (LSTM)', color='red')
plt.title('LSTM: Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Save the LSTM model
lstm_model.save('lstm_stock_model.h5')

# ------------------ Step 4: Model Comparison ------------------

# Compare the performance of the tuned Random Forest and LSTM
print(f"Optimized Random Forest MSE: {mse_tuned}")
mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
print(f"LSTM Model MSE: {mse_lstm}")
