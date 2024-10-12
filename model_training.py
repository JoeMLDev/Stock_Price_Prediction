# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

#Step 1: Load Preprocessed Data 
stock_data = pd.read_csv('preprocessed_stock_data.csv', index_col='Date', parse_dates=True)

# Define features and target variable
target = 'Close'
features = ['Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Volume', 'MA50', 'MA200']

X = stock_data[features]
y = stock_data[target]

# Step 2: Split Data into Training and Testing Sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Step 3: Build and Train Models 

# 3.1. Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr}")

# 3.2. Decision Tree Regressor
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print(f"Decision Tree MSE: {mse_tree}")

# 3.3. Random Forest Regressor
forest_model = RandomForestRegressor(n_estimators=100)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)
mse_forest = mean_squared_error(y_test, y_pred_forest)
print(f"Random Forest MSE: {mse_forest}")

# Step 4: Evaluate Model Performance 

# Plot actual vs predicted for the best model (assuming Random Forest performs best)
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_pred_forest, label='Predicted', color='red')
plt.title('Random Forest: Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price (Scaled)')
plt.legend()
plt.show()

# Step 5: Save the Best Model 

# Save the Random Forest model (assuming it's the best performing model)
joblib.dump(forest_model, 'random_forest_model.pkl')

# To load the model later for prediction, use the following code:
# forest_model = joblib.load('random_forest_model.pkl')
