import pandas as pd
import numpy as np
import math
import datetime as dt
from itertools import cycle
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

#Load the dataset
df1= pd.read_csv(r"models\eth\ETH_DATA_1h_2024.csv")
#Display the last few rows to inspect the data
df1.tail()

# Calculate the 10-period Exponential Moving Average (EMA) of the 'open' price
df1['EMA_Open_10'] = df1['open'].ewm(span=10, adjust=False).mean()

# Calculate and print the mean and standard deviation of the 'oprn' price
open_mean= df1['open'].mean()
open_std= df1['open'].std()
print (open_mean)
print (open_std)

from sklearn.preprocessing import StandardScaler
# Define the columns to be scaled
columns= ['open', 'high', 'low', 'close', 'volume',
       'quote_asset_volume', 'RSI14', 'RSI30', 'RSI200', 'EMA10', 'EMA30',
       'EMA200', 'MOM10', 'MOM30', 'EMA12', 'EMA26', 'MACD', 'PROC9',
       '%K', '%K10', '%K30', '%K200','EMA_Open_10']
# Initialize StandardScaler
scaler = StandardScaler()

# Scale the selected columns
df1_scaled = scaler.fit_transform(df1[columns])

# Create a new DataFrame with scaled columns
df1 = pd.DataFrame(df1_scaled, columns=columns, index=df1.index)



df1.dropna(inplace=True)

# Define features (X) and target (y) for 'open' price prediction
X = df1[[ 'volume',
       'quote_asset_volume', 'RSI14', 'RSI30', 'RSI200', 'MOM10', 'MOM30', 'MACD',
        'PROC9', 'EMA10', 'EMA30', 'EMA200',
       '%K10', '%K30', '%K200']]
y = df1[['open']]

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
#Predicting open price
# Initialize the XGBoost regressor
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=400, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_open = model.predict(X_test)

# Evaluate the model
print ("Error on normalized data:")
mse_open = mean_squared_error(y_test, y_pred_open)
print(f'MSE: {mse_open}')

rmse_open = np.sqrt(mse_open)
print(f"RMSE: {rmse_open}")

r2_score_open= r2_score(y_test, y_pred_open)
print(f"R2 Score: {r2_score_open}")

mae_open = mean_absolute_error(y_test, y_pred_open)
print(f"MAE: {mae_open}")

# Calculate and print final error metrics (unscaled)
print ("\nFinal error:")
print ("MAE ", mae_open*open_std)
print ("RMSE ", rmse_open*open_std)


import json

stats = {
    "mean": open_mean,
    "std": open_std
}

with open("models/eth_stats_open.json", "w") as f:
    json.dump(stats, f)