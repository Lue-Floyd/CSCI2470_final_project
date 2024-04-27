# @Time : 4/27/24
# @Auther : Yuan Cao
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Function to preprocess the data
def preprocess_data(data, feature_cols, target_col, time_steps=60, test_size=0.2):
    # Extract features and target
    features = data[feature_cols].values
    target = data[target_col].values.reshape(-1,1)

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(target)
    features_scaled = scaler.fit_transform(target)

    # Create time steps for LSTM model
    X, y = [], []
    for i in range(time_steps, len(features_scaled)):
        X.append(features_scaled[i-time_steps:i])
        y.append(features_scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

    return X_train, X_test, y_train, y_test
    # return X,y

# Using the function
# data = pd.read_csv('../data/data.csv')
# feature_cols = ['Open', 'High', 'Low', 'Close']  # or just ['Close'] for using only the closing price
# target_col = 'Close'
# time_steps = 60
# test_size = 0.2
#
# X_train, X_test, y_train, y_test= preprocess_data(data, feature_cols, target_col, time_steps, test_size)
#
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)
