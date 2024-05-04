
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Preprocess the data
def preprocess_data(data, feature_cols, target_col, time_steps=60, test_size=0.2):

    features = data[feature_cols].values
    target = data[target_col].values.reshape(-1, 1)

    # Normalize the features and the target
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target)


    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(features_scaled[i:i + time_steps])
        y.append(target_scaled[i + time_steps])
    X, y = np.array(X), np.array(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler

