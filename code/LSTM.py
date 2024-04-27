# lstm_model.py

import numpy as np
import pandas as pd
from preprocessing import preprocess_data
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('../data/data.csv')
feature_cols = ['Open', 'High', 'Low', 'Close']
target_col = 'Close'
time_steps = 60
test_size = 0.2

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(data, feature_cols, target_col, time_steps, test_size)

# X_train, y_train = preprocess_data(data, feature_cols, target_col, time_steps, test_size)

# Define the LSTM model
def build_lstm_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dense(units=25, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1))  # Output layer

    return model

# Define input shape
input_shape = (X_train.shape[1], X_train.shape[2])

# Build the LSTM model
model = build_lstm_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=64)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print("Train Loss:", train_loss)
print("Test Loss:", test_loss)

# Predictions for test set
y_pred = model.predict(X_test)

# Visualize the predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('LSTM Model Predictions vs Actual')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()

