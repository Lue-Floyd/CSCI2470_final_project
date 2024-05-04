
import numpy as np
import pandas as pd
from preprocessing import preprocess_data
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


# Define the LSTM model
def build_lstm_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=200, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(units=200, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=200, return_sequences=False))
    model.add(tf.keras.layers.Dense(units=1))  # Output layer

    return model


def main():
    data = pd.read_csv('../data/data.csv')
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Rate']
    target_col = 'Close'
    time_steps = 60
    test_size = 0.2

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data, feature_cols, target_col, time_steps, test_size)
    # Define input shape
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build the LSTM model
    model = build_lstm_model(input_shape)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=128)

    # Evaluate the model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    print("Train Loss:", train_loss)
    print("Test Loss:", test_loss)

    # Predictions for test set
    y_pred = model.predict(X_test)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[target_col].values.reshape(-1, 1))  # Fit scaler on original target data
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))

    # Visualize the predictions vs actual values

    end_date = datetime(2023, 12, 31)  # Adjust this if needed
    start_date = end_date - timedelta(days=len(y_test) - 1)

    # Generate date range for x-axis
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    plt.figure(figsize=(12, 6))
    plt.plot(date_range,y_test_original, label='Actual')
    plt.plot(date_range, y_pred_original, label='Predicted')
    plt.title('LSTM Model Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

