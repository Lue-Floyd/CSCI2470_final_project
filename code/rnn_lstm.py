import pandas as pd
import numpy as np
import tensorflow as tf
from preprocessing_latest import preprocess_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from tensorflow.keras.optimizers.legacy import Adam
from datetime import datetime, timedelta


# Custom LSTM model using subclassing
class MyRNN(tf.keras.Model):
    def __init__(self, units=50, input_shape=()):
        super(MyRNN, self).__init__()
        self.lstm1 = LSTM(units, return_sequences=True, input_shape=input_shape)
        # self.dropout1 = Dropout(0.7)
        self.lstm2 = LSTM(units, return_sequences=False)
        # self.dropout2 = Dropout(0.7)
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm1(inputs)
        # x = self.dropout1(x)
        x = self.lstm2(x)
        # x = self.dropout2(x)
        return self.dense(x)


def train_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = MyRNN(units=200, input_shape=input_shape)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    return model, history


def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    data = pd.read_csv('../data/data.csv')
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Rate']
    target_col = 'Close'
    X_train, X_test, y_train, y_test, target_scaler = preprocess_data(data, feature_cols, target_col, time_steps=60, test_size=0.2)
    model, history = train_model(X_train, y_train, X_test, y_test, epochs=80, batch_size=64)
    plot_history(history)  # Visualize the training and validation loss

    # Make predictions
    y_pred_scaled = model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_actual = target_scaler.inverse_transform(y_test)

    end_date = datetime(2023, 12, 31)  # Adjust this if needed
    start_date = end_date - timedelta(days=len(y_test) - 1)

    # Generate date range for x-axis
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 5))
    plt.plot(date_range, y_actual, label='Actual Close Price')
    plt.plot(date_range, y_pred, label='Predicted Close Price', color='red')
    plt.title('Actual vs Predicted Close Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return model, history


if __name__ == '__main__':
    model, history = main()
    model.save_weights('final_model_weights.h5')
