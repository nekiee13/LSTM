import os
import pandas as pd
import cupy as cp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from tensorflow.keras.mixed_precision import Policy
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import json
import datetime
import time
import sys

print(tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING logs

# Enable mixed precision
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Import libraries for logging and visualization
import structlog
from rich.progress import Progress
from rich.console import Console
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
PATIENCE = int(os.getenv("PATIENCE", 15))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 48))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", 10000))
LEARNING_RATE_STEP = float(os.getenv("LEARNING_RATE_STEP", 0.000001))

# Logging setup
log = structlog.get_logger()
console = Console()

# Helper functions
def save_json(filepath, content):
    with open(filepath, "w") as f:
        json.dump(content, f, indent=4)

def timestamp_filename(base_filename):
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    return f"{base_filename}_{timestamp}"

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return cp.array(X), cp.array(y)

def save_results(config, metrics, filepath):
    with open(filepath, "a") as f:
        f.write(f"# Config: {config}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

# Custom callback to log metrics at the end of each epoch
class LogEpochMetrics(Callback):
    def __init__(self, validation_data, results_path, ax, r2_line, background, fig, config):
        super().__init__()
        self.validation_data = validation_data
        self.results_path = results_path
        self.ax = ax
        self.r2_line = r2_line
        self.background = background
        self.fig = fig
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        metrics = {
            "Epoch": epoch + 1,
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }
        log.msg("Metrics", **metrics)
        save_results(self.results_path, metrics, self.results_path)

        # Update plot
        self.fig.canvas.restore_region(self.background)
        self.r2_line.set_xdata(np.append(self.r2_line.get_xdata(), epoch + 1))
        self.r2_line.set_ydata(np.append(self.r2_line.get_ydata(), r2))
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.draw_artist(self.r2_line)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

# Model builder
def build_model(model_type, first_layer, second_layer, dense_layer, lr, input_shape):
    model = Sequential()
    if model_type == 1:  # Basic
        model.add(LSTM(first_layer, input_shape=input_shape, return_sequences=True))
    elif model_type == 2:  # Semi-Advanced
        model.add(LSTM(first_layer, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
    else:  # Advanced
        model.add(Bidirectional(LSTM(first_layer, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(0.001))))
        model.add(BatchNormalization())

    model.add(Dropout(0.2))
    model.add(LSTM(second_layer))
    if model_type > 1:
        model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(dense_layer, activation='relu'))
    model.add(Dense(input_shape[-1], activation='linear'))
    model.compile(loss=Huber(), optimizer=tf.keras.optimizers.Nadam(learning_rate=lr))
    return model

# Training function
def train_model(model, train_dataset, val_dataset, validation_data, config, results_path, ax, r2_line, background, fig, min_epochs=500):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    log_metrics = LogEpochMetrics(validation_data, results_path, ax, r2_line, background, fig, config)
    start_time = time.time()
    history = model.fit(train_dataset, epochs=MAX_EPOCHS, validation_data=val_dataset, callbacks=[early_stop, log_metrics], verbose=0)
    end_time = time.time()

    metrics = {
        "Training Time": round(end_time - start_time, 2)
    }
    log.msg("Metrics", **metrics)
    save_results(config, metrics, results_path)

    return history, r2_line.get_ydata()[-1]  # Return the final R2 score

# Main function
def main():
    console.print("[bold green]Loading configuration and data...[/bold green]")

    # Load data
    data = pd.read_csv("05_DATA.csv")
    if 'Date' in data.columns:
        data = data.drop(columns=['Date'])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = create_sequences(cp.array(data_scaled), sequence_length=7)
    X_train, X_val, y_train, y_val = train_test_split(cp.asnumpy(X), cp.asnumpy(y), test_size=0.2)

    # Create efficient data pipelines
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Load LSTM setup
    lstm_config = {}
    with open("00_LSTM_Setup.txt") as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=', 1)
                lstm_config[k.strip()] = eval(v)

    results_file = "xResults.txt"
    model_type = int(input("Select Model (1-Basic, 2-Semi-Advanced, 3-Advanced): "))

    # Initialize plot
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R2 Score')
    ax.set_title('R2 Score Over Epochs')
    ax.set_xlim(0, 50)
    ax.set_ylim(-1, 1)
    r2_line, = ax.plot([], [], label='R2 Score')
    ax.legend()

    # Draw the canvas initially
    fig.canvas.draw()

    # Generate learning rates using NumPy
    learning_rates = np.geomspace(0.001, 0.00001, num=10)

    for first_layer in lstm_config['first_layer_values']:
        for second_layer in lstm_config['second_layer_values']:
            for dense_layer in lstm_config['dense_layer_values']:
                for lr in learning_rates:
                    config = (model_type, first_layer, second_layer, dense_layer, lr)
                    log.msg("Training Configuration", config=config)

                    # Clear the existing graph
                    ax.cla()
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('R2 Score')
                    ax.set_title(f'R2 Score Over Epochs\nConfig: {config}')
                    ax.set_xlim(0, 50)
                    ax.set_ylim(-1, 1)
                    r2_line, = ax.plot([], [], label='R2 Score')
                    ax.legend()

                    # Capture the background of the figure
                    background = fig.canvas.copy_from_bbox(ax.bbox)

                    model = build_model(model_type, first_layer, second_layer, dense_layer, lr, X_train.shape[1:])
                    history, final_r2 = train_model(model, train_dataset, val_dataset, (X_val, y_val), config, results_file, ax, r2_line, background, fig)

                    if final_r2 < 0:
                        console.print(f"[red]Early Termination: R2 < 0 for config {config}[/red]")
                    else:
                        # Save the chart as a PNG file
                        chart_filename = timestamp_filename("r2_chart") + ".png"
                        fig.savefig(chart_filename)
                        console.print(f"[green]Chart saved as {chart_filename}[/green]")

                    if len(history.history['loss']) >= 250:
                        console.print(f"[red]Early Termination: Reached 500 epochs for config {config}[/red]")
                        break

    console.print("[bold green]Training Completed. Results saved.[/bold green]")

    # Close plots
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
