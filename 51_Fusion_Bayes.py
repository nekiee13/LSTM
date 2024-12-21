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
import optuna
import matplotlib.pyplot as plt

print(tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING logs

# Enable mixed precision
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Import libraries for logging and visualization
import structlog
from rich.progress import Progress
from rich.console import Console

# Load environment variables
load_dotenv()
PATIENCE = int(os.getenv("PATIENCE", 25))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 48))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", 10000))
LEARNING_RATE_STEP = float(os.getenv("LEARNING_RATE_STEP", 0.0000001))
CHART_LIVE = os.getenv("CHART_LIVE", "true").lower() in ("true", "1", "t")

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
def build_model(trial, input_shape):
    model_type = trial.suggest_int('model_type', 1, 3)
    first_layer = trial.suggest_int('first_layer', 16, 320)
    second_layer = trial.suggest_int('second_layer', 16, 320)
    dense_layer = trial.suggest_int('dense_layer', 16, 320)
    lr = trial.suggest_float('lr', 1e-8, 1e-1, log=True)  # Updated

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
def train_model(model, train_dataset, val_dataset, validation_data, config, results_path, ax, r2_line, background, fig, min_epochs=800):
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

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    model_type = trial.suggest_int('model_type', 1, 3)
    first_layer = trial.suggest_int('first_layer', 32, 256)
    second_layer = trial.suggest_int('second_layer', 32, 256)
    dense_layer = trial.suggest_int('dense_layer', 32, 256)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)  # Updated

    config = (model_type, first_layer, second_layer, dense_layer, lr)
    log.msg("Training Configuration", config=config)

    if CHART_LIVE:
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

    model = build_model(trial, X_train.shape[1:])
    if CHART_LIVE:
        history, final_r2 = train_model(model, train_dataset, val_dataset, (X_val, y_val), config, results_file, ax, r2_line, background, fig)
    else:
        history, final_r2 = train_model(model, train_dataset, val_dataset, (X_val, y_val), config, results_file, None, None, None, None)

    if final_r2 < 0:
        console.print(f"[red]Early Termination: R2 < 0 for config {config}[/red]")
    else:
        if CHART_LIVE:
            # Save the chart as a PNG file
            chart_filename = timestamp_filename("r2_chart") + ".png"
            fig.savefig(chart_filename)
            console.print(f"[green]Chart saved as {chart_filename}[/green]")

    return final_r2

# Function to plot optimization history using Matplotlib
def plot_optimization_history_matplotlib(study):
    plt.figure()
    plt.plot([t.number for t in study.trials], [t.value for t in study.trials], 'o-')
    plt.xlabel('Trial number')
    plt.ylabel('Objective value')
    plt.title('Optimization history')
    plt.show()

# Function to plot parameter importances using Matplotlib
def plot_param_importances_matplotlib(study):
    plt.figure()
    optuna.visualization.plot_param_importances(study).show()

# Function to save top N results to a file
def save_top_n_results(study, n, filepath):
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    top_n_trials = sorted_trials[:n]
    with open(filepath, "w") as f:
        for i, trial in enumerate(top_n_trials):
            f.write(f"Top {i+1}: Trial {trial.number}, Value: {trial.value}, Params: {trial.params}\n")

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
    global X_train, X_val, y_train, y_val, train_dataset, val_dataset, results_file, fig, ax
    X_train, X_val, y_train, y_val = train_test_split(cp.asnumpy(X), cp.asnumpy(y), test_size=0.2)

    # Create efficient data pipelines
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    results_file = "xResults.txt"

    if CHART_LIVE:
        # Initialize plot
        plt.ion()
        fig, ax = plt.subplots()

    # Create a study object
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=8000, timeout=54000)

    console.print("[bold green]Optimization Completed. Results saved.[/bold green]")

    # Retrieve and print the best trial
    best_trial = study.best_trial
    console.print(f"[bold blue]Best Trial:[/bold blue] Trial {best_trial.number}, Value: {best_trial.value}, Params: {best_trial.params}")

    # Retrieve and print the top 10 trials
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    top_10_trials = sorted_trials[:10]
    console.print("[bold blue]Top 10 Trials:[/bold blue]")
    for i, trial in enumerate(top_10_trials):
        console.print(f"Top {i+1}: Trial {trial.number}, Value: {trial.value}, Params: {trial.params}")

    # Save top 10 results to a file
    save_top_n_results(study, 10, "top_10_results.txt")

    # Visualize the optimization process
    if CHART_LIVE:
        plot_optimization_history_matplotlib(study)
        plot_param_importances_matplotlib(study)

    # Close plots
    if CHART_LIVE:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()