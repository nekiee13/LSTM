import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.losses import Huber
import json
import time
import sys
import datetime
import matplotlib.pyplot as plt
import optuna

# ======================
# Record Start Time
# ======================
script_start_time = time.time()

# ======================
# Constants
# ======================
EPOCHS = 200  # Number of training epochs
PATIENCE = 50  # Patience for early stopping
SEQUENCE_LENGTH = 7  # Length of input sequences
N_TRIALS = 20  # Increased number of Optuna trials
TEST_SIZE = 0.2  # Proportion of data to use for testing

# ======================
# Track Predictions and Trends
# ======================
all_predictions = []  # Store all predictions
all_trends = []  # Store all trends

# ======================
# Helper Functions
# ======================
def add_timestamp(filename):
    now = datetime.datetime.now()
    timestamp = now.strftime("%y%m%d%H%M%S")
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{timestamp}{ext}"
    return new_filename

# ======================
# File Paths and Checks
# ======================
input_folder = r"D:\LSTM\TF\xPrj"
export_folder = os.path.join(input_folder, "Export")
data_file = os.path.join(input_folder, "06_DATA_Predict.csv")
lstm_config_file = os.path.join(input_folder, "30_LSTM2.txt")
results_file = os.path.join(input_folder, "08_Predictions.txt")

if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file '{data_file}' not found.")
if not os.path.exists(lstm_config_file):
    raise FileNotFoundError(f"LSTM configuration file '{lstm_config_file}' not found.")

# ======================
# Load Data
# ======================
data = pd.read_csv(data_file)
data = data.drop(columns=data.columns[0])

# ======================
# Load LSTM Configurations
# ======================
def load_lstm_configs():
    with open(lstm_config_file, "r") as f:
        return json.load(f)

lstm_configs = load_lstm_configs()

# ======================
# Data Preprocessing
# ======================
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQUENCE_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

# ======================
# Model Creation with Increased Complexity
# ======================
def create_model(first_layer, second_layer, third_layer, dense_layer, learning_rate, dropout_rate):
    model = Sequential()
    model.add(LSTM(first_layer, input_shape=(SEQUENCE_LENGTH, X_train.shape[2]), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(second_layer, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(third_layer))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_layer, activation='relu'))
    model.add(Dense(X_train.shape[2], activation='linear'))
    model.compile(loss=Huber(), optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate))
    return model

# ======================
# Trend Prediction Function
# ======================
def predict_trends(last_recorded, predicted):
    trends = []
    for i in range(len(last_recorded)):
        if predicted[i] > last_recorded[i]:
            trends.append("Up")
        elif predicted[i] < last_recorded[i]:
            trends.append("Down")
        else:
            trends.append("Equal")
    return trends

# ======================
# Optuna Objective Function
# ======================
def objective(trial):
    # Define hyperparameters to optimize
    first_layer = trial.suggest_int("first_layer", 64, 256)
    second_layer = trial.suggest_int("second_layer", 32, 128)
    third_layer = trial.suggest_int("third_layer", 16, 64)
    dense_layer = trial.suggest_int("dense_layer", 8, 32)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 16, 128)

    # Create and train the model
    model = create_model(first_layer, second_layer, third_layer, dense_layer, learning_rate, dropout_rate)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )

    # Return the validation loss
    return history.history['val_loss'][-1]

# ======================
# Optimize Hyperparameters using Optuna
# ======================
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

# Get the best hyperparameters
best_params = study.best_params
print(f"Optimized Hyperparameters: {best_params}")

# ======================
# Train and Predict with Optimized Hyperparameters
# ======================
if not os.path.exists(results_file):
    open(results_file, 'w').close()

total_configs = len(lstm_configs)

with open(results_file, 'a') as result_file:
    for config_idx, config in enumerate(lstm_configs):
        first_layer = config['first_layer']
        second_layer = config['second_layer']
        dense_layer = config['dense_layer']
        learning_rate = config['learning_rate']

        print(f"Training with configuration {config_idx + 1}/{total_configs}: first_layer={first_layer}, second_layer={second_layer}, dense_layer={dense_layer}, learning_rate={learning_rate}")

        model = create_model(first_layer, second_layer, best_params['third_layer'], dense_layer, learning_rate, dropout_rate=0.2)

        start_time = time.time()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=best_params['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )
        end_time = time.time()

        # Evaluate on the test set
        test_predictions = model.predict(X_test, verbose=0)
        test_predictions_rescaled = scaler.inverse_transform(test_predictions)
        y_test_rescaled = scaler.inverse_transform(y_test)

        # Calculate R² score on the test set
        r2 = r2_score(y_test_rescaled, test_predictions_rescaled)

        # Predict trends
        trends = predict_trends(y_test_rescaled[-1], test_predictions_rescaled[-1])
        all_predictions.append(test_predictions_rescaled[-1])
        all_trends.append(trends)

        # Calculate total script duration
        script_duration = time.time() - script_start_time
        hours, remainder = divmod(script_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

        result = f"\n# LSTM Input: ({first_layer}, {second_layer}, {dense_layer}, {learning_rate:.2e})\n"
        result += "| Series | Number 1 | Number 2 | Number 3 | Number 4 | Number 5 | Star 1 | Star 2 |\n"
        result += "|---|---|---|---|---|---|---|---|\n"
        result += f"| Last Recorded | {' | '.join(map(str, np.round(y_test_rescaled[-1]).astype(int)))} |\n"
        result += f"| Prediction | {' | '.join(map(str, np.round(test_predictions_rescaled[-1]).astype(int)))} |\n"
        result += f"| Prediction Trend | {' | '.join(trends)} |\n"
        result += f"\nFirst layer: {first_layer}\n"
        result += f"Second layer: {second_layer}\n"
        result += f"Third layer: {best_params['third_layer']}\n"
        result += f"Dense layer: {dense_layer}\n"
        result += f"Learning rate: {learning_rate:.2e}\n"
        result += f"Training Time: {end_time - start_time:.2f} seconds\n"
        result += f"R² Score: {r2:.4f}\n"
        result += f"Total Script Duration: {duration_str}\n\n"

        result_file.write(result)
        print(result)

# ======================
# User Analysis
# ======================
while True:
    print("Select an option for further analysis:")
    print("[0] Trends")
    for i in range(1, 8):
        print(f"[{i}] Number {i}")

    user_choice = int(input("Enter your choice (0-7): "))

    if user_choice == 0:
        # Display percentage of trends
        trend_stats = {f"Number {i+1}": {"Up": 0, "Equal": 0, "Down": 0} for i in range(7)}

        for trend in all_trends:
            for i, t in enumerate(trend):
                trend_stats[f"Number {i+1}"][t] += 1

        total_predictions = len(all_trends)

        for number, stats in trend_stats.items():
            up_percentage = (stats["Up"] / total_predictions) * 100
            equal_percentage = (stats["Equal"] / total_predictions) * 100
            down_percentage = (stats["Down"] / total_predictions) * 100
            print(f"{number} - Up = {up_percentage:.1f}% ; Equal = {equal_percentage:.1f}% ; Down = {down_percentage:.1f}%")

    else:
        # Display histogram of predicted values
        selected_number_idx = user_choice - 1
        predicted_values = [pred[selected_number_idx] for pred in all_predictions]

        plt.hist(predicted_values, bins=20, edgecolor='black')
        plt.title(f"Histogram of Predicted Number {user_choice}")
        plt.xlabel(f"Predicted Values of Number {user_choice}")
        plt.ylabel("Frequency")
        
        # Save the histogram
        save_filename = f"Histogram_{user_choice}.png"
        plt.savefig(save_filename)
        print(f"Histogram saved as {save_filename}")
        
        plt.show()

    # Ask the user if they want to re-choose
    print("Do you want to re-choose an option?")
    print("0. No [0]")
    print("1. Yes [1]")
    re_choose = input("Enter your choice (0 or 1): ").strip()

    if re_choose == "0":
        break

# ======================
# Completion
# ======================
print("Analysis completed.")
sys.exit(0)