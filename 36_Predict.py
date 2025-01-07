import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.losses import Huber
import json
import time
import sys
import datetime
import re
import matplotlib.pyplot as plt

# ======================
# Constants
# ======================
EPOCHS = 12500  # Number of training epochs
PATIENCE = 75  # Patience for early stopping
BATCH_SIZE = 48  # Batch size for training
RANDOM_OFFSET = 0.035  # Randomness added to predictions
SEQUENCE_LENGTH = 7  # Length of input sequences

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
X_train, y_train = X, y

# ======================
# Model Creation
# ======================
def create_model(first_layer, second_layer, dense_layer, learning_rate):
    model = Sequential()
    model.add(LSTM(first_layer, input_shape=(SEQUENCE_LENGTH, X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(second_layer))
    model.add(Dropout(0.2))
    model.add(Dense(dense_layer, activation='relu'))
    model.add(Dense(X_train.shape[2], activation='linear'))
    model.compile(loss=Huber(), optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate))
    return model

# ======================
# Trend Prediction
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
# Track Predictions
# ======================
all_predictions = []
all_trends = []

# ======================
# Train and Predict
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

        model = create_model(first_layer, second_layer, dense_layer, learning_rate)

        start_time = time.time()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], verbose=0)
        end_time = time.time()

        last_sequence = X_train[-1:]
        prediction = model.predict(last_sequence, verbose=0)
        prediction_rescaled = scaler.inverse_transform(prediction.reshape(-1, prediction.shape[-1]))

        # Add randomness to predictions
        prediction_rescaled += RANDOM_OFFSET * np.random.normal(loc=0.0, scale=1.0, size=prediction_rescaled.shape)

        # Correctly extract and rescale the last recorded values
        last_recorded_scaled = y_train[-1:].reshape(-1, y_train.shape[-1])
        last_recorded_rescaled = scaler.inverse_transform(last_recorded_scaled)

        # Calculate R² score
        r2 = r2_score(last_recorded_rescaled[0], prediction_rescaled[0])

        trends = predict_trends(last_recorded_rescaled[0], prediction_rescaled[0])
        all_predictions.append(prediction_rescaled[0])
        all_trends.append(trends)

        result = f"\n# LSTM Input: ({first_layer}, {second_layer}, {dense_layer}, {learning_rate:.2e})\n"
        result += "| Series | Number 1 | Number 2 | Number 3 | Number 4 | Number 5 | Star 1 | Star 2 |\n"
        result += "|---|---|---|---|---|---|---|---|\n"
        result += f"| Last Recorded | {' | '.join(map(str, np.round(last_recorded_rescaled[0]).astype(int)))} |\n"
        result += f"| Prediction | {' | '.join(map(str, np.round(prediction_rescaled[0]).astype(int)))} |\n"
        result += f"| Prediction Trend | {' | '.join(trends)} |\n"
        result += f"\nFirst layer: {first_layer}\n"
        result += f"Second layer: {second_layer}\n"
        result += f"Dense layer: {dense_layer}\n"
        result += f"Learning rate: {learning_rate:.2e}\n"
        result += f"Training Time: {end_time - start_time:.2f} seconds\n"
        result += f"R² Score: {r2:.4f}\n\n"

        result_file.write(result)
        print(result)

# ======================
# Load and Export Configurations
# ======================
def load_top_min_configs(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found. Please check the file path.")
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Ensure integers for first_layer, second_layer, and dense_layer
    for config in data:
        config['first_layer'] = int(config['first_layer'])
        config['second_layer'] = int(config['second_layer'])
        config['dense_layer'] = int(config['dense_layer'])
    
    return data

def find_trimmed_filename(base_name, directory):
    regex = re.compile(rf"^{base_name}(_\d+)?\.json$")
    for file in os.listdir(directory):
        if regex.match(file):
            return file
    raise FileNotFoundError(f"File starting with '{base_name}' not found in directory '{directory}'.")

def export_huber_configs(configs, file_name_prefix="Export_Huber"):
    timestamp = time.strftime("%y%m%d%H%M%S")
    file_name = f"{file_name_prefix}_{timestamp}.json"
    with open(file_name, 'w') as f:
        json.dump(configs, f, indent=4)
    print(f"\nHuber configurations exported to {file_name}")

# Load and export Huber configurations
top_min_config_base = "Export_Huber"
top_min_config_file = find_trimmed_filename(top_min_config_base, export_folder)

try:
    huber_configs = load_top_min_configs(os.path.join(export_folder, top_min_config_file))
    export_huber_configs(huber_configs)
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

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
    re_choose = input("Do you want to re-choose an option? (yes/no): ").strip().lower()
    if re_choose != "yes":
        break

# ======================
# Completion
# ======================
print("Analysis completed.")
sys.exit(0)