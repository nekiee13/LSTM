import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import Huber
import json
import time
import sys
import datetime
import re
import matplotlib.pyplot as plt

# Step 1: Define helper functions for adding a timestamp to filenames
def add_timestamp(filename):
    now = datetime.datetime.now()
    timestamp = now.strftime("%y%m%d%H%M%S")
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{timestamp}{ext}"
    return new_filename

# Step 2: Check for file existence
input_folder = r"D:\LSTM\TF\xPrj"
export_folder = os.path.join(input_folder, "Export")
data_file = os.path.join(input_folder, "06_DATA_Predict.csv")
lstm_config_file = os.path.join(input_folder, "30_LSTM2.txt")
results_file = os.path.join(input_folder, "08_Predictions.txt")

if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file '{data_file}' not found.")
if not os.path.exists(lstm_config_file):
    raise FileNotFoundError(f"LSTM configuration file '{lstm_config_file}' not found.")

# Step 3: Load Data from 06_DATA_Predict.csv
data = pd.read_csv(data_file)
data = data.drop(columns=data.columns[0])

# Step 4: Load LSTM configurations from 30_LSTM2.txt (predefined configurations)
def load_lstm_configs():
    with open(lstm_config_file, "r") as f:
        return json.load(f)

lstm_configs = load_lstm_configs()

# Step 5: Data Preprocessing
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Function to estimate probability density using a histogram
def estimate_probability_density(data, bins=20):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

# Function to calculate weights based on inverse probability density with smoothing factor α
def calculate_weights(data, bin_centers, hist, alpha=0.8):
    weights = np.zeros_like(data)
    for i, value in enumerate(data):
        idx = np.argmin(np.abs(bin_centers - value))
        # Apply smoothing factor α
        weights[i] = (1.0 / hist[idx]) ** alpha if hist[idx] > 0 else 1.0
    return weights

# Estimate probability density for each feature column
weights = []
for col in range(data_scaled.shape[1]):  # Loop over each feature column
    bin_centers, hist = estimate_probability_density(data_scaled[:, col], bins=20)
    col_weights = calculate_weights(data_scaled[:, col], bin_centers, hist, alpha=0.8)
    weights.append(col_weights)

# Combine weights into a single array and aggregate across features
weights = np.array(weights).T  # Transpose to match the shape of data_scaled
sample_weights = np.mean(weights, axis=1)  # Aggregate weights across features

# Function to create sequences for LSTM input
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 7
X, y = create_sequences(data_scaled, sequence_length)

X_train, y_train = X, y

# Step 6: Model creation function
def create_model(first_layer, second_layer, dense_layer, learning_rate):
    model = Sequential()
    model.add(LSTM(first_layer, input_shape=(sequence_length, X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(second_layer))
    model.add(Dropout(0.2))
    model.add(Dense(dense_layer, activation='relu'))
    model.add(Dense(X_train.shape[2], activation='linear'))
    model.compile(loss=Huber(), optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate))
    return model

# Step 7: Trend prediction function
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

# Track all predictions for user analysis
all_predictions = []
all_trends = []

# Step 8: Train the model and make predictions using LSTM configurations
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
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=75, restore_best_weights=True)
        
        # Use the aggregated sample weights in the training process
        history = model.fit(X_train, y_train, epochs=12500, batch_size=48, callbacks=[early_stopping], verbose=0, sample_weight=sample_weights[:len(y_train)])
        end_time = time.time()

        last_sequence = X_train[-1:]
        prediction = model.predict(last_sequence, verbose=0)
        prediction_rescaled = scaler.inverse_transform(prediction.reshape(-1, prediction.shape[-1]))

        last_recorded_rescaled = scaler.inverse_transform(last_sequence.reshape(-1, last_sequence.shape[-1]))

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
        result += f"Training Time: {end_time - start_time:.2f} seconds\n\n"

        result_file.write(result)
        print(result)

# Step 9: Function to handle JSON structure and return a list of configurations
def load_top_min_configs(file_path):
    """Load configurations from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found. Please check the file path.")
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Directly assume the data is a list of configurations
    huber_configs = data  

    # Ensure integers for first_layer, second_layer, and dense_layer
    for config in huber_configs:
        config['first_layer'] = int(config['first_layer'])
        config['second_layer'] = int(config['second_layer'])
        config['dense_layer'] = int(config['dense_layer'])
    
    return huber_configs

# Step 10: Function to find and trim file names by removing the timestamp
def find_trimmed_filename(base_name, directory):
    """Find a file with the given base name and return the name without timestamp."""
    regex = re.compile(rf"^{base_name}(_\d+)?\.json$")
    for file in os.listdir(directory):
        if regex.match(file):
            return file
    raise FileNotFoundError(f"File starting with '{base_name}' not found in directory '{directory}'.")

# Step 11: Export function for Huber-only configurations
def export_huber_configs(configs, file_name_prefix="Export_Huber"):
    """Export filtered Huber configs in the JSON-like format with integer layers."""
    timestamp = time.strftime("%y%m%d%H%M%S")
    file_name = f"{file_name_prefix}_{timestamp}.json"
    with open(file_name, 'w') as f:
        json.dump(configs, f, indent=4)
    print(f"\nHuber configurations exported to {file_name}")

# Use the updated function to load from the JSON file and export only Huber configurations
top_min_config_base = "Export_Huber"
# Change the search directory to the correct export folder
top_min_config_file = find_trimmed_filename(top_min_config_base, export_folder)

try:
    huber_configs = load_top_min_configs(os.path.join(export_folder, top_min_config_file))
    export_huber_configs(huber_configs)
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

# Step 12: User Input for Analysis
while True:
    print("Select an option for further analysis:")
    print("[0] Trends")
    for i in range(1, 8):
        print(f"[{i}] Number {i}")

    user_choice = int(input("Enter your choice (0-7): "))

    if user_choice == 0:
        # Option 0: Display percentage of trends
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
        # Options 1-7: Display histogram of predicted values and save as image
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

# Step 13: Notify completion
print("Analysis completed.")
sys.exit(0)