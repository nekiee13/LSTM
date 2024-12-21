import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.losses import Huber
import json
import time
import sys  # Required for sys.exit()
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Step 0: Define Temp folder and file to store progress
temp_folder = "Temp"
progress_file = os.path.join(temp_folder, "Temp.txt")

# Create the Temp folder if it doesn't exist
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

# Function to update progress in a file
def update_progress(progress_percentage):
    with open(progress_file, "w") as f:
        f.write(f"{progress_percentage:.2f}")

# Function to add a timestamp to filenames
def add_timestamp(filename):
    now = datetime.datetime.now()
    timestamp = now.strftime("%y%m%d%H%M%S")
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{timestamp}{ext}"
    return new_filename

# Step 1: Check for file existence
input_folder = r"D:\LSTM\TF\xPrj"
data_file = os.path.join(input_folder, "05_DATA.csv")
learning_rate_file = os.path.join(input_folder, "00_S1.txt")
lstm_config_file = os.path.join(input_folder, "LSTM1.txt")
actual_values_file = os.path.join(input_folder, "03_ACTUAL.txt")
lr_config_file = os.path.join(input_folder, "00_S0.txt")
lstm_setup_file = os.path.join(input_folder, "00_LSTM_Setup.txt")
results_file = os.path.join(input_folder, "xResults.txt")
results_file_bckp = os.path.join(input_folder, "xResults_bckp.txt")

if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file '{data_file}' not found.")
if not os.path.exists(actual_values_file):
    raise FileNotFoundError(f"Actual values file '{actual_values_file}' not found.")
if not os.path.exists(lr_config_file):
    raise FileNotFoundError(f"Learning rate config file '{lr_config_file}' not found.")
if not os.path.exists(lstm_setup_file):
    raise FileNotFoundError(f"LSTM setup file '{lstm_setup_file}' not found.")

# Step 2: Load Data
data = pd.read_csv(data_file)

# Load actual values from actual_values_file
if os.stat(actual_values_file).st_size == 0:
    actual_values = []
else:
    actual_values = pd.read_csv(actual_values_file, header=None).values.flatten()

# Drop the date column
data = data.drop(columns=data.columns[0])

# Plotting histograms for each series
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12))
fig.tight_layout(pad=3.0)
for i, column in enumerate(data.columns):
    sns.histplot(data[column], ax=axes[i // 2, i % 2], kde=True)
    axes[i // 2, i % 2].set_title(f"Histogram of {column}")
plt.savefig('D:\\LSTM\\TF\\xPrj\\appHistograms.png')
print("Histograms saved at: D:\\LSTM\\TF\\xPrj\\appHistograms.png")
plt.close(fig)

# Step 3: Load or generate learning rate configuration
def load_lr_parameters():
    with open(lr_config_file, "r") as f:
        try:
            config = json.load(f)
            upper_limit = config.get('upper_limit', 1)
            lower_limit = config.get('lower_limit', 0)
            step = config.get('step', 0.1)
            return upper_limit, lower_limit, step
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in 00_S0.txt")

upper_limit, lower_limit, step = load_lr_parameters()

# Step 4: Generate Learning Rate Range
def dynamic_rounding(value, step):
    step_str = f'{step:.10f}'.rstrip('0')
    if '.' in step_str:
        decimal_places = len(step_str.split('.')[1])
    else:
        decimal_places = 0
    return decimal_places

LR = upper_limit
LR_range_list = []

while LR > lower_limit and LR > 0:
    LR_rounded = round(LR, dynamic_rounding(LR, step))
    decimal_places = dynamic_rounding(LR_rounded, step)
    print(f"LR = {LR_rounded:.{decimal_places}f}")
    LR_range_list.append(LR_rounded)
    LR -= step

# Step 5: Save learning rate configuration
def save_lr_config(upper_limit, lower_limit, step):
    with open(learning_rate_file, "w") as f:
        json.dump({"upper_limit": upper_limit, "lower_limit": lower_limit, "step": step}, f)

save_lr_config(upper_limit, lower_limit, step)

# Step 6: Load LSTM Setup from file
def load_lstm_setup():
    lstm_setup = {}
    with open(lstm_setup_file, "r") as f:
        for line in f:
            key, value = line.strip().split(' = ', 1)
            lstm_setup[key.strip()] = eval(value)
    return lstm_setup

lstm_setup = load_lstm_setup()
first_layer_values = lstm_setup['first_layer_values']
second_layer_values = lstm_setup['second_layer_values']
dense_layer_values = lstm_setup['dense_layer_values']

# Step 7: Data Preprocessing
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 7
X, y = create_sequences(data_scaled, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Step 8: Dynamically Generate LSTM Configurations
def generate_lstm_configs():
    lstm_configs = set()
    for first_layer in first_layer_values:
        for second_layer in second_layer_values:
            for dense_layer in dense_layer_values:
                for learning_rate in LR_range_list:
                    config = (first_layer, second_layer, dense_layer, learning_rate)
                    lstm_configs.add(config)
    return lstm_configs

lstm_configs = generate_lstm_configs()

# Step 9: Model creation function
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

# Step 10: Train and Save Results
early_terminated_configs = []
total_configs = len(lstm_configs)

if not os.path.exists(results_file):
    open(results_file, 'w').close()

for config_idx, config in enumerate(lstm_configs):
    first_layer, second_layer, dense_layer, learning_rate = config

    if learning_rate == 0:
        print(f"Skipping configuration {config_idx + 1}/{total_configs} due to learning_rate=0")
        continue

    print(f"Training with configuration {config_idx + 1}/{total_configs}: first_layer={first_layer}, second_layer={second_layer}, dense_layer={dense_layer}, learning_rate={learning_rate}")
    model = create_model(first_layer, second_layer, dense_layer, learning_rate)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=10000, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    end_time = time.time()

    if len(history.epoch) < 10000:
        early_terminated_configs.append(config)

    y_pred = model.predict(X_test, verbose=0)
    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[-1]))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[-1]))

    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    huber_loss = Huber()(y_test_rescaled, y_pred_rescaled).numpy()

    last_recorded_sequence = data.iloc[-1].values
    predicted_sequence = y_pred_rescaled[-1]
    trends = {"prediction_trend": [], "actual_trend": []}
    for i in range(len(last_recorded_sequence)):
        trends["prediction_trend"].append("Up" if predicted_sequence[i] > last_recorded_sequence[i] else ("Down" if predicted_sequence[i] < last_recorded_sequence[i] else "Equal"))
        trends["actual_trend"].append("Up" if actual_values[i] > last_recorded_sequence[i] else ("Down" if actual_values[i] < last_recorded_sequence[i] else "Equal"))

    actual_series_rounded = np.round(actual_values).astype(int) if len(actual_values) > 0 else [""] * 7
    last_recorded_sequence_rounded = np.round(last_recorded_sequence).astype(int)
    predicted_sequence_rounded = np.round(predicted_sequence).astype(int)

    result = "\n| Series | Number 1 | Number 2 | Number 3 | Number 4 | Number 5 | Star 1 | Star 2 |\n"
    result += "|---|---|---|---|---|---|---|---|\n"
    result += f"| Last Recorded | {last_recorded_sequence_rounded[0]} | {last_recorded_sequence_rounded[1]} | {last_recorded_sequence_rounded[2]} | {last_recorded_sequence_rounded[3]} | {last_recorded_sequence_rounded[4]} | {last_recorded_sequence_rounded[5]} | {last_recorded_sequence_rounded[6]} |\n"
    result += f"| Prediction | {predicted_sequence_rounded[0]} | {predicted_sequence_rounded[1]} | {predicted_sequence_rounded[2]} | {predicted_sequence_rounded[3]} | {predicted_sequence_rounded[4]} | {predicted_sequence_rounded[5]} | {predicted_sequence_rounded[6]} |\n"
    result += f"| Prediction Trend | {trends['prediction_trend'][0]} | {trends['prediction_trend'][1]} | {trends['prediction_trend'][2]} | {trends['prediction_trend'][3]} | {trends['prediction_trend'][4]} | {trends['prediction_trend'][5]} | {trends['prediction_trend'][6]} |\n"
    result += f"| Actual Trend | {trends['actual_trend'][0]} | {trends['actual_trend'][1]} | {trends['actual_trend'][2]} | {trends['actual_trend'][3]} | {trends['actual_trend'][4]} | {trends['actual_trend'][5]} | {trends['actual_trend'][6]} |\n"
    result += f"| Actual Series | {actual_series_rounded[0]} | {actual_series_rounded[1]} | {actual_series_rounded[2]} | {actual_series_rounded[3]} | {actual_series_rounded[4]} | {actual_series_rounded[5]} | {actual_series_rounded[6]} |\n"

    print(result)
    print(f"Learning rate: {learning_rate}")
    print(f"Training Time: {end_time - start_time:.2f} seconds \n")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Huber Loss: {huber_loss}\n")

    with open(results_file, "a") as f:
        f.write(f"\n# LSTM Input: {config}")
        f.write(result)
        f.write(f"First layer: {first_layer}\n")
        f.write(f"Second_layer: {second_layer}\n")
        f.write(f"Dense_layer: {dense_layer}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Training Time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Mean Squared Error (MSE): {mse}\n")
        f.write(f"Mean Absolute Error (MAE): {mae}\n")
        f.write(f"Huber Loss: {huber_loss}\n\n")

    # Step 20: Calculate and write progress percentage to file
    progress_percentage = (config_idx + 1) / total_configs * 100
    update_progress(progress_percentage)  # Write to Temp file
    print(f"Progress: {progress_percentage:.2f}% completed.")

# Step 21: Backup results once, after all configurations are done
timestamped_bckp_file = add_timestamp(results_file_bckp)
os.system(f'copy "{results_file}" "{timestamped_bckp_file}"')
print(f"Backup saved as: {timestamped_bckp_file}")

# Step 22: Notify completion and display early terminated configurations
if early_terminated_configs:
    print("Early terminated configurations:")
    for config in early_terminated_configs:
        print(config)

print("All LSTM configurations and learning rates processed. Exiting.")
sys.exit(0)
