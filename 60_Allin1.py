import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import json
import optuna

# Constants
EPOCHS = 65
PATIENCE = 15
N_TRIALS = 800
TEST_SIZE = 0.2

# Helper functions
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def build_model(trial, input_shape):
    model_type = trial.suggest_int('model_type', 1, 3)
    first_layer = trial.suggest_int('first_layer', 64, 512)
    second_layer = trial.suggest_int('second_layer', 32, 256)
    dense_layer = trial.suggest_int('dense_layer', 16, 128)
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    inputs = Input(shape=input_shape)
    if model_type == 1:  # Basic LSTM
        x = LSTM(first_layer, return_sequences=True)(inputs)
    elif model_type == 2:  # CNN + LSTM
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = LSTM(first_layer, return_sequences=True)(x)
    elif model_type == 3:  # LSTM with MultiHeadAttention
        x = LSTM(first_layer, return_sequences=True)(inputs)
        # Add MultiHeadAttention layer
        attention_output = MultiHeadAttention(num_heads=4, key_dim=first_layer)(x, x)
        x = Concatenate()([x, attention_output])
    
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = LSTM(second_layer)(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Dense(dense_layer, activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = Dense(input_shape[-1], activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(loss=Huber(), optimizer=tf.keras.optimizers.Nadam(learning_rate=lr))
    return model

def objective(trial):
    sequence_length = trial.suggest_int('sequence_length', 5, 20)
    X, y = create_sequences(data_scaled[:-1], sequence_length)  # Exclude the last row for training
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    model = build_model(trial, X_train.shape[1:])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=trial.suggest_int('batch_size', 16, 128),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Predict on the validation set
    y_val_pred = model.predict(X_val)
    
    # Calculate R² score on the validation set
    r2 = r2_score(y_val, y_val_pred)
    
    # Predict on the last sequence
    last_sequence = data_scaled[-sequence_length - 1:-1]  # Use the sequence before the last row
    last_sequence_pred = model.predict(last_sequence[np.newaxis, ...])
    
    # Inverse transform the prediction to the original scale
    last_sequence_pred_original = scaler.inverse_transform(last_sequence_pred)
    
    # Calculate prediction error on the last sequence (e.g., Mean Absolute Error)
    last_sequence_actual = data_scaled[-1]  # Actual value for the last row
    last_sequence_actual_original = scaler.inverse_transform(last_sequence_actual[np.newaxis, ...])
    prediction_error = mean_absolute_error(last_sequence_actual_original, last_sequence_pred_original)
    
    # Save the model if it performs well on both the validation set and the last sequence
    if r2 > 0.66 and prediction_error < 5.0:  # Adjust thresholds as needed
        # Save the model
        model.save(f"model_r2_{r2:.4f}_error_{prediction_error:.4f}.h5")
        
        # Save the prediction along with the model's parameters and metrics
        model_info = {
            'r2_score': r2,
            'prediction_error': prediction_error,
            'params': trial.params,
            'last_sequence_prediction': last_sequence_pred_original.tolist(),
            'last_sequence_actual': last_sequence_actual_original.tolist()
        }
        
        with open(f"model_r2_{r2:.4f}_error_{prediction_error:.4f}_info.json", "w") as f:
            json.dump(model_info, f, indent=4)
    
    # Return a combined score (e.g., negative R² score + prediction error)
    combined_score = -r2 + prediction_error  # Adjust weighting as needed
    return combined_score

# Load data
data = pd.read_csv("06_DATA_Predict.csv")
data = data.drop(columns=data.columns[0])  # Drop the first column (date)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Optuna optimization
study = optuna.create_study(direction='minimize')  # Minimize the combined score
study.optimize(objective, n_trials=N_TRIALS)

# Save top models
def save_top_n_results(study, n, filepath):
    sorted_trials = sorted(study.trials, key=lambda t: t.value)  # Sort by combined score (ascending)
    top_n_trials = sorted_trials[:n]
    top_n_params = [t.params for t in top_n_trials]
    
    # Add sequence_length to each model's parameters
    for params in top_n_params:
        params['sequence_length'] = study.best_params['sequence_length']
    
    with open(filepath, "w") as f:
        json.dump(top_n_params, f, indent=4)

save_top_n_results(study, 10, "top_models_params.json")

# Use all good models to predict the next unknown sequence
def predict_next_unknown_sequence():
    # Load all models that meet the criteria
    good_models = []
    for file in os.listdir():
        if file.startswith("model_r2_") and file.endswith(".h5"):
            # Extract R² score and prediction error from the filename
            r2_score = float(file.split("_")[2])
            prediction_error = float(file.split("_")[4].replace(".h5", ""))
            
            # Load the model
            model = tf.keras.models.load_model(file)
            
            # Load the model info
            with open(file.replace(".h5", "_info.json"), "r") as f:
                model_info = json.load(f)
            
            good_models.append({
                'model': model,
                'r2_score': r2_score,
                'prediction_error': prediction_error,
                'params': model_info['params'],
                'last_sequence_prediction': model_info['last_sequence_prediction'],
                'last_sequence_actual': model_info['last_sequence_actual']
            })
    
    # Predict the next unknown sequence using each good model
    for model_data in good_models:
        model = model_data['model']
        sequence_length = model_data['params']['sequence_length']
        
        # Use the last known sequence (including the most recent record) to predict the next value
        last_sequence = data_scaled[-sequence_length:]
        next_value_pred = model.predict(last_sequence[np.newaxis, ...])
        
        # Inverse transform the prediction to the original scale
        next_value_pred_original = scaler.inverse_transform(next_value_pred)
        
        # Display the results
        print(f"Model with R² score: {model_data['r2_score']:.4f}, Prediction Error: {model_data['prediction_error']:.4f}")
        print(f"Predicted next value: {next_value_pred_original}")
        print(f"Model parameters: {model_data['params']}")
        print("-" * 50)

# Predict the next unknown sequence using all good models
predict_next_unknown_sequence()