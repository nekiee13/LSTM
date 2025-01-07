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
from sklearn.metrics import r2_score
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
    X, y = create_sequences(data_scaled, sequence_length)
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
    
    # Calculate R² score
    r2 = r2_score(y_val, y_val_pred)
    
    # Return negative R² score (since Optuna minimizes the objective function)
    return -r2

# Load data
data = pd.read_csv("05_DATA.csv")
data = data.drop(columns=data.columns[0])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Optuna optimization
study = optuna.create_study(direction='maximize')  # Maximize R² score
study.optimize(objective, n_trials=N_TRIALS)

# Save top models
def save_top_n_results(study, n, filepath):
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)  # Sort by R² score (descending)
    top_n_trials = sorted_trials[:n]
    top_n_params = [t.params for t in top_n_trials]
    
    # Add sequence_length to each model's parameters
    for params in top_n_params:
        params['sequence_length'] = study.best_params['sequence_length']
    
    with open(filepath, "w") as f:
        json.dump(top_n_params, f, indent=4)

save_top_n_results(study, 10, "top_r2_models.json")