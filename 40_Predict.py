import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Add, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt

# Constants
EPOCHS = 75
PATIENCE = 23
TEST_SIZE = 0.48
TEMPERATURE = 9.5
STD_DEV = 12.0
SEED = 42
DROP_TOP_PERCENT = 0.8

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Custom Loss Function with Diversity Penalty
def custom_loss(y_true, y_pred, diversity_weight=1.0):  # Reduced diversity weight
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    diversity_penalty = -tf.reduce_mean(tf.math.reduce_variance(y_pred, axis=0))
    return mse_loss + diversity_weight * diversity_penalty

# Load prediction data
data_predict = pd.read_csv("06_DATA_Predict.csv")
data_predict = data_predict.drop(columns=data_predict.columns[0])  # Drop the first column (index)
data_predict_np = data_predict.to_numpy()

# Scale the data
scaler = StandardScaler()
data_scaled_predict = scaler.fit_transform(data_predict_np)

# Load best architectures
with open("top_r2_models.json") as f:
    best_architectures = json.load(f)

# Add sequence_length if missing
if 'sequence_length' not in best_architectures[0]:
    sequence_length = 10
    for arch in best_architectures:
        arch['sequence_length'] = sequence_length

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Build and compile the model
def build_and_compile_model(params):
    inputs = Input(shape=(params['sequence_length'], data_scaled_predict.shape[1]))
    
    # Bayesian LSTM (First Layer)
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)  # Reduced units
    x = Dropout(0.3)(x, training=True)  # Moderate dropout rate
    x = BatchNormalization()(x)  # Add batch normalization
    
    # Residual Connection
    residual = x
    
    # Encoder-Decoder Architecture (Second Layer)
    encoder_output, state_h, state_c = LSTM(64, return_sequences=True, return_state=True, kernel_regularizer=l2(0.01))(x)
    decoder_input = Concatenate()([encoder_output, residual])  # Skip connection
    decoder_output = LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01))(decoder_input, initial_state=[state_h, state_c])
    decoder_output = BatchNormalization()(decoder_output)  # Add batch normalization
    
    # Output Layer
    outputs = Dense(data_scaled_predict.shape[1], activation='linear')(decoder_output)
    
    # Define the model
    model = Model(inputs, outputs)
    
    # Compile the model with a lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Use Adam optimizer
                  loss=custom_loss)
    return model

# Boosted Monte Carlo Dropout Prediction
def boosted_monte_carlo_dropout_predict(model, X, n_samples=100, noise_scale=1.0):  # Reduced noise scale
    predictions = []
    for _ in range(n_samples):
        noisy_X = X + noise_scale * np.random.normal(size=X.shape)
        pred = model(noisy_X, training=True)  # Enable dropout during inference
        predictions.append(pred.numpy())
    return np.mean(predictions, axis=0), np.std(predictions, axis=0)

# Clip predictions to valid range
def clip_predictions(predictions, min_val, max_val):
    return np.clip(predictions, min_val, max_val)

# Ensemble predictions with clipping
def ensemble_predictions(models, X, n_samples=100, min_val=None, max_val=None):
    all_predictions = []
    for model in models:
        mean_pred, _ = boosted_monte_carlo_dropout_predict(model, X, n_samples=n_samples)
        if min_val is not None and max_val is not None:
            mean_pred = clip_predictions(mean_pred, min_val, max_val)
        all_predictions.append(mean_pred)
    return np.mean(all_predictions, axis=0)

# Create sequences
sequence_length = best_architectures[0]['sequence_length']
X_predict, y_predict = create_sequences(data_scaled_predict, sequence_length)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_predict, y_predict, test_size=TEST_SIZE, shuffle=False)

# Train and predict using the best architectures
models = []
for i, arch in enumerate(best_architectures):
    print(f"Training and predicting with model {i + 1}/{len(best_architectures)}")
    model = build_and_compile_model(arch)
    
    # Early stopping and learning rate scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=arch['batch_size'],
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model {i + 1} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    models.append(model)

# Get min and max values from training data
min_val = np.min(data_scaled_predict, axis=0)
max_val = np.max(data_scaled_predict, axis=0)

# Ensemble predictions with clipping
final_prediction_scaled = ensemble_predictions(
    models, 
    X_predict[-1].reshape(1, sequence_length, data_scaled_predict.shape[1]),
    min_val=min_val,
    max_val=max_val
)
final_prediction_original = scaler.inverse_transform(final_prediction_scaled)
print("Final Ensemble Prediction (Original Scale):", final_prediction_original)