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
import json

# Constants
EPOCHS = 200
PATIENCE = 15

# Helper functions
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def build_model(params, input_shape):
    inputs = Input(shape=input_shape)
    if params['model_type'] == 1:  # Basic LSTM
        x = LSTM(params['first_layer'], return_sequences=True)(inputs)
    elif params['model_type'] == 2:  # CNN + LSTM
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = LSTM(params['first_layer'], return_sequences=True)(x)
    elif params['model_type'] == 3:  # LSTM with MultiHeadAttention
        x = LSTM(params['first_layer'], return_sequences=True)(inputs)
        # Add MultiHeadAttention layer
        attention_output = MultiHeadAttention(num_heads=4, key_dim=params['first_layer'])(x, x)
        x = Concatenate()([x, attention_output])
    
    x = Dropout(params['dropout_rate'])(x)
    x = BatchNormalization()(x)
    x = LSTM(params['second_layer'])(x)
    x = Dropout(params['dropout_rate'])(x)
    x = BatchNormalization()(x)
    x = Dense(params['dense_layer'], activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = Dense(input_shape[-1], activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(loss=Huber(), optimizer=tf.keras.optimizers.Nadam(learning_rate=params['lr']))
    return model

def boosted_monte_carlo_dropout_predict(model, X_test, n_samples=100, noise_scale=2):
    """
    Boosted Monte Carlo Dropout with increased noise scale.
    """
    np.random.seed(42)  # For reproducibility
    predictions = []
    for _ in range(n_samples):
        # Add boosted noise to the model's predictions
        prediction = model.predict(X_test, verbose=0)
        noise = np.random.normal(loc=0, scale=noise_scale, size=prediction.shape)
        predictions.append(prediction + noise)
    return np.mean(predictions, axis=0), np.std(predictions, axis=0)

def update_model(model, new_data, sequence_length):
    X_new, y_new = create_sequences(new_data, sequence_length)
    model.fit(X_new, y_new, epochs=10, batch_size=32, verbose=0)

def rescale_and_round_predictions(predictions, scaler, min_value=1, max_value=49):
    """
    Rescale predictions to the original scale and round them to integers.
    Ensure the numbers fall within the valid range [min_value, max_value].
    """
    rescaled_predictions = scaler.inverse_transform(predictions)
    rounded_predictions = np.round(rescaled_predictions).astype(int)
    # Clip values to ensure they are within the valid range
    rounded_predictions = np.clip(rounded_predictions, min_value, max_value)
    return rounded_predictions

def add_noise(predictions, noise_std=0.1):
    """
    Add Gaussian noise to the predictions to introduce variability.
    """
    noise = np.random.normal(0, noise_std, predictions.shape)
    return predictions + noise

def diversify_predictions(predictions, scaler, n_variants=5, noise_std=0.1):
    """
    Generate multiple variants of predictions by adding noise and rescaling.
    """
    diversified_predictions = []
    for _ in range(n_variants):
        noisy_predictions = add_noise(predictions, noise_std)
        rescaled_predictions = rescale_and_round_predictions(noisy_predictions, scaler)
        diversified_predictions.append(rescaled_predictions)
    return diversified_predictions

def select_most_diverse_variant(diversified_predictions):
    """
    Select the variant with the most diverse Star 1 & 2 predictions.
    Diversity is measured as the absolute difference between Star 1 and Star 2.
    """
    diversity_scores = []
    for variant in diversified_predictions:
        star1, star2 = variant[0][-2], variant[0][-1]  # Star 1 & 2 are the last two numbers
        diversity_scores.append(abs(star1 - star2))
    most_diverse_index = np.argmax(diversity_scores)
    return diversified_predictions[most_diverse_index]

# Load data
data = pd.read_csv("06_DATA_Predict.csv")
data = data.drop(columns=data.columns[0])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Load top 10 models
with open("top_10_models.json", "r") as f:
    top_models = json.load(f)

# Train and predict with each model
for i, params in enumerate(top_models):
    # print(f"Training and predicting with model {i + 1}")
    
    # Use sequence_length from params
    sequence_length = params['sequence_length']
    X, y = create_sequences(data_scaled, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_model(params, X_train.shape[1:])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=params['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )

    # Predict with boosted Monte Carlo Dropout
    predictions, uncertainty = boosted_monte_carlo_dropout_predict(model, X_test, noise_scale=2)
    # print(f"Raw Predictions: {predictions[-1]}")
    # print(f"Uncertainty: {uncertainty[-1]}")

    # Rescale and round predictions to get the predicted sequence of numbers
    predicted_numbers = rescale_and_round_predictions(predictions[-1].reshape(1, -1), scaler)
    #print(f"Predicted Numbers: {predicted_numbers[0]}")

    # Diversify predictions by adding noise and generating multiple variants
    diversified_predictions = diversify_predictions(predictions[-1].reshape(1, -1), scaler, n_variants=5)
    
    # Select the most diverse variant for Star 1 & 2
    most_diverse_variant = select_most_diverse_variant(diversified_predictions)
    print(f"Most Diverse Variant (Star 1 & 2): {most_diverse_variant[0]}")

    # Update model with new data
    update_model(model, data_scaled, sequence_length)