import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the preprocessed data
def load_data():
    # Assuming the data is stored in a numpy array format
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    return X_train, y_train, X_test, y_test
X_train, y_train, X_test, y_test = load_data()

# Ensure data is in float32 format for compatibility
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Define the hybrid model
inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
cnn = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
cnn = layers.MaxPooling1D(pool_size=2)(cnn)
cnn = layers.GlobalAveragePooling1D()(cnn)

lstm = layers.LSTM(64, activation='tanh', return_sequences=True)(inputs)
attention = layers.Attention()([lstm, lstm])
attention = layers.GlobalAveragePooling1D()(attention)
merged = layers.concatenate([cnn, attention])
dense = layers.Dense(64, activation='relu')(merged)
outputs = layers.Dense(1)(dense)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model for full epochs
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

# Print evaluation metrics
print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R^2 Score: {r2:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')

# Save the model
model.save('models/hybrid_model.h5')