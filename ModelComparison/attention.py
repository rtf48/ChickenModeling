import tensorflow as tf
from tensorflow.keras import layers, models
from scikeras.wrappers import KerasRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import dataset
import listStorage as ls
import analysis

# Generate synthetic regression data
X, y = dataset.get_data(ls.target_features_comp,'akp U per ml')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a custom model with an attention mechanism
def build_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))

    # Expand to sequence for attention (1 timestep)
    x = layers.Reshape((1, input_dim))(inputs)

    # Self-attention mechanism
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=4)(x, x)
    x = layers.Add()([x, attention])  # Residual connection
    x = layers.GlobalAveragePooling1D()(x)

    # Output layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Wrap the Keras model with SciKeras
regressor = KerasRegressor(
    model=build_model,
    model__input_dim=X.shape[1],
    epochs=500,
    batch_size=200,
    verbose=0
)

# Train the model
regressor.fit(X_train, y_train)

print(analysis.evaluate(regressor, X_val, y_val))
