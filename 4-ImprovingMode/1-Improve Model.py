import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os


class WeatherPredictor:
    def __init__(self, input_sequence_length=30, prediction_days=7):
        self.input_sequence_length = input_sequence_length
        self.prediction_days = prediction_days
        self.feature_columns = [
            'TemperatureMax', 'TemperatureMin',
            'HumidityMax', 'HumidityMin',
            'AirPressureMax', 'AirPressureMin'
        ]
        os.makedirs('models', exist_ok=True)

    def create_model(self, input_shape):
        """Create model with fixed loss function"""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=input_shape),

            # Conv1D layers
            tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # LSTM layer
            tf.keras.layers.LSTM(32, return_sequences=False),

            # Dense layers
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(32, activation='relu'),

            # Output layer
            tf.keras.layers.Dense(self.prediction_days * len(self.feature_columns)),
            tf.keras.layers.Reshape((self.prediction_days, len(self.feature_columns)))
        ])

        # Using built-in Huber loss instead of custom loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mae']
        )

        return model

    def train_model(self, X, y, epochs=100, batch_size=32):
        """Train the model"""
        model = self.create_model((X.shape[1], X.shape[2]))

        print("\nModel Architecture:")
        model.summary()

        callbacks = [
            # Learning rate reduction
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),

            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),

            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/best_weather_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train model
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        return model, history

    def convert_to_tflite(self, model):
        """Convert to TFLite"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        try:
            print("Converting model to TFLite format...")
            tflite_model = converter.convert()

            # Save TFLite model
            with open('models/weather_model.tflite', 'wb') as f:
                f.write(tflite_model)

            # Save C header
            with open('models/weather_model.h', 'w') as f:
                f.write("const unsigned char weather_model[] = {")
                f.write(",".join([f"0x{b:02x}" for b in tflite_model]))
                f.write("};\n")
                f.write(f"const unsigned int weather_model_size = {len(tflite_model)};")

            return tflite_model

        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            raise


def prepare_data(file_path, sequence_length=30, prediction_days=7):
    """Prepare data for training"""
    print("Loading and preparing data...")

    # Load data
    df = pd.read_csv(file_path)

    # Define features
    features = [
        'TemperatureMax', 'TemperatureMin',
        'HumidityMax', 'HumidityMin',
        'AirPressureMax', 'AirPressureMin'
    ]

    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Create sequences
    X = []
    y = []

    for i in range(len(scaled_data) - sequence_length - prediction_days + 1):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[(i + sequence_length):(i + sequence_length + prediction_days)])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler


def main():
    try:
        # Prepare data
        X, y, scaler = prepare_data('thai_weather.csv')
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")

        # Create and train model
        predictor = WeatherPredictor()
        print("\nTraining model...")
        model, history = predictor.train_model(X, y)

        # Convert to TFLite
        print("\nConverting to TFLite...")
        tflite_model = predictor.convert_to_tflite(model)

        print(f"\nFinal model size: {len(tflite_model) / 1024:.2f} KB")
        print("Model training and conversion completed successfully!")

        # Save scaler
        import joblib
        joblib.dump(scaler, 'models/weather_scaler.joblib')
        print("Scaler saved for future predictions")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()