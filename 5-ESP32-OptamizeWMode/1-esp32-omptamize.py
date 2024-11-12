import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os


class ESP32WeatherPredictor:
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
        """Create ESP32-optimized model without LSTM"""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=input_shape),

            # Use 1D convolutions instead of LSTM
            tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling1D(2),

            tf.keras.layers.Conv1D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling1D(2),

            # Flatten for dense layers
            tf.keras.layers.Flatten(),

            # Dense layers
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(32, activation='relu'),

            # Output layer
            tf.keras.layers.Dense(self.prediction_days * len(self.feature_columns)),
            tf.keras.layers.Reshape((self.prediction_days, len(self.feature_columns)))
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train_model(self, X, y, epochs=100, batch_size=32):
        """Train the model"""
        model = self.create_model((X.shape[1], X.shape[2]))

        print("\nModel Architecture:")
        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/esp32_weather_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

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
        """Convert to TFLite with microcontroller optimizations"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Optimize for size and speed
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Use only built-in ops
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

        # Quantize to int8
        converter.representative_dataset = self._representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        try:
            print("Converting model to TFLite format...")
            tflite_model = converter.convert()

            # Save TFLite model
            tflite_path = 'models/esp32_weather_model.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            # Generate C header
            with open('models/esp32_weather_model.h', 'w') as f:
                f.write("#ifndef ESP32_WEATHER_MODEL_H\n")
                f.write("#define ESP32_WEATHER_MODEL_H\n\n")
                f.write("#include <pgmspace.h>\n\n")
                f.write("alignas(8) ")
                f.write("const unsigned char weather_model[] PROGMEM = {")
                f.write(",".join([f"0x{b:02x}" for b in tflite_model]))
                f.write("};\n\n")
                f.write(f"const unsigned int weather_model_size = {len(tflite_model)};\n\n")
                f.write("#endif  // ESP32_WEATHER_MODEL_H\n")

            print(f"\nModel successfully saved:")
            print(f"- TFLite model: {tflite_path}")
            print(f"- C header: models/esp32_weather_model.h")
            print(f"- Model size: {len(tflite_model) / 1024:.2f} KB")

            return tflite_model

        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            raise

    def _representative_dataset_gen(self):
        """Generate representative dataset for quantization"""
        for _ in range(100):
            yield [np.random.randn(1, self.input_sequence_length, len(self.feature_columns)).astype(np.float32)]


def prepare_data(file_path, sequence_length=30, prediction_days=7):
    """Prepare data for training"""
    df = pd.read_csv(file_path)

    features = [
        'TemperatureMax', 'TemperatureMin',
        'HumidityMax', 'HumidityMin',
        'AirPressureMax', 'AirPressureMin'
    ]

    scaler = MinMaxScaler(feature_range=(-1, 1))  # Better for int8 quantization
    scaled_data = scaler.fit_transform(df[features])

    X = []
    y = []

    for i in range(len(scaled_data) - sequence_length - prediction_days + 1):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[(i + sequence_length):(i + sequence_length + prediction_days)])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler


def main():
    try:
        # Prepare data
        print("Preparing data...")
        X, y, scaler = prepare_data('thai_weather.csv')
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")

        # Create and train model
        predictor = ESP32WeatherPredictor()
        print("\nTraining model...")
        model, history = predictor.train_model(X, y)

        # Convert to TFLite
        print("\nConverting to TFLite...")
        tflite_model = predictor.convert_to_tflite(model)

        # Save scaler
        import joblib
        joblib.dump(scaler, 'models/esp32_weather_scaler.joblib')
        print("\nScaler saved for future predictions")

        # Print final metrics
        print("\nTraining completed successfully!")
        print(f"Final validation loss: {min(history.history['val_loss']):.4f}")
        print(f"Final validation MAE: {min(history.history['val_mae']):.4f}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()