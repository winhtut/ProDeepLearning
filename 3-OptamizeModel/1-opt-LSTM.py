import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class SimpleOptimizedWeatherPredictor:
    def __init__(self, input_sequence_length=30, prediction_days=7):
        self.input_sequence_length = input_sequence_length
        self.prediction_days = prediction_days
        self.scalers = {}
        self.feature_columns = [
            'TemperatureMax', 'TemperatureMin',
            'HumidityMax', 'HumidityMin',
            'AirPressureMax', 'AirPressureMin'
        ]

    def create_optimized_model(self, input_shape):
        """Create a lightweight model optimized for ESP32"""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=input_shape),

            # Use simpler layers instead of LSTM
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),

            # Smaller dense layers
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.prediction_days * len(self.feature_columns)),
            tf.keras.layers.Reshape((self.prediction_days, len(self.feature_columns)))
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def convert_to_tflite(self, model):
        """Convert to TFLite with optimization options"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Enable optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Enable quantization
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        # Convert model
        try:
            print("Converting model to TFLite format...")
            tflite_model = converter.convert()

            # Save the model
            with open('weather_model_small.tflite', 'wb') as f:
                f.write(tflite_model)

            # Generate C header file
            self._generate_c_header(tflite_model)

            print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
            return tflite_model

        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            raise

    def _generate_c_header(self, tflite_model):
        """Generate C header file for ESP32"""
        with open('weather_model_small.h', 'w') as f:
            f.write("#ifndef WEATHER_MODEL_SMALL_H\n")
            f.write("#define WEATHER_MODEL_SMALL_H\n\n")

            # Align data for better memory access
            f.write("#include <pgmspace.h>\n\n")
            f.write("alignas(8) ")
            f.write("const unsigned char weather_model[] PROGMEM = {\n")

            # Write hex data in chunks
            chunk_size = 12
            hex_data = [f"0x{b:02x}" for b in tflite_model]
            chunks = [", ".join(hex_data[i:i + chunk_size])
                      for i in range(0, len(hex_data), chunk_size)]

            f.write(",\n".join(f"    {chunk}" for chunk in chunks))
            f.write("\n};\n\n")

            # Add size constant
            f.write(f"const unsigned int weather_model_size = {len(tflite_model)};\n\n")
            f.write("#endif\n")


def optimize_and_test():
    """Main function to optimize and test the model"""
    try:
        # Initialize predictor
        predictor = SimpleOptimizedWeatherPredictor()

        # Create the optimized model
        print("Creating optimized model...")
        input_shape = (30, 9)  # Adjust based on your features
        model = predictor.create_optimized_model(input_shape)

        # Print model summary
        print("\nModel Architecture:")
        model.summary()

        # Convert to TFLite
        print("\nConverting to TFLite...")
        tflite_model = predictor.convert_to_tflite(model)

        print("\nOptimization complete!")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")

        # Test the TFLite model
        print("\nTesting TFLite model...")
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("\nModel Details:")
        print(f"Input Shape: {input_details[0]['shape']}")
        print(f"Output Shape: {output_details[0]['shape']}")

    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        raise


if __name__ == "__main__":
    optimize_and_test()