import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class WeatherPredictor:
    def __init__(self, input_sequence_length=30, prediction_days=7):
        self.input_sequence_length = input_sequence_length
        self.prediction_days = prediction_days
        self.scalers = {}
        self.feature_columns = [
            'TemperatureMax', 'TemperatureMin',
            'HumidityMax', 'HumidityMin',
            'AirPressureMax', 'AirPressureMin'
        ]

    def prepare_data(self, file_path):
        """
        Prepare the data for training
        """
        try:
            # Load the data
            print(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)

            # Print the first few rows and columns for debugging
            print("\nFirst few rows of the data:")
            print(df.head())
            print("\nColumns in the dataset:")
            print(df.columns.tolist())

            # Check for required columns
            missing_columns = [col for col in self.feature_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Convert timestamp to datetime
            print("\nConverting timestamp...")
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values('Timestamp')

            # Add time-based features
            print("Adding time-based features...")
            df['Day'] = df['Timestamp'].dt.day
            df['Month'] = df['Timestamp'].dt.month
            df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

            # Handle missing values if any
            if df.isnull().any().any():
                print("\nHandling missing values...")
                df = df.interpolate(method='linear')

            # Scale the features
            print("Scaling features...")
            scaled_features = []

            for column in self.feature_columns + ['Day', 'Month', 'DayOfWeek']:
                scaler = MinMaxScaler()
                scaled_values = scaler.fit_transform(df[column].values.reshape(-1, 1))
                self.scalers[column] = scaler
                scaled_features.append(scaled_values.flatten())

            # Create a new dataframe with scaled values
            scaled_df = pd.DataFrame(
                np.column_stack(scaled_features),
                columns=self.feature_columns + ['Day', 'Month', 'DayOfWeek']
            )

            # Create sequences
            print("Creating sequences...")
            X, y = self._create_sequences(scaled_df)

            print(f"\nSequences created: {X.shape}, {y.shape}")

            return X, y, df['Timestamp']

        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            raise

    def _create_sequences(self, data):
        """
        Create sequences for training
        """
        X, y = [], []

        for i in range(len(data) - self.input_sequence_length - self.prediction_days + 1):
            # Input sequence
            X.append(data.iloc[i:(i + self.input_sequence_length)].values)

            # Output sequence (next 7 days, only weather parameters)
            target_sequence = data.iloc[
                              (i + self.input_sequence_length):(i + self.input_sequence_length + self.prediction_days)
                              ][self.feature_columns].values
            y.append(target_sequence)

        return np.array(X), np.array(y)

    def create_model(self, input_shape):
        """Create the LSTM model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.prediction_days * len(self.feature_columns)),
            tf.keras.layers.Reshape((self.prediction_days, len(self.feature_columns)))
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train_model(self, X, y):
        """Train the model"""
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = self.create_model((X.shape[1], X.shape[2]))
        print("\nModel Architecture:")
        model.summary()

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        return model, history, (X_test, y_test)

    def predict_next_week(self, model, last_sequence):
        """Predict the next 7 days"""
        prediction = model.predict(last_sequence.reshape(1, self.input_sequence_length, -1))
        prediction = prediction.reshape(self.prediction_days, len(self.feature_columns))

        results = pd.DataFrame()

        for i, feature in enumerate(self.feature_columns):
            results[feature] = self.scalers[feature].inverse_transform(
                prediction[:, i].reshape(-1, 1)).flatten()

        return results

    def plot_predictions(self, history, predictions):
        """Plot training history and predictions"""
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Plot predictions
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        days = range(self.prediction_days)

        # Temperature plot
        ax1.plot(days, predictions['TemperatureMax'], 'r-', label='Max Temperature')
        ax1.plot(days, predictions['TemperatureMin'], 'b-', label='Min Temperature')
        ax1.set_title('Predicted Temperature for Next 7 Days')
        ax1.set_xlabel('Days from now')
        ax1.set_ylabel('Temperature (°C)')
        ax1.legend()
        ax1.grid(True)

        # Humidity plot
        ax2.plot(days, predictions['HumidityMax'], 'r-', label='Max Humidity')
        ax2.plot(days, predictions['HumidityMin'], 'b-', label='Min Humidity')
        ax2.set_title('Predicted Humidity for Next 7 Days')
        ax2.set_xlabel('Days from now')
        ax2.set_ylabel('Humidity (%)')
        ax2.legend()
        ax2.grid(True)

        # Air Pressure plot
        ax3.plot(days, predictions['AirPressureMax'], 'r-', label='Max Air Pressure')
        ax3.plot(days, predictions['AirPressureMin'], 'b-', label='Min Air Pressure')
        ax3.set_title('Predicted Air Pressure for Next 7 Days')
        ax3.set_xlabel('Days from now')
        ax3.set_ylabel('Air Pressure (hPa)')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

        # Table view of predictions
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.axis('tight')
        ax.axis('off')

        table_data = predictions.round(2)
        table_data.index = [f'Day {i + 1}' for i in range(len(table_data))]

        table = ax.table(cellText=table_data.values,
                         colLabels=table_data.columns,
                         rowLabels=table_data.index,
                         cellLoc='center',
                         loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        plt.title('Detailed Predictions for Next 7 Days')
        plt.show()


def convert_to_tflite(model):
    """Convert Keras model to TFLite format with proper settings for LSTM"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable experimental features needed for LSTM
    converter.experimental_enable_resource_variables = True

    # Add TF ops and disable tensor list ops lowering
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    # Set optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]

    # Convert the model
    try:
        print("Converting model to TFLite format...")
        tflite_model = converter.convert()

        # Save the model
        with open('thai_weather_model.tflite', 'wb') as f:
            f.write(tflite_model)

        # Save as C array for ESP32
        c_file_content = "const unsigned char weather_model[] = {"
        c_file_content += ",".join([f"0x{b:02x}" for b in tflite_model])
        c_file_content += "};\n"
        c_file_content += f"const unsigned int weather_model_size = {len(tflite_model)};\n"

        with open('weather_model.h', 'w') as f:
            f.write(c_file_content)

        print(f"Model successfully converted and saved!")
        print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")

        return tflite_model

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise
def main():
    # Initialize predictor
    predictor = WeatherPredictor(input_sequence_length=30, prediction_days=7)

    try:
        # Load and prepare data
        print("Step 1: Loading and preparing data...")
        X, y, timestamps = predictor.prepare_data('thai_weather.csv')
        print(f"Data shape: Input={X.shape}, Output={y.shape}")

        # Train model
        print("\nStep 2: Training model...")
        model, history, test_data = predictor.train_model(X, y)

        # Make predictions
        print("\nStep 3: Making predictions...")
        last_sequence = X[-1]
        predictions = predictor.predict_next_week(model, last_sequence)

        # Plot results
        print("\nStep 4: Plotting results...")
        predictor.plot_predictions(history, predictions)

        # Print predictions
        print("\nPredictions for next 7 days:")
        print(predictions.round(2))

        # Save model with new conversion function
        print("\nStep 5: Converting model for ESP32...")
        tflite_model = convert_to_tflite(model)

        print("Process completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()