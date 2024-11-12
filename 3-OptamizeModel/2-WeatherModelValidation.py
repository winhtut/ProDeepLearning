import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


class WeatherModelValidator:
    def __init__(self, model_path='weather_model_small.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.scalers = {}

    def prepare_data(self, df):
        """Prepare and scale the data"""
        # Define feature columns
        feature_columns = [
            'TemperatureMax', 'TemperatureMin',
            'HumidityMax', 'HumidityMin',
            'AirPressureMax', 'AirPressureMin'
        ]

        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Add time-based features
        df['Day'] = df['Timestamp'].dt.day
        df['Month'] = df['Timestamp'].dt.month
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

        # Scale all features
        scaled_features = []
        for column in feature_columns + ['Day', 'Month', 'DayOfWeek']:
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(df[column].values.reshape(-1, 1))
            self.scalers[column] = scaler
            scaled_features.append(scaled_values.flatten())

        # Create scaled dataframe
        scaled_df = pd.DataFrame(
            np.column_stack(scaled_features),
            columns=feature_columns + ['Day', 'Month', 'DayOfWeek']
        )

        return scaled_df, feature_columns

    def create_sequences(self, data, sequence_length=30, prediction_days=7):
        """Create sequences for validation"""
        X, y = [], []

        for i in range(len(data) - sequence_length - prediction_days + 1):
            # Input sequence
            X.append(data.iloc[i:(i + sequence_length)].values)
            # Output sequence (only weather parameters, not time features)
            y.append(data.iloc[(i + sequence_length):(i + sequence_length + prediction_days)]
                     .iloc[:, :6].values)  # First 6 columns are weather parameters

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def validate_predictions(self, X_test, y_test):
        """Validate model predictions against actual values"""
        predictions = []

        for i in range(len(X_test)):
            # Prepare input data
            input_data = X_test[i:i + 1]
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

            # Run inference
            self.interpreter.invoke()

            # Get prediction
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions.append(prediction[0])

        predictions = np.array(predictions)

        # Unscale predictions and actual values for meaningful metrics
        unscaled_predictions = self.unscale_predictions(predictions)
        unscaled_actual = self.unscale_predictions(y_test)

        # Calculate metrics for each feature
        metrics = {}
        feature_names = ['TemperatureMax', 'TemperatureMin', 'HumidityMax',
                         'HumidityMin', 'AirPressureMax', 'AirPressureMin']

        for i, feature in enumerate(feature_names):
            mae = mean_absolute_error(unscaled_actual[:, :, i], unscaled_predictions[:, :, i])
            rmse = np.sqrt(mean_squared_error(unscaled_actual[:, :, i], unscaled_predictions[:, :, i]))
            metrics[feature] = {'mae': mae, 'rmse': rmse}

        return {
            'metrics': metrics,
            'predictions': unscaled_predictions,
            'actual': unscaled_actual
        }

    def unscale_predictions(self, scaled_data):
        """Unscale the predictions back to original values"""
        feature_names = ['TemperatureMax', 'TemperatureMin', 'HumidityMax',
                         'HumidityMin', 'AirPressureMax', 'AirPressureMin']

        # Reshape data for unscaling
        batch_size, time_steps, n_features = scaled_data.shape
        unscaled_data = np.zeros_like(scaled_data)

        for i, feature in enumerate(feature_names):
            reshaped_data = scaled_data[:, :, i].reshape(-1, 1)
            unscaled_values = self.scalers[feature].inverse_transform(reshaped_data)
            unscaled_data[:, :, i] = unscaled_values.reshape(batch_size, time_steps)

        return unscaled_data

    def plot_validation_results(self, results, days_to_plot=7):
        """Plot actual vs predicted values"""
        actual = results['actual'][0]
        predicted = results['predictions'][0]

        feature_names = ['Temperature Max (°C)', 'Temperature Min (°C)',
                         'Humidity Max (%)', 'Humidity Min (%)',
                         'Air Pressure Max (hPa)', 'Air Pressure Min (hPa)']

        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        axes = axes.flatten()

        for i, (feature, ax) in enumerate(zip(feature_names, axes)):
            ax.plot(range(days_to_plot), actual[:days_to_plot, i],
                    'b-o', label='Actual', linewidth=2)
            ax.plot(range(days_to_plot), predicted[:days_to_plot, i],
                    'r--x', label='Predicted', linewidth=2)
            ax.set_title(feature)
            ax.set_xlabel('Days')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()


def validate_model():
    try:
        # Load your test data
        print("Loading data...")
        df = pd.read_csv('thai_weather.csv')

        # Initialize validator
        print("Initializing validator...")
        validator = WeatherModelValidator()

        # Prepare data
        print("Preparing data...")
        scaled_df, feature_columns = validator.prepare_data(df)

        # Create sequences
        print("Creating sequences...")
        X, y = validator.create_sequences(scaled_df)

        # Validate model
        print("Validating model...")
        results = validator.validate_predictions(X, y)

        # Print metrics
        print("\nValidation Results:")
        for feature, metrics in results['metrics'].items():
            print(f"\n{feature}:")
            print(f"  Mean Absolute Error: {metrics['mae']:.2f}")
            print(f"  Root Mean Square Error: {metrics['rmse']:.2f}")

        # Plot results
        print("\nPlotting results...")
        validator.plot_validation_results(results)

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    validate_model()