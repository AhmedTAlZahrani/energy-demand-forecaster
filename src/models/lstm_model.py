import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


class LSTMForecaster:
    """LSTM-based time series forecaster using TensorFlow/Keras.

    Two-layer LSTM with dropout regularization, trained on
    sliding window sequences of configurable length.

    Parameters
    ----------
    seq_length : int
        Number of timesteps in each input sequence.
    n_features : int
        Number of input features per timestep.
    """

    def __init__(self, seq_length=168, n_features=1):
        self.seq_length = seq_length
        self.n_features = n_features
        self._model = None
        self._scaler = MinMaxScaler()

    def build_model(self):
        """Construct the LSTM architecture.

        Returns
        -------
        tensorflow.keras.Model
            Compiled Keras sequential model.
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential([
            LSTM(128, return_sequences=True,
                 input_shape=(self.seq_length, self.n_features)),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        self._model = model
        return model

    def create_sequences(self, data):
        """Create sliding window sequences for training.

        Parameters
        ----------
        data : numpy.ndarray
            1D array of scaled values.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            X and y arrays for supervised learning.
        """
        X, y = [], []
        for i in range(self.seq_length, len(data)):
            X.append(data[i - self.seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(self, series, epochs=50, batch_size=64, validation_split=0.1):
        """Train the LSTM model on a time series.

        Parameters
        ----------
        series : numpy.ndarray
            1D array of raw values.
        epochs : int
            Number of training epochs.
        batch_size : int
            Training batch size.
        validation_split : float
            Fraction of data for validation.

        Returns
        -------
        tensorflow.keras.callbacks.History
            Training history object.
        """
        from tensorflow.keras.callbacks import EarlyStopping

        scaled = self._scaler.fit_transform(series.reshape(-1, 1)).flatten()
        X, y = self.create_sequences(scaled)
        X = X.reshape(-1, self.seq_length, self.n_features)

        if self._model is None:
            self.build_model()

        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        history = self._model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1,
        )
        return history

    def predict(self, series):
        """Generate predictions for a series.

        Parameters
        ----------
        series : numpy.ndarray
            1D array of raw values (must be >= seq_length).

        Returns
        -------
        numpy.ndarray
            1D array of predictions (inverse-scaled).
        """
        scaled = self._scaler.transform(series.reshape(-1, 1)).flatten()
        X, _ = self.create_sequences(scaled)
        X = X.reshape(-1, self.seq_length, self.n_features)

        pred_scaled = self._model.predict(X, verbose=0).flatten()
        predictions = self._scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return predictions

    @staticmethod
    def evaluate(y_true, y_pred):
        """Compute forecast error metrics.

        Parameters
        ----------
        y_true : array-like
            Actual values.
        y_pred : array-like
            Predicted values.

        Returns
        -------
        dict
            Dict with RMSE, MAE, and MAPE.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE": round(mape, 2)}
