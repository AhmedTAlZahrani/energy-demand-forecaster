import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ARIMAForecaster:
    """Auto-ARIMA forecaster using pmdarima.

    Automatically selects (p,d,q) parameters via AIC minimization.

    Parameters
    ----------
    seasonal : bool
        Whether to include seasonal component.
    m : int
        Seasonal period (24 for hourly data).
    """

    def __init__(self, seasonal=True, m=24):
        self.seasonal = seasonal
        self.m = m
        self._model = None

    def fit(self, series):
        """Fit auto-ARIMA on a time series.

        Parameters
        ----------
        series : array-like
            1D array or Series of observed values.
        """
        import pmdarima as pm

        self._model = pm.auto_arima(
            series,
            seasonal=self.seasonal,
            m=self.m,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=5, max_q=5,
            max_P=2, max_Q=2,
        )
        print(f"ARIMA order: {self._model.order}, seasonal: {self._model.seasonal_order}")

    def predict(self, steps):
        """Forecast n steps ahead.

        Parameters
        ----------
        steps : int
            Number of future time steps to forecast.

        Returns
        -------
        numpy.ndarray
            Array of predictions.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(n_periods=steps)

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
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE": round(mape, 2)}
