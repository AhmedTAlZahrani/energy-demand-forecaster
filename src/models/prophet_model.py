import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ProphetForecaster:
    """Facebook Prophet forecaster with daily/weekly seasonality.

    Handles trend changepoints, seasonality decomposition, and
    holiday effects automatically.

    Parameters
    ----------
    daily_seasonality : bool
        Whether to include daily seasonality component.
    weekly_seasonality : bool
        Whether to include weekly seasonality component.
    """

    def __init__(self, daily_seasonality=True, weekly_seasonality=True):
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self._model = None

    def fit(self, df, ds_col="timestamp", y_col="consumption"):
        """Fit Prophet on a time series DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with timestamp and value columns.
        ds_col : str
            Name of the datetime column.
        y_col : str
            Name of the value column.
        """
        from fbprophet import Prophet

        prophet_df = df[[ds_col, y_col]].rename(columns={ds_col: "ds", y_col: "y"})

        self._model = Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=True,
        )
        self._model.fit(prophet_df)
        print("Prophet model fitted.")

    def predict(self, periods, freq="h"):
        """Forecast future periods.

        Parameters
        ----------
        periods : int
            Number of future time steps.
        freq : str
            Frequency string (e.g., 'h' for hourly).

        Returns
        -------
        pandas.DataFrame
            DataFrame with ds, yhat, yhat_lower, yhat_upper.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        future = self._model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self._model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)

    def plot_components(self):
        """Return Prophet component plots (trend, seasonality).

        Returns
        -------
        matplotlib.figure.Figure
            Component decomposition plot.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        future = self._model.make_future_dataframe(periods=0)
        forecast = self._model.predict(future)
        return self._model.plot_components(forecast)

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
