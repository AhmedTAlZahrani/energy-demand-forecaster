import pandas as pd
import plotly.graph_objects as go

from .models.arima_model import ARIMAForecaster
from .models.prophet_model import ProphetForecaster
from .models.lstm_model import LSTMForecaster


class ModelComparison:
    """Run and compare all three forecasting models."""

    def __init__(self):
        self.results = {}
        self.predictions = {}

    def run_all_models(self, train_df, test_df, forecast_steps=None):
        """Train and evaluate all models.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training DataFrame with timestamp and consumption columns.
        test_df : pandas.DataFrame
            Test DataFrame for evaluation.
        forecast_steps : int or None
            Number of steps to forecast. Defaults to len(test_df).

        Returns
        -------
        pandas.DataFrame
            Comparison DataFrame with metrics for each model.
        """
        steps = forecast_steps or len(test_df)
        y_true = test_df["consumption"].values[:steps]

        # ARIMA
        print("\n--- ARIMA ---")
        arima = ARIMAForecaster(seasonal=True, m=24)
        arima.fit(train_df["consumption"].values)
        arima_pred = arima.predict(steps)
        self.predictions["ARIMA"] = arima_pred
        self.results["ARIMA"] = arima.evaluate(y_true, arima_pred)

        # Prophet
        print("\n--- Prophet ---")
        prophet = ProphetForecaster()
        prophet.fit(train_df)
        prophet_forecast = prophet.predict(steps, freq="h")
        prophet_pred = prophet_forecast["yhat"].values
        self.predictions["Prophet"] = prophet_pred
        self.results["Prophet"] = ProphetForecaster.evaluate(y_true, prophet_pred)

        # LSTM
        print("\n--- LSTM ---")
        lstm = LSTMForecaster(seq_length=168, n_features=1)
        lstm.fit(train_df["consumption"].values, epochs=30, batch_size=64)
        lstm_pred = lstm.predict(test_df["consumption"].values)
        self.predictions["LSTM"] = lstm_pred
        self.results["LSTM"] = LSTMForecaster.evaluate(
            y_true[:len(lstm_pred)], lstm_pred
        )

        return self.comparison_table()

    def comparison_table(self):
        """Return metrics as a sorted DataFrame.

        Returns
        -------
        pandas.DataFrame
            Metrics table sorted by MAPE.
        """
        rows = [{"Model": name, **metrics} for name, metrics in self.results.items()]
        return pd.DataFrame(rows).sort_values("MAPE")

    def plot_forecasts(self, test_df):
        """Overlay all model predictions against actuals.

        Parameters
        ----------
        test_df : pandas.DataFrame
            Test DataFrame with timestamp and consumption.

        Returns
        -------
        plotly.graph_objects.Figure
            Plotly Figure with forecast overlay.
        """
        fig = go.Figure()

        timestamps = test_df["timestamp"].values
        fig.add_trace(go.Scatter(
            x=timestamps, y=test_df["consumption"].values,
            name="Actual", line=dict(color="white", width=2),
        ))

        colors = {"ARIMA": "#636EFA", "Prophet": "#EF553B", "LSTM": "#00CC96"}
        for name, pred in self.predictions.items():
            n = min(len(pred), len(timestamps))
            fig.add_trace(go.Scatter(
                x=timestamps[:n], y=pred[:n],
                name=name, line=dict(color=colors.get(name, "gray")),
            ))

        fig.update_layout(
            template="plotly_dark", height=500,
            title="Forecast Comparison: All Models vs Actual",
            xaxis_title="Time", yaxis_title="Consumption (MW)",
        )
        return fig
