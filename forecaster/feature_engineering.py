import numpy as np
import pandas as pd


# TODO: revisit after upgrading to sklearn 1.5
class DemandFeatures:
    """Feature engineering for energy demand time series forecasting.

    Creates lag features, rolling statistics, and cyclical time encodings
    to capture temporal patterns in hourly consumption data.

    Parameters
    ----------
    lags : list of int or None
        Lag offsets in hours for the target column.
    rolling_windows : list of int or None
        Window sizes in hours for rolling statistics.
    """

    def __init__(self, lags=None, rolling_windows=None):
        self.lags = lags or [1, 24, 168]
        self.rolling_windows = rolling_windows or [24, 168]

    def create_lag_features(self, df, column="consumption"):
        """Create lag features for the target column.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with the target column.
        column : str
            Name of the column to lag.

        Returns
        -------
        pandas.DataFrame
            DataFrame with lag columns added.
        """
        df = df.copy()
        for lag in self.lags:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)
        return df

    def create_rolling_features(self, df, column="consumption"):
        """Create rolling mean and standard deviation features.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with the target column.
        column : str
            Name of the column.

        Returns
        -------
        pandas.DataFrame
            DataFrame with rolling feature columns added.
        """
        df = df.copy()
        for window in self.rolling_windows:
            df[f"{column}_roll_mean_{window}"] = df[column].rolling(window).mean()
            df[f"{column}_roll_std_{window}"] = df[column].rolling(window).std()
        return df

    @staticmethod
    def create_cyclical_features(df):
        """Encode hour and day_of_week as sin/cos pairs.

        Cyclical encoding preserves the circular nature of time
        (e.g., hour 23 is close to hour 0).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with hour and day_of_week columns.

        Returns
        -------
        pandas.DataFrame
            DataFrame with sin/cos encoded time features.
        """
        df = df.copy()

        if "hour" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        if "day_of_week" in df.columns:
            df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        return df

    def build_features(self, df, column="consumption"):
        """Apply all feature engineering steps.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.
        column : str
            Target column name.

        Returns
        -------
        pandas.DataFrame
            DataFrame with all engineered features, NaN rows dropped.
        """
        df = self.create_lag_features(df, column)
        df = self.create_rolling_features(df, column)
        df = self.create_cyclical_features(df)
        df = df.dropna().reset_index(drop=True)
        return df
