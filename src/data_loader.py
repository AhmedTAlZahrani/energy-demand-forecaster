import pandas as pd
import numpy as np


def load_energy_data(path="data/energy.csv", datetime_col="Datetime", value_col="AEP_MW"):
    """Load and prepare hourly energy consumption data.

    Parses timestamps, sorts chronologically, handles missing values
    via linear interpolation, and adds time-based features.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    datetime_col : str
        Name of the timestamp column.
    value_col : str
        Name of the consumption column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with timestamp index and derived time features.
    """
    df = pd.read_csv(path)

    if datetime_col in df.columns:
        df["timestamp"] = pd.to_datetime(df[datetime_col])
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.to_datetime(df.iloc[:, 0])

    if value_col in df.columns:
        df["consumption"] = pd.to_numeric(df[value_col], errors="coerce")
    elif "consumption" not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df["consumption"] = df[numeric_cols[0]]

    df = df[["timestamp", "consumption"]].sort_values("timestamp").reset_index(drop=True)
    df["consumption"] = df["consumption"].interpolate(method="linear")

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"] = df["timestamp"].dt.quarter

    print("Loaded {} records from {} to {}".format(
        len(df), df["timestamp"].min(), df["timestamp"].max()))
    return df


def split_by_date(df, test_days=30):
    """Split data chronologically into train and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with timestamp column.
    test_days : int
        Number of days to reserve for testing.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        Train and test DataFrames.
    """
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train = df[df["timestamp"] <= cutoff].copy()
    test = df[df["timestamp"] > cutoff].copy()
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test
