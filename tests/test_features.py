import numpy as np
import pandas as pd
import pytest

from forecaster.data_loader import load_energy_data, split_by_date
from forecaster.feature_engineering import DemandFeatures


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hourly_df():
    """Build a small hourly consumption DataFrame spanning two weeks.

    Returns
    -------
    pandas.DataFrame
        DataFrame with timestamp, consumption, hour, day_of_week, month,
        is_weekend, and quarter columns (same schema as load_energy_data).
    """
    n_hours = 24 * 14  # two weeks
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    rng = np.random.default_rng(42)
    consumption = 14000 + 2000 * np.sin(2 * np.pi * np.arange(n_hours) / 24) + rng.normal(0, 200, n_hours)

    df = pd.DataFrame({"timestamp": timestamps, "consumption": consumption})
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"] = df["timestamp"].dt.quarter
    return df


@pytest.fixture
def tmp_csv(tmp_path, hourly_df):
    """Write the hourly fixture to a temporary CSV for data_loader tests.

    Returns
    -------
    pathlib.Path
        Path to the CSV file.
    """
    csv_path = tmp_path / "energy.csv"
    out = hourly_df[["timestamp", "consumption"]].rename(
        columns={"timestamp": "Datetime", "consumption": "AEP_MW"}
    )
    out.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# DemandFeatures — init defaults
# ---------------------------------------------------------------------------

class TestDemandFeaturesInit:
    """Verify constructor defaults and custom overrides."""

    def test_default_lags(self):
        """Default lags should be [1, 24, 168]."""
        fe = DemandFeatures()
        assert fe.lags == [1, 24, 168]

    def test_default_rolling_windows(self):
        """Default rolling windows should be [24, 168]."""
        fe = DemandFeatures()
        assert fe.rolling_windows == [24, 168]

    def test_custom_lags(self):
        """Custom lags passed at init should be preserved."""
        fe = DemandFeatures(lags=[3, 6])
        assert fe.lags == [3, 6]

    def test_custom_rolling_windows(self):
        """Custom rolling windows passed at init should be preserved."""
        fe = DemandFeatures(rolling_windows=[12])
        assert fe.rolling_windows == [12]


# ---------------------------------------------------------------------------
# DemandFeatures — lag features
# ---------------------------------------------------------------------------

class TestLagFeatures:
    """Test create_lag_features for various lag configurations."""

    @pytest.mark.parametrize("lags", [
        [1],
        [1, 24],
        [1, 24, 168],
    ], ids=["lag-1", "lag-1-24", "lag-1-24-168"])
    def test_lag_columns_created(self, hourly_df, lags):
        """Each requested lag should produce a corresponding column.

        Parameters
        ----------
        hourly_df : pandas.DataFrame
            Fixture with hourly consumption data.
        lags : list of int
            Lag offsets to test.
        """
        fe = DemandFeatures(lags=lags)
        result = fe.create_lag_features(hourly_df)
        for lag in lags:
            assert f"consumption_lag_{lag}" in result.columns

    def test_lag_values_correct(self, hourly_df):
        """Lag-1 column should equal the consumption column shifted by 1."""
        fe = DemandFeatures(lags=[1])
        result = fe.create_lag_features(hourly_df)
        expected = hourly_df["consumption"].shift(1)
        pd.testing.assert_series_equal(
            result["consumption_lag_1"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_lag_does_not_mutate_input(self, hourly_df):
        """The original DataFrame should not be modified."""
        original_cols = set(hourly_df.columns)
        fe = DemandFeatures(lags=[1])
        fe.create_lag_features(hourly_df)
        assert set(hourly_df.columns) == original_cols

    @pytest.mark.parametrize("column", ["consumption", "hour"])
    def test_lag_custom_column(self, hourly_df, column):
        """Lag features should work on arbitrary numeric columns.

        Parameters
        ----------
        hourly_df : pandas.DataFrame
            Fixture.
        column : str
            Column to create lags for.
        """
        fe = DemandFeatures(lags=[1])
        result = fe.create_lag_features(hourly_df, column=column)
        assert f"{column}_lag_1" in result.columns


# ---------------------------------------------------------------------------
# DemandFeatures — rolling features
# ---------------------------------------------------------------------------

class TestRollingFeatures:
    """Test create_rolling_features for various window sizes."""

    @pytest.mark.parametrize("windows", [
        [24],
        [24, 168],
    ], ids=["win-24", "win-24-168"])
    def test_rolling_columns_created(self, hourly_df, windows):
        """Rolling mean and std columns should appear for each window.

        Parameters
        ----------
        hourly_df : pandas.DataFrame
            Fixture.
        windows : list of int
            Window sizes in hours.
        """
        fe = DemandFeatures(rolling_windows=windows)
        result = fe.create_rolling_features(hourly_df)
        for w in windows:
            assert f"consumption_roll_mean_{w}" in result.columns
            assert f"consumption_roll_std_{w}" in result.columns

    def test_rolling_mean_value(self, hourly_df):
        """Rolling mean with window=3 should match pandas rolling(3).mean()."""
        fe = DemandFeatures(rolling_windows=[3])
        result = fe.create_rolling_features(hourly_df)
        expected = hourly_df["consumption"].rolling(3).mean()
        pd.testing.assert_series_equal(
            result["consumption_roll_mean_3"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_rolling_does_not_mutate_input(self, hourly_df):
        """The original DataFrame should not be modified."""
        original_cols = set(hourly_df.columns)
        fe = DemandFeatures(rolling_windows=[24])
        fe.create_rolling_features(hourly_df)
        assert set(hourly_df.columns) == original_cols


# ---------------------------------------------------------------------------
# DemandFeatures — cyclical features
# ---------------------------------------------------------------------------

class TestCyclicalFeatures:
    """Test create_cyclical_features for sin/cos encoding."""

    def test_cyclical_columns_created(self, hourly_df):
        """hour_sin, hour_cos, dow_sin, dow_cos should all appear."""
        result = DemandFeatures.create_cyclical_features(hourly_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert col in result.columns

    @pytest.mark.parametrize("hour,expected_sin", [
        (0, 0.0),
        (6, 1.0),
        (12, 0.0),
        (18, -1.0),
    ], ids=["midnight", "6am", "noon", "6pm"])
    def test_hour_sin_values(self, hour, expected_sin):
        """Cyclical hour_sin should follow sin(2*pi*hour/24).

        Parameters
        ----------
        hour : int
            Hour of day.
        expected_sin : float
            Expected sine value.
        """
        df = pd.DataFrame({"hour": [hour], "day_of_week": [0]})
        result = DemandFeatures.create_cyclical_features(df)
        assert result["hour_sin"].iloc[0] == pytest.approx(expected_sin, abs=1e-10)

    def test_cyclical_range(self, hourly_df):
        """All sin/cos values should stay within [-1, 1]."""
        result = DemandFeatures.create_cyclical_features(hourly_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert result[col].min() >= -1.0
            assert result[col].max() <= 1.0

    def test_cyclical_missing_hour_column(self):
        """If hour column is absent, hour_sin/hour_cos should not appear."""
        df = pd.DataFrame({"day_of_week": [0, 1, 2]})
        result = DemandFeatures.create_cyclical_features(df)
        assert "hour_sin" not in result.columns
        assert "dow_sin" in result.columns

    def test_cyclical_does_not_mutate_input(self, hourly_df):
        """The original DataFrame should not be modified."""
        original_cols = set(hourly_df.columns)
        DemandFeatures.create_cyclical_features(hourly_df)
        assert set(hourly_df.columns) == original_cols


# ---------------------------------------------------------------------------
# DemandFeatures — build_features (full pipeline)
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    """Test the full build_features pipeline."""

    def test_build_features_no_nans(self, hourly_df):
        """Output of build_features should contain no NaN values."""
        fe = DemandFeatures()
        result = fe.build_features(hourly_df)
        assert result.isna().sum().sum() == 0

    def test_build_features_row_count(self, hourly_df):
        """Rows should be fewer than input due to lag/rolling NaN drop."""
        fe = DemandFeatures()
        result = fe.build_features(hourly_df)
        assert len(result) < len(hourly_df)
        assert len(result) > 0

    @pytest.mark.parametrize("lags,windows", [
        ([1], [3]),
        ([1, 24], [24]),
        ([1, 24, 168], [24, 168]),
    ], ids=["small", "medium", "default-like"])
    def test_build_features_expected_columns(self, hourly_df, lags, windows):
        """All lag, rolling, and cyclical columns should be present.

        Parameters
        ----------
        hourly_df : pandas.DataFrame
            Fixture.
        lags : list of int
            Lag offsets.
        windows : list of int
            Rolling window sizes.
        """
        fe = DemandFeatures(lags=lags, rolling_windows=windows)
        result = fe.build_features(hourly_df)
        for lag in lags:
            assert f"consumption_lag_{lag}" in result.columns
        for w in windows:
            assert f"consumption_roll_mean_{w}" in result.columns
            assert f"consumption_roll_std_{w}" in result.columns
        assert "hour_sin" in result.columns


# ---------------------------------------------------------------------------
# data_loader — load_energy_data
# ---------------------------------------------------------------------------

class TestLoadEnergyData:
    """Test the CSV loading function."""

    def test_loads_correct_row_count(self, tmp_csv):
        """Should load all rows from the CSV."""
        df = load_energy_data(path=str(tmp_csv))
        assert len(df) == 24 * 14

    def test_derived_columns_present(self, tmp_csv):
        """Time-derived columns should be added."""
        df = load_energy_data(path=str(tmp_csv))
        for col in ["hour", "day_of_week", "month", "is_weekend", "quarter"]:
            assert col in df.columns

    def test_sorted_chronologically(self, tmp_csv):
        """Rows should be sorted by timestamp ascending."""
        df = load_energy_data(path=str(tmp_csv))
        assert df["timestamp"].is_monotonic_increasing

    def test_no_nans_after_interpolation(self, tmp_csv):
        """Consumption should have no NaN after interpolation."""
        df = load_energy_data(path=str(tmp_csv))
        assert df["consumption"].isna().sum() == 0

    @pytest.mark.parametrize("col_map", [
        {"timestamp": "Datetime", "consumption": "AEP_MW"},
        {"timestamp": "timestamp", "consumption": "value"},
    ], ids=["default-cols", "alt-cols"])
    def test_flexible_column_names(self, tmp_path, col_map):
        """Loader should handle different column naming conventions.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Pytest temp directory.
        col_map : dict
            Mapping of semantic name to CSV column name.
        """
        n = 48
        ts = pd.date_range("2024-06-01", periods=n, freq="h")
        data = pd.DataFrame({col_map["timestamp"]: ts, col_map["consumption"]: range(n)})
        csv_path = tmp_path / "alt.csv"
        data.to_csv(csv_path, index=False)

        df = load_energy_data(
            path=str(csv_path),
            datetime_col=col_map["timestamp"],
            value_col=col_map["consumption"],
        )
        assert len(df) == n
        assert "consumption" in df.columns


# ---------------------------------------------------------------------------
# data_loader — split_by_date
# ---------------------------------------------------------------------------

class TestSplitByDate:
    """Test chronological train/test splitting."""

    @pytest.mark.parametrize("test_days", [1, 3, 7])
    def test_split_sizes(self, hourly_df, test_days):
        """Train + test should equal original row count.

        Parameters
        ----------
        hourly_df : pandas.DataFrame
            Fixture.
        test_days : int
            Days to reserve for test set.
        """
        train, test = split_by_date(hourly_df, test_days=test_days)
        assert len(train) + len(test) == len(hourly_df)
        assert len(test) > 0
        assert len(train) > 0

    def test_no_temporal_leakage(self, hourly_df):
        """All train timestamps must precede all test timestamps."""
        train, test = split_by_date(hourly_df, test_days=3)
        assert train["timestamp"].max() < test["timestamp"].min()
