# energy-demand-forecaster

Time series forecasting for energy demand. Batch analysis project.

## Environment

    python -m venv .venv
    source .venv/Scripts/activate
    pip install -r requirements.txt

## Tests

    pytest tests/ -v

## Run

    python -m forecaster.main

## Architecture

- `forecaster/data_loader.py` handles ingestion; `forecaster/feature_engineering.py` defines `DemandFeatures`
- NumPy docstrings; descriptive commit messages; range-based requirement pinning
