"""Configuration loader using configparser for the energy demand forecaster.

Reads settings from ``config.ini`` at the project root. Each section maps
to a dict accessible via :func:`get_config`.
"""

import configparser
from pathlib import Path

_DEFAULT_INI = Path(__file__).resolve().parent.parent / "config.ini"


def get_config(path=None):
    """Load configuration from an INI file.

    Parameters
    ----------
    path : str or Path, optional
        Path to the INI file.  Defaults to ``config.ini`` in the project root.

    Returns
    -------
    configparser.ConfigParser
        Parsed configuration object.
    """
    path = Path(path) if path else _DEFAULT_INI
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    parser = configparser.ConfigParser()
    parser.read(path)
    return parser


def data_settings(cfg=None):
    """Return the [data] section as a dict.

    Parameters
    ----------
    cfg : configparser.ConfigParser, optional
        Pre-loaded config.  Loads default if *None*.

    Returns
    -------
    dict
        Data-related settings.
    """
    cfg = cfg or get_config()
    return dict(cfg["data"])


def model_settings(cfg=None):
    """Return the [model] section as a dict with typed values.

    Parameters
    ----------
    cfg : configparser.ConfigParser, optional
        Pre-loaded config.

    Returns
    -------
    dict
        Model hyperparameters with numeric values cast appropriately.
    """
    cfg = cfg or get_config()
    raw = dict(cfg["model"])
    raw["n_estimators"] = int(raw["n_estimators"])
    raw["max_depth"] = int(raw["max_depth"])
    raw["learning_rate"] = float(raw["learning_rate"])
    return raw


def output_settings(cfg=None):
    """Return the [output] section as a dict.

    Parameters
    ----------
    cfg : configparser.ConfigParser, optional
        Pre-loaded config.

    Returns
    -------
    dict
        Output path settings.
    """
    cfg = cfg or get_config()
    return dict(cfg["output"])
