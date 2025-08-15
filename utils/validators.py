# 1) Check that DataFrames have the right data before we use them.
# 2) This helps catch mistakes early (like missing data or columns).

from typing import Iterable
import pandas as pd
import logging


def assert_nonempty_df(df: pd.DataFrame, name: str, logger: logging.Logger) -> None:
    """
    Check if the DataFrame:
    - Exists (is not None)
    - Is actually a DataFrame
    - Has at least one row of data

    If any of these checks fail → log an error and stop the program.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.error("%s is empty or not a DataFrame", name)
        raise ValueError(f"{name} is empty or not a DataFrame")


def require_columns(df: pd.DataFrame, required: Iterable[str], logger: logging.Logger) -> None:
    """
    Check if the DataFrame has all the required columns.

    Steps:
    1. Compare the DataFrame's columns to the list of required ones.
    2. If any are missing → log an error and stop the program.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing required columns %s", missing)
        raise KeyError(f"Missing required columns: {missing}")