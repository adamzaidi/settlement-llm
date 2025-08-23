# -----------------------------------------------------------------------------
# Purpose:
# I use this module to load the processed CSV into a pandas DataFrame
# so I can pass it into modeling and visualization steps.
# -----------------------------------------------------------------------------

import logging
import pandas as pd

logger = logging.getLogger("pipeline")

# Default location of the processed dataset
PROCESSED_PATH = "data/processed/processed_data.csv"


def load_data(path: str = PROCESSED_PATH) -> pd.DataFrame:
    """
    Load the processed data into a pandas DataFrame.

    Args:
        path (str): The CSV file path to read from.
                    By default it loads from data/processed/processed_data.csv.

    Returns:
        pd.DataFrame: The processed data ready for analysis.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.EmptyDataError: If the file exists but is empty.
        pd.errors.ParserError: If the file cannot be parsed as CSV.
    """
    logger.info("Entering load_data")
    df = pd.read_csv(path)
    logger.info("Loaded processed data for analysis (rows=%d, cols=%d)", df.shape[0], df.shape[1])
    return df