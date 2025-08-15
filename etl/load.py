#Loads the processed CSV into a DataFrame for modeling/visualization.
import logging
logger = logging.getLogger("pipeline")

import pandas as pd

PROCESSED_PATH = "data/processed/processed_data.csv"

def load_data(path: str = PROCESSED_PATH) -> pd.DataFrame:
    """
    Read processed CSV. Raises if missing.
    """
    logger.info("Entering load_data")
    df = pd.read_csv(path)
    logger.info("Loaded processed data for analysis")
    return df