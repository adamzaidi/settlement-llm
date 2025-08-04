import pandas as pd

def load_data():
    """
    Loads processed data from data/processed for analysis.
    """
    processed_path = "data/processed/processed_data.csv"
    df = pd.read_csv(processed_path)
    print("Loaded processed data for analysis.")
    return df