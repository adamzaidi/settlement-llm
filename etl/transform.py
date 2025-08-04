import pandas as pd
import numpy as np
import os

def transform_data():
    """
    Cleans and prepares extracted data for analysis:
    - Fills missing values
    - Drops large text fields (plain_text_url)
    - Encodes temp outcome_code
    """

    raw_path = "data/extracted/raw_data.csv"
    df = pd.read_csv(raw_path)

    # Standardizes column names
    df.columns = df.columns.str.lower()

    # Fills missing values for key fields
    df["case_name"] = df["case_name"].fillna("Unknown")
    df["court"] = df["court"].fillna("Unknown")
    df["citation"] = df["citation"].fillna("Unknown")

    # Drop plain_text_url column
    if "plain_text_url" in df.columns:
        df = df.drop(columns=["plain_text_url"])

    # Assigns random 0/1 outcomes to allow training (TEMP)
    df["outcome_code"] = np.random.randint(0, 2, size=len(df))

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/processed_data.csv", index=False)
    print(f"Processed {len(df)} corporate cases to data/processed/processed_data.csv")
    return df