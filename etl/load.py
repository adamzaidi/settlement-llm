# -----------------------------------------------------------------------------
# Purpose:
# I use this module to load the processed CSV into a pandas DataFrame
# so I can pass it into modeling and visualization steps.
#
# CHANGE:
# - Adds optional SQLite persistence so the project "feels like a tool":
#   * load CSV -> DataFrame (default)
#   * optionally write to SQLite (local warehouse)
#   * optionally read from SQLite instead of CSV
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Literal

import pandas as pd

logger = logging.getLogger("pipeline")

# Default location of the processed dataset
PROCESSED_PATH = "data/processed/processed_data.csv"

# CHANGE: local warehouse defaults
SQLITE_PATH = os.getenv("SQLITE_PATH", "data/warehouse/inst414.db")
SQLITE_TABLE = os.getenv("SQLITE_TABLE", "processed_cases")


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _connect_sqlite(db_path: str) -> sqlite3.Connection:
    _ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path)
    # CHANGE: make SQLite a bit more resilient/performance-friendly for local use
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def _write_df_to_sqlite(df: pd.DataFrame, db_path: str, table: str, if_exists: str = "replace") -> None:
    """
    CHANGE: persist processed dataset into SQLite.
    """
    conn = _connect_sqlite(db_path)
    try:
        df.to_sql(table, conn, if_exists=if_exists, index=False)

        # CHANGE: add a few helpful indexes for typical analysis queries
        cols = set(c.lower() for c in df.columns)
        if "opinion_id" in cols:
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_opinion_id ON {table}(opinion_id);')
        if "court" in cols:
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_court ON {table}(court);')
        if "jurisdiction_state" in cols:
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_state ON {table}(jurisdiction_state);')
        if "opinion_year" in cols:
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_year ON {table}(opinion_year);')
        if "outcome_label_fine" in cols:
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_outcome_fine ON {table}(outcome_label_fine);')
        if "outcome_code" in cols:
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_outcome_code ON {table}(outcome_code);')

        conn.commit()
        logger.info("Wrote SQLite table '%s' to %s (rows=%d)", table, db_path, len(df))
    finally:
        conn.close()


def _read_df_from_sqlite(db_path: str, table: str) -> pd.DataFrame:
    """
    CHANGE: load dataset from SQLite instead of CSV (useful for demos).
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"SQLite DB not found at: {db_path}")

    conn = _connect_sqlite(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
        return df
    finally:
        conn.close()


def load_data(
    path: str = PROCESSED_PATH,
    *,
    source: Literal["csv", "sqlite", "auto"] = "auto",
    sqlite_path: str = SQLITE_PATH,
    sqlite_table: str = SQLITE_TABLE,
    write_sqlite: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Load the processed data into a pandas DataFrame.

    Default behavior remains CSV -> DataFrame.

    CHANGE: optional SQLite behaviors:
      - source="sqlite": read from SQLite table
      - source="auto": prefer SQLite if it exists, else CSV
      - write_sqlite=True: after reading CSV, write to SQLite warehouse

    Args:
        path (str): CSV file path to read from (default processed CSV).
        source: "csv" | "sqlite" | "auto"
        sqlite_path: location for local SQLite DB
        sqlite_table: table name inside SQLite
        write_sqlite: if True, persist the loaded df to SQLite.
                      if None, respects env LOAD_TO_SQLITE=1.

    Returns:
        pd.DataFrame: The processed data ready for analysis.
    """
    logger.info("Entering load_data")

    # CHANGE: allow env-based control without changing code elsewhere
    if write_sqlite is None:
        write_sqlite = os.getenv("LOAD_TO_SQLITE", "0").strip() == "1"

    if source == "auto":
        # Prefer SQLite if it exists and the table seems present
        if Path(sqlite_path).exists():
            try:
                df = _read_df_from_sqlite(sqlite_path, sqlite_table)
                logger.info(
                    "Loaded processed data from SQLite (db=%s, table=%s, rows=%d, cols=%d)",
                    sqlite_path, sqlite_table, df.shape[0], df.shape[1]
                )
                return df
            except Exception as e:
                logger.warning("SQLite auto-load failed (%s); falling back to CSV.", e)

        source = "csv"

    if source == "sqlite":
        df = _read_df_from_sqlite(sqlite_path, sqlite_table)
        logger.info(
            "Loaded processed data from SQLite (db=%s, table=%s, rows=%d, cols=%d)",
            sqlite_path, sqlite_table, df.shape[0], df.shape[1]
        )
        return df

    # Default: CSV
    df = pd.read_csv(path)
    logger.info("Loaded processed data from CSV (rows=%d, cols=%d)", df.shape[0], df.shape[1])

    # CHANGE: optional persistence step
    if write_sqlite:
        try:
            _write_df_to_sqlite(df, sqlite_path, sqlite_table, if_exists="replace")
        except Exception as e:
            logger.warning("Failed writing to SQLite (%s). Continuing with DataFrame.", e)

    return df