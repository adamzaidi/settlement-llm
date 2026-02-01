# analysis/sql/build_warehouse.py
import os
import sqlite3
import pandas as pd

PROCESSED_CSV = "data/processed/processed_data.csv"
DB_PATH = "data/warehouse.db"

def build_warehouse(processed_csv: str = PROCESSED_CSV, db_path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    df = pd.read_csv(processed_csv)

    # Minimal, analyst-friendly tables (denormalized enough to query easily)
    opinions = df[[
        "case_id",
        "court",
        "jurisdiction_state",
        "opinion_year",
        "text_word_count",
        "per_curiam",
        "citation",
        "date_created",
        "date_modified",
    ]].copy()

    outcomes = df[[
        "case_id",
        "outcome_code",
        "outcome_label_fine",
        "outcome_code_fine",
        "outcome_confidence",
        "needs_review",
        "disposition_zone_found",
        "evidence_contains_strong_phrase",
        "evidence_match_position",
        "outcome_evidence",
    ]].copy()

    # “Courts” table can be super lightweight for now
    courts = (
        df[["court", "jurisdiction_state"]]
        .drop_duplicates()
        .copy()
    )

    # Write to SQLite
    con = sqlite3.connect(db_path)
    try:
        opinions.to_sql("opinions", con, if_exists="replace", index=False)
        outcomes.to_sql("outcomes", con, if_exists="replace", index=False)
        courts.to_sql("courts", con, if_exists="replace", index=False)

        # Helpful indexes
        con.execute("CREATE INDEX IF NOT EXISTS idx_opinions_case_id ON opinions(case_id);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_case_id ON outcomes(case_id);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_opinions_court ON opinions(court);")
        con.commit()
    finally:
        con.close()

    print(f"✅ Built SQLite warehouse at: {db_path}")
    print("Tables: opinions, outcomes, courts")

if __name__ == "__main__":
    build_warehouse()