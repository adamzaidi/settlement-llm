# 1) Set up logging to see progress + errors.
# 2) Runs the ETL pipeline: Extract -> Transform -> Load. 
# 3) Trains + evaluates models with coarse + fine labels.
# 4) Save visualizations for the report.

import logging
from utils.logging_setup import setup_logger

# 1) ETL imports
from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data

# 2) Modeling + visuals
from analysis.model import run_models
from vis.visualizations import generate_visualizations


def run_pipeline():

    """
    Runs the full data pipeline end-to-end.
    """

    # STEP 0: Logger (prints + saves to logs/pipeline.log)
    logger = setup_logger(name="pipeline")
    logger.info("Pipeline start")

    try:
        # STEP 1: Extract raw data from API (writes to data/extracted/)
        logger.info("Step 1: Extract data")
        extract_data(query="corporation", max_cases=1000)  # adjust as needed
        logger.info("Extract complete")

        # STEP 2: Transform to clean, tidy data (writes to data/processed/)
        logger.info("Step 2: Transform data")
        transform_data()
        logger.info("Transform complete")

        # STEP 3: Load processed data to a DataFrame for analysis
        logger.info("Step 3: Load processed data")
        df = load_data()
        logger.info("Load complete: %d rows", len(df))

        # STEP 4: Train + evaluate models (saves reports to data/model-eval/)
        logger.info("Step 4: Model training & evaluation")

        # STEP 4a. Coarse labels: Loss/Win/Settlement
        run_models(df, label="coarse")

        # STEP 4b. Fine labels: Loss/Win/Mixed/Partial/Settlement/Other (if available)
        try:
            run_models(df, label="fine")
        except Exception as e:
            logger.warning("Fine-label training skipped: %s", e)

        logger.info("Modeling complete")

        # STEP 5: Visualizations (saves charts to data/outputs/)
        logger.info("Step 5: Generate visualizations")
        generate_visualizations(df)
        logger.info("Visualizations complete")

    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise

    logger.info("Pipeline finished")


if __name__ == "__main__":
    run_pipeline()
