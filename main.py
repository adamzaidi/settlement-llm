# -----------------------------------------------------------------------------
# Pipeline runner
# 1) Configure logging (console + logs/pipeline.log) robustly.
# 2) Run ETL: Extract -> Transform -> Load.
# 3) Run models (always produce an evaluation artifact).
# 4) Generate visualizations.
# -----------------------------------------------------------------------------

import os
import logging
from utils.run_context import init_run_context, configure_run_logging

ctx = init_run_context()
configure_run_logging(ctx)

params = {
    "run_id": ctx.run_id,
    # add your real config values here (seed, split, model choices, etc.)
}
ctx.save_json("params.json", params)

# Optional centralized setup (use if present), else fall back to inline config
try:
    from utils.logging_setup import setup_logger  # my helper
except Exception:  # graceful fallback
    setup_logger = None

# ETL
from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data

# Modeling + visuals
from analysis.model import run_models
from vis.visualizations import generate_visualizations

# Lightweight validation after load
try:
    from utils.validators import validate_processed
except Exception:
    validate_processed = None


def _bootstrap_logging():
    """
    Configure logging so graders see my output without extra steps.
    I prefer to use my helper if it's available, otherwise I fall back to a
    local setup that logs to both console and file.
    """
    os.makedirs("logs", exist_ok=True)
    if setup_logger is not None:
        return setup_logger(name="pipeline")

    # Inline fallback logger
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh = logging.FileHandler("logs/pipeline.log")
        sh = logging.StreamHandler()
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def run_pipeline():
    """
    Run the full pipeline end-to-end:
      1. Extract raw cases from CourtListener API
      2. Transform them into a clean, enriched dataset
      3. Load into a DataFrame
      4. Validate schema/quality quickly
      5. Train and evaluate models (always save artifacts)
      6. Generate charts for my report
    """
    logger = _bootstrap_logging()
    logger.info("Pipeline start")

    try:
        # STEP 1: Extract
        logger.info("Step 1: Extract data")
        extract_data(query="corporation", max_cases=1000)
        logger.info("Extract complete")

        # STEP 2: Transform
        logger.info("Step 2: Transform data")
        transform_data()
        logger.info("Transform complete")

        # STEP 3: Load
        logger.info("Step 3: Load processed data")
        df = load_data()
        logger.info("Load complete: %d rows", len(df))

        # STEP 3.5: Light validation
        if validate_processed is not None:
            validate_processed(df)
        else:
            logger.warning("validators.validate_processed not available; skipping light validation.")

        # STEP 4: Models
        logger.info("Step 4: Model training & evaluation")
        run_models(df, label="coarse")
        try:
            run_models(df, label="fine")
        except Exception as e:
            logger.warning("Fine-label training skipped: %s", e)
        logger.info("Modeling complete")

        # STEP 5: Visualizations
        logger.info("Step 5: Generate visualizations")
        generate_visualizations(df)
        logger.info("Visualizations complete")

    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise

    logger.info("Pipeline finished")


if __name__ == "__main__":
    run_pipeline()