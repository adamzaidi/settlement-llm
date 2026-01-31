# -----------------------------------------------------------------------------
# Pipeline runner (TOOL MODE)
# 1) Configure logging (console + logs/pipeline.log) robustly.
# 2) Run ETL: Extract -> Transform -> Load.
# 3) Run models (always produce an evaluation artifact).
# 4) Generate visualizations.
#
# CHANGE: adds a real CLI with subcommands + run artifact packaging
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Optional

from utils.run_context import init_run_context, configure_run_logging

# Optional centralized setup (use if present), else fall back to inline config
try:
    from utils.logging_setup import setup_logger  # helper
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


# -----------------------------------------------------------------------------
# Logging / Run context
# -----------------------------------------------------------------------------
def _bootstrap_logging() -> logging.Logger:
    """
    Configure logging so graders see output without extra steps.

    CHANGE: avoids double handlers and duplicate logs by only attaching
    handlers if missing.
    """
    os.makedirs("logs", exist_ok=True)

    if setup_logger is not None:
        return setup_logger(name="pipeline")

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


def _ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: str | Path, dst: str | Path, logger: logging.Logger) -> None:
    srcp = Path(src)
    dstp = Path(dst)
    if not srcp.exists():
        return
    _ensure_dir(dstp.parent)
    try:
        if srcp.is_dir():
            # Python 3.11+: dirs_exist_ok supported
            shutil.copytree(srcp, dstp, dirs_exist_ok=True)
        else:
            shutil.copy2(srcp, dstp)
        logger.info("Packaged artifact: %s -> %s", str(srcp), str(dstp))
    except Exception as e:
        logger.warning("Failed to package artifact %s: %s", str(srcp), e)


def _package_run_artifacts(ctx, logger: logging.Logger) -> None:
    """
    CHANGE: copy key outputs into runs/<run_id>/artifacts so a run is self-contained.
    """
    artifacts_dir = Path(ctx.run_dir) / "artifacts"
    _ensure_dir(artifacts_dir)

    # Core data outputs
    _copy_if_exists("data/extracted/raw_data.csv", artifacts_dir / "raw_data.csv", logger)
    _copy_if_exists("data/extracted/raw_data_partial.csv", artifacts_dir / "raw_data_partial.csv", logger)
    _copy_if_exists("data/processed/processed_data.csv", artifacts_dir / "processed_data.csv", logger)
    _copy_if_exists("data/processed/review_queue.csv", artifacts_dir / "review_queue.csv", logger)

    # Visualizations + evaluation
    _copy_if_exists("data/outputs", artifacts_dir / "outputs", logger)
    _copy_if_exists("data/model-eval", artifacts_dir / "model-eval", logger)

    # Logs
    _copy_if_exists("logs/pipeline.log", Path(ctx.run_dir) / "logs" / "pipeline.log", logger)

    # Quick run summary
    summary = {
        "run_id": getattr(ctx, "run_id", None),
        "run_dir": str(getattr(ctx, "run_dir", "")),
        "artifacts_dir": str(artifacts_dir),
        "outputs": {
            "raw_data_csv": str(artifacts_dir / "raw_data.csv"),
            "processed_data_csv": str(artifacts_dir / "processed_data.csv"),
            "review_queue_csv": str(artifacts_dir / "review_queue.csv"),
            "plots_dir": str(artifacts_dir / "outputs"),
            "model_eval_dir": str(artifacts_dir / "model-eval"),
        },
    }
    try:
        (Path(ctx.run_dir) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Wrote run summary: %s", str(Path(ctx.run_dir) / "summary.json"))
    except Exception as e:
        logger.warning("Could not write summary.json: %s", e)


# -----------------------------------------------------------------------------
# CLI wiring
# -----------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """
    CHANGE: 'tool feel' CLI. Subcommands are easy to demo and resume-friendly.
    """
    p = argparse.ArgumentParser(
        prog="inst414-pipeline",
        description="CourtListener ETL + outcome labeling + modeling + visuals",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared args
    def add_common_run_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--query", default="corporation", help="Keyword query passed to CourtListener /search/?q=...")
        sp.add_argument("--max-cases", type=int, default=1000, help="Hard cap on number of opinions extracted")
        sp.add_argument("--max-pages", type=int, default=0, help="Hard cap on /search/ pages (0 = no cap)")
        sp.add_argument("--fast-mode", action="store_true", help="Fast mode (less text processing)")
        sp.add_argument("--page-size", type=int, default=0, help="Search page_size override (0 uses default)")
        sp.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
        sp.add_argument("--no-models", action="store_true", help="Skip model training/eval")
        sp.add_argument("--label", choices=["coarse", "fine", "both"], default="both", help="Which labels to model")

    # run (full pipeline)
    sp_run = sub.add_parser("run", help="Run the full pipeline end-to-end")
    add_common_run_args(sp_run)

    # extract only
    sp_ex = sub.add_parser("extract", help="Run extract step only")
    sp_ex.add_argument("--query", default="corporation")
    sp_ex.add_argument("--max-cases", type=int, default=1000)
    sp_ex.add_argument("--max-pages", type=int, default=0)
    sp_ex.add_argument("--fast-mode", action="store_true")
    sp_ex.add_argument("--page-size", type=int, default=0)

    # transform only
    sub.add_parser("transform", help="Run transform step only (reads data/extracted/raw_data.csv)")

    # load only
    sub.add_parser("load", help="Run load step only (reads data/processed/processed_data.csv)")

    # model only
    sp_m = sub.add_parser("model", help="Run model training/eval only")
    sp_m.add_argument("--label", choices=["coarse", "fine"], default="coarse")

    # viz only
    sub.add_parser("viz", help="Generate visualizations only")

    # doctor
    sub.add_parser("doctor", help="Quick health check: verify expected files/dirs exist")

    return p


def _apply_runtime_env(max_pages: int, logger: logging.Logger) -> None:
    """
    CHANGE: let CLI control behavior without rewriting ETL code.
    Your extractor already reads MAX_SEARCH_PAGES from env.
    """
    if max_pages and max_pages > 0:
        os.environ["MAX_SEARCH_PAGES"] = str(max_pages)
        logger.info("Set env MAX_SEARCH_PAGES=%s", max_pages)


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
def cmd_doctor(logger: logging.Logger) -> int:
    checks = [
        ("etl/ folder", Path("etl").exists()),
        ("data/extracted/", Path("data/extracted").exists()),
        ("data/processed/", Path("data/processed").exists()),
        ("raw_data.csv", Path("data/extracted/raw_data.csv").exists()),
        ("processed_data.csv", Path("data/processed/processed_data.csv").exists()),
        ("review_queue.csv", Path("data/processed/review_queue.csv").exists()),
    ]
    ok = True
    for name, exists in checks:
        logger.info("[doctor] %-22s : %s", name, "OK" if exists else "MISSING")
        ok = ok and bool(exists)
    return 0 if ok else 2


def cmd_extract(args, logger: logging.Logger) -> None:
    _apply_runtime_env(args.max_pages, logger)

    page_size = args.page_size if getattr(args, "page_size", 0) else None
    extract_data(
        query=args.query,
        max_cases=args.max_cases,
        fast_mode=bool(args.fast_mode),
        page_size=page_size,
    )


def cmd_transform(logger: logging.Logger) -> None:
    transform_data()


def cmd_load(logger: logging.Logger):
    df = load_data()
    if validate_processed is not None:
        validate_processed(df)
    else:
        logger.warning("validators.validate_processed not available; skipping light validation.")
    return df


def cmd_model(df, label: str, logger: logging.Logger) -> None:
    if label == "coarse":
        run_models(df, label="coarse")
    elif label == "fine":
        run_models(df, label="fine")
    else:
        # should never happen because argparse validates
        run_models(df, label="coarse")


def cmd_viz(df, logger: logging.Logger) -> None:
    generate_visualizations(df)


def run_pipeline_from_args(args, logger: logging.Logger, ctx) -> None:
    """
    CHANGE: pipeline now accepts CLI args so runs are reproducible + demo-able.
    """
    logger.info("Pipeline start")
    logger.info("Args: %s", vars(args))

    # Save run params inside runs/<run_id> (already do this, but now includes CLI)
    params = {"run_id": ctx.run_id, **vars(args)}
    ctx.save_json("params.json", params)

    try:
        # STEP 1: Extract
        logger.info("Step 1: Extract data")
        cmd_extract(args, logger)
        logger.info("Extract complete")

        # STEP 2: Transform
        logger.info("Step 2: Transform data")
        cmd_transform(logger)
        logger.info("Transform complete")

        # STEP 3: Load (+ validate)
        logger.info("Step 3: Load processed data")
        df = cmd_load(logger)
        logger.info("Load complete: %d rows", len(df))

        # STEP 4: Models
        if not getattr(args, "no_models", False):
            logger.info("Step 4: Model training & evaluation")
            if args.label in ("coarse", "both"):
                cmd_model(df, "coarse", logger)
            if args.label in ("fine", "both"):
                try:
                    cmd_model(df, "fine", logger)
                except Exception as e:
                    logger.warning("Fine-label training skipped: %s", e)
            logger.info("Modeling complete")
        else:
            logger.info("Skipping models (--no-models)")

        # STEP 5: Visualizations
        if not getattr(args, "no_viz", False):
            logger.info("Step 5: Generate visualizations")
            cmd_viz(df, logger)
            logger.info("Visualizations complete")
        else:
            logger.info("Skipping visualizations (--no-viz)")

    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise
    finally:
        # CHANGE: package artifacts even if something fails mid-run (best effort)
        try:
            _package_run_artifacts(ctx, logger)
        except Exception as e:
            logger.warning("Artifact packaging failed: %s", e)

    logger.info("Pipeline finished")


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    logger = _bootstrap_logging()

    # CHANGE: unify run context logging with pipeline logger output
    # (run_context already writes to runs/<run_id>/logs/run.log)
    ctx = init_run_context()
    configure_run_logging(ctx)

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "doctor":
        return cmd_doctor(logger)

    if args.cmd == "extract":
        cmd_extract(args, logger)
        _package_run_artifacts(ctx, logger)
        return 0

    if args.cmd == "transform":
        cmd_transform(logger)
        _package_run_artifacts(ctx, logger)
        return 0

    if args.cmd == "load":
        _ = cmd_load(logger)
        _package_run_artifacts(ctx, logger)
        return 0

    if args.cmd == "model":
        df = load_data()
        cmd_model(df, args.label, logger)
        _package_run_artifacts(ctx, logger)
        return 0

    if args.cmd == "viz":
        df = load_data()
        cmd_viz(df, logger)
        _package_run_artifacts(ctx, logger)
        return 0

    if args.cmd == "run":
        run_pipeline_from_args(args, logger, ctx)
        return 0

    logger.error("Unknown command: %s", args.cmd)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())