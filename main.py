# -----------------------------------------------------------------------------
# Pipeline runner (TOOL MODE)
# 1) Configure logging (console + logs/pipeline.log) robustly.
# 2) Run ETL: Extract -> Transform -> Load.
# 3) Run models (always produce an evaluation artifact).
# 4) Generate visualizations.
#
# CHANGE: adds a real CLI with subcommands + run artifact packaging
# CHANGE (env-first ergonomics):
# - CLI args default to .env when present (QUERY, MAX_CASES, FAST_MODE, PAGE_SIZE, MAX_SEARCH_PAGES)
# - Precedence: CLI flags > env > code defaults
# - Passing --max-pages 0 clears any env cap (sets MAX_SEARCH_PAGES=0)
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
# .env defaults (only used if CLI does not provide values)
# -----------------------------------------------------------------------------
ENV_QUERY = os.getenv("QUERY")  # default inside extract.py is "corporation"
ENV_MAX_CASES = os.getenv("MAX_CASES")
ENV_FAST_MODE = os.getenv("FAST_MODE")  # "true"/"false"/"1"/"0"
ENV_PAGE_SIZE = os.getenv("PAGE_SIZE")
ENV_MAX_SEARCH_PAGES = os.getenv("MAX_SEARCH_PAGES")  # "0" means no cap


def _env_bool(val: Optional[str], default: Optional[bool] = None) -> Optional[bool]:
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    return default


def _env_int(val: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        return default


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
            shutil.copytree(srcp, dstp, dirs_exist_ok=True)
        else:
            shutil.copy2(srcp, dstp)
        logger.info("Packaged artifact: %s -> %s", str(srcp), str(dstp))
    except Exception as e:
        logger.warning("Failed to package artifact %s: %s", str(srcp), e)


def _package_run_artifacts(ctx, logger: logging.Logger) -> None:
    """
    Copy key outputs into runs/<run_id>/artifacts so a run is self-contained.
    """
    artifacts_dir = Path(ctx.run_dir) / "artifacts"
    _ensure_dir(artifacts_dir)

    _copy_if_exists("data/extracted/raw_data.csv", artifacts_dir / "raw_data.csv", logger)
    _copy_if_exists("data/extracted/raw_data_partial.csv", artifacts_dir / "raw_data_partial.csv", logger)
    _copy_if_exists("data/processed/processed_data.csv", artifacts_dir / "processed_data.csv", logger)
    _copy_if_exists("data/processed/review_queue.csv", artifacts_dir / "review_queue.csv", logger)

    _copy_if_exists("data/outputs", artifacts_dir / "outputs", logger)
    _copy_if_exists("data/model-eval", artifacts_dir / "model-eval", logger)

    _copy_if_exists("logs/pipeline.log", Path(ctx.run_dir) / "logs" / "pipeline.log", logger)

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
    Tool-feel CLI. Defaults flow from .env unless flags are provided.
    """
    p = argparse.ArgumentParser(
        prog="inst414-pipeline",
        description="CourtListener ETL + outcome labeling + modeling + visuals",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_run_args(sp: argparse.ArgumentParser) -> None:
        # NOTE: default=None so extract.py can use env defaults when not provided.
        sp.add_argument("--query", default=ENV_QUERY, help="Keyword query. Defaults to env QUERY if set.")
        sp.add_argument(
            "--max-cases",
            type=int,
            default=_env_int(ENV_MAX_CASES, None),
            help="Hard cap on number of opinions extracted. Defaults to env MAX_CASES if set.",
        )
        sp.add_argument(
            "--max-pages",
            type=int,
            default=_env_int(ENV_MAX_SEARCH_PAGES, None),
            help="Hard cap on /search/ pages. 0 = no cap. Defaults to env MAX_SEARCH_PAGES if set.",
        )
        # BooleanOptionalAction gives you --fast-mode / --no-fast-mode (nice)
        sp.add_argument(
            "--fast-mode",
            action=argparse.BooleanOptionalAction,
            default=_env_bool(ENV_FAST_MODE, None),
            help="Fast mode. Defaults to env FAST_MODE if set.",
        )
        sp.add_argument(
            "--page-size",
            type=int,
            default=_env_int(ENV_PAGE_SIZE, None),
            help="Search page_size override. Defaults to env PAGE_SIZE if set.",
        )
        sp.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
        sp.add_argument("--no-models", action="store_true", help="Skip model training/eval")
        sp.add_argument("--label", choices=["coarse", "fine", "both"], default="both", help="Which labels to model")

    sp_run = sub.add_parser("run", help="Run the full pipeline end-to-end")
    add_common_run_args(sp_run)

    sp_ex = sub.add_parser("extract", help="Run extract step only")
    sp_ex.add_argument("--query", default=ENV_QUERY)
    sp_ex.add_argument("--max-cases", type=int, default=_env_int(ENV_MAX_CASES, None))
    sp_ex.add_argument("--max-pages", type=int, default=_env_int(ENV_MAX_SEARCH_PAGES, None))
    sp_ex.add_argument("--fast-mode", action=argparse.BooleanOptionalAction, default=_env_bool(ENV_FAST_MODE, None))
    sp_ex.add_argument("--page-size", type=int, default=_env_int(ENV_PAGE_SIZE, None))

    sub.add_parser("transform", help="Run transform step only (reads data/extracted/raw_data.csv)")
    sub.add_parser("load", help="Run load step only (reads data/processed/processed_data.csv)")

    sp_m = sub.add_parser("model", help="Run model training/eval only")
    sp_m.add_argument("--label", choices=["coarse", "fine"], default="coarse")

    sub.add_parser("viz", help="Generate visualizations only")
    sub.add_parser("doctor", help="Quick health check: verify expected files/dirs exist")

    return p


def _apply_runtime_env(max_pages: Optional[int], logger: logging.Logger) -> None:
    """
    Let CLI control extractor behavior via env MAX_SEARCH_PAGES.
    - If max_pages is None: do nothing (use whatever env/.env has)
    - If max_pages is provided (including 0): set env explicitly
      (0 clears any cap because extractor treats 0 -> None)
    """
    if max_pages is None:
        return
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
    _apply_runtime_env(getattr(args, "max_pages", None), logger)

    extract_data(
        query=getattr(args, "query", None),
        max_cases=getattr(args, "max_cases", None),
        fast_mode=getattr(args, "fast_mode", None),
        page_size=getattr(args, "page_size", None),
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
        run_models(df, label="coarse")


def cmd_viz(df, logger: logging.Logger) -> None:
    generate_visualizations(df)


def run_pipeline_from_args(args, logger: logging.Logger, ctx) -> None:
    logger.info("Pipeline start")
    logger.info("Args: %s", vars(args))

    params = {"run_id": ctx.run_id, **vars(args)}
    ctx.save_json("params.json", params)

    try:
        logger.info("Step 1: Extract data")
        cmd_extract(args, logger)
        logger.info("Extract complete")

        logger.info("Step 2: Transform data")
        cmd_transform(logger)
        logger.info("Transform complete")

        logger.info("Step 3: Load processed data")
        df = cmd_load(logger)
        logger.info("Load complete: %d rows", len(df))

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