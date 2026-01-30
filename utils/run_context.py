from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def make_run_id() -> str:
    # Timestamp-based is simplest and readable
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def save_json(self, filename: str, payload: Dict[str, Any]) -> Path:
        out_path = self.path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return out_path


def init_run_context(base_dir: str = "runs", run_id: Optional[str] = None) -> RunContext:
    rid = run_id or make_run_id()
    run_dir = Path(base_dir) / rid

    # create standard structure
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    return RunContext(run_id=rid, run_dir=run_dir)


def configure_run_logging(ctx: RunContext) -> None:
    log_path = ctx.logs_dir / "run.log"

    # Avoid duplicate handlers if re-run in same interpreter session
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger(__name__).info("Run logging initialized: %s", log_path)