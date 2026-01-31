from pathlib import Path
import logging

def setup_logger(
    name: str = "pipeline",
    log_path: str = "logs/pipeline.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging so I can see messages while the program runs
    and also keep a log file for later.

    Args:
        name: the logger name (default "pipeline")
        log_path: path for the log file
        level: logging level (INFO, DEBUG, etc.)

    Returns:
        A configured logger instance that writes to both console and file.
    """

    # Make sure the logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Create or get the logger object
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove old handlers so I don’t get duplicate log lines on re-run
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # File handler: writes to disk
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")

    # Console handler: shows messages in stdout/stderr
    stream_handler = logging.StreamHandler()

    # Formatter controls how each log line looks
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)

    # Attach both handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # IMPORTANT:
    # Prevent log messages from propagating to the root logger.
    # This avoids duplicate log lines when other modules
    # (e.g., run_context) also configure logging.
    logger.propagate = False

    return logger