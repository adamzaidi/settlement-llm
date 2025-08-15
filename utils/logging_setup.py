# Returns logger so other files can use it
from pathlib import Path
import logging

def setup_logger(name: str = "pipeline", log_path: str = "logs/pipeline.log", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger so we can see messages while the program runs
    AND save them to a file for later.
    """

    # 1) Make sure a "logs" folder exists to store the log file
    Path("logs").mkdir(exist_ok=True)

    # 2) Create a logger object with the given name (default is "pipeline")
    logger = logging.getLogger(name)
    logger.setLevel(level)  # Set how much detail to show (INFO, DEBUG, etc.)

    # 3) Remove old handlers so logs don't repeat if we run the script again
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # 4) Create a file handler (writes logs to a file)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")

    # 5) Create a stream handler (shows logs in the console)
    stream_handler = logging.StreamHandler()

    # 6) Choose how the log messages will look
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)

    # 7) Add both handlers (file + console) to our logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger