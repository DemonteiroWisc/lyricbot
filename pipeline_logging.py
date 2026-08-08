import glob
import logging
import os
import time

LOG_RETENTION_DAYS = 30

_LOGGER_NAME = "lyricbot"


def setup_logging(config) -> logging.Logger:
    """
    Configures a per-run log file under <ASSET_FOLDER>/logs/, and deletes
    log files older than LOG_RETENTION_DAYS. Kept to milestone-level
    messages (not a mirror of every print()) so files stay small.
    """
    log_dir = os.path.join(config["ASSET_FOLDER"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    _cleanup_old_logs(log_dir)

    log_path = os.path.join(log_dir, f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)

    return logger


def _cleanup_old_logs(log_dir: str) -> None:
    cutoff = time.time() - LOG_RETENTION_DAYS * 86400
    for path in glob.glob(os.path.join(log_dir, "run_*.log")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass
