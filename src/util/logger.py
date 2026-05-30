import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
log_path = log_dir / log_filename

if not logger.handlers:
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
