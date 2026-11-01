"""Rotating persistent logs, also available in console-free Windows builds."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from core.config import VERSION, SUBTITLE
from services.resources import resource_root
from services.userdata import user_data_dir


def configure_logging(directory: Path | None = None):
    folder = (directory if directory is not None else user_data_dir()) / "logs"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(folder / "SnakeInfinite.log", maxBytes=1_000_000,
                                      backupCount=3, encoding="utf-8")
        handler.setFormatter(formatter)
        root.addHandler(handler)
    except OSError:
        if sys.stderr is not None:
            logging.exception("Could not create persistent diagnostic log at %s", folder)
    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        root.addHandler(console)
    logging.info("SNAKE INFINITE %s — %s", VERSION, SUBTITLE)
    logging.info("Runtime executable=%s frozen=%s cwd=%s resources=%s user_data=%s",
                 sys.executable, bool(getattr(sys, "frozen", False)), Path.cwd(), resource_root(),
                 directory if directory is not None else user_data_dir())
