"""Lightweight discovery. Ordinary gameplay must never import PyTorch."""
import importlib.util


def torch_available():
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False
