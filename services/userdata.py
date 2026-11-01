"""Writable data locations and copy-only migration from earlier releases."""
import logging
import os
from pathlib import Path
import shutil
import sys

USER_FILES = ("settings.json", "records.json")


def user_data_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "SnakeInfinite"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "SnakeInfinite"


def legacy_data_dirs() -> tuple[Path, ...]:
    old_windows = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "snake-ai-laboratory"
    old_other = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "snake-ai-laboratory"
    app_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
    return old_windows, old_other, app_dir


def migrate_user_data(destination: Path | None = None, sources=None) -> list[Path]:
    """Copy each missing file from the first available legacy location.

    Existing destination files and all original files are left untouched.
    Custom --settings paths intentionally bypass automatic migration.
    """
    target = destination if destination is not None else user_data_dir()
    copied = []
    for source in legacy_data_dirs() if sources is None else sources:
        for name in USER_FILES:
            original, output = Path(source) / name, target / name
            if not original.is_file() or output.exists() or original.resolve() == output.resolve():
                continue
            created = False
            try:
                target.mkdir(parents=True, exist_ok=True)
                with original.open("rb") as incoming, output.open("xb") as outgoing:
                    created = True
                    shutil.copyfileobj(incoming, outgoing)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                copied.append(output)
                logging.info("Migrated user data (original preserved): %s -> %s", original, output)
            except FileExistsError:
                pass
            except OSError as error:
                logging.warning("User-data migration failed for %s: %s", original, error)
                if created:
                    try:
                        output.unlink(missing_ok=True)
                    except OSError:
                        logging.exception("Could not clean incomplete migration file %s", output)
    return copied
