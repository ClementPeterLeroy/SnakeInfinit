"""Validate preferences and atomically persist them in the user config folder."""
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import tempfile
from core.config import SPEEDS
from services.userdata import user_data_dir, migrate_user_data


@dataclass
class Settings:
    music: bool = True
    sound_effects: bool = True
    animations: bool = True
    speed: str = "NORMAL"
    voice_announcements: bool = True


def default_path() -> Path:
    return user_data_dir() / "settings.json"


class SettingsStore:
    def __init__(self, path: Path | None = None):
        if path is None:
            migrate_user_data()
        self.path = path if path is not None else default_path()

    def load(self) -> Settings:
        settings = Settings()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key in ("music", "sound_effects", "animations", "voice_announcements"):
                    if type(raw.get(key)) is bool:
                        setattr(settings, key, raw[key])
                if isinstance(raw.get("speed"), str) and raw["speed"] in SPEEDS:
                    settings.speed = raw["speed"]
        except (OSError, ValueError):
            pass
        self.save(settings)
        return settings

    def save(self, settings: Settings) -> bool:
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             dir=self.path.parent, delete=False) as file:
                temporary = Path(file.name)
                json.dump(asdict(settings), file, indent=2)
                file.write("\n")
            temporary.replace(self.path)
            return True
        except OSError as error:
            logging.warning("Could not save settings: %s", error)
            return False
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    logging.warning("Could not clean temporary settings file: %s", error)
