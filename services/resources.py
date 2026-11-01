"""Read-only application resources, independent of the process working directory."""
import logging
from pathlib import Path
import sys

IMAGES = ("logo.png", "Icon.png", "apple.png", "star.png")
SOUNDS = ("synth_wave_background.wav", "eating_apple.wav", "eating_star.wav",
          "game_over.wav", "3.wav", "2.wav", "1.wav", "go.wav",
          "super_voice.wav", "game_over_voice.wav")
VIDEO_NAMES = ("assets/video/SnakeInfinite_Startup_6s.mp4",
               "assets/px/SnakeInfinite_Startup_6s.mp4",
               "assets/SnakeInfinite_Startup_6s.mp4",
               "assets/video/SnakeInfinite_Startup.mp4",
               "assets/px/SnakeInfinite_Startup.mp4")


def resource_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parent.parent


def resource_path(*parts) -> Path:
    return resource_root().joinpath(*parts)


def startup_video_path() -> Path:
    for name in VIDEO_NAMES:
        candidate = resource_path(name)
        if candidate.is_file():
            logging.info("Startup video resolved: %s", candidate)
            return candidate
    logging.error("Startup video missing. Searched: %s", ", ".join(str(resource_path(name)) for name in VIDEO_NAMES))
    return resource_path(VIDEO_NAMES[0])


def bundled_assets() -> dict[str, Path]:
    return {**{name: resource_path("assets", "px", name) for name in IMAGES},
            **{name: resource_path("assets", "snd", name) for name in SOUNDS},
            "startup_video": startup_video_path()}
