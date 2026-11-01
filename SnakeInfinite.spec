# Reproducible onedir build. SNAKE_BUILD_DIAGNOSTIC=1 enables its console.
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

root = Path(SPECPATH)
diagnostic = os.environ.get("SNAKE_BUILD_DIAGNOSTIC") == "1"
av_datas, av_binaries, av_imports = collect_all("av", include_py_files=False)
images = ("logo.png", "Icon.png", "apple.png", "star.png")
sounds = ("synth_wave_background.wav", "eating_apple.wav", "eating_star.wav",
          "game_over.wav", "3.wav", "2.wav", "1.wav", "go.wav",
          "super_voice.wav", "game_over_voice.wav")
data = [(str(root / "assets/px" / name), "assets/px") for name in images]
data += [(str(root / "assets/snd" / name), "assets/snd") for name in sounds]
# Use the approved file under its actual supplied name. No asset duplication.
videos = ("assets/video/SnakeInfinite_Startup_6s.mp4", "assets/px/SnakeInfinite_Startup_6s.mp4",
          "assets/SnakeInfinite_Startup_6s.mp4", "assets/video/SnakeInfinite_Startup.mp4",
          "assets/px/SnakeInfinite_Startup.mp4")
video = next((root / name for name in videos if (root / name).is_file()), None)
if video is None:
    raise FileNotFoundError("Official startup video is required to build the distribution")
data.append((str(video), str(video.relative_to(root).parent)))
icon = root / "assets/branding/snake_infinite_icon.ico"
if not icon.is_file():
    raise FileNotFoundError("Run tools/build_icon.py to prepare the official ICO")

a = Analysis([str(root / "main.py")], pathex=[str(root)],
             binaries=av_binaries, datas=data + av_datas, hiddenimports=av_imports,
             excludes=["pytest", "_pytest", "tests", "tkinter", "numpy", "PIL",
                       "matplotlib", "scipy", "torch", "setuptools", "pip"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SnakeInfinite",
          debug=False, strip=False, upx=False, console=diagnostic, icon=str(icon),
          version=str(root / "packaging/windows_version.txt"))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SnakeInfinite")
