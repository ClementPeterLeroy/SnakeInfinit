"""Resource roots, persistent profiles, migration, and decoder failure handling."""
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from services import resources, userdata
from services.persistence import SettingsStore
from services.video import StartupVideo


def test_development_resources_do_not_depend_on_cwd(tmp_path, monkeypatch):
    source = tmp_path / "source"
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    monkeypatch.setattr(resources, "__file__", str(source / "services" / "resources.py"))
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.chdir(unrelated)
    assert resources.resource_path("assets", "px", "logo.png") == source / "assets/px/logo.png"


def test_frozen_resource_root_and_asset_manifest(tmp_path, monkeypatch):
    bundle = tmp_path / "distribution" / "_internal"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    video = bundle / "assets/video/SnakeInfinite_Startup_6s.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"approved video placeholder")
    manifest = resources.bundled_assets()
    for path in manifest.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"asset")
    assert len(manifest) == 15
    assert all(path.is_file() and path.is_relative_to(bundle) for path in manifest.values())
    assert manifest["startup_video"] == video


def test_supplied_legacy_video_name_is_discovered_without_renaming():
    assert resources.startup_video_path().name in ("SnakeInfinite_Startup.mp4", "SnakeInfinite_Startup_6s.mp4")
    assert all(path.is_file() for path in resources.bundled_assets().values())


def test_preferred_video_name_has_priority_and_missing_assets_are_reported(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(resources, "resource_root", lambda: tmp_path)
    missing = resources.startup_video_path()
    assert missing.name == "SnakeInfinite_Startup_6s.mp4"
    assert "Startup video missing" in caplog.text
    legacy = tmp_path / "assets/px/SnakeInfinite_Startup.mp4"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"existing")
    assert resources.startup_video_path() == legacy
    missing.parent.mkdir(parents=True)
    missing.write_bytes(b"preferred")
    assert resources.startup_video_path() == missing


def test_windows_user_data_prefers_localappdata(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setattr(sys, "platform", "win32")
    assert userdata.user_data_dir() == tmp_path / "local/SnakeInfinite"


def test_migration_copies_settings_and_records_without_deleting_or_overwriting(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    settings = b'{"music":false,"speed":"FAST"}'
    records = b'{"leaderboards":{},"statistics":{"games":12},"challenges":{}}'
    (source / "settings.json").write_bytes(settings)
    (source / "records.json").write_bytes(records)
    assert len(userdata.migrate_user_data(destination, [source])) == 2
    for name in ("settings.json", "records.json"):
        assert (destination / name).read_bytes() == (source / name).read_bytes()
    (destination / "settings.json").write_text("new profile")
    assert userdata.migrate_user_data(destination, [source]) == []
    assert (destination / "settings.json").read_text() == "new profile"
    assert (source / "settings.json").read_bytes() == settings


def test_default_settings_migrate_before_defaults_are_created(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    old = tmp_path / "roaming/snake-ai-laboratory"
    old.mkdir(parents=True)
    original = '{"music":false,"speed":"FAST"}'
    (old / "settings.json").write_text(original)
    (old / "records.json").write_text('{"statistics":{"games":8}}')
    store = SettingsStore()
    settings = store.load()
    assert not settings.music and settings.speed == "FAST"
    assert store.path.parent == tmp_path / "local/SnakeInfinite"
    assert (store.path.parent / "records.json").read_text() == '{"statistics":{"games":8}}'
    assert (old / "settings.json").read_text() == original


def test_custom_settings_do_not_trigger_migration(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("Custom profile must not migrate normal user records")
    monkeypatch.setattr("services.persistence.migrate_user_data", forbidden)
    assert SettingsStore(tmp_path / "custom.json").load().music


def test_failed_migration_preserves_original_and_cleans_partial_copy(tmp_path, monkeypatch):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    (source / "settings.json").write_text("original")
    def fail(incoming, outgoing):
        outgoing.write(b"partial")
        raise OSError("simulated write failure")
    monkeypatch.setattr(userdata.shutil, "copyfileobj", fail)
    assert userdata.migrate_user_data(destination, [source]) == []
    assert (source / "settings.json").read_text() == "original"
    assert not (destination / "settings.json").exists()


def test_decoder_initialization_failure_logs_and_falls_back(monkeypatch, caplog):
    import av
    def fail(*args, **kwargs):
        raise OSError("decoder initialization failed")
    monkeypatch.setattr(av, "open", fail)
    video = StartupVideo(resources.startup_video_path())
    video.thread.join(timeout=5)
    assert video.update(0.01)
    assert "decoder initialization failed" in video.error
    assert "Continuing to menu" in caplog.text
    video.close()
    assert not video.thread.is_alive()


def test_missing_pyav_is_recorded_as_a_decoder_error(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "av", None)
    video = StartupVideo(resources.startup_video_path())
    video.thread.join(timeout=5)
    assert video.update(0.01) and video.error
    assert "av" in video.error and video.current is None
    video.close()


def test_irregular_frame_timestamps_control_presentation(monkeypatch):
    class Plane:
        line_size = 6
        def __init__(self, color):
            self.color = color
        def __bytes__(self):
            return bytes(self.color) * 2
    class Frame:
        width, height = 2, 1
        def __init__(self, pts, color):
            self.time = pts
            self.planes = [Plane(color)]
        def reformat(self, **kwargs):
            return self
    stream = SimpleNamespace(duration=10, time_base=Fraction(1, 10), average_rate=100,
                             width=2, height=1, codec_context=SimpleNamespace(name="fake"))
    class Container:
        streams = SimpleNamespace(video=[stream], audio=[])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def decode(self, **kwargs):
            return iter((Frame(0, (255,0,0)), Frame(0.25, (0,255,0)), Frame(0.9, (0,0,255))))
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(__version__="test", open=lambda path: Container()))
    video = StartupVideo(Path("timestamp-test.mp4"))
    video.thread.join(timeout=5)
    assert not video.update(0)
    assert tuple(video.current.get_at((0,0)))[:3] == (255,0,0)
    video.update(0.2)
    assert tuple(video.current.get_at((0,0)))[:3] == (255,0,0)
    video.update(0.1)
    assert tuple(video.current.get_at((0,0)))[:3] == (0,255,0)
    video.update(0.65)
    assert tuple(video.current.get_at((0,0)))[:3] == (0,0,255)
    assert video.update(0.2)
    video.close()
