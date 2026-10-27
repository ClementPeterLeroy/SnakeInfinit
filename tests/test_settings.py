import json
import pytest
from services.persistence import Settings, SettingsStore


def test_missing_file_created_and_roundtrip(tmp_path):
    store = SettingsStore(tmp_path / "nested" / "settings.json")
    assert store.load() == Settings()
    assert store.path.exists()
    selected = Settings(False, False, False, "FAST")
    assert store.save(selected)
    assert store.load() == selected


@pytest.mark.parametrize("contents", ["{broken", "[]", "null", "42", '"text"'])
def test_corrupt_or_wrong_shape_recovers(tmp_path, contents):
    store = SettingsStore(tmp_path / "settings.json")
    store.path.write_text(contents, encoding="utf-8")
    assert store.load() == Settings()
    assert json.loads(store.path.read_text())["speed"] == "NORMAL"


def test_invalid_fields_default_without_discarding_valid_fields(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.path.write_text(json.dumps({"music": False, "animations": 1,
                                     "sound_effects": "off", "speed": []}))
    assert store.load() == Settings(music=False)


def test_unwritable_location_does_not_crash(tmp_path):
    parent = tmp_path / "file"
    parent.write_text("not a directory")
    store = SettingsStore(parent / "settings.json")
    assert store.load() == Settings()
    assert not store.save(Settings())
