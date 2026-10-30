import json
from core.challenges import BY_ID
from core.game import Game
from core.types import GameStatus
from services.records import RecordsStore
from services.persistence import Settings, SettingsStore


def test_leaderboard_creation_sorting_top_ten_and_reload(tmp_path):
    store = RecordsStore(tmp_path / "records.json")
    assert store.path.exists()
    for score in range(1, 16):
        store.add_score("arcade", "NAME LONGER THAN TWELVE", score, "arcade")
    rows = store.data["leaderboards"]["arcade"]
    assert [row["score"] for row in rows] == list(range(15, 5, -1))
    assert all(len(row["name"]) <= 12 for row in rows)
    assert not store.qualifies("arcade", 6)
    assert store.qualifies("arcade", 7)
    assert RecordsStore(store.path).data == store.data
    assert not store.data["leaderboards"]["classic_NORMAL"]


def test_corrupt_records_backed_up_and_valid_rows_salvaged(tmp_path):
    path = tmp_path / "records.json"
    original = {"leaderboards": {"arcade": [dict(name="VALID", score=22, date="2026-10-30"),
                                                     dict(name="BAD", score="oops")]},
                "statistics": {"games": 4, "apples": "broken"}}
    path.write_text(json.dumps(original))
    store = RecordsStore(path)
    assert store.data["leaderboards"]["arcade"][0]["name"] == "VALID"
    assert store.data["statistics"]["games"] == 4
    backups = list(tmp_path.glob("records.recovered-*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text()) == original


def test_invalid_json_retained_before_recovery(tmp_path):
    path = tmp_path / "records.json"
    path.write_text("{broken data")
    store = RecordsStore(path)
    backup = next(tmp_path.glob("records.recovered-*.json"))
    assert backup.read_text() == "{broken data"
    assert store.data["statistics"]["games"] == 0


def test_challenge_and_statistics_persistence(tmp_path):
    store = RecordsStore(tmp_path / "records.json")
    game = Game(rules=BY_ID["centurion"].rules)
    game.score = 70
    game.apples_eaten = 70
    game.elapsed = 88.0
    game.status = GameStatus.WON
    store.record_game(game)
    game.elapsed = 80
    store.record_game(game)
    loaded = RecordsStore(store.path)
    record = loaded.data["challenges"]["centurion"]
    assert record["completed"] and record["victories"] == record["attempts"] == 2
    assert record["best_time"] == 80 and record["best_survival"] == 88
    assert record["best_score"] == 70 and record["last_completion"]
    stats = loaded.data["statistics"]
    assert stats["games"] == 2 and stats["apples"] == stats["score"] == 140
    assert stats["challenge_completions"] == 2 and stats["deaths"] == 0
    assert loaded.average_score == 70
    assert not any(loaded.data["leaderboards"].values())


def test_voice_setting_roundtrips_and_old_settings_migrate(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.path.write_text('{"music": false, "speed": "FAST"}')
    settings = store.load()
    assert settings.voice_announcements and not settings.music
    settings.voice_announcements = False
    store.save(settings)
    assert not store.load().voice_announcements


def test_failed_backup_keeps_original_records(tmp_path, monkeypatch):
    import shutil
    path = tmp_path / "records.json"
    path.write_text("unrecoverable")
    def fail(*args):
        raise OSError("denied")
    monkeypatch.setattr(shutil, "copy2", fail)
    store = RecordsStore(path)
    assert not store.save()
    assert path.read_text() == "unrecoverable"
