"""Local results, aggregate statistics, and extensible challenge records."""
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

BOARDS = ("classic_CASUAL", "classic_NORMAL", "classic_FAST", "arcade")
COUNTERS = ("games", "apples", "stars", "score", "highest_score", "deaths",
            "challenge_attempts", "challenge_completions")


def number(value, default=0):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else default


class RecordsStore:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"version": 1, "leaderboards": {key: [] for key in BOARDS},
                     "statistics": {key: 0 for key in (*COUNTERS, "playtime")}, "challenges": {}}
        self.writable = True
        self.last_save_ok = True
        self.load()

    def load(self):
        invalid = False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Expected a records object")
            invalid |= not isinstance(raw.get("leaderboards", {}), dict)
            for key in BOARDS:
                rows = raw.get("leaderboards", {}).get(key, []) if isinstance(raw.get("leaderboards"), dict) else []
                clean = []
                if not isinstance(rows, list):
                    rows, invalid = [], True
                for row in rows:
                    if (isinstance(row, dict) and isinstance(row.get("name"), str)
                            and type(row.get("score")) is int and row["score"] >= 0
                            and isinstance(row.get("date"), str)):
                        clean.append(row)
                    else:
                        invalid = True
                self.data["leaderboards"][key] = sorted(clean, key=lambda row: row["score"], reverse=True)[:10]
            stats = raw.get("statistics", {})
            if isinstance(stats, dict):
                for key in self.data["statistics"]:
                    value = stats.get(key, 0)
                    self.data["statistics"][key] = (number(value) if key == "playtime"
                                                    else value if type(value) is int and value >= 0 else 0)
                    invalid |= value != self.data["statistics"][key]
            else:
                invalid = True
            challenges = raw.get("challenges", {})
            if isinstance(challenges, dict):
                for key, record in challenges.items():
                    if isinstance(record, dict):
                        clean = {field: number(record.get(field)) for field in
                                 ("attempts", "victories", "best_score", "best_survival")}
                        clean.update(id=key, completed=record.get("completed") is True,
                                     best_time=number(record.get("best_time"), None),
                                     last_completion=record.get("last_completion") if isinstance(record.get("last_completion"), str) else None)
                        if "last_death_cause" in record:
                            clean["last_death_cause"] = record["last_death_cause"] if isinstance(record["last_death_cause"], str) else None
                        self.data["challenges"][key] = clean
                        invalid |= clean != record
                    else:
                        invalid = True
            else:
                invalid = True
        except FileNotFoundError:
            self.save()
            return
        except (ValueError, OSError) as error:
            logging.warning("Records recovery: %s", error)
            invalid = self.path.exists()
        if invalid:
            try:
                backup = self.path.with_name(self.path.stem + ".recovered-" + uuid4().hex + ".json")
                shutil.copy2(self.path, backup)
                logging.warning("Original records preserved at %s", backup)
            except OSError as error:
                self.writable = False
                logging.warning("Cannot preserve damaged records; leaving original untouched: %s", error)
            if self.writable:
                self.save()

    def save(self):
        if not self.writable:
            self.last_save_ok = False
            return False
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as file:
                temporary = Path(file.name)
                json.dump(self.data, file, indent=2, allow_nan=False)
                file.write("\n")
            temporary.replace(self.path)
            self.last_save_ok = True
        except OSError as error:
            logging.warning("Could not save records: %s", error)
            self.last_save_ok = False
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    logging.warning("Could not clean temporary records file: %s", error)
        return self.last_save_ok

    @property
    def average_score(self):
        stats = self.data["statistics"]
        return stats["score"] / stats["games"] if stats["games"] else 0.0

    def qualifies(self, key, score):
        rows = self.data["leaderboards"][key]
        return score > 0 and (len(rows) < 10 or score > rows[-1]["score"])

    def add_score(self, key, name, score, mode, speed=None, date=None):
        name = "".join(char for char in name if char.isprintable()).strip()[:12] or "PLAYER"
        if not self.qualifies(key, score):
            return False
        row = dict(name=name, score=score, mode=mode, speed=speed,
                   date=date or datetime.now(timezone.utc).isoformat())
        rows = self.data["leaderboards"][key]
        rows.append(row)
        rows.sort(key=lambda entry: entry["score"], reverse=True)
        del rows[10:]
        return self.save()

    def record_game(self, game):
        """Called once by the application at terminal/abandon, never on redraw."""
        from core.types import GameStatus
        stats = self.data["statistics"]
        stats["games"] += 1
        stats["apples"] += game.apples_eaten
        stats["stars"] += game.stars_collected
        stats["score"] += game.score
        stats["highest_score"] = max(stats["highest_score"], game.score)
        stats["playtime"] += game.elapsed
        stats["deaths"] += game.status == GameStatus.LOST
        if game.rules.challenge_id:
            key = game.rules.challenge_id
            won = game.status == GameStatus.WON
            stats["challenge_attempts"] += 1
            stats["challenge_completions"] += won
            record = self.data["challenges"].setdefault(key, dict(id=key, attempts=0, victories=0,
                completed=False, best_score=0, best_time=None, best_survival=0, last_completion=None))
            record["attempts"] += 1
            record["victories"] += won
            record["best_score"] = max(record["best_score"], game.score)
            record["best_survival"] = max(record["best_survival"], game.elapsed)
            if game.status == GameStatus.LOST:
                record["last_death_cause"] = game.death_cause
            if won:
                record["completed"] = True
                record["best_time"] = min(record["best_time"], game.elapsed) if record["best_time"] is not None else game.elapsed
                record["last_completion"] = datetime.now(timezone.utc).isoformat()
        self.save()
