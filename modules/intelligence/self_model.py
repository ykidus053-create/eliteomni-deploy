import json, time, sqlite3, re
from pathlib import Path
from typing import List

DB = Path.home() / "eliteomni_self_model.db"

def _conn():
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS capability_scores (
        skill TEXT, domain TEXT, metric TEXT,
        score REAL, sample_count INTEGER, updated REAL,
        PRIMARY KEY(skill, domain, metric))""")
    c.execute("""CREATE TABLE IF NOT EXISTS failure_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        skill TEXT, pattern TEXT, frequency INTEGER, last_seen REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS calibration_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        confidence_stated REAL, was_correct INTEGER, ts REAL)""")
    c.commit()
    return c

class SelfModel:
    def __init__(self):
        self._db = _conn()

    def record_outcome(self, skill: str, domain: str, success: bool,
                       prm_score: float, user_rating: int = 0):
        combined = prm_score * 0.6 + (user_rating / 5.0) * 0.4 if user_rating else prm_score
        try:
            row = self._db.execute(
                "SELECT score, sample_count FROM capability_scores WHERE skill=? AND domain=? AND metric=?",
                (skill, domain, "quality")).fetchone()
            if row:
                old_score, n = row
                new_score = (old_score * n + combined) / (n + 1)
                self._db.execute(
                    "UPDATE capability_scores SET score=?, sample_count=?, updated=? WHERE skill=? AND domain=? AND metric=?",
                    (new_score, n+1, time.time(), skill, domain, "quality"))
            else:
                self._db.execute(
                    "INSERT INTO capability_scores VALUES (?,?,?,?,?,?)",
                    (skill, domain, "quality", combined, 1, time.time()))
            self._db.commit()
        except Exception as e:
            print(f"[SelfModel] {e}")

    def get_capability(self, skill: str, domain: str) -> float:
        row = self._db.execute(
            "SELECT score FROM capability_scores WHERE skill=? AND domain=? AND sample_count >= 3",
            (skill, domain)).fetchone()
        return row[0] if row else 0.75

    def record_failure_pattern(self, skill: str, pattern: str):
        try:
            row = self._db.execute(
                "SELECT id, frequency FROM failure_patterns WHERE skill=? AND pattern=?",
                (skill, pattern[:100])).fetchone()
            if row:
                self._db.execute(
                    "UPDATE failure_patterns SET frequency=?, last_seen=? WHERE id=?",
                    (row[1]+1, time.time(), row[0]))
            else:
                self._db.execute(
                    "INSERT INTO failure_patterns VALUES (NULL,?,?,1,?)",
                    (skill, pattern[:100], time.time()))
            self._db.commit()
        except: pass

    def get_known_weaknesses(self, skill: str) -> List[str]:
        try:
            rows = self._db.execute(
                "SELECT pattern FROM failure_patterns WHERE skill=? AND frequency >= 2 ORDER BY frequency DESC LIMIT 4",
                (skill,)).fetchall()
            return [r[0] for r in rows]
        except: return []

    def record_calibration(self, confidence_stated: float, was_correct: bool):
        try:
            self._db.execute(
                "INSERT INTO calibration_log VALUES (NULL,?,?,?)",
                (confidence_stated, 1 if was_correct else 0, time.time()))
            self._db.commit()
        except: pass

    def get_calibration_error(self) -> float:
        try:
            rows = self._db.execute(
                "SELECT confidence_stated, was_correct FROM calibration_log ORDER BY ts DESC LIMIT 50"
            ).fetchall()
            if len(rows) < 5: return 0.0
            errors = [abs(r[0] - r[1]) for r in rows]
            return sum(errors) / len(errors)
        except: return 0.0

    def build_self_awareness_injection(self, skill: str, domain: str) -> str:
        cap = self.get_capability(skill, domain)
        weaknesses = self.get_known_weaknesses(skill)
        cal_error = self.get_calibration_error()
        parts = []
        if cap < 0.6:
            parts.append(f"SELF-AWARENESS: Historical performance in {skill}/{domain} is {cap:.0%}. Apply extra verification.")
        if weaknesses:
            parts.append(f"KNOWN FAILURE PATTERNS in {skill}: " + "; ".join(weaknesses[:3]))
        if cal_error > 0.2:
            parts.append(f"CALIBRATION WARNING: Confidence overestimation detected ({cal_error:.0%} avg error). State uncertainty more explicitly.")
        if not parts:
            return ""
        return "\n<self_awareness>\n" + "\n".join(parts) + "\n</self_awareness>"

_self_model = None

def get_self_model() -> SelfModel:
    global _self_model
    if _self_model is None:
        _self_model = SelfModel()
    return _self_model
