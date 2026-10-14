"""Gamification store (SQLite): points ledger, streaks, unlocks, badges, profile and scan history."""
import json
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

CHECKIN_POINTS = 10
BOTH_SESSIONS_BONUS = 5
STREAK_7_BONUS = 50
FIRST_SCAN_POINTS = 50
WEEKLY_SCAN_POINTS = 20
POINTS_PER_LEVEL = 100


class GameStore:
    def __init__(self, db_path):
        self.db_path = str(Path(db_path))
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS ledger(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE, ts REAL, day TEXT, kind TEXT, points INTEGER, meta TEXT);
                CREATE TABLE IF NOT EXISTS scans(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, day TEXT, data TEXT);
                CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS products(id TEXT PRIMARY KEY, data TEXT);
                """
            )

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    # ---- ledger ----------------------------------------------------
    def _award(self, c, key, kind, points, day, meta=""):
        cur = c.execute(
            "INSERT OR IGNORE INTO ledger(key, ts, day, kind, points, meta) VALUES(?,?,?,?,?,?)",
            (key, time.time(), day, kind, points, meta),
        )
        return cur.rowcount == 1

    def _streak(self, c, today: date) -> int:
        days = {r["day"] for r in c.execute("SELECT DISTINCT day FROM ledger WHERE kind='checkin'")}
        cur = today if today.isoformat() in days else today - timedelta(days=1)
        n = 0
        while cur.isoformat() in days:
            n += 1
            cur -= timedelta(days=1)
        return n

    def checkin(self, session: str, today: date | None = None) -> dict:
        if session not in ("am", "pm"):
            raise ValueError("session must be 'am' or 'pm'")
        today = today or date.today()
        day = today.isoformat()
        gained, messages = 0, []
        with self._conn() as c:
            if not self._award(c, f"checkin:{day}:{session}", "checkin", CHECKIN_POINTS, day, session):
                return {"awarded": 0, "message": f"You already checked in for {session.upper()} today.",
                        "state": self._state(c, today)}
            gained += CHECKIN_POINTS
            messages.append(f"+{CHECKIN_POINTS} for your {session.upper()} routine")
            other = "pm" if session == "am" else "am"
            if c.execute("SELECT 1 FROM ledger WHERE key=?", (f"checkin:{day}:{other}",)).fetchone():
                if self._award(c, f"bonus:{day}", "bonus", BOTH_SESSIONS_BONUS, day):
                    gained += BOTH_SESSIONS_BONUS
                    messages.append(f"+{BOTH_SESSIONS_BONUS} full-day bonus")
            streak = self._streak(c, today)
            if streak and streak % 7 == 0:
                if self._award(c, f"streak7:{day}", "streak", STREAK_7_BONUS, day):
                    gained += STREAK_7_BONUS
                    messages.append(f"+{STREAK_7_BONUS} for a {streak}-day streak")
            return {"awarded": gained, "message": ", ".join(messages), "state": self._state(c, today)}

    def record_scan(self, skin: dict, today: date | None = None) -> int:
        today = today or date.today()
        day = today.isoformat()
        gained = 0
        with self._conn() as c:
            last = c.execute("SELECT day FROM scans ORDER BY id DESC LIMIT 1").fetchone()
            c.execute("INSERT INTO scans(ts, day, data) VALUES(?,?,?)", (time.time(), day, json.dumps(skin)))
            if last is None:
                if self._award(c, "scan:first", "scan", FIRST_SCAN_POINTS, day, "first scan"):
                    gained = FIRST_SCAN_POINTS
            elif (today - date.fromisoformat(last["day"])).days >= 7:
                if self._award(c, f"scan:{day}", "scan", WEEKLY_SCAN_POINTS, day, "progress scan"):
                    gained = WEEKLY_SCAN_POINTS
        return gained

    def unlock(self, item_id: str, cost: int, today: date | None = None) -> dict:
        today = today or date.today()
        with self._conn() as c:
            if item_id in self._unlocked(c):
                return {"ok": True, "message": "Already unlocked.", "state": self._state(c, today)}
            balance = self._balance(c)
            if balance < cost:
                return {"ok": False, "message": f"You need {cost - balance} more points.",
                        "state": self._state(c, today)}
            self._award(c, f"unlock:{item_id}", "unlock", -cost, today.isoformat(), item_id)
            return {"ok": True, "message": "Unlocked!", "state": self._state(c, today)}

    # ---- state -----------------------------------------------------
    def _balance(self, c) -> int:
        return c.execute("SELECT COALESCE(SUM(points),0) FROM ledger").fetchone()[0]

    def _unlocked(self, c) -> set[str]:
        return {r["meta"] for r in c.execute("SELECT meta FROM ledger WHERE kind='unlock'")}

    def unlocked(self) -> set[str]:
        with self._conn() as c:
            return self._unlocked(c)

    def _state(self, c, today: date) -> dict:
        day = today.isoformat()
        earned = c.execute("SELECT COALESCE(SUM(points),0) FROM ledger WHERE points > 0").fetchone()[0]
        today_keys = {r["key"] for r in c.execute(
            "SELECT key FROM ledger WHERE kind='checkin' AND day=?", (day,))}
        streak = self._streak(c, today)
        badges = []
        if c.execute("SELECT 1 FROM scans LIMIT 1").fetchone():
            badges.append("First Scan")
        if c.execute("SELECT 1 FROM ledger WHERE kind='streak' LIMIT 1").fetchone() or streak >= 7:
            badges.append("7-Day Streak")
        if earned >= 500:
            badges.append("Glow Getter")
        return {
            "points": self._balance(c),
            "earned_total": earned,
            "level": 1 + earned // POINTS_PER_LEVEL,
            "points_to_next_level": POINTS_PER_LEVEL - (earned % POINTS_PER_LEVEL),
            "streak": streak,
            "checked_in_today": {"am": f"checkin:{day}:am" in today_keys, "pm": f"checkin:{day}:pm" in today_keys},
            "unlocked": sorted(self._unlocked(c)),
            "badges": badges,
            "scan_count": c.execute("SELECT COUNT(*) FROM scans").fetchone()[0],
        }

    def state(self, today: date | None = None) -> dict:
        with self._conn() as c:
            return self._state(c, today or date.today())

    # ---- profile & scans ---------------------------------------------
    def get_profile(self) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT value FROM kv WHERE key='profile'").fetchone()
        return json.loads(row["value"]) if row else {}

    def set_profile(self, profile: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO kv(key, value) VALUES('profile', ?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(profile),))

    def latest_scan(self) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT day, data FROM scans ORDER BY id DESC LIMIT 1").fetchone()
        return {"day": row["day"], **json.loads(row["data"])} if row else None

    # ---- imported / custom products --------------------------------------
    def save_product(self, product: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO products(id, data) VALUES(?, ?) "
                      "ON CONFLICT(id) DO UPDATE SET data=excluded.data", (product["id"], json.dumps(product)))

    def list_products(self) -> list[dict]:
        with self._conn() as c:
            return [json.loads(r["data"]) for r in c.execute("SELECT data FROM products ORDER BY rowid")]

    def delete_product(self, product_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM products WHERE id=?", (product_id,))
