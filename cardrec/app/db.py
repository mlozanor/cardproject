"""Tiny SQLite store for the user's collection and a price cache."""
import json
import sqlite3
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    # Columns added after the first version (older collection files get them on start-up).
    # language: printed language of the user's copy (en / es / ja ...), from the photo or set by hand
    # condition: JSON of the last visual condition estimate (condition.assess), '' if none
    EXTRA_COLUMNS = {"language": "TEXT DEFAULT ''", "condition": "TEXT DEFAULT ''"}

    def __init__(self, path):
        self.path = str(path)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS collection(
                id INTEGER PRIMARY KEY AUTOINCREMENT, card_id TEXT, name TEXT, set_name TEXT,
                number TEXT, quantity INTEGER DEFAULT 1, price REAL, price_source TEXT,
                price_updated TEXT, photo TEXT, method TEXT, score REAL, notes TEXT DEFAULT '',
                added_at TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS price_cache(
                card_id TEXT PRIMARY KEY, payload TEXT, fetched_at TEXT)""")
            # columns added later: upgrade collection files created by an older version
            have = {r["name"] for r in c.execute("PRAGMA table_info(collection)")}
            for col, sql_type in self.EXTRA_COLUMNS.items():
                if col not in have:
                    c.execute(f"ALTER TABLE collection ADD COLUMN {col} {sql_type}")

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    # ---- collection ----
    def list(self):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM collection ORDER BY added_at DESC, id DESC")]

    def get(self, item_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM collection WHERE id=?", (item_id,)).fetchone()
            return dict(r) if r else None

    def add(self, meta, photo, method, score, quantity=1, price=None, price_source=None, language=""):
        """Save a card. If the same card_id in the same language is already tracked, increase its
        quantity instead (an English and a Spanish copy are separate items)."""
        with self._conn() as c:
            row = c.execute("SELECT id, quantity FROM collection WHERE card_id=? AND language=?",
                            (meta["id"], language)).fetchone()
            if row:
                c.execute("UPDATE collection SET quantity=?, photo=COALESCE(?, photo) WHERE id=?",
                          (row["quantity"] + quantity, photo, row["id"]))
                return row["id"]
            cur = c.execute(
                """INSERT INTO collection(card_id,name,set_name,number,quantity,price,price_source,
                   price_updated,photo,method,score,added_at,language) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (meta["id"], meta.get("name", ""), meta.get("set_name", ""), meta.get("number", ""),
                 quantity, price, price_source, _now() if price is not None else None,
                 photo, method, score, _now(), language))
            return cur.lastrowid

    def update(self, item_id, **fields):
        allowed = {"price", "price_source", "quantity", "notes", "language", "condition"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if not fields:
            return
        if "price" in fields:
            fields["price_updated"] = _now()
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE collection SET {sets} WHERE id=?", (*fields.values(), item_id))

    def delete(self, item_id):
        with self._conn() as c:
            c.execute("DELETE FROM collection WHERE id=?", (item_id,))

    def summary(self):
        rows = self.list()
        return {"unique": len(rows), "total_cards": sum(r["quantity"] or 0 for r in rows),
                "total_value": round(sum((r["price"] or 0) * (r["quantity"] or 0) for r in rows), 2)}

    # ---- price cache ----
    def cache_get(self, card_id, max_age_hours=24):
        with self._conn() as c:
            r = c.execute("SELECT payload, fetched_at FROM price_cache WHERE card_id=?", (card_id,)).fetchone()
        if not r:
            return None
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(r["fetched_at"])).total_seconds() / 3600
        return json.loads(r["payload"]) if age <= max_age_hours else None

    def cache_put(self, card_id, payload):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO price_cache(card_id,payload,fetched_at) VALUES(?,?,?)",
                      (card_id, json.dumps(payload), _now()))
