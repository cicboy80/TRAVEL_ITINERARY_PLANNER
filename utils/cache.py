# utils/cache.py
import json, sqlite3, time
import threading
from typing import Any, Optional

class SQLiteCache:
    def __init__(
        self, 
        path: str = "cache.sqlite",
        default_ttl_seconds: int = 3600,
        sqlite_timeout_seconds: int = 10,
        busy_timeout_ms: int = 5000,
        use_wal: bool = True,
    ):
        self.path = path
        self.default_ttl_seconds = default_ttl_seconds
        self.sqlite_timeout_seconds = sqlite_timeout_seconds
        self.busy_timeout_ms = busy_timeout_ms
        self.use_wal = use_wal
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=self.sqlite_timeout_seconds)
        if self.use_wal:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
        con.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")

    def _init_db(self):
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL,
                    expires_at INTEGER NOT NULL
                )
            """)
            con.commit()

    def get(self, key: str) -> Optional[Any]:
        now = int(time.time())
        with self.connect() as con:
            row = con.execute("SELECT v, expires_at FROM cache WHERE k=?", (key,)).fetchone()
        if not row:
            return None
        v, expires_at = row
        if expires_at < now:
            self.delete(key)
            return None
        
        try:
            return json.loads(v)
        except Exception:
            self.delete(key)
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = self.default_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        expires_at = int(time.time()) + ttl

        payload = json.dumps(value)

        with self._lock:
            with self.connect() as con:
                con.execute(
                    "INSERT OR REPLACE INTO cache (k, v, expires_at) VALUES (?, ?, ?)",
                    (key, payload, expires_at)
                )
                con.commit()

    def delete(self, key: str) -> None:
        with self._lock:
            with self.connect() as con:
                con.execute("DELETE FROM cache WHERE k=?", (key,))
                con.commit()

    def set_negative(self, key: str, ttl_seconds: int = 600, marker: str = "__no_route__") -> None:
        self.set(key, {marker: True}, ttl_seconds=ttl_seconds)