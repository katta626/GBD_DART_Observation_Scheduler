import sqlite3
from typing import Optional, Dict, Any


class DataStore:
    def __init__(self, db_path: str = "data_store.db"):
        """Initialize connection and create tables if needed."""
        self.db_path = db_path
        self._create_tables()
    def connect(self):
        return sqlite3.connect(self.db_path)

    def _get_conn(self):
        """Create a connection (safe for concurrent access)."""
        return sqlite3.connect(self.db_path, timeout=10)

    def _create_tables(self):
        """Ensure required tables exist."""
        conn = self._get_conn()
        cur = conn.cursor()

        # Observations table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            name TEXT PRIMARY KEY,
            duration INTEGER,
            status TEXT,
            ra_start TEXT,
            count_down REAL,
            started_at TEXT
        )
        ''')

        # System status table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS system_status (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        ''')

        self._ensure_column(cur, "observations", "started_at", "TEXT")
        conn.commit()
        conn.close()

    def _ensure_column(self, cur, table_name: str, column_name: str, column_type: str):
        cur.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in cur.fetchall()}
        if column_name not in existing_columns:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    # ------------------------------
    # Observation Operations
    # ------------------------------
    def add_or_update_observation(self, name: str, duration: int, status: str,
                                  ra_start: str, count_down: float, started_at: str | None = None):
        """Insert or update an observation."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute('''
        INSERT INTO observations (name, duration, status, ra_start, count_down, started_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            duration=excluded.duration,
            status=excluded.status,
            ra_start=excluded.ra_start,
            count_down=excluded.count_down,
            started_at=excluded.started_at
        ''', (name, duration, status, ra_start, count_down, started_at))
        conn.commit()
        conn.close()

    def update_observation(self, name: str, **fields):
        """Update specific fields for an observation."""
        if not fields:
            return
        keys = ", ".join([f"{k}=?" for k in fields])
        values = list(fields.values()) + [name]
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(f'UPDATE observations SET {keys} WHERE name=?', values)
        conn.commit()
        conn.close()

    def get_observation(self, name: str) -> Optional[Dict[str, Any]]:
        """Get one observation by name."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute('SELECT * FROM observations WHERE name=?', (name,))
        row = cur.fetchone()
        conn.close()

        if row:
            return {
                "name": row[0],
                "duration": row[1],
                "status": row[2],
                "ra_start": row[3],
                "count_down": row[4],
                "started_at": row[5],
            }
        return None

    def get_all_observations(self, order_by_countdown=False):
        """Return all observations as a list of dicts, optionally sorted by countdown."""
        conn = self._get_conn()
        cur = conn.cursor()
        if order_by_countdown:
            cur.execute('SELECT * FROM observations ORDER BY count_down ASC')
        else:
            cur.execute('SELECT * FROM observations')
        rows = cur.fetchall()
        conn.close()
        return [
            {"name": r[0], "duration": r[1], "status": r[2],
             "ra_start": r[3], "count_down": r[4], "started_at": r[5]}
            for r in rows
        ]
    def delete_observation(self, name: str):
        """Delete a pulsar observation from the database by name."""
        conn = self._get_conn()  # get a valid connection
        with conn:  # use the same connection here
            conn.execute(
                "DELETE FROM observations WHERE name = ?",
                (name,)
            )
        conn.close()  # close it after done
        print(f"✅ Observation '{name}' deleted from DB.")
        
    # ------------------------------
    # System Status Operations
    # ------------------------------
    def set_system_status(self, key: str, value: str):
        """Store a system status entry directly as TEXT."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute('''
        INSERT OR REPLACE INTO system_status (key, value)
        VALUES (?, ?)
        ''', (key, value))
        conn.commit()
        conn.close()

    def get_system_status(self, key):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT value FROM system_status WHERE key=?", (key,))
            result = cur.fetchone()
        finally:
            conn.close()

        if not result:
            return None

        val = result[0]

        # ONLY treat Log_Current as list
        if key == "Log_Current":
            return val.split(",") if val else []

        # Otherwise return plain string (important!)
        return val

        
        
    def update_system_status(self, key, value):
        """Store lists as comma-separated strings; plain strings stay as-is."""
        if isinstance(value, list):
            value = ",".join(value)
        elif isinstance(value, dict) and "current_file" in value:
            # Special case for Log_Current dicts
            value = ",".join(value["current_file"])

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO system_status (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (key, value))
            conn.commit()
        finally:
            conn.close()
