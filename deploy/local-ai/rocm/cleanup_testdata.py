"""Delete all throwaway users created during acceptance checks.

Every user in this local dev database was created by an automated check; there is no
real account here. Deletion goes through the project's own `delete_user_data` (the
same path `DELETE /api/account` uses), so no orphan rows are left and the tables
that must be kept for rollups are anonymized instead.

Afterward a single fixed account, `test-runner`, is created via the backend's dev
login so future checks always reuse the same user instead of making new ones.

The database is backed up first, and the backend must be stopped while this runs.
"""
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"F:\eidolon-sc")
sys.path.insert(0, str(ROOT / "backend" / "src"))
from eidolon.database import delete_user_data  # noqa: E402

DB = ROOT / "backend" / "data" / "eidolon.db"
BACKUP = ROOT / "backend" / "data" / "eidolon.db.before-cleanup"

conn = sqlite3.connect(DB)
conn.execute("PRAGMA foreign_keys = ON")
users = conn.execute("SELECT id, name FROM users ORDER BY created_at").fetchall()

if not users:
    print("no users; nothing to do")
    sys.exit(0)

shutil.copyfile(DB, BACKUP)
print(f"backup: {BACKUP} ({BACKUP.stat().st_size / 1e6:.1f} MB)")
print(f"deleting {len(users)} test users:")
for uid, name in users:
    print(f"  {uid[:8]}  {name}")

with conn:
    for uid, _ in users:
        delete_user_data(conn, uid)
conn.commit()

print(f"\nusers left: {conn.execute('SELECT count(*) FROM users').fetchone()[0]}")
for table in ("conversations", "sessions", "usage", "activity_records", "messages", "memories"):
    n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    print(f"  {table:20} {n}")

# A fixed account for all future checks. Created directly, mirroring what the dev
# login endpoint does, so no check ever needs to invent another throwaway user.
import secrets  # noqa: E402

uid = secrets.token_hex(16)
now = datetime.now(timezone.utc).isoformat()
with conn:
    conn.execute(
        "INSERT INTO users (id, name, created_at, consented_at) VALUES (?, ?, ?, ?)",
        (uid, "test-runner", now, now),
    )
conn.commit()
print(f"\ncreated fixed test account: test-runner ({uid[:8]})")
print(f"users now: {conn.execute('SELECT count(*) FROM users').fetchone()[0]}")
conn.close()
