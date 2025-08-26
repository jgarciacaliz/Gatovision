import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import json
from .config import MEMORY_DB, MIN_SAMPLES_PER_ID

SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    meta TEXT,
    FOREIGN KEY(identity_id) REFERENCES identities(id)
);
"""

def _connect():
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as c:
        c.executescript(SCHEMA)

def add_identity(identity_id: str, display_name: str):
    with _connect() as c:
        c.execute("INSERT OR IGNORE INTO identities(id, display_name) VALUES (?,?)",
                  (identity_id, display_name))

def rename_identity(identity_id: str, new_name: str):
    with _connect() as c:
        c.execute("UPDATE identities SET display_name=? WHERE id=?", (new_name, identity_id))

def list_identities() -> List[Tuple[str,str]]:
    with _connect() as c:
        cur = c.execute("SELECT id, display_name FROM identities ORDER BY id")
        return cur.fetchall()

def add_sample(identity_id: str, embedding: np.ndarray, meta: dict = None):
    if meta is None: meta = {}
    blob = embedding.astype("float32").tobytes()
    with _connect() as c:
        c.execute("INSERT INTO samples(identity_id, embedding, meta) VALUES (?,?,?)",
                  (identity_id, blob, json.dumps(meta)))

def get_samples(identity_id: str) -> np.ndarray:
    with _connect() as c:
        cur = c.execute("SELECT embedding FROM samples WHERE identity_id=?", (identity_id,))
        rows = cur.fetchall()
    if not rows:
        return np.zeros((0, 2048), dtype="float32")
    emb_list = [np.frombuffer(r[0], dtype="float32") for r in rows]
    return np.vstack(emb_list)

def get_all_identities_with_embs() -> List[Tuple[str, str, np.ndarray]]:
    """
    Retorna lista de (identity_id, display_name, embeddings_matrix)
    """
    out = []
    with _connect() as c:
        cur = c.execute("SELECT id, display_name FROM identities")
        id_rows = cur.fetchall()
    for ident_id, name in id_rows:
        embs = get_samples(ident_id)
        out.append((ident_id, name, embs))
    return out

def get_prototype(embs: np.ndarray) -> Optional[np.ndarray]:
    if embs.shape[0] == 0:
        return None
    if embs.shape[0] >= MIN_SAMPLES_PER_ID:
        return np.mean(embs, axis=0)
    else:
        return np.median(embs, axis=0)
