"""Open SQLite connections for the local video catalog."""

import sqlite3


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)
