"""Persist and load video descriptors and dashboard reviews."""

import csv
from contextlib import closing
from hashlib import sha256
import io
from pathlib import Path

import numpy as np

from ..features.feature_utils import FEATURE_COUNT, GROUP_SLICES
from .connection import connect

SCHEMA = Path(__file__).with_name("schema.sql")
RATINGS = ("similar", "different", "unsure")


def prepare_database(path, source):
    """Import an extraction once, retaining ratings for the same video IDs."""
    feature_file = source / "features.npz"
    manifest_file = source / "manifest.csv"
    fingerprint = sha256(feature_file.read_bytes() + manifest_file.read_bytes()).hexdigest()
    with closing(connect(path)) as db, db:
        db.executescript(SCHEMA.read_text(encoding="utf-8"))
        if db.execute("SELECT digest FROM dashboard_source").fetchone() == (fingerprint,):
            return
        with manifest_file.open(encoding="utf-8", newline="") as stream:
            manifest = list(csv.DictReader(stream))
        with np.load(feature_file, allow_pickle=False) as archive:
            ids = archive["video_ids"].tolist()
            names = archive["feature_names"].tolist()
            vectors = archive["vectors"].astype("<f4", copy=False)
            valid = archive["valid"].astype(np.bool_, copy=False)
        if (ids != [item["video_id"] for item in manifest] or not ids
                or len(set(ids)) != len(ids)
                or vectors.shape != valid.shape or vectors.shape != (len(ids), FEATURE_COUNT)
                or not np.isfinite(vectors).all()
                or any(any(not name.startswith(group + "_") for name in names[span])
                       for group, span in GROUP_SLICES.items())):
            raise ValueError("Extraction manifest and vector schema are inconsistent")
        rows = []
        for item, vector, mask in zip(manifest, vectors, valid):
            key = item["video_id"]
            action, filename = key.split("/")
            person, _, setting = Path(filename).stem.split("_")
            rows.append((key, action, person, setting, int(item["frames"]), float(item["fps"]),
                         int(item["valid_pairs"]), int(mask.sum()), vector.tobytes(), mask.tobytes()))
        db.execute("DELETE FROM videos")
        db.executemany("INSERT INTO videos VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        db.execute("DELETE FROM ratings WHERE query NOT IN (SELECT id FROM videos) OR candidate NOT IN (SELECT id FROM videos)")
        db.execute("DELETE FROM dashboard_source")
        db.execute("INSERT INTO dashboard_source VALUES (?)", (fingerprint,))


class VideoRepository:
    def __init__(self, path):
        self.path = path

    def connect(self):
        return connect(self.path)

    def load_catalog(self):
        with closing(self.connect()) as db:
            records = db.execute("SELECT * FROM videos ORDER BY id").fetchall()
        if not records:
            raise ValueError("No indexed videos. Run scripts.build_database first.")
        items = [
            dict(zip(("id", "action", "person", "setting", "frames", "fps", "valid_pairs", "dimensions"), row[:8]))
            for row in records
        ]
        vectors = np.stack([np.frombuffer(row[8], dtype="<f4") for row in records]).astype(float)
        valid = np.stack([np.frombuffer(row[9], dtype=np.bool_) for row in records])
        return items, vectors, valid

    def ratings_for_query(self, key):
        with closing(self.connect()) as db:
            ratings = dict(db.execute("SELECT candidate,verdict FROM ratings WHERE query=?", (key,)))
            count = db.execute("SELECT count(*) FROM ratings").fetchone()[0]
        return ratings, count

    def rate(self, query, candidate, verdict):
        with closing(self.connect()) as db, db:
            if verdict is None:
                db.execute("DELETE FROM ratings WHERE query=? AND candidate=?", (query, candidate))
            else:
                db.execute("""
                    INSERT INTO ratings (query, candidate, verdict) VALUES (?, ?, ?)
                    ON CONFLICT(query,candidate) DO UPDATE SET
                      verdict=excluded.verdict, updated_at=CURRENT_TIMESTAMP
                """, (query, candidate, verdict))
            return db.execute("SELECT count(*) FROM ratings").fetchone()[0]

    def rating_csv(self):
        with closing(self.connect()) as db:
            rows = db.execute("SELECT query,candidate,verdict,updated_at FROM ratings ORDER BY query,candidate").fetchall()
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(("video_truy_van", "video_ket_qua", "nhan_xet", "cap_nhat_utc"))
        labels = {"similar": "Tương tự", "different": "Khác", "unsure": "Chưa chắc"}
        writer.writerows((query, candidate, labels[verdict], stamp) for query, candidate, verdict, stamp in rows)
        return stream.getvalue().encode("utf-8-sig")
