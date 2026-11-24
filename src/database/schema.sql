CREATE TABLE IF NOT EXISTS dashboard_source (digest TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY, action TEXT NOT NULL, person TEXT NOT NULL,
    setting TEXT NOT NULL, frames INTEGER NOT NULL, fps REAL NOT NULL,
    valid_pairs INTEGER NOT NULL, dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL CHECK(length(vector) = 1296),
    valid BLOB NOT NULL CHECK(length(valid) = 324)
);
CREATE TABLE IF NOT EXISTS ratings (
    query TEXT NOT NULL, candidate TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK(verdict IN ('similar','different','unsure')),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (query, candidate)
);
