"""Paths and fixed pilot samples for the local video retrieval project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = (ROOT / "data" / "dataset").resolve()
CONFIG_PATH = ROOT / "data" / "extraction_config.json"
DATABASE_PATH = ROOT / "database" / "videos.db"
OUTPUT = ROOT / "outputs" / "intermediate"
WEB = ROOT / "src" / "visualization"
LEGACY_RESULTS = ROOT.parent / "results"
DEFAULT_QUERY = "boxing/person01_boxing_d1.mp4"

PILOT = (
    "boxing/person01_boxing_d1.mp4",
    "boxing/person16_boxing_d3.mp4",
    "handclapping/person01_handclapping_d2.mp4",
    "handclapping/person16_handclapping_d4.mp4",
    "handwaving/person01_handwaving_d3.mp4",
    "handwaving/person16_handwaving_d1.mp4",
    "jogging/person01_jogging_d4.mp4",
    "jogging/person16_jogging_d2.mp4",
    "running/person01_running_d1.mp4",
    "running/person16_running_d3.mp4",
    "walking/person01_walking_d2.mp4",
    "walking/person16_walking_d4.mp4",
)

def feature_folder(mode):
    """Prefer local extraction artifacts; reuse the previously exported dataset if present."""
    local = OUTPUT / mode
    old = LEGACY_RESULTS / mode
    return old if not (local / "features.npz").is_file() and (old / "features.npz").is_file() else local
