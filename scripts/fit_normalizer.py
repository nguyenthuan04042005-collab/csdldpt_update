"""Inspect the mean, standard deviation and usable dimensions for one query."""

import argparse
from pathlib import Path

from src.config import DATABASE_PATH, feature_folder
from src.retrieval.normalizer import fit_normalizer
from src.services.query_service import QueryService
from src.visualization.result_exporter import json_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", help="Catalog ID used to select matching candidates")
    parser.add_argument("--db", type=Path, default=DATABASE_PATH)
    parser.add_argument("--other-people", action="store_true")
    args = parser.parse_args()
    engine = QueryService(args.db.resolve(), feature_folder("full")).engine
    if args.video_id not in engine.index:
        parser.error("Video ID not found in catalog")
    pool = engine.candidates(args.video_id, args.other_people)
    mean, std, usable = fit_normalizer(engine.vectors[pool], engine.valid[pool])
    print(json_text({
        "video_id": args.video_id, "candidates": len(pool),
        "mean": mean.tolist(), "std": std.tolist(), "usable": usable.tolist(),
    }))


if __name__ == "__main__":
    main()
