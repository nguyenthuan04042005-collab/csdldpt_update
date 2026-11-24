"""Run the existing catalog-video search at the command line."""

import argparse
from pathlib import Path

from src.config import DATABASE_PATH, feature_folder
from src.retrieval.similarity import METRICS
from src.services.query_service import QueryService
from src.visualization.result_exporter import json_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", help="Catalog ID, for example boxing/person01_boxing_d1.mp4")
    parser.add_argument("--db", type=Path, default=DATABASE_PATH)
    parser.add_argument("--metric", choices=METRICS, default="manhattan")
    parser.add_argument("--without-d", action="store_true")
    parser.add_argument("--other-people", action="store_true")
    args = parser.parse_args()
    service = QueryService(args.db.resolve(), feature_folder("full"))
    if args.video_id not in service.engine.index:
        parser.error("Video ID not found in catalog")
    result = service.search(args.video_id, args.metric, not args.without_d, args.other_people)
    print(json_text(result))


if __name__ == "__main__":
    main()
