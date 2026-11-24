"""Import previously extracted results or run the existing A-F extractor first."""

import argparse
import os
from pathlib import Path

# Set native thread limits before importing the extraction stack.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

from src.config import CONFIG_PATH, DATABASE_PATH, DATASET, OUTPUT, feature_folder
from src.database.repository import prepare_database
from src.services.indexing_service import extract_dataset
from src.visualization.result_exporter import json_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), help="Extract videos before importing")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path, help="Output folder for extraction, or source folder for import")
    parser.add_argument("--db", type=Path, default=DATABASE_PATH)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.mode:
            source = args.out or OUTPUT / args.mode
            result = extract_dataset(args.dataset, args.config, source, args.mode, args.workers)
            print(json_text(result))
        else:
            source = args.out or feature_folder("full")
        if args.mode != "pilot":
            if not (source / "features.npz").is_file():
                parser.error("No saved features. Run with --mode full to extract first.")
            prepare_database(args.db.resolve(), source)
            print(f"SQLite: {args.db.resolve()}")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
