"""Run the inherited extraction verification."""

import argparse
from pathlib import Path

from src.config import DATASET, ROOT, feature_folder
from src.evaluation.evaluator import verify
from src.visualization.result_exporter import json_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    folder = args.out or feature_folder(args.mode)
    pilot = args.pilot or folder.with_name("pilot")
    report = ROOT / "outputs" / "evaluation" / f"verification_{args.mode}.json"
    print(json_text(verify(folder, args.dataset, args.mode, pilot, report_path=report)))


if __name__ == "__main__":
    main()
