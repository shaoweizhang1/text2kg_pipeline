"""Entry point for the medical text-to-KG pipeline."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.config import OUTPUT_DIR as DEFAULT_OUTPUT_DIR
from src.structured_inference import run


load_dotenv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N articles")
    parser.add_argument("--resume", action="store_true",
                        help="Skip articles already in final_triplets.jsonl")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: output/)")
    args = parser.parse_args()
    run(limit=args.limit, resume=args.resume, output_dir=args.output_dir)
