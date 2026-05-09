"""
Medical KG extraction pipeline.
StatPearls nurse articles → LLM triples → UMLS normalization → JSONL.

Usage:
    python pipeline.py                  # run all 117 articles
    python pipeline.py --limit 3        # dry run, first 3 articles only
    python pipeline.py --resume         # skip already-processed articles
"""

import os
import json
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from parse_nxml import load_all_nurse_articles
from extractor import Extractor
from aligner import Aligner
from umls_linker import UMLSLinker

LOG_DIR = Path("output")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "pipeline.log"),
    ],
)
logger = logging.getLogger("pipeline")

DATA_DIR = "statpearls_NBK430685"
OUTPUT_DIR = "output"


def load_processed(output_dir: str) -> set:
    """Return set of already-processed filenames (for resume)."""
    done = set()
    p = Path(output_dir) / "triplets_enriched.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                t = json.loads(line)
                done.add(t.get("source", ""))
            except Exception:
                pass
    return done


def run(limit: int = None, resume: bool = False):
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    umls_key = os.environ.get("UMLS_API_KEY", "")

    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    if not umls_key:
        raise ValueError("UMLS_API_KEY not set")

    extractor = Extractor(api_key=anthropic_key)
    aligner = Aligner(save_dir=OUTPUT_DIR)
    linker = UMLSLinker(api_key=umls_key)

    articles = load_all_nurse_articles(DATA_DIR)
    if limit:
        articles = articles[:limit]
        logger.info("Dry run: processing %d articles", limit)

    processed = load_processed(OUTPUT_DIR) if resume else set()
    if processed:
        logger.info("Resuming: %d articles already done", len(processed))

    enriched_path = Path(OUTPUT_DIR) / "triplets_enriched.jsonl"
    mode = "a" if resume else "w"

    with open(enriched_path, mode) as out:
        total = 0

        for idx, article in enumerate(articles):
            if article.filename in processed:
                logger.info("[%d/%d] skip (done) %s", idx + 1, len(articles), article.filename)
                continue

            logger.info("[%d/%d] %s", idx + 1, len(articles), article.title)

            for sec in article.sections:
                if len(sec.text) < 30:
                    continue

                # Stage 1: LLM triple extraction
                try:
                    raw_triplets = extractor.extract_triplets(sec.text)
                except Exception as e:
                    logger.error("Extraction failed for %s / %s: %s", article.filename, sec.title, e)
                    continue

                for t in raw_triplets:
                    try:
                        # Stage 2: UMLS normalization
                        s_cui, s_name = linker.lookup(t.get("subject", ""))
                        o_cui, o_name = linker.lookup(t.get("object", ""))

                        t["subject"]     = aligner.normalize_entity(t["subject"], s_cui, s_name)
                        t["object"]      = aligner.normalize_entity(t["object"],  o_cui, o_name)
                        t["relation"]    = aligner.normalize_relation(t["relation"])
                        t["subject_cui"] = s_cui
                        t["object_cui"]  = o_cui
                        t["source"]      = article.filename
                        t["section"]     = sec.title

                        aligner.add_triplet(t)
                        out.write(json.dumps(t) + "\n")
                        out.flush()
                        total += 1

                    except Exception as e:
                        logger.warning("Skip triplet %s: %s", t, e)

            if (idx + 1) % 10 == 0:
                aligner.save()
                logger.info("Checkpoint: %d triplets total", total)

    aligner.save()
    logger.info("Done. Total triplets: %d", total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N articles")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed articles")
    args = parser.parse_args()
    run(limit=args.limit, resume=args.resume)
