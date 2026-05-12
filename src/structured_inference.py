"""Runtime orchestration for the ontology-aware Wikontic-style pipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .config import DATA_DIR, OUTPUT_DIR as DEFAULT_OUTPUT_DIR
from .extractor import Extractor
from .parse_nxml import load_all_nurse_articles
from .structured_aligner import Aligner, GraphState
from .umls_linker import UMLSLinker


WIKONTIC_OUTPUT_FILES = {
    "initial_triplets": "initial_triplets.jsonl",
    "final_triplets": "final_triplets.jsonl",
    "filtered_triplets": "filtered_triplets.jsonl",
    "ontology_filtered_triplets": "ontology_filtered_triplets.jsonl",
}


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    file_handler = logging.FileHandler(output_dir / "pipeline.log")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    return logging.getLogger("pipeline")


def load_processed(final_path: Path) -> set[str]:
    done = set()
    if final_path.exists():
        for line in final_path.read_text().splitlines():
            try:
                done.add(json.loads(line).get("source", ""))
            except Exception:
                pass
    return done


def write_filtered(files, counts, record: dict):
    files["filtered_triplets"].write(json.dumps(record, ensure_ascii=False) + "\n")
    files["filtered_triplets"].flush()
    counts["filtered_triplets"] += 1


def run(limit=None, resume=False, output_dir=DEFAULT_OUTPUT_DIR):
    out_dir = Path(output_dir)
    logger = setup_logger(out_dir)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    umls_key = os.environ.get("UMLS_API_KEY", "")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    if not umls_key:
        raise ValueError("UMLS_API_KEY not set")

    extractor = Extractor(api_key=anthropic_key)
    graph_state = GraphState(save_dir=out_dir)
    aligner = Aligner(extractor=extractor, graph_state=graph_state)
    linker = UMLSLinker(api_key=umls_key)

    paths = {
        name: out_dir / filename
        for name, filename in WIKONTIC_OUTPUT_FILES.items()
    }

    if resume:
        processed = load_processed(paths["final_triplets"])
        loaded = graph_state.load_triplets(paths["final_triplets"])
        if processed:
            logger.info("Resuming: %d articles already processed", len(processed))
        if loaded:
            logger.info("Loaded %d existing final triplets into graph state", loaded)
    else:
        processed = set()
        for path in paths.values():
            if path.exists():
                path.unlink()

    aligner.seed_from_graph_state()

    articles = load_all_nurse_articles(DATA_DIR)
    if limit:
        articles = articles[:limit]
        logger.info("Dry run: processing %d articles", limit)

    files = {name: open(path, "a") for name, path in paths.items()}
    counts = {name: 0 for name in paths}

    try:
        for idx, article in enumerate(articles):
            if article.filename in processed:
                logger.info("[%d/%d] skip (done) %s", idx + 1, len(articles), article.filename)
                continue
            logger.info("[%d/%d] %s", idx + 1, len(articles), article.title)

            for section in article.sections:
                if len(section.text) < 30:
                    continue

                try:
                    raw_triplets = extractor.extract_triplets(section.text)
                except Exception as exc:
                    logger.error("Extraction failed %s / %s: %s", article.filename, section.title, exc)
                    write_filtered(files, counts, {
                        "source": article.filename,
                        "section": section.title,
                        "source_text": section.text,
                        "exception_text": str(exc),
                        "stage": "extract_triplets",
                    })
                    continue

                for triplet in raw_triplets:
                    initial = dict(triplet)
                    initial["source"] = article.filename
                    initial["section"] = section.title
                    files["initial_triplets"].write(json.dumps(initial, ensure_ascii=False) + "\n")
                    counts["initial_triplets"] += 1

                    try:
                        out = _process_triplet(
                            text=section.text,
                            triplet=triplet,
                            source=article.filename,
                            section=section.title,
                            aligner=aligner,
                            graph_state=graph_state,
                            linker=linker,
                        )

                        ok, reason = aligner.validate_backbone_triplet(
                            out["subject_type_id"],
                            out["relation_label"],
                            out["object_type_id"],
                        )
                        if ok:
                            graph_state.record_relation(out["relation"])
                            graph_state.add_triplet(out)
                            files["final_triplets"].write(json.dumps(out, ensure_ascii=False) + "\n")
                            counts["final_triplets"] += 1
                        else:
                            out["filtered_reason"] = reason
                            out["exception_text"] = reason
                            files["ontology_filtered_triplets"].write(json.dumps(out, ensure_ascii=False) + "\n")
                            counts["ontology_filtered_triplets"] += 1

                    except Exception as exc:
                        err = dict(initial)
                        err["exception_text"] = str(exc)
                        err["stage"] = "process_triplet"
                        write_filtered(files, counts, err)
                        logger.warning("Skip triplet %s: %s", triplet, exc)

                for file in files.values():
                    file.flush()

            if (idx + 1) % 10 == 0:
                graph_state.save()
                logger.info("Checkpoint: %s", counts)
    finally:
        for file in files.values():
            file.close()
        graph_state.save()

    logger.info("Done. Counts: %s", counts)


def _process_triplet(
    text: str,
    triplet: dict,
    source: str,
    section: str,
    aligner: Aligner,
    graph_state: GraphState,
    linker: UMLSLinker,
) -> dict:
    subject_type_id, subject_type_label = aligner.align_entity_type(triplet.get("subject_type", ""))
    object_type_id, object_type_label = aligner.align_entity_type(triplet.get("object_type", ""))
    relation_label, relation_id = aligner.align_relation(
        text=text,
        triplet=triplet,
        subj_type_id=subject_type_id,
        obj_type_id=object_type_id,
    )

    subject_cui, subject_name = linker.lookup(triplet.get("subject", ""), triplet.get("subject_type"))
    object_cui, object_name = linker.lookup(triplet.get("object", ""), triplet.get("object_type"))

    out = dict(triplet)
    out["source"] = source
    out["section"] = section
    out["source_text"] = text
    out["subject"] = subject_name if subject_cui and subject_name else triplet["subject"]
    out["object"] = object_name if object_cui and object_name else triplet["object"]
    out["subject_cui"] = subject_cui
    out["object_cui"] = object_cui
    out["subject_type_id"] = subject_type_id
    out["object_type_id"] = object_type_id
    out["subject_type_label"] = subject_type_label
    out["object_type_label"] = object_type_label
    out["relation_label"] = relation_label
    out["relation_id"] = relation_id
    out["relation_raw"] = triplet.get("relation")
    out["relation"] = relation_label if relation_label else triplet.get("relation", "").strip().lower()

    out["subject"] = aligner.refine_entity_name(text=text, triplet=out, is_object=False)
    out["object"] = aligner.refine_entity_name(text=text, triplet=out, is_object=True)
    out["subject_cui"] = graph_state.cui_for_name(out["subject"]) or subject_cui
    out["object_cui"] = graph_state.cui_for_name(out["object"]) or object_cui
    graph_state.record_entity(triplet["subject"], out["subject"], out["subject_cui"])
    graph_state.record_entity(triplet["object"], out["object"], out["object_cui"])
    return out
