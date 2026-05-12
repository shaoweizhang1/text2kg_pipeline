"""Visualize local KG triplets with PyVis, following Wikontic's simple style."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pyvis.network import Network


TYPE_COLORS = {
    "Disease or Syndrome": "#d73027",
    "Pathologic Function": "#fc8d59",
    "Sign or Symptom": "#fdae61",
    "Finding": "#fee08b",
    "Therapeutic or Preventive Procedure": "#1a9850",
    "Diagnostic Procedure": "#66bd63",
    "Pharmacologic Substance": "#4575b4",
    "Clinical Drug": "#74add1",
    "Body Part, Organ, or Organ Component": "#984ea3",
    "Anatomical Structure": "#b276b2",
}
DEFAULT_COLOR = "#C7C8CC"


def iter_jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def load_rows(path: Path, source: str | None = None) -> list[dict]:
    rows = list(iter_jsonl(path))
    if source:
        rows = [row for row in rows if row.get("source") == source]
    return rows


def filter_rows(rows: list[dict], limit_edges: int | None, min_degree: int) -> list[dict]:
    if limit_edges is not None:
        rows = rows[:limit_edges]

    if min_degree <= 0:
        return rows

    degree = Counter()
    for row in rows:
        degree[row.get("subject")] += 1
        degree[row.get("object")] += 1

    keep = {node for node, count in degree.items() if count >= min_degree}
    return [
        row for row in rows
        if row.get("subject") in keep and row.get("object") in keep
    ]


def node_color(type_label: str | None) -> str:
    return TYPE_COLORS.get(type_label, DEFAULT_COLOR)


def tooltip(role: str, row: dict) -> str:
    return "<br>".join([
        f"{role}: {row.get(role, '')}",
        f"CUI: {row.get(f'{role}_cui') or ''}",
        f"Type: {row.get(f'{role}_type_label') or ''}",
        f"Type ID: {row.get(f'{role}_type_id') or ''}",
        f"Source: {row.get('source') or ''}",
        f"Section: {row.get('section') or ''}",
    ])


def edge_tooltip(row: dict) -> str:
    return "<br>".join([
        f"Relation: {row.get('relation') or row.get('relation_label') or ''}",
        f"Source: {row.get('source') or ''}",
        f"Section: {row.get('section') or ''}",
    ])


def visualize(rows: list[dict], output_file: Path):
    net = Network(
        height="780px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )
    net.barnes_hut()
    added_nodes = set()

    for row in rows:
        subject = row.get("subject")
        obj = row.get("object")
        relation = row.get("relation") or row.get("relation_label")
        if not subject or not obj or not relation:
            continue

        if subject not in added_nodes:
            net.add_node(
                subject,
                label=subject,
                color=node_color(row.get("subject_type_label")),
                title=tooltip("subject", row),
            )
            added_nodes.add(subject)

        if obj not in added_nodes:
            net.add_node(
                obj,
                label=obj,
                color=node_color(row.get("object_type_label")),
                title=tooltip("object", row),
            )
            added_nodes.add(obj)

        net.add_edge(
            subject,
            obj,
            label=relation,
            color="#000000",
            title=edge_tooltip(row),
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to final_triplets.jsonl or triplets.jsonl")
    parser.add_argument("--output", type=Path, required=True,
                        help="HTML file to write")
    parser.add_argument("--source", default=None,
                        help="Only include rows from one source filename")
    parser.add_argument("--limit_edges", type=int, default=None,
                        help="Keep only the first N edges")
    parser.add_argument("--min_degree", type=int, default=0,
                        help="Drop nodes whose degree is below this value")
    args = parser.parse_args()

    rows = load_rows(args.input, source=args.source)
    rows = filter_rows(rows, limit_edges=args.limit_edges, min_degree=args.min_degree)
    visualize(rows, args.output)
    print(f"Wrote {args.output} ({len(rows)} edges)")


if __name__ == "__main__":
    main()
