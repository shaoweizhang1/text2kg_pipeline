"""Build the UMLS Semantic Network ontology mappings used by the aligner.

Mirrors Wikontic's ontology_mappings/ structure (entity_type2*, prop2*) but the
source ontology is the UMLS Semantic Network (NLM) instead of Wikidata.

One-shot: hits the UMLS REST endpoint `/semantic-network/current` once
(returns all ~181 records: 127 SemanticTypes + 54 Relations in one shot)
and writes the JSON tables into src/ontology_mappings/.

Run from project root:
    python src/build_umls_ontology.py
"""

import json
import os
from collections import defaultdict

import requests
from dotenv import load_dotenv

from src.config import ONTOLOGY_DIR

load_dotenv()

OUT_DIR = ONTOLOGY_DIR
OUT_DIR.mkdir(exist_ok=True)

API = "https://uts-ws.nlm.nih.gov/rest/semantic-network/current"


def fetch_all() -> list:
    key = os.environ["UMLS_API_KEY"]
    r = requests.get(API, params={"apiKey": key, "pageSize": 500}, timeout=30)
    r.raise_for_status()
    return r.json()["result"]


def is_st(rec: dict) -> bool:
    return rec["treeNumber"][:1] in ("A", "B", "H")


def is_rel(rec: dict) -> bool:
    return rec["treeNumber"].startswith("R")


def parent_tree_no(tree_no: str) -> str:
    """A1.1.3.3 -> A1.1.3 ; A1 -> '' (root)."""
    if "." not in tree_no:
        return ""
    return tree_no.rsplit(".", 1)[0]


def build_hierarchy(records: list) -> dict:
    """TUI -> [ancestor TUIs] (from root-most down to direct parent)."""
    by_tree = {r["treeNumber"]: r["ui"] for r in records}
    out = {}
    for r in records:
        ancestors = []
        tn = parent_tree_no(r["treeNumber"])
        while tn:
            if tn in by_tree:
                ancestors.append(by_tree[tn])
            tn = parent_tree_no(tn)
        out[r["ui"]] = list(reversed(ancestors))  # root → ... → parent
    return out


def build_st_constraints(records: list, name_to_tui: dict) -> dict:
    """For each Relation, list (subject_TUI, object_TUI) pairs that are valid.

    We use the direct `relations` field on each ST: each entry there is
    (this_ST is the subject)  --[entry.type]-->  (entry.relation is the object).
    """
    constraints: dict = defaultdict(lambda: {"subject_types": set(), "object_types": set(),
                                              "pairs": []})
    for st in (r for r in records if is_st(r)):
        for rel_entry in st.get("relations", []):
            rel_name = rel_entry["type"]
            obj_name = rel_entry["relation"]
            if obj_name not in name_to_tui:
                continue
            constraints[rel_name]["subject_types"].add(st["ui"])
            constraints[rel_name]["object_types"].add(name_to_tui[obj_name])
            constraints[rel_name]["pairs"].append([st["ui"], name_to_tui[obj_name]])
    return {
        rel: {
            "subject_types": sorted(v["subject_types"]),
            "object_types":  sorted(v["object_types"]),
            "pairs":         v["pairs"],
        }
        for rel, v in constraints.items()
    }


def main():
    records = fetch_all()
    sts  = [r for r in records if is_st(r)]
    rels = [r for r in records if is_rel(r)]
    print(f"fetched {len(records)} records: {len(sts)} STs, {len(rels)} relations")

    # Name -> TUI lookups
    name_to_tui = {r["name"]: r["ui"] for r in records}

    # ---- Semantic Types ----
    st2label    = {r["ui"]: r["name"]                                for r in sts}
    st2abbrev   = {r["ui"]: r["abbreviation"]                        for r in sts}
    st2def      = {r["ui"]: r["definition"]                          for r in sts}
    st2aliases  = {r["ui"]: sorted({r["name"], r["abbreviation"]})   for r in sts}
    st2hierarchy = {ui: anc for ui, anc in build_hierarchy(sts).items()}

    # ---- Relations ----
    rel2label   = {r["ui"]: r["name"]                                for r in rels}
    rel2abbrev  = {r["ui"]: r["abbreviation"]                        for r in rels}
    rel2def     = {r["ui"]: r["definition"]                          for r in rels}
    # Aliases: include the snake_case name and a space-separated form for embedding lookup
    def aliases_for_rel(name: str, abbrev: str) -> list:
        spaced = name.replace("_", " ")
        return sorted({name, spaced, abbrev})
    rel2aliases = {r["ui"]: aliases_for_rel(r["name"], r["abbreviation"]) for r in rels}
    rel2hierarchy = build_hierarchy(rels)

    # ---- Constraints (valid (subject_ST, REL, object_ST) pairs) ----
    # Keyed by relation NAME (snake_case) for convenience when matching against LLM output.
    constraints = build_st_constraints(records, name_to_tui)

    # ---- write ----
    files = {
        "st2label.json":         st2label,
        "st2abbrev.json":        st2abbrev,
        "st2definition.json":    st2def,
        "st2aliases.json":       st2aliases,
        "st2hierarchy.json":     st2hierarchy,
        "rel2label.json":        rel2label,
        "rel2abbrev.json":       rel2abbrev,
        "rel2definition.json":   rel2def,
        "rel2aliases.json":      rel2aliases,
        "rel2hierarchy.json":    rel2hierarchy,
        "rel2constraints.json":  constraints,
    }
    for name, data in files.items():
        path = OUT_DIR / name
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  wrote {path.name:<26} {len(data):>4} entries")


if __name__ == "__main__":
    main()
