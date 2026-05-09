"""
Entity/relation deduplication via UMLS CUI.
Two entities that map to the same CUI are treated as the same entity.
No local embedding model required.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple


class Aligner:
    def __init__(self, save_dir: str = "output"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # cui → preferred_name (canonical label)
        self.cui_to_name: dict[str, str] = {}
        # surface_form → (cui, canonical_name)
        self.entity_cache: dict[str, Tuple[Optional[str], str]] = {}
        # canonical relation labels seen so far
        self.relations: set[str] = set()
        # deduplicated triplets
        self.triplets: List[dict] = []

    def normalize_entity(self, surface: str, cui: Optional[str], preferred_name: Optional[str]) -> str:
        """Return canonical label: UMLS preferred name if CUI found, else surface form."""
        if cui and preferred_name:
            self.cui_to_name[cui] = preferred_name
            self.entity_cache[surface] = (cui, preferred_name)
            return preferred_name
        self.entity_cache[surface] = (None, surface)
        return surface

    def normalize_relation(self, relation: str) -> str:
        """Lower-case and deduplicate relations."""
        normalized = relation.strip().lower()
        self.relations.add(normalized)
        return normalized

    def add_triplet(self, triplet: dict):
        key = (triplet["subject"], triplet["relation"], triplet["object"])
        if not any((t["subject"], t["relation"], t["object"]) == key for t in self.triplets):
            self.triplets.append(triplet)

    def save(self):
        (self.save_dir / "triplets.jsonl").write_text(
            "\n".join(json.dumps(t) for t in self.triplets)
        )
        (self.save_dir / "entities.json").write_text(
            json.dumps(self.cui_to_name, indent=2)
        )
        (self.save_dir / "relations.json").write_text(
            json.dumps(sorted(self.relations), indent=2)
        )
        print(f"[aligner] saved {len(self.triplets)} triplets, "
              f"{len(self.cui_to_name)} UMLS entities, "
              f"{len(self.relations)} relations")
