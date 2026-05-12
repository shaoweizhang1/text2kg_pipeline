"""Local version of Wikontic's ontology-aware structured_aligner."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config import (
    ONTOLOGY_DIR,
    STRUCTURED_ALIGNER_MODEL,
    STRUCTURED_ALIGNER_THRESHOLD,
    STRUCTURED_ALIGNER_TOP_K,
)

logger = logging.getLogger("StructuredAligner")


TYPE_SEED: Dict[str, str] = {
    "disease": "T047",
    "medical condition": "T047",
    "syndrome": "T047",
    "infection": "T047",
    "symptom": "T184",
    "sign": "T184",
    "finding": "T033",
    "medical procedure": "T061",
    "procedure": "T061",
    "treatment": "T061",
    "therapy": "T061",
    "diagnostic procedure": "T060",
    "diagnostic test": "T060",
    "medication": "T121",
    "drug": "T121",
    "clinical drug": "T200",
    "anatomy": "T023",
    "body part": "T023",
    "organ": "T023",
    "patient": "T016",
    "patient group": "T101",
    "population group": "T098",
    "behavior": "T053",
    "measurement": "T081",
    "medical device": "T074",
}


def _sanitize_string(text: str) -> str:
    return (text or "").strip()


def _normalize_key(text: str) -> str:
    return re.sub(r"[\W_]+", " ", (text or "").lower()).strip()


def _normalize_relation(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", (text or "").strip().lower())


class GraphState:
    """In-memory replacement for Wikontic's triplets DB collections."""

    def __init__(self, save_dir: str | Path = "output"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cui_to_name: dict[str, str] = {}
        self.name_to_cui: dict[str, str] = {}
        self.entity_cache: dict[str, Tuple[Optional[str], str]] = {}
        self.relations: set[str] = set()
        self.triplets: List[dict] = []

    def record_entity(self, surface: str, canonical_name: str, cui: Optional[str] = None) -> str:
        canonical = canonical_name or surface
        if cui and canonical:
            self.cui_to_name[cui] = canonical
            self.name_to_cui[canonical] = cui
        self.entity_cache[surface] = (cui, canonical)
        return canonical

    def cui_for_name(self, name: str) -> Optional[str]:
        return self.name_to_cui.get(name)

    def record_relation(self, relation: Optional[str]) -> str:
        normalized = (relation or "").strip()
        if normalized:
            self.relations.add(normalized)
        return normalized

    def add_triplet(self, triplet: dict):
        key = (triplet["subject"], triplet["relation"], triplet["object"])
        if not any((t["subject"], t["relation"], t["object"]) == key for t in self.triplets):
            self.triplets.append(triplet)

    def load_triplets(self, path: Path) -> int:
        if not path.exists():
            return 0
        loaded = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                triplet = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.add_triplet(triplet)
            self.record_relation(triplet.get("relation"))
            for role in ("subject", "object"):
                cui = triplet.get(f"{role}_cui")
                name = triplet.get(role)
                if cui and name:
                    self.cui_to_name[cui] = name
                    self.name_to_cui[name] = cui
            loaded += 1
        return loaded

    def save(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "triplets.jsonl").write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in self.triplets)
        )
        (self.save_dir / "entities.json").write_text(
            json.dumps(self.cui_to_name, indent=2, ensure_ascii=False)
        )
        (self.save_dir / "relations.json").write_text(
            json.dumps(sorted(self.relations), indent=2, ensure_ascii=False)
        )
        print(f"[aligner] saved {len(self.triplets)} triplets, "
              f"{len(self.cui_to_name)} UMLS entities, "
              f"{len(self.relations)} relations")


class Aligner:
    """Ontology-aware entity alias retriever.

    Wikontic's `structured_aligner.Aligner.retrieve_entity_by_type` retrieves
    generated KG entity aliases constrained by entity type. This local version
    keeps the same method names and stores aliases in an online FAISS index
    instead of MongoDB.
    """

    def __init__(
        self,
        extractor,
        graph_state,
        model_name: str = STRUCTURED_ALIGNER_MODEL,
        k: int = STRUCTURED_ALIGNER_TOP_K,
        threshold: float = STRUCTURED_ALIGNER_THRESHOLD,
    ):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError as e:
            raise RuntimeError(
                "Wikontic structured alignment requires sentence-transformers, numpy, and faiss-cpu. "
                "Install dependencies from requirements.txt first."
            ) from e

        self.extractor = extractor
        self.graph_state = graph_state
        self.model = SentenceTransformer(model_name)
        self.faiss = faiss
        self.np = np
        self.k = k
        self.threshold = threshold

        self.entity_type_collection_name = "entity_types"
        self.entity_type_aliases_collection_name = "entity_type_aliases"
        self.property_collection_name = "properties"
        self.property_aliases_collection_name = "property_aliases"
        self.entity_aliases_collection_name = "entity_aliases"
        self.triplets_collection_name = "triplets"
        self.filtered_triplets_collection_name = "filtered_triplets"
        self.ontology_filtered_triplets_collection_name = "ontology_filtered_triplets"
        self.initial_triplets_collection_name = "initial_triplets"

        self.aliases: list[str] = []
        self.labels: list[str] = []
        self.entity_types: list[str] = []
        self.alias_to_label: dict[tuple[str, str], str] = {}
        self.entity_aliases_index = None
        self.embedding_dim: Optional[int] = None

        self.t2label = json.loads((ONTOLOGY_DIR / "st2label.json").read_text())
        self.t2aliases = json.loads((ONTOLOGY_DIR / "st2aliases.json").read_text())
        self.t2def = json.loads((ONTOLOGY_DIR / "st2definition.json").read_text())
        self.t2parents = json.loads((ONTOLOGY_DIR / "st2hierarchy.json").read_text())
        self.rel2label = json.loads((ONTOLOGY_DIR / "rel2label.json").read_text())
        self.rel2def = json.loads((ONTOLOGY_DIR / "rel2definition.json").read_text())
        self.rel2constraints = json.loads((ONTOLOGY_DIR / "rel2constraints.json").read_text())

        self.label2id: Dict[str, str] = {label: rid for rid, label in self.rel2label.items()}
        for label in self.rel2constraints:
            self.label2id.setdefault(label, label)

        self._type_alias_index: Dict[str, str] = {}
        for tcode, label in self.t2label.items():
            self._type_alias_index[_normalize_key(label)] = tcode
        for tcode, aliases in self.t2aliases.items():
            for alias in aliases:
                self._type_alias_index.setdefault(_normalize_key(alias), tcode)
        for alias, tcode in TYPE_SEED.items():
            if tcode in self.t2label:
                self._type_alias_index.setdefault(_normalize_key(alias), tcode)
        self._type_cache: Dict[str, Optional[str]] = {}
        self._relation_candidate_cache: Dict[Tuple[str, str], List[str]] = {}

    def seed_from_graph_state(self):
        for triplet in self.graph_state.triplets:
            subject = triplet.get("subject")
            subject_type = triplet.get("subject_type_id")
            if subject and subject_type:
                self.add_entity(subject, subject, subject_type)

            obj = triplet.get("object")
            object_type = triplet.get("object_type_id")
            if obj and object_type:
                self.add_entity(obj, obj, object_type)

    def align_entity_type(self, freeform: str) -> Tuple[Optional[str], Optional[str]]:
        if not freeform or not isinstance(freeform, str):
            return None, None
        norm = _normalize_key(freeform)
        if norm in self._type_cache:
            tcode = self._type_cache[norm]
            return tcode, self.t2label.get(tcode) if tcode else None
        if norm in self._type_alias_index:
            tcode = self._type_alias_index[norm]
            self._type_cache[norm] = tcode
            return tcode, self.t2label[tcode]

        lines = []
        for tcode in sorted(self.t2label):
            label = self.t2label[tcode]
            definition = self.t2def.get(tcode, "") or ""
            definition = (definition[:140].rstrip() + "...") if len(definition) > 140 else definition
            lines.append(f"{tcode} | {label} - {definition}")
        picked = self.extractor.pick_semantic_type(freeform, "\n".join(lines))
        tcode = picked if picked in self.t2label else None
        self._type_cache[norm] = tcode
        return tcode, self.t2label.get(tcode) if tcode else None

    def expand_hierarchy(self, tcode: Optional[str]) -> Set[str]:
        if not tcode:
            return set()
        return {tcode} | set(self.t2parents.get(tcode, []))

    def relation_candidates(self, subj_type_id: Optional[str], obj_type_id: Optional[str]) -> List[str]:
        key = (subj_type_id or "", obj_type_id or "")
        if key in self._relation_candidate_cache:
            return self._relation_candidate_cache[key]
        if not subj_type_id or not obj_type_id:
            self._relation_candidate_cache[key] = []
            return []

        subj_set = self.expand_hierarchy(subj_type_id)
        obj_set = self.expand_hierarchy(obj_type_id)
        candidates = []
        for label, constraints in self.rel2constraints.items():
            if subj_set & set(constraints.get("subject_types", [])) and obj_set & set(constraints.get("object_types", [])):
                candidates.append(label)
        self._relation_candidate_cache[key] = candidates
        return candidates

    def align_relation(
        self,
        text: str,
        triplet: dict,
        subj_type_id: Optional[str],
        obj_type_id: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        candidates = self.relation_candidates(subj_type_id, obj_type_id)
        if not candidates:
            return None, None
        relation = _normalize_relation(triplet.get("relation") or "")
        if relation in candidates:
            return relation, self.label2id.get(relation, relation)

        lines = []
        for label in candidates:
            rid = self.label2id.get(label)
            definition = self.rel2def.get(rid, "") if rid else ""
            definition = (definition[:160].rstrip() + "...") if len(definition) > 160 else definition
            lines.append(f"{label} - {definition}")
        picked = self.extractor.pick_relation_from_candidates(text, triplet, "\n".join(lines))
        if picked and picked in candidates:
            return picked, self.label2id.get(picked, picked)
        return None, None

    def validate_backbone_triplet(
        self,
        subj_type_id: Optional[str],
        relation_label: Optional[str],
        obj_type_id: Optional[str],
    ) -> Tuple[bool, str]:
        if not subj_type_id:
            return False, "missing subject type"
        if not obj_type_id:
            return False, "missing object type"
        if not relation_label:
            return False, "missing relation"
        constraints = self.rel2constraints.get(relation_label)
        if constraints is None:
            return False, f"relation {relation_label!r} has no constraint definition"

        subj_set = self.expand_hierarchy(subj_type_id)
        obj_set = self.expand_hierarchy(obj_type_id)
        for subj, obj in constraints.get("pairs", []):
            if subj in subj_set and obj in obj_set:
                return True, ""
        return False, (
            f"({subj_type_id},{relation_label},{obj_type_id}) "
            f"not in valid type-pair list for this relation"
        )

    def refine_entity_name(
        self,
        text: str,
        triplet: dict,
        sample_id: Optional[str] = None,
        is_object: bool = False,
    ) -> str:
        if is_object:
            entity = self.sanitize_string(triplet.get("object", ""))
            entity_type = triplet.get("object_type_id")
        else:
            entity = self.sanitize_string(triplet.get("subject", ""))
            entity_type = triplet.get("subject_type_id")

        if not entity or not entity_type:
            return entity

        similar_entities = self.retrieve_entity_by_type(
            entity_name=entity,
            entity_type=entity_type,
            sample_id=sample_id,
        )

        if len(similar_entities) > 0:
            if entity in similar_entities:
                updated_entity = similar_entities[entity]
            else:
                updated_entity = self.extractor.refine_entity(
                    text=text,
                    triplet=triplet,
                    candidates=list(similar_entities.values()),
                    is_object=is_object,
                )
                updated_entity = self.sanitize_string(updated_entity)
                if re.sub(r"[^\w\s]", "", updated_entity).strip().lower() == "none":
                    updated_entity = entity
                elif updated_entity not in similar_entities.values():
                    updated_entity = entity
        else:
            updated_entity = entity

        self.add_entity(
            entity_name=updated_entity,
            alias=entity,
            entity_type=entity_type,
            sample_id=sample_id,
        )
        return updated_entity

    def retrieve_entity_by_type(
        self,
        entity_name: str,
        entity_type: str,
        sample_id: Optional[str] = None,
        k: Optional[int] = None,
    ) -> dict[str, str]:
        if self.entity_aliases_index is None or not self.aliases:
            return {}

        limit = k or self.k
        query_embedding = self.get_embedding(entity_name)
        search_limit = min(max(limit * 2, limit), len(self.aliases))

        result: dict[str, str] = {}
        query_key = _normalize_key(entity_name)
        while len(result) < limit and search_limit <= len(self.aliases):
            scores, indices = self.entity_aliases_index.search(
                query_embedding.reshape(1, -1),
                search_limit,
            )
            for score, idx in zip(scores[0], indices[0]):
                idx = int(idx)
                if idx < 0 or float(score) < self.threshold:
                    continue
                alias = self.aliases[idx]
                label = self.labels[idx]
                candidate_type = self.entity_types[idx]
                if candidate_type != entity_type:
                    continue
                if _normalize_key(alias) == query_key:
                    result[alias] = label
                    return result
                result.setdefault(alias, label)
                if len(result) >= limit:
                    break
            if search_limit == len(self.aliases):
                break
            search_limit = min(search_limit * 2, len(self.aliases))
        return result

    def retrieve_similar_entity_names(
        self,
        entity_name: str,
        k: Optional[int] = None,
        sample_id: Optional[str] = None,
    ) -> list[dict[str, str]]:
        if self.entity_aliases_index is None or not self.aliases:
            return []

        limit = k or self.k
        query_embedding = self.get_embedding(entity_name)
        search_limit = min(max(limit * 2, limit), len(self.aliases))

        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        while len(result) < limit and search_limit <= len(self.aliases):
            scores, indices = self.entity_aliases_index.search(
                query_embedding.reshape(1, -1),
                search_limit,
            )
            for score, idx in zip(scores[0], indices[0]):
                idx = int(idx)
                if idx < 0 or float(score) < self.threshold:
                    continue
                label = self.labels[idx]
                entity_type = self.entity_types[idx]
                key = (label, entity_type)
                if key in seen:
                    continue
                result.append({"entity": label, "entity_type": entity_type})
                seen.add(key)
                if len(result) >= limit:
                    break
            if search_limit == len(self.aliases):
                break
            search_limit = min(search_limit * 2, len(self.aliases))
        return result

    def add_entity(
        self,
        entity_name: str,
        alias: str,
        entity_type: str,
        sample_id: Optional[str] = None,
    ):
        entity_name = self.sanitize_string(entity_name)
        alias = self.sanitize_string(alias)
        if not entity_name or not alias or not entity_type:
            return

        alias_key = (_normalize_key(alias), entity_type)
        if self.alias_to_label.get(alias_key) == entity_name:
            return

        self.alias_to_label[alias_key] = entity_name
        self.aliases.append(alias)
        self.labels.append(entity_name)
        self.entity_types.append(entity_type)

        embedding = self.get_embedding(alias).reshape(1, -1)
        if self.entity_aliases_index is None:
            self.embedding_dim = int(embedding.shape[1])
            self.entity_aliases_index = self.faiss.IndexFlatIP(self.embedding_dim)
        self.entity_aliases_index.add(embedding)

    def get_embedding(self, text: str):
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self.np.asarray(embedding[0], dtype="float32")

    @staticmethod
    def sanitize_string(text: str) -> str:
        return _sanitize_string(text)
