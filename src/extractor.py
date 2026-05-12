"""
LLM triple extractor using Claude API.
Adapted from Wikontic's openai_utils.py — same prompts, same logic, different client.
"""

import json
import time
import logging
from typing import List, Dict, Union

import anthropic

from .config import LLM_MAX_TOKENS, LLM_MODEL, LLM_RATE_LIMIT_DELAY, PROMPT_DIR

logger = logging.getLogger("Extractor")


class Extractor:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.prompts = {
            "triplet_extraction": (PROMPT_DIR / "triplet_extraction.txt").read_text(),
            "subject_ranker":     (PROMPT_DIR / "rank_subject_names.txt").read_text(),
            "object_ranker":      (PROMPT_DIR / "rank_object_names.txt").read_text(),
            "relation_ranker":    (PROMPT_DIR / "rank_relation_names.txt").read_text(),
            "map_type":           (PROMPT_DIR / "map_type_to_ontology.txt").read_text(),
            "map_relation":       (PROMPT_DIR / "map_relation_to_ontology.txt").read_text(),
        }
        self._last_call = 0.0

    def _wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < LLM_RATE_LIMIT_DELAY:
            time.sleep(LLM_RATE_LIMIT_DELAY - elapsed)
        self._last_call = time.time()

    def _call(self, system: str, user: str, as_json: bool = True) -> Union[dict, str]:
        self._wait()
        chunks = []
        with self.client.messages.stream(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)
        content = "".join(chunks).strip()
        if not as_json:
            return content
        # raw_decode tolerates trailing chatter after a valid JSON prefix.
        decoder = json.JSONDecoder()
        candidates = [content]
        for i, ch in enumerate(content):
            if ch in "{[":
                candidates.append(content[i:])
                break
        for c in candidates:
            try:
                obj, _ = decoder.raw_decode(c)
                return obj
            except json.JSONDecodeError:
                continue
        logger.warning("Could not parse JSON from LLM output: %s", content[:200])
        return {}

    def extract_triplets(self, text: str) -> List[Dict]:
        result = self._call(
            system=self.prompts["triplet_extraction"],
            user=f'Text: "{text}"',
        )
        if isinstance(result, dict) and "triplets" in result:
            return result["triplets"]
        if isinstance(result, list):
            return result
        return []

    def refine_entity(self, text: str, triplet: Dict, candidates: List[str], is_object: bool) -> str:
        role = "object" if is_object else "subject"
        original = triplet[role]
        prompt_key = "object_ranker" if is_object else "subject_ranker"
        triplet_str = json.dumps({k: triplet[k] for k in ["subject", "relation", "object"]})
        result = self._call(
            system=self.prompts[prompt_key],
            user=(
                f'Text: "{text}"\n'
                f"Extracted Triplet: {triplet_str}\n"
                f"Original {role.capitalize()}: {original}\n"
                f"Candidate {role.capitalize()}s: {json.dumps(candidates)}"
            ),
            as_json=False,
        )
        return result.strip()

    def refine_relation(self, text: str, triplet: Dict, candidates: List[str]) -> str:
        original = triplet["relation"]
        triplet_str = json.dumps({k: triplet[k] for k in ["subject", "relation", "object"]})
        result = self._call(
            system=self.prompts["relation_ranker"],
            user=(
                f'Text: "{text}"\n'
                f"Extracted Triplet: {triplet_str}\n"
                f"Original Relation: {original}\n"
                f"Candidate Relations: {json.dumps(candidates)}"
            ),
            as_json=False,
        )
        return result.strip()

    def pick_semantic_type(self, freeform: str, candidates_block: str):
        """Map a freeform type string to a UMLS Semantic Type T-code."""
        result = self._call(
            system=self.prompts["map_type"],
            user=(
                f"Freeform type: {freeform}\n\n"
                f"Candidates:\n{candidates_block}"
            ),
            as_json=True,
        )
        if isinstance(result, dict):
            t = result.get("t_code")
            if isinstance(t, str) and t.strip() and t.strip().lower() != "none":
                return t.strip()
        return None

    def pick_relation_from_candidates(self, text: str, triplet: Dict, candidates_block: str):
        """Pick a UMLS SN relation label from a constraint-filtered candidate list."""
        triplet_str = json.dumps(
            {k: triplet.get(k) for k in ["subject", "relation", "object", "subject_type", "object_type"]},
            ensure_ascii=False,
        )
        result = self._call(
            system=self.prompts["map_relation"],
            user=(
                f'Text: "{text}"\n'
                f"Extracted Triplet: {triplet_str}\n"
                f"Candidate Relations:\n{candidates_block}"
            ),
            as_json=True,
        )
        if isinstance(result, dict):
            r = result.get("relation")
            if isinstance(r, str) and r.strip() and r.strip().lower() != "none":
                return r.strip()
        return None
