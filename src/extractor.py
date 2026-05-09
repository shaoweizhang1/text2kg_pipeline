"""
LLM triple extractor using Claude API.
Adapted from Wikontic's openai_utils.py — same prompts, same logic, different client.
"""

import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Union

import anthropic

logger = logging.getLogger("Extractor")

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
MODEL = "claude-haiku-4-5-20251001"
RATE_LIMIT_DELAY = 0.5  # 500ms between LLM calls


class Extractor:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.prompts = {
            "triplet_extraction": (PROMPT_DIR / "triplet_extraction.txt").read_text(),
            "subject_ranker":     (PROMPT_DIR / "rank_subject_names.txt").read_text(),
            "object_ranker":      (PROMPT_DIR / "rank_object_names.txt").read_text(),
            "relation_ranker":    (PROMPT_DIR / "rank_relation_names.txt").read_text(),
        }
        self._last_call = 0.0

    def _wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_call = time.time()

    def _call(self, system: str, user: str, as_json: bool = True) -> Union[dict, str]:
        self._wait()
        msg = self.client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        content = msg.content[0].text.strip()
        if not as_json:
            return content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
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
