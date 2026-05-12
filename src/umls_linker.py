"""UMLS REST API entity linker — maps entity strings to UMLS CUIs."""

import time
import requests
import logging
from typing import Optional, Tuple

from .config import (
    UMLS_API_BASE,
    UMLS_PAGE_SIZE,
    UMLS_RATE_LIMIT_DELAY,
    UMLS_SOURCE_VOCABS,
    UMLS_TIMEOUT_SEC,
)

logger = logging.getLogger("UMLSLinker")
UMLS_MAX_RETRIES = 3
UMLS_RETRY_DELAY_SEC = 1.0


class UMLSLinker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: dict = {}
        self._last_call = 0.0

    def _wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < UMLS_RATE_LIMIT_DELAY:
            time.sleep(UMLS_RATE_LIMIT_DELAY - elapsed)
        self._last_call = time.time()

    def lookup(self, entity: str, type_hint: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """Return (CUI, preferred_name) or (None, None).

        ``type_hint`` is accepted for compatibility with the pipeline call site,
        but UMLS candidate selection stays top-1. Entity deduplication is handled
        by the ontology-aware structured aligner.
        """
        key = entity.lower().strip()
        if key in self._cache:
            return self._cache[key]

        for attempt in range(1, UMLS_MAX_RETRIES + 1):
            self._wait()
            try:
                r = requests.get(
                    f"{UMLS_API_BASE}/search/current",
                    params={
                        "string": entity,
                        "apiKey": self.api_key,
                        "returnIdType": "concept",
                        "searchType": "normalizedWords",
                        "sabs": UMLS_SOURCE_VOCABS,
                        "pageSize": UMLS_PAGE_SIZE,
                    },
                    timeout=UMLS_TIMEOUT_SEC,
                )
                r.raise_for_status()
                results = [x for x in r.json().get("result", {}).get("results", [])
                           if x.get("ui") not in ("NONE", "", None)]

                if results:
                    pick = results[0]
                    cui = pick["ui"]
                    name = pick.get("name", entity)
                    self._cache[key] = (cui, name)
                    return cui, name
                break
            except requests.exceptions.RequestException as e:
                if attempt < UMLS_MAX_RETRIES:
                    logger.warning(
                        "UMLS lookup failed for '%s' on attempt %d/%d: %s",
                        entity, attempt, UMLS_MAX_RETRIES, e,
                    )
                    time.sleep(UMLS_RETRY_DELAY_SEC * attempt)
                    continue
                logger.warning("UMLS lookup failed for '%s': %s", entity, e)
            except Exception as e:
                logger.warning("UMLS lookup failed for '%s': %s", entity, e)
                break

        self._cache[key] = (None, None)
        return None, None
