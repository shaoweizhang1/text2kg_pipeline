"""UMLS REST API entity linker — maps entity strings to UMLS CUIs."""

import time
import requests
import logging
from typing import Optional, Tuple

logger = logging.getLogger("UMLSLinker")
BASE = "https://uts-ws.nlm.nih.gov/rest"
RATE_LIMIT_DELAY = 0.1  # 100ms between calls (~10 req/s)


class UMLSLinker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: dict = {}
        self._last_call = 0.0

    def _wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_call = time.time()

    def lookup(self, entity: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (CUI, preferred_name) or (None, None)."""
        key = entity.lower().strip()
        if key in self._cache:
            return self._cache[key]

        self._wait()
        try:
            r = requests.get(
                f"{BASE}/search/current",
                params={
                    "string": entity,
                    "apiKey": self.api_key,
                    "returnIdType": "concept",
                    "pageSize": 1,
                },
                timeout=15,
            )
            r.raise_for_status()
            results = r.json().get("result", {}).get("results", [])
            if results and results[0].get("ui") not in ("NONE", "", None):
                cui = results[0]["ui"]
                name = results[0].get("name", entity)
                self._cache[key] = (cui, name)
                return cui, name
        except requests.exceptions.Timeout:
            logger.warning("UMLS timeout for '%s', retrying once", entity)
            time.sleep(1.0)
            return self.lookup(entity)
        except Exception as e:
            logger.warning("UMLS lookup failed for '%s': %s", entity, e)

        self._cache[key] = (None, None)
        return None, None
