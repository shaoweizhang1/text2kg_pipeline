"""Central configuration for the text2kg pipeline.

All paths and tunable constants live here so they can be changed in one place.
Modules import named values from this file rather than redeclaring or
hard-coding them.
"""

from pathlib import Path

# ---- Paths --------------------------------------------------------------

SRC_DIR       = Path(__file__).resolve().parent
PROJECT_ROOT  = SRC_DIR.parent

# Assets that ship with the package live inside src/.
PROMPT_DIR    = SRC_DIR / "prompts"
ONTOLOGY_DIR  = SRC_DIR / "ontology_mappings"

# External corpus + runtime output live at project root.
DATA_DIR      = PROJECT_ROOT / "statpearls_NBK430685"
OUTPUT_DIR    = PROJECT_ROOT / "output"


# ---- LLM (Anthropic) ----------------------------------------------------

LLM_MODEL              = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS         = 16000
LLM_RATE_LIMIT_DELAY   = 0.5   # seconds between LLM calls


# ---- UMLS UTS API -------------------------------------------------------

UMLS_API_BASE          = "https://uts-ws.nlm.nih.gov/rest"
UMLS_RATE_LIMIT_DELAY  = 0.1   # seconds between UMLS calls (~10 req/s)
UMLS_PAGE_SIZE         = 5
UMLS_TIMEOUT_SEC       = 15

# Restrict entity linking to clean clinical source vocabularies (drops noisy
# ICD-10-PCS / CPT / PsycINFO results).
UMLS_SOURCE_VOCABS     = "SNOMEDCT_US,MSH,ICD10CM,RXNORM,NCI"


# ---- Wikontic structured aligner ---------------------------------------

STRUCTURED_ALIGNER_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
STRUCTURED_ALIGNER_TOP_K      = 10
STRUCTURED_ALIGNER_THRESHOLD  = 0.78
