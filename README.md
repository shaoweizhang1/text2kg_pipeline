# text2kg

A simple medical KG extraction pipeline built on top of [Wikontic](https://github.com/screemix/Wikontic)'s 3-stage idea
(LLM extraction → ontology alignment → entity dedup).

```
StatPearls nurse articles → LLM-extracted triples → UMLS-normalized KG
```

## Data

**StatPearls** (NCBI Bookshelf, collection NBK430685) — physician/nurse-authored
clinical reference articles. We use only the 117 `nurse-article-*.nxml` files,
which are shorter and more clinically actionable (assessment / interventions /
monitoring) than the full physician encyclopedia. Coverage spans the major
specialties: cardiology, neurology, endocrinology, pulmonology, etc.

Format is BITS/JATS XML; `src/parse_nxml.py` flattens each article into
`(section_title, section_text)` pairs.

## Ontology

**UMLS Metathesaurus** (2026AA), accessed via the UTS REST API
(`https://uts-ws.nlm.nih.gov/rest/search/current`). For each entity string the
LLM extracts, we look up the best-matching UMLS concept and replace the surface
form with the preferred name; the CUI is kept alongside.

Two entities that resolve to the same CUI are treated as the same node
(e.g. "heart failure" and "cardiac failure" both → `C0018801`). This is what
gives us deduplication without needing a local embedding model.

## Pipeline

```
nurse-article-*.nxml
        │   src/parse_nxml.py
        ▼
  Article(title, sections=[Section(title, text), ...])
        │   src/extractor.py          (Claude API, Wikontic prompts)
        ▼
  raw triples: [{subject, relation, object, ...}]
        │   src/umls_linker.py        (UMLS REST API)
        ▼
  (CUI, preferred_name) for each entity
        │   src/aligner.py            (CUI-based dedup)
        ▼
  output/triplets_enriched.jsonl, entities.json, relations.json
```

## Layout

```
text2kg_pipeline/
├── pipeline.py            # entry point
├── prompts/               # Wikontic system prompts (verbatim)
├── src/
│   ├── parse_nxml.py      # BITS/JATS → Article/Section dataclasses
│   ├── extractor.py       # Anthropic client + extraction logic
│   ├── umls_linker.py     # UMLS REST API wrapper, in-memory cache
│   └── aligner.py         # CUI-based entity/relation normalization + dedup
├── requirements.txt
└── README.md
```

## Run

```bash
pip install -r requirements.txt
# fill .env with ANTHROPIC_API_KEY, UMLS_USERNAME, UMLS_API_KEY
python pipeline.py --limit 3      # dry run
python pipeline.py                # full run on all 117 articles
python pipeline.py --resume       # resume after interruption
```

## Results

_TBD — to be filled in after the first full run._
