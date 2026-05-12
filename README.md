# text2kg

A simple medical KG extraction pipeline built on top of [Wikontic](https://github.com/screemix/Wikontic)'s ontology-aware online generation loop
(LLM extraction → ontology alignment → structured entity refinement → graph update).
Wikontic's MongoDB vector search is replaced with an online local FAISS index.

```
StatPearls nurse articles → LLM-extracted triples → UMLS + structured KG refinement
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
LLM extracts, we look up the best-matching UMLS concept and keep the CUI
alongside the generated triplet. The UMLS preferred name is used as the first
normalized label before Wikontic-style structured refinement.

## Pipeline

```
nurse-article-*.nxml
        │   src/parse_nxml.py
        ▼
  Article(title, sections=[Section(title, text), ...])
        │   src/extractor.py          (Claude API, Wikontic prompts)
        ▼
  raw triples: [{subject, relation, object, ...}]
        │   src/structured_aligner.py
        ▼
  ontology-aligned types and relation
        │   src/umls_linker.py        (UMLS REST API)
        ▼
  (CUI, preferred_name) for each entity
        │   src/structured_aligner.py (typed FAISS entity retrieval)
        ▼
  refined entity names
        │   src/structured_aligner.py
        ▼
  Wikontic-style JSONL outputs + entities.json + relations.json
```

## Layout

```
text2kg_pipeline/
├── pipeline.py            # entry point
├── scripts/
│   └── visualize_graph.py # PyVis graph export from local JSONL files
├── src/
│   ├── prompts/           # Wikontic-style prompts
│   ├── parse_nxml.py      # BITS/JATS → Article/Section dataclasses
│   ├── extractor.py       # Anthropic client + extraction logic
│   ├── structured_inference.py
│   ├── structured_aligner.py
│   ├── umls_linker.py     # UMLS REST API wrapper, in-memory cache
│   └── build_umls_ontology.py
├── requirements.txt
└── README.md
```

Primary online outputs:

- `initial_triplets.jsonl`
- `final_triplets.jsonl`
- `filtered_triplets.jsonl`
- `ontology_filtered_triplets.jsonl`

## Run

```bash
pip install -r requirements.txt
# fill .env with ANTHROPIC_API_KEY, UMLS_USERNAME, UMLS_API_KEY
python pipeline.py --limit 3      # dry run
python pipeline.py                # full run on all 117 articles
python pipeline.py --resume       # resume after interruption
```

## Visualize

Generate a PyVis HTML graph from local JSONL outputs:

```bash
python scripts/visualize_graph.py \
  --input output/dry_run_1_faiss_online_16000/final_triplets.jsonl \
  --output output/dry_run_1_faiss_online_16000/graph.html
```

For a smaller graph:

```bash
python scripts/visualize_graph.py \
  --input output/dry_run_1_faiss_online_16000/final_triplets.jsonl \
  --output output/dry_run_1_faiss_online_16000/graph_min_degree_2.html \
  --min_degree 2
```

The visualizer reads local KG snapshots such as `final_triplets.jsonl`

## Results

_TBD — to be filled in after the first full run._
