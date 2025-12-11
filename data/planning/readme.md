# Search strategy for planning-level studies

This document records the search terms and database-specific queries that we
used to identify planning-level studies on power systems with electric
vehicles (EVs). The focus of these searches was on planning, expansion,
siting/sizing and related optimization / learning methods.

Across all databases we consistently relied on four conceptual term blocks:

- **EV block**
- **Power system block**
- **Planning / siting / sizing block**
- **Methods block**

In each platform, these blocks were combined with logical AND/OR operators as
described below. The queries documented here correspond exactly to those that
were executed when building the planning-level corpus.

---

## 1. Conceptual term blocks actually used

### 1.1 EV block

To capture different ways EVs appear in titles and abstracts, we used:

```text
("electric vehicle*" OR "plug-in vehicle*" OR "plug-in hybrid" OR EV)
```

### 1.2 Power system block

To restrict the context to electric power systems, we used:

```text
("power system*" OR "electric grid" OR "distribution network" OR "transmission network" OR "smart grid")
```

### 1.3 Planning / siting / sizing block

To retrieve planning, expansion, and siting/sizing problems (e.g., charging
stations, network reinforcement, generation expansion), we used:

```text
(planning OR "expansion planning" OR "generation expansion" OR
 "capacity expansion" OR "transmission expansion" OR "distribution expansion" OR
 "unit commitment" OR "generation scheduling" OR investment OR "long-term" OR
 siting OR "siting and sizing" OR sizing)
```

In practice, this block allowed us to cover:

- Broad mentions of `planning` in titles/abstracts.
- Long-term investment problems via `"expansion planning"`, `"generation expansion"`,
  `"capacity expansion"`.
- Day-ahead / medium-term planning formulations through `"unit commitment"` and
  `"generation scheduling"`.
- Location and capacity decisions (charging infrastructure, network assets)
  via `siting`, `"siting and sizing"`, and `sizing`.

### 1.4 Methods block

To focus on optimization and learning methods relevant for planning-level
decisions, we used the following block:

```text
("stochastic programming" OR "robust optimization" OR scenario* OR "chance-constrained"
 OR metaheuristic* OR "genetic algorithm" OR GA OR "particle swarm" OR PSO
 OR "differential evolution" OR "simulated annealing"
 OR "machine learning" OR "reinforcement learning" OR "deep reinforcement learning"
 OR optimization OR MILP OR MINLP)
```

This block effectively captured:

- Stochastic formulations (`"stochastic programming"`, `scenario*`,
  `"chance-constrained"`).
- Robust formulations (`"robust optimization"`).
- Metaheuristic approaches (`metaheuristic*`, `"genetic algorithm"`, `PSO`,
  `"differential evolution"`, `"simulated annealing"`).
- Learning-based approaches (`"machine learning"`, `"reinforcement learning"`,
  `"deep reinforcement learning"`).
- General optimization formulations via `optimization`, `MILP`, `MINLP`.

---

## 2. Core planning query (generic form)

Before adapting to each database, we first defined a generic core query that
combines the four blocks above. All platform-specific queries are direct
instantiations of this expression:

```text
("electric vehicle*" OR "plug-in vehicle*" OR "plug-in hybrid" OR EV)
AND
("power system*" OR "electric grid" OR "distribution network" OR "transmission network" OR "smart grid")
AND
(planning OR "expansion planning" OR "generation expansion" OR
 "capacity expansion" OR "transmission expansion" OR "distribution expansion" OR
 "unit commitment" OR "generation scheduling" OR investment OR "long-term" OR
 siting OR "siting and sizing" OR sizing)
AND
("stochastic programming" OR "robust optimization" OR scenario* OR "chance-constrained"
 OR metaheuristic* OR "genetic algorithm" OR GA OR "particle swarm" OR PSO
 OR "differential evolution" OR "simulated annealing"
 OR "machine learning" OR "reinforcement learning" OR "deep reinforcement learning"
 OR optimization OR MILP OR MINLP)
```

When volume was too high (e.g., WoS returning tens of thousands), we also ran
a **narrow variant without the methods block**, keeping only EV + power system
+ planning/siting terms. The WoS query below reflects that narrower variant.

---

## 3. Scopus search

### 3.1 Field and filters used

In Scopus we searched in:

- **Field**: `TITLE-ABS-KEY` (title, abstract, keywords).

Filters applied during the search:

- Publication year: 2015–2025
- Document type: Article
- Language: English
- Subject area: Energy; Engineering (and related), excluding unrelated areas (e.g., Medicine, Chemistry)

### 3.2 Scopus query executed

The exact Scopus query we ran (narrow variant, aligned with the WoS run) was:

```text
TITLE-ABS-KEY(
  ("electric vehicle*" OR "plug-in vehicle*" OR "plug-in hybrid" OR EV)
  AND
  ("power system*" OR "electric grid" OR "distribution network" OR "transmission network" OR "smart grid")
  AND
  ("expansion planning" OR "generation expansion" OR "capacity expansion"
   OR "transmission expansion" OR "distribution expansion"
   OR "unit commitment" OR "generation scheduling"
   OR siting OR "siting and sizing" OR sizing OR planning)
)
```

The resulting records were exported as CSV and RIS files and saved under
`data/planning/`.

---

## 4. Web of Science (Core Collection) search

### 4.1 Field and filters used

In Web of Science Core Collection we searched using the topic field and set
the main filters directly in the Advanced Search expression, then refined in
the interface. Configuration:

- **Field**: `TS=` (Topic: title, abstract, author keywords, Keywords Plus).

Filters enforced in the query:

- Timespan: 2015–2025 (`PY=2015-2025`)
- Document type: Article (`DT=(Article)`)
- Web of Science Categories (OR):
  - `ENGINEERING ELECTRICAL ELECTRONIC`
  - `COMPUTER SCIENCE INFORMATION SYSTEMS`
  - `TRANSPORTATION SCIENCE TECHNOLOGY`

Filters applied via the interface (Refine results):

- Language: English
- Research Areas: `ENGINEERING`
- Citation Topics Meso: `4.18 Power Systems & Electric Vehicles`
- Citation Topics Micro: `4.18.204 Smart Grid Optimization` OR `4.18.788 Electric Vehicles` OR `4.18.2795 Real-time Power Simulation`

### 4.2 Web of Science query executed

The exact WoS Advanced Search query we ran was:

```text
TS=(
  ("electric vehicle*" OR "plug-in vehicle*" OR "plug-in hybrid" OR EV)
  AND
  ("power system*" OR "electric grid" OR "distribution network" OR "transmission network" OR "smart grid")
  AND
  ("expansion planning" OR "generation expansion" OR "capacity expansion"
   OR "transmission expansion" OR "distribution expansion"
   OR "unit commitment" OR "generation scheduling"
   OR siting OR "siting and sizing" OR sizing OR planning)
)
AND PY=2015-2025
AND DT=(Article)
AND WC=(ENGINEERING ELECTRICAL ELECTRONIC)
```

Language was set to English via the interface. The exported CSV and RIS
files from the refined result set were stored in `data/planning/`.

---

## 5. IEEE Xplore search

### 5.1 Field and filters used

In IEEE Xplore we searched in the metadata fields (Title, Abstract, Index
Terms when available) and applied the following filters:

- Content type: Journals & Magazines (journal articles)
- Publication years: 2015–2025
- Language: English
- Optional subject filters (when available): Power & Energy, Transportation, Smart Grid

### 5.2 IEEE Xplore query executed

The exact IEEE Xplore query we ran (narrow variant, aligned with the WoS run) was:

```text
("electric vehicle" OR "plug-in vehicle" OR "plug-in hybrid" OR EV)
AND
("power system" OR "electric grid" OR "distribution network" OR "transmission network" OR "smart grid")
AND
("expansion planning" OR "generation expansion" OR "capacity expansion"
 OR "transmission expansion" OR "distribution expansion"
 OR "unit commitment" OR "generation scheduling"
 OR siting OR "siting and sizing" OR sizing OR planning)
```
The resulting metadata were exported and saved under `data/planning/`.

---

## 6. ScienceDirect (Elsevier) search

### 6.1 Field and filters used

In ScienceDirect we searched within:

- **Search in**: Title, abstract, keywords.

Filters applied:

- Year: 2015–2025
- Article type: Research articles
- Subject area: Energy, Engineering (used to refine results)
- Language: English

### 6.2 ScienceDirect queries executed (split due to boolean limit)

ScienceDirect impõe limite de ~8 conectores booleanos por campo. Rodamos buscas
mais curtas e depois exportamos e deduplicamos:

- Q1 (expansão / siting)
  ```text
  ("electric vehicle" OR EV)
  AND ("power system" OR "electric grid")
  AND ("expansion planning" OR "capacity expansion" OR siting)
  ```

- Q2 (UC / scheduling / planning geral)
  ```text
  ("electric vehicle" OR EV)
  AND ("power system" OR "smart grid")
  AND ("unit commitment" OR "generation scheduling" OR planning)
  ```

- Q3 (transmissão / distribuição)
  ```text
  ("electric vehicle" OR EV)
  AND ("power system" OR "electric grid")
  AND ("transmission expansion" OR "distribution expansion")
  ```

- Q4 (sizing)
  ```text
  ("electric vehicle" OR EV)
  AND ("power system" OR "smart grid")
  AND ("siting and sizing" OR sizing)
  ```

Filtros: 2015–2025; Research articles; Subject area Energy/Engineering; English.
As exportações (CSV e BibTeX/RIS) de cada busca foram armazenadas em
`data/planning/` e deduplicadas posteriormente.

---

## 7. Reproducibility notes

For transparency and reproducibility in the review paper, this file preserves
the exact queries and filters used in each database. In addition:

- For each database, we kept a record of:
  - The query string shown above.
  - The date when the search was executed.
  - The filters applied (years, document type, language, subject area).
- All exported search results (CSV and RIS/BibTeX files) from these planning
  searches are stored in this repository under:
  - `data/planning/`
- Post-processing steps recorded in the repo:
  - `data/planning/merge_dedup.py`: merges all CSV/BibTeX exports, normalizes
    fields, deduplicates (DOI or title+year), and writes `merged_dedup.csv`.
    The script reports how many raw records were loaded and how many were
    removed by deduplication.
  - `data/planning/prioritize.py`: reads `merged_dedup.csv`, assigns a
    `priority_score`/`priority_bucket` based on keywords and recency, and
    writes `merged_prioritized.csv` sorted by priority and year. Use this
    file to start manual screening (title/abstract) from the highest scores.
- Any future refinements to the planning-level search will be versioned and
  briefly documented in this file so that the search process remains
  reproducible.



