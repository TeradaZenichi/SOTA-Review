# Search strategy for distribution-level EV siting/sizing studies

This document defines the **updated search strategy** we will use to rebuild
the planning corpus. Compared to the previous version, the scope is now more
focused and coherent:

> We focus on planning, siting and sizing of EV-related charging
> infrastructure (public/semi-public and smart-home / residential) in **power
> distribution networks**, explicitly considering network constraints (power
> flow, voltages, feeder/line capacities), while keeping a broad view of
> modeling / solution methods to support a taxonomic analysis.

Across all databases, we rely on a set of conceptual term blocks:

- **EV block**
- **Distribution network block**
- **EV infrastructure / siting/sizing block** (public charging and smart-home contexts)
- **Network modeling block** (power flow, voltages, hosting capacity, etc.)
- **Methods block** (broad, to capture different optimization / learning families)

These blocks are combined with logical AND/OR operators, with minor
adaptations to each platform. This file describes the *planned* queries; once
the new searches are executed, we will record run dates and any additional
refinements here.

---

## 1. Conceptual term blocks (new scope)

### 1.1 EV block

To capture different ways EVs appear in titles and abstracts:

```text
("electric vehicle*" OR "plug-in vehicle*" OR "plug-in hybrid" OR EV)
```

### 1.2 Distribution network block

To restrict the physical context to power **distribution** systems (LV/MV
grids, feeders):

```text
("distribution network" OR "distribution networks" OR
 "distribution system" OR "distribution systems" OR
 "distribution grid" OR "distribution feeder*" OR "radial feeder*" OR
 "low-voltage network" OR "LV network" OR "low voltage network" OR
 "medium-voltage network" OR "MV network" OR "medium voltage network")
```

This block intentionally excludes generic "power system"/"transmission"
terms so that we focus on works that clearly operate at the distribution
level.

### 1.3 EV infrastructure / siting / sizing block

To focus on planning, siting and sizing decisions for EV-related assets,
covering both public/semi-public infrastructure and smart-home / residential
contexts:

```text
(
  "charging station" OR "charging stations" OR
  "charging infrastructure" OR "charging facility" OR "charging facilities" OR
  "charging hub" OR "charging hubs" OR
  "electric vehicle supply equipment" OR EVSE OR EVCS OR
  "home charging" OR "residential charging" OR
  "home energy management" OR "smart home" OR "smart-home" OR
  "home battery" OR "residential storage" OR
  siting OR "siting and sizing" OR sizing OR
  "location-allocation" OR "location allocation" OR
  "facility location" OR placement OR allocation OR
  "infrastructure planning" OR "network planning"
)
```

Here we deliberately avoid terms tied to **generation/transmission expansion**
(`"generation expansion"`, `"transmission expansion"`) and **unit
commitment**/`"generation scheduling"`, since those are now out of scope.

### 1.4 Methods block

To keep a wide range of modeling and solution approaches (for taxonomy), we
use a broad methods block:

```text
(
  "stochastic programming" OR "robust optimization" OR scenario* OR "chance-constrained" OR
  optimization OR "mixed-integer" OR MILP OR MINLP OR
  metaheuristic* OR heuristic* OR
  "genetic algorithm" OR GA OR "particle swarm" OR PSO OR
  "differential evolution" OR "simulated annealing" OR
  "multi-objective" OR "multiobjective" OR
  "machine learning" OR "reinforcement learning" OR "deep reinforcement learning"
)
```

This block is not meant to be exclusive; it just ensures that retrieved
papers do include some form of explicit optimization, heuristic search or
learning-based decision model.

### 1.5 Network modeling block

To focus on studies that explicitly model the electrical network (power
flows, voltages, capacity, reinforcement, hosting capacity):

```text
(
  "power flow" OR "load flow" OR
  "power-flow" OR "load-flow" OR
  "voltage profile" OR "voltage constraint" OR "voltage constraints" OR
  "hosting capacity" OR "network reinforcement" OR
  "line capacity" OR "feeder capacity"
)
```

---

## 2. Generic core query (to be specialized per database)

The generic expression that guides all database-specific queries is:

```text
("electric vehicle*" OR "plug-in vehicle*" OR "plug-in hybrid" OR EV)
AND
"distribution network" OR "distribution networks" OR
 "distribution system" OR "distribution systems" OR
 "distribution grid" OR "distribution feeder*" OR "radial feeder*" OR
 "low-voltage network" OR "LV network" OR "low voltage network" OR
 "medium-voltage network" OR "MV network" OR "medium voltage network")
AND
(
  "charging station" OR "charging stations" OR
  "charging infrastructure" OR "charging facility" OR "charging facilities" OR
  "charging hub" OR "charging hubs" OR
  "electric vehicle supply equipment" OR EVSE OR EVCS OR
  "home charging" OR "residential charging" OR
  "home energy management" OR "smart home" OR "smart-home" OR
  "home battery" OR "residential storage" OR
  siting OR "siting and sizing" OR sizing OR
  "location-allocation" OR "location allocation" OR
  "facility location" OR placement OR allocation OR
  "infrastructure planning" OR "network planning"
)
AND
(
  "power flow" OR "load flow" OR
  "power-flow" OR "load-flow" OR
  "voltage profile" OR "voltage constraint" OR "voltage constraints" OR
  "hosting capacity" OR "network reinforcement" OR
  "line capacity" OR "feeder capacity"
)
AND
(
  "stochastic programming" OR "robust optimization" OR scenario* OR "chance-constrained" OR
  optimization OR "mixed-integer" OR MILP OR MINLP OR
  metaheuristic* OR heuristic* OR
  "genetic algorithm" OR GA OR "particle swarm" OR PSO OR
  "differential evolution" OR "simulated annealing" OR
  "multi-objective" OR "multiobjective" OR
  "machine learning" OR "reinforcement learning" OR "deep reinforcement learning"
)
```

Depending on the volume returned in each platform, we may (i) remove the
methods block to obtain a broader initial set, and then (ii) re-apply a
methods filter during manual screening.

---

## 3. Scopus search (new run)

### 3.1 Field and filters to use

- **Field**: `TITLE-ABS-KEY` (title, abstract, keywords).
- Filters during search:
  - Publication year: 2015–2025
  - Document type: Article
  - Language: English
  - Subject area: Energy; Engineering (and related), excluding unrelated
    areas (e.g., Medicine, Chemistry).

### 3.2 Proposed Scopus query

```text
TITLE-ABS-KEY(
  ("electric vehicle*" OR "plug-in vehicle*" OR "plug-in hybrid" OR EV)
  AND
  ("distribution network" OR "distribution networks" OR
   "distribution system" OR "distribution systems" OR
   "distribution grid" OR "distribution feeder*" OR "radial feeder*" OR
   "low-voltage network" OR "LV network" OR "low voltage network" OR
   "medium-voltage network" OR "MV network" OR "medium voltage network")
  AND
  (
    "charging station" OR "charging stations" OR
    "charging infrastructure" OR "charging facility" OR "charging facilities" OR
    "charging hub" OR "charging hubs" OR
     "electric vehicle supply equipment" OR EVSE OR EVCS OR
     "home charging" OR "residential charging" OR
     "home energy management" OR "smart home" OR "smart-home" OR
     "home battery" OR "residential storage" OR
     siting OR "siting and sizing" OR sizing OR
     "location-allocation" OR "location allocation" OR
     "facility location" OR placement OR allocation OR
     "infrastructure planning" OR "network planning"
    )
    AND
    (
      "power flow" OR "load flow" OR
      "power-flow" OR "load-flow" OR
      "voltage profile" OR "voltage constraint" OR "voltage constraints" OR
      "hosting capacity" OR "network reinforcement" OR
      "line capacity" OR "feeder capacity"
  )
)
```

If this still returns an unmanageable number of records, we will:

1. Add the methods block at the end with `AND (...)`, or
2. Refine by subject areas / source titles in the Scopus interface.

---

## 4. Web of Science (Core Collection) search (new run)

### 4.1 Field and filters to use

- **Field**: `TS=` (Topic: title, abstract, author keywords, Keywords Plus).
- Filters enforced in the query:
  - Timespan: 2015–2025 (`PY=2015-2025`)
  - Document type: Article (`DT=(Article)`)
  - Web of Science Categories (OR), e.g.:
    - `ENGINEERING ELECTRICAL ELECTRONIC`
    - `ENERGY FUELS`
    - `TRANSPORTATION SCIENCE TECHNOLOGY`
- Filters via interface (Refine results):
  - Language: English
  - Research Areas: `ENGINEERING`, `ENERGY & FUELS`.

### 4.2 Proposed WoS Advanced Search query

```text
TS=(
  ("electric vehicle*" OR "plug-in vehicle*" OR "plug-in hybrid" OR EV)
  AND
  ("distribution network" OR "distribution networks" OR
   "distribution system" OR "distribution systems" OR
   "distribution grid" OR "distribution feeder*" OR "radial feeder*" OR
   "low-voltage network" OR "LV network" OR "low voltage network" OR
   "medium-voltage network" OR "MV network" OR "medium voltage network")
  AND
  (
    "charging station" OR "charging stations" OR
    "charging infrastructure" OR "charging facility" OR "charging facilities" OR
    "charging hub" OR "charging hubs" OR
    "electric vehicle supply equipment" OR EVSE OR EVCS OR
    "home charging" OR "residential charging" OR
    "home energy management" OR "smart home" OR "smart-home" OR
    "home battery" OR "residential storage" OR
    siting OR "siting and sizing" OR sizing OR
    "location-allocation" OR "location allocation" OR
    "facility location" OR placement OR allocation OR
    "infrastructure planning" OR "network planning"
  )
  AND
  (
    "power flow" OR "load flow" OR
    "power-flow" OR "load-flow" OR
    "voltage profile" OR "voltage constraint" OR "voltage constraints" OR
    "hosting capacity" OR "network reinforcement" OR
    "line capacity" OR "feeder capacity"
  )
)
AND PY=2015-2025
AND DT=(Article)
AND WC=(ENGINEERING ELECTRICAL ELECTRONIC OR ENERGY FUELS OR TRANSPORTATION SCIENCE TECHNOLOGY)
```

If necessary, we may further refine using Citation Topics (e.g., Smart Grid,
EVs) in the interface.

---

## 5. IEEE Xplore search (new run)

### 5.1 Field and filters to use

- Search fields: metadata (Title, Abstract, Index Terms when available).
- Filters:
  - Content type: Journals & Magazines (journal articles)
  - Publication years: 2015–2025
  - Language: English
  - Subject areas (when available): Power & Energy, Transportation, Smart Grid.

### 5.2 Proposed IEEE Xplore query

```text
("electric vehicle" OR "plug-in vehicle" OR "plug-in hybrid" OR EV)
AND
(
  "distribution network" OR "distribution system" OR
  "distribution grid" OR "distribution feeder" OR "radial feeder" OR
  "low-voltage network" OR "LV network" OR
  "medium-voltage network" OR "MV network"
)
AND
(
  "charging station" OR "charging stations" OR
  "charging infrastructure" OR "charging facility" OR
  "charging hub" OR "electric vehicle supply equipment" OR EVSE OR EVCS OR
  "home charging" OR "residential charging" OR
  "home energy management" OR "smart home" OR "smart-home" OR
  "home battery" OR "residential storage"
)
AND
(
  "power flow" OR "load flow" OR
  "power-flow" OR "load-flow" OR
  "voltage profile" OR "voltage constraint" OR "voltage constraints" OR
  "hosting capacity" OR "network reinforcement" OR
  "line capacity" OR "feeder capacity"
)
AND
(
  optimization OR "stochastic programming" OR "robust optimization" OR
  "mixed-integer" OR MILP OR MINLP OR
  metaheuristic* OR heuristic* OR
  "genetic algorithm" OR "particle swarm" OR PSO OR
  "differential evolution" OR "simulated annealing" OR
  "machine learning" OR "reinforcement learning"
)
```

---

## 6. ScienceDirect (Elsevier) search (new run)

### 6.1 Field and filters to use

- **Search in**: Title, abstract, keywords.
- Filters:
  - Year: 2015–2025
  - Article type: Research articles
  - Subject area: Energy, Engineering (for refinement)
  - Language: English

### 6.2 Proposed ScienceDirect queries (respecting boolean limit)

Because ScienceDirect limits the number of boolean operators (AND/OR/NOT) per
field, we use **shorter queries**, each with **no more than 8 connectors**,
and rely on multiple runs to collectively approximate the broader scope.

- Q1 – core distribution + charging station planning (includes home contexts)

  ```text
  ("electric vehicle" OR EV)
  AND "distribution network"
  AND ("charging station" OR "charging infrastructure")
  AND (planning OR siting OR sizing)
  ```

- Q2 – distribution feeders + charging + explicit network modeling

  ```text
  ("electric vehicle" OR EV)
  AND "distribution feeder"
  AND ("charging station" OR "charging hub")
  AND ("power flow" OR "voltage profile" OR "hosting capacity")
  ```

- Q3 – optimization/heuristic formulations for siting/sizing

  ```text
  ("electric vehicle" OR EV)
  AND "distribution network"
  AND ("charging station" OR EVSE)
  AND (optimization OR MILP OR heuristic*)
  ```

The exports (CSV and BibTeX/RIS) from each query will be stored under
`data/planning/` and deduplicated using the existing scripts
(`merge_dedup.py`, `prioritize.py`).

---

## 7. Reproducibility notes for the new search

Once the new database searches are executed, we will **append** the following
information to this file (or a companion log):

- For each database:
  - Exact query string used (possibly adjusted from the proposals above).
  - Date of execution.
  - Filters applied (years, document type, language, subject area, citation
    topics).
- Locations of all exported search results (CSV and RIS/BibTeX files) under
  `data/planning/`.
- Post-processing scripts already in the repo:
  - `data/planning/merge_dedup.py`: merges CSV/BibTeX exports, normalizes
    fields, deduplicates (DOI or title+year), and writes
    `merged_dedup.csv`.
  - `data/planning/prioritize.py`: reads `merged_dedup.csv`, assigns a
    `priority_score` / `priority_bucket` (keywords + recency), and writes
    `merged_prioritized.csv` sorted by priority and year.

This ensures that the **new, narrower distribution-level EV siting/sizing
search** remains fully reproducible in the review paper.



